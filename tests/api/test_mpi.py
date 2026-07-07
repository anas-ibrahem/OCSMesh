"""
MPI tests for HfunCollector.

Tests that do NOT require mpirun (mock-based) are in TestMPIModeProperty.
Tests that REQUIRE mpirun are in TestMPIWriteHfun.

Running instructions:

    # Unit tests (no mpirun needed)
    python -m pytest tests/api/test_mpi.py::TestMPIModeProperty -v

    # MPI integration tests (require mpirun)
    mpirun -n 2 python -m pytest tests/api/test_mpi.py::TestMPIWriteHfun -v
    mpirun -n 4 python -m pytest tests/api/test_mpi.py::TestMPIWriteHfun -v
"""

import unittest
import tempfile
import shutil
import gc
import os
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.utils import raster_from_numpy
from ocsmesh.hfun.collector import _mpi_dispatch, _meshdata_task_worker

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


def _is_under_mpirun():
    """True only when running under mpirun with >1 rank."""
    if not HAS_MPI:
        return False
    return MPI.COMM_WORLD.Get_size() > 1


def _create_test_rasters(base_dir):
    """Create two synthetic DEM rasters for testing.

    Returns list of Raster objects.
    """
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10  # Values from -10 to 10

    dem1_path = base_dir / 'dem1.tif'
    dem2_path = base_dir / 'dem2.tif'
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1_path), Raster(dem2_path)]


# ════════════════════════════════════════════════════════════════════
# Unit tests — no mpirun needed
# ════════════════════════════════════════════════════════════════════

class TestMPIModeProperty(unittest.TestCase):
    """Test execution_mode='mpi' property behavior without mpirun."""

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.raster_list = _create_test_rasters(self.tdir)

    def tearDown(self):
        self.raster_list = None
        gc.collect()
        try:
            shutil.rmtree(self.tdir)
        except PermissionError:
            pass

    def test_execution_mode_accepts_mpi_string(self):
        """'mpi' is a valid mode string (even if it falls back)."""
        hfun = Hfun(self.raster_list, nprocs=2)
        # When not under mpirun, should warn and fall back
        with self.assertWarns(UserWarning):
            hfun.execution_mode = 'mpi'
        # Should have fallen back to 'parallel'
        self.assertEqual(hfun.execution_mode, 'parallel')

    def test_mpi_mode_fallback_not_under_mpirun(self):
        """Setting mode='mpi' outside mpirun falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = 'mpi'
        self.assertIn('Falling back', str(cm.warning))

    @patch('ocsmesh.hfun.collector._HAS_MPI', False)
    def test_mpi_mode_fallback_no_mpi4py(self):
        """Setting mode='mpi' without mpi4py falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = 'mpi'
        self.assertIn('mpi4py is not installed', str(cm.warning))
        self.assertEqual(hfun.execution_mode, 'parallel')

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertRaises(ValueError):
            hfun.execution_mode = 'distributed'

    # def test_pid_guard_prevents_foreign_cleanup(self):
    #     """__del__ with foreign PID does not delete _work_dir."""
    #     hfun = Hfun(self.raster_list, nprocs=2)
    #     work_dir = hfun._work_dir
    #     self.assertTrue(os.path.exists(work_dir))
    #     self.assertEqual(hfun._creator_pid, os.getpid())

    #     # Simulate foreign PID (MPI worker rank)
    #     hfun._creator_pid = -1
    #     del hfun
    #     gc.collect()
    #     self.assertTrue(os.path.exists(work_dir),
    #                     "PID guard failed: foreign PID deleted _work_dir!")
    #     # Manual cleanup
    #     shutil.rmtree(work_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
# MPI integration tests — require mpirun
# ════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_MPI and _is_under_mpirun(),
                     "Requires mpirun with >1 rank")
class TestMPIWriteHfun(unittest.TestCase):
    """Test _calculate_and_write_hfun_to_disk under MPI.

    Run with: mpirun -n 2 python -m pytest tests/api/test_mpi.py::TestMPIWriteHfun -v
    """

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        # Only rank 0 creates test data — broadcast path to all ranks
        if self.rank == 0:
            self.tdir = Path(tempfile.mkdtemp())
            self.raster_list = _create_test_rasters(self.tdir)
        else:
            self.tdir = None
            self.raster_list = None

        self.tdir = comm.bcast(self.tdir, root=0)

    def tearDown(self):
        comm = MPI.COMM_WORLD
        comm.Barrier()
        self.raster_list = None
        gc.collect()
        if self.rank == 0:
            try:
                shutil.rmtree(self.tdir)
            except (PermissionError, FileNotFoundError):
                pass

    def test_serial_vs_mpi_write_hfun_equivalence(self):
        """Numerical equivalence: serial meshdata == MPI meshdata.

        Only rank 0 creates the Hfun objects and calls meshdata().
        Worker ranks participate in _mpi_dispatch collective ops.
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            # ── SERIAL baseline ──
            hfun_serial = Hfun(self.raster_list, nprocs=2,
                               hmin=10, hmax=1000)
            hfun_serial.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_serial.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_serial = hfun_serial.meshdata()
            values_serial = meshdata_serial.values

            # ── MPI execution ──
            hfun_mpi = Hfun(self.raster_list, nprocs=2,
                            hmin=10, hmax=1000)
            hfun_mpi.execution_mode = 'mpi'
            hfun_mpi.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_mpi.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_mpi = hfun_mpi.meshdata()
            values_mpi = meshdata_mpi.values

            # ── COMPARE ──
            # Node count within 1%
            self.assertAlmostEqual(
                len(values_serial), len(values_mpi),
                delta=len(values_serial) * 0.01)

            # Statistical properties must be nearly identical
            npt.assert_allclose(
                np.min(values_serial), np.min(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.max(values_serial), np.max(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.mean(values_serial), np.mean(values_mpi), rtol=1e-5)
        else:
            # Worker ranks: meshdata() on Rank 0 calls _mpi_dispatch internally.
            # Workers must participate in the collective operation.
            _mpi_dispatch(None, _meshdata_task_worker)

        comm.Barrier()

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output."""
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            hfun = Hfun(self.raster_list, nprocs=1,
                        hmin=10, hmax=1000)
            hfun.execution_mode = 'mpi'
            meshdata = hfun.meshdata()

            self.assertIsNotNone(meshdata)
            self.assertTrue(len(meshdata.values) > 0)
            self.assertTrue(len(meshdata.coords) > 0)
        else:
            # Worker ranks must participate in collective _mpi_dispatch
            _mpi_dispatch(None, _meshdata_task_worker)

        comm.Barrier()

    def test_real_user_scenario_bcast_hfun(self):
        """
        Simulate a real user script where Rank 0 builds the Hfun object
        and broadcasts it to workers so they can participate in meshdata().
        
        If the PID guard is commented out, the workers' copies of Hfun
        will delete Rank 0's _work_dir when they go out of scope, 
        proving the bug exists in real MPI usage!
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            print(f"\n[Rank 0] Setting up Hfun configuration...", flush=True)
            hfun = Hfun(self.raster_list, nprocs=2, hmin=10, hmax=1000)
            hfun.execution_mode = 'mpi'
            hfun.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
            work_dir = hfun._work_dir
            creator_pid = hfun._creator_pid
            print(f"[Rank 0] Created _work_dir: {work_dir}", flush=True)
            self.assertTrue(os.path.exists(work_dir))
        else:
            print(f"\n[Rank {self.rank}] Waiting to receive Hfun configuration...", flush=True)
            hfun = None
            work_dir = None
            creator_pid = None

        # User broadcasts the _work_dir path and creator_pid to workers
        print(f"[Rank {self.rank}] Participating in bcast...", flush=True)
        work_dir = comm.bcast(work_dir, root=0)
        creator_pid = comm.bcast(creator_pid, root=0)

        if self.rank != 0:
            print(f"[Rank {self.rank}] Received _work_dir: {work_dir}", flush=True)
            # Worker rank: artificially construct an Hfun pointing to Rank 0's _work_dir.
            # This perfectly mimics what happens if the object was received via MPI
            # or if the user explicitly configured workers to use a shared directory.
            from ocsmesh.hfun.collector import HfunCollector
            hfun_worker = HfunCollector([], nprocs=2, hmin=10, hmax=1000)
            
            # Clean up the worker's own dummy directory
            import shutil
            shutil.rmtree(hfun_worker._work_dir, ignore_errors=True)
            
            # Point worker's Hfun to Rank 0's directory and PID!
            # This perfectly simulates what unpickling from Rank 0 would do.
            hfun_worker._work_dir = work_dir
            hfun_worker._creator_pid = creator_pid
            # (If PID guard is commented out, the worker deletes this dir anyway)
            
            print(f"[Rank {self.rank}] Worker finished. Explicitly deleting Hfun to trigger Garbage Collection...", flush=True)
            del hfun_worker
            gc.collect()
            print(f"[Rank {self.rank}] Garbage collection complete.", flush=True)

        # Synchronize: Rank 0 waits until workers have triggered GC
        print(f"[Rank {self.rank}] Waiting at barrier...", flush=True)
        comm.Barrier()

        if self.rank == 0:
            # Rank 0 checks if its directory survived.
            print(f"[Rank 0] Barrier crossed. Checking if _work_dir still exists...", flush=True)
            # If the PID guard is missing, this will fail!
            exists = os.path.exists(work_dir)
            print(f"[Rank 0] _work_dir exists: {exists}", flush=True)
            
            self.assertTrue(
                exists,
                "BUG PROVED: Worker rank deleted Rank 0's _work_dir "
                "during garbage collection because the PID guard is missing!"
            )


if __name__ == '__main__':
    unittest.main()
