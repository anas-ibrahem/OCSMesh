"""
MPI tests for HfunCollector (Approach 2 — dynamic point-to-point).

Tests that do NOT require mpirun are in TestMPIModeProperty and
TestMPIDynamicDispatchUnit (the latter exercises the size==1 coordinator
fallback of the dynamic scheduler).
Tests that REQUIRE mpirun are in TestMPIWriteHfun.

Running instructions:

    # Unit tests (no mpirun needed)
    PYTHONPATH=. python -m pytest tests/api/test_mpi.py::TestMPIModeProperty -v
    PYTHONPATH=. python -m pytest tests/api/test_mpi.py::TestMPIDynamicDispatchUnit -v

    # MPI integration tests (require mpirun; rank 0 is a dedicated coordinator,
    # so use n>=2 — with -n 2 there is exactly ONE compute worker)
    PYTHONPATH=. mpirun -n 2 python -m pytest tests/api/test_mpi.py::TestMPIWriteHfun -v -s
    PYTHONPATH=. mpirun -n 4 python -m pytest tests/api/test_mpi.py::TestMPIWriteHfun -v -s
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
from ocsmesh.hfun.collector import (
    _mpi_dispatch,            # legacy Approach 1 (kept for back-compat)
    _mpi_dynamic_dispatch,    # Approach 2 dynamic scheduler
    _mpi_shutdown_workers,    # Approach 2 worker shutdown
    _meshdata_task_worker,
    mpi_worker_loop,
    _MPI_OP_MESHDATA,
)

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
# Dynamic-scheduler unit tests — no mpirun needed (size == 1 fallback)
# ════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_MPI, "requires mpi4py")
class TestMPIDynamicDispatchUnit(unittest.TestCase):
    """Exercise _mpi_dynamic_dispatch's single-rank coordinator fallback.

    When there is no separate worker (size == 1, i.e. run WITHOUT mpirun),
    the coordinator runs tasks locally through the same op registry and
    error channel used by the point-to-point workers. This lets us test
    the scheduler's routing/error handling without launching mpirun.
    """

    def setUp(self):
        if MPI.COMM_WORLD.Get_size() != 1:
            self.skipTest("only meaningful without mpirun (size == 1)")

    def test_unknown_op_returns_structured_error(self):
        """A task naming an unregistered op yields a structured error."""
        tasks = [{'op': 'does_not_exist', 'original_index': 0}]
        results = _mpi_dynamic_dispatch(tasks)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'error')
        self.assertEqual(results[0]['original_index'], 0)

    def test_meshdata_failure_is_structured(self):
        """A meshdata task with bad paths fails as a structured error,
        not an unhandled exception (worker catches its own errors)."""
        tasks = [{
            'op': _MPI_OP_MESHDATA,
            'type': 'raster',
            'original_index': 7,
            'topo_path': '/no/such/file.tif',
            'hfun_input_path': '/no/such/input',
            'output_path': '/tmp/should_not_be_written',
            'hmin': 10,
            'hmax': 1000,
            'meshdata_kwargs': {},
        }]
        results = _mpi_dynamic_dispatch(tasks)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'error')
        self.assertEqual(results[0]['original_index'], 7)

    def test_empty_task_list_returns_empty(self):
        """No tasks → no results, no error."""
        self.assertEqual(_mpi_dynamic_dispatch([]), [])


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

    # ────────────────────────────────────────────────────────────────
    # OLD TESTS (used manual _mpi_dispatch calls — fragile pattern)
    #
    # Problem: workers had to manually call _mpi_dispatch(None, fn)
    # at exactly the right time. Real users would not know to do this
    # and would get deadlocks. Replaced by mpi_worker_loop tests.
    # ────────────────────────────────────────────────────────────────

    # def test_serial_vs_mpi_write_hfun_equivalence_OLD(self):
    #     """Numerical equivalence: serial meshdata == MPI meshdata."""
    #     comm = MPI.COMM_WORLD
    #
    #     if self.rank == 0:
    #         hfun_serial = Hfun(self.raster_list, nprocs=2,
    #                            hmin=10, hmax=1000)
    #         hfun_serial.add_subtidal_flow_limiter(
    #             hmin=50, lower_bound=-5, upper_bound=5)
    #         hfun_serial.add_constant_value(
    #             value=200, lower_bound=5, upper_bound=10)
    #         meshdata_serial = hfun_serial.meshdata()
    #
    #         hfun_mpi = Hfun(self.raster_list, nprocs=2,
    #                         hmin=10, hmax=1000)
    #         hfun_mpi.execution_mode = 'mpi'
    #         hfun_mpi.add_subtidal_flow_limiter(
    #             hmin=50, lower_bound=-5, upper_bound=5)
    #         hfun_mpi.add_constant_value(
    #             value=200, lower_bound=5, upper_bound=10)
    #         meshdata_mpi = hfun_mpi.meshdata()
    #
    #         self.assertAlmostEqual(
    #             len(meshdata_serial.values), len(meshdata_mpi.values),
    #             delta=len(meshdata_serial.values) * 0.01)
    #         npt.assert_allclose(
    #             np.min(meshdata_serial.values),
    #             np.min(meshdata_mpi.values), rtol=1e-5)
    #     else:
    #         _mpi_dispatch(None, _meshdata_task_worker)
    #
    #     comm.Barrier()

    # def test_mpi_write_hfun_basic_OLD(self):
    #     """Basic smoke test: MPI meshdata() produces valid output."""
    #     comm = MPI.COMM_WORLD
    #
    #     if self.rank == 0:
    #         hfun = Hfun(self.raster_list, nprocs=1,
    #                     hmin=10, hmax=1000)
    #         hfun.execution_mode = 'mpi'
    #         meshdata = hfun.meshdata()
    #
    #         self.assertIsNotNone(meshdata)
    #         self.assertTrue(len(meshdata.values) > 0)
    #     else:
    #         _mpi_dispatch(None, _meshdata_task_worker)
    #
    #     comm.Barrier()

    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # Approach 2 tests — workers use the point-to-point mpi_worker_loop()
    #
    # Rank 0 is the dedicated coordinator: it calls normal methods
    # (hfun.meshdata()), which internally stream tasks to workers via
    # send()/recv() and release them with TAG_STOP. Workers just call
    # mpi_worker_loop() and return once the coordinator stops them.
    # No manual dispatch calls, no collective-ordering deadlock risk.
    # ────────────────────────────────────────────────────────────────

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output.

        Workers use mpi_worker_loop() — no manual _mpi_dispatch needed.
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            print(f"\n[Rank 0] Creating Hfun and calling meshdata()...",
                  flush=True)
            hfun = Hfun(self.raster_list, nprocs=1,
                        hmin=10, hmax=1000)
            hfun.execution_mode = 'mpi'
            meshdata = hfun.meshdata()

            print(f"[Rank 0] meshdata() returned {len(meshdata.values)} "
                  f"nodes", flush=True)
            self.assertIsNotNone(meshdata)
            self.assertTrue(len(meshdata.values) > 0)
            self.assertTrue(len(meshdata.coords) > 0)
        else:
            # Workers just call mpi_worker_loop() — it handles
            # all collective operations automatically via tags.
            print(f"[Rank {self.rank}] Entering mpi_worker_loop()...",
                  flush=True)
            mpi_worker_loop()
            print(f"[Rank {self.rank}] mpi_worker_loop() returned.",
                  flush=True)

        comm.Barrier()

    def test_serial_vs_mpi_write_hfun_equivalence(self):
        """Numerical equivalence: serial meshdata == MPI meshdata.

        Workers use mpi_worker_loop() — no manual _mpi_dispatch needed.
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            # ── SERIAL baseline ──
            print(f"\n[Rank 0] Running serial baseline...", flush=True)
            hfun_serial = Hfun(self.raster_list, nprocs=2,
                               hmin=10, hmax=1000)
            hfun_serial.execution_mode = 'serial'
            hfun_serial.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_serial.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_serial = hfun_serial.meshdata()
            values_serial = meshdata_serial.values
            print(f"[Rank 0] Serial: {len(values_serial)} nodes", flush=True)

            # ── MPI execution ──
            print(f"[Rank 0] Running MPI execution...", flush=True)
            hfun_mpi = Hfun(self.raster_list, nprocs=2,
                            hmin=10, hmax=1000)
            hfun_mpi.execution_mode = 'mpi'
            hfun_mpi.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_mpi.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_mpi = hfun_mpi.meshdata()
            values_mpi = meshdata_mpi.values
            print(f"[Rank 0] MPI: {len(values_mpi)} nodes", flush=True)

            # ── COMPARE ──
            self.assertAlmostEqual(
                len(values_serial), len(values_mpi),
                delta=len(values_serial) * 0.01)
            npt.assert_allclose(
                np.min(values_serial), np.min(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.max(values_serial), np.max(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.mean(values_serial), np.mean(values_mpi), rtol=1e-5)
            print(f"[Rank 0] ✅ Serial and MPI results match!", flush=True)
        else:
            # Workers participate in the MPI meshdata collective
            # automatically via the tag-based loop.
            print(f"[Rank {self.rank}] Entering mpi_worker_loop()...",
                  flush=True)
            mpi_worker_loop()
            print(f"[Rank {self.rank}] mpi_worker_loop() returned.",
                  flush=True)

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
            # Use getattr: _creator_pid only exists when PID guard
            # fix is active. Fall back to current PID so the test
            # can still run (and prove the bug) either way.
            creator_pid = getattr(hfun, '_creator_pid', os.getpid())
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
