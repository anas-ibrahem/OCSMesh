#!/usr/bin/env python
"""
Manual MPI test for _calculate_and_write_hfun_to_disk.

This script runs the full meshdata() pipeline under MPI and compares
the result against a serial baseline. It prints clear pass/fail output.

Usage:
    # Serial baseline only (no mpirun)
    python tests/api/manual_mpi_test.py

    # MPI test with 2 ranks
    mpirun -n 2 python tests/api/manual_mpi_test.py

    # MPI test with 4 ranks
    mpirun -n 4 python tests/api/manual_mpi_test.py
"""

import os
import sys
import tempfile
import shutil
import gc
import time
from pathlib import Path

import numpy as np

# ── Setup ──────────────────────────────────────────────────────────

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None

from ocsmesh import Hfun, Raster
from ocsmesh.utils import raster_from_numpy
from ocsmesh.hfun.collector import _mpi_dispatch, _meshdata_task_worker


def log(msg):
    """Print with rank prefix."""
    print(f"[Rank {rank}/{size}] {msg}", flush=True)


def create_test_data(base_dir):
    """Create synthetic DEM rasters."""
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10

    dem1 = base_dir / 'dem1.tif'
    dem2 = base_dir / 'dem2.tif'
    raster_from_numpy(dem1, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1), Raster(dem2)]


# ── Main ───────────────────────────────────────────────────────────

def main():
    tdir = None

    if rank == 0:
        tdir = Path(tempfile.mkdtemp(prefix='mpi_test_'))
        log(f"Test directory: {tdir}")

    # Share test dir path with all ranks
    if comm is not None:
        tdir = comm.bcast(tdir, root=0)

    # Only rank 0 creates rasters and Hfun objects
    if rank == 0:
        raster_list = create_test_data(tdir)

        # ── SERIAL BASELINE ──
        log("Running SERIAL baseline...")
        t0 = time.time()
        hfun_serial = Hfun(raster_list, nprocs=2, hmin=10, hmax=1000)
        hfun_serial.add_subtidal_flow_limiter(
            hmin=50, lower_bound=-5, upper_bound=5)
        hfun_serial.add_constant_value(
            value=200, lower_bound=5, upper_bound=10)
        meshdata_serial = hfun_serial.meshdata()
        serial_time = time.time() - t0
        log(f"Serial: {len(meshdata_serial.values)} nodes, "
            f"{serial_time:.2f}s")

        if size > 1:
            # ── MPI EXECUTION ──
            log(f"Running MPI with {size} ranks...")
            t0 = time.time()
            hfun_mpi = Hfun(raster_list, nprocs=2, hmin=10, hmax=1000)
            hfun_mpi.execution_mode = 'mpi'
            hfun_mpi.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_mpi.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_mpi = hfun_mpi.meshdata()
            mpi_time = time.time() - t0
            log(f"MPI:    {len(meshdata_mpi.values)} nodes, "
                f"{mpi_time:.2f}s")

            # ── COMPARE ──
            v_serial = meshdata_serial.values
            v_mpi = meshdata_mpi.values

            node_diff = abs(len(v_serial) - len(v_mpi))
            node_pct = node_diff / len(v_serial) * 100

            min_ok = np.isclose(np.min(v_serial), np.min(v_mpi), rtol=1e-5)
            max_ok = np.isclose(np.max(v_serial), np.max(v_mpi), rtol=1e-5)
            mean_ok = np.isclose(np.mean(v_serial), np.mean(v_mpi), rtol=1e-5)

            print()
            print("=" * 60)
            print("  RESULTS")
            print("=" * 60)
            print(f"  Serial nodes: {len(v_serial)}")
            print(f"  MPI nodes:    {len(v_mpi)}")
            print(f"  Node diff:    {node_diff} ({node_pct:.2f}%)")
            print()
            print(f"  Serial min/max/mean: "
                  f"{np.min(v_serial):.4f} / "
                  f"{np.max(v_serial):.4f} / "
                  f"{np.mean(v_serial):.4f}")
            print(f"  MPI    min/max/mean: "
                  f"{np.min(v_mpi):.4f} / "
                  f"{np.max(v_mpi):.4f} / "
                  f"{np.mean(v_mpi):.4f}")
            print()
            print(f"  Min match:  {'✅' if min_ok else '❌'}")
            print(f"  Max match:  {'✅' if max_ok else '❌'}")
            print(f"  Mean match: {'✅' if mean_ok else '❌'}")
            print()

            all_pass = min_ok and max_ok and mean_ok and node_pct < 1.0
            if all_pass:
                print("  ✅ ALL CHECKS PASSED — MPI matches serial")
            else:
                print("  ❌ MISMATCH DETECTED")
            print("=" * 60)
            print()
        else:
            print()
            print("=" * 60)
            print("  Running without mpirun (single rank)")
            print("  Serial baseline completed successfully")
            print("  Re-run with: mpirun -n 2 python "
                  "tests/api/manual_mpi_test.py")
            print("=" * 60)
            print()
    else:
        # Worker ranks: participate in MPI collective operations
        # that meshdata() triggers internally via _mpi_dispatch
        log("Worker rank waiting for MPI dispatch...")
        _mpi_dispatch(None, _meshdata_task_worker)
        log("Worker rank done.")

    # Sync all ranks before cleanup
    if comm is not None:
        comm.Barrier()

    # Cleanup
    if rank == 0 and tdir is not None:
        raster_list = None
        gc.collect()
        shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    main()
