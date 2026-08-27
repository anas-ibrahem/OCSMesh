"""
MPI Benchmark for the meshdata() pipeline.

Standalone script (not pytest) — must be run under mpiexec::

    mpiexec -n 2 python tests/benchmarks/benchmark_mpi.py
    mpiexec -n 4 python tests/benchmarks/benchmark_mpi.py

Measures the full hfun.meshdata() pipeline under MPI execution mode
and compares results against the serial baseline for equivalence.

All ranks participate collectively, but only Rank 0 performs timing
measurements, prints results, and runs the serial baseline.
"""

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.mpi import MPIExecutor, _is_mpi_env_detected
from ocsmesh.utils import raster_from_numpy


# ─── Configuration ──────────────────────────────────────────────────
BENCHMARK_ROUNDS = 3
EQUIVALENCE_RTOL = 1e-5


# ─── Helpers ─────────────────────────────────────────────────────────

def _create_benchmark_rasters(base_dir):
    """Create 4 tiled rasters with 20% overlap for benchmarking.

    Layout is a 2x2 grid covering [0,1] x [0,1]. Each tile spans
    60% of each axis (50% base + 10% overlap per edge), giving 20%
    overlap between adjacent tiles.
    """
    tile_ranges = [
        (0.0, 0.6, 0.0, 0.6),  # top-left
        (0.4, 1.0, 0.0, 0.6),  # top-right
        (0.0, 0.6, 0.4, 1.0),  # bottom-left
        (0.4, 1.0, 0.4, 1.0),  # bottom-right
    ]

    raster_list = []
    for i, (x0, x1, y0, y1) in enumerate(tile_ranges):
        gx, gy = np.mgrid[x0:x1:60j, y0:y1:60j]
        dem_data = (gx * 20) - 10
        p = base_dir / f'dem_{i}.tif'
        raster_from_numpy(p, dem_data, (gx, gy), 4326)
        raster_list.append(Raster(p))

    return raster_list


def _build_hfun(raster_list, execution_mode):
    """Create an Hfun with 3 refinements and the given execution_mode."""
    hfun = Hfun(raster_list, nprocs=4, hmin=10, hmax=1000)
    hfun.execution_mode = execution_mode
    hfun.add_topo_bound_constraint(
        value=100, upper_bound=5, lower_bound=-5, value_type='min')
    hfun.add_topo_bound_constraint(
        value=200, upper_bound=8, lower_bound=2, value_type='max')
    hfun.add_constant_value(value=500, lower_bound=-10, upper_bound=-5)
    return hfun


def _extract_stats(meshdata):
    """Extract comparable statistics from a meshdata result."""
    return {
        "num_nodes": len(meshdata.coords),
        "min": float(np.min(meshdata.values)),
        "max": float(np.max(meshdata.values)),
        "mean": float(np.mean(meshdata.values)),
        "coords": meshdata.coords,
    }


def _rel_diff(a, b):
    """Compute the relative difference between two scalars."""
    return abs(a - b) / abs(a) if a != 0 else 0.0


def _assert_stats_equal(baseline, current, mode_label, rtol):
    """Assert that two stat dicts are numerically equivalent."""
    assert baseline["num_nodes"] == current["num_nodes"], (
        f"[{mode_label}] Node count mismatch: "
        f"{baseline['num_nodes']} vs {current['num_nodes']}"
    )

    npt.assert_allclose(
        baseline["coords"], current["coords"],
        rtol=rtol,
        err_msg=f"[{mode_label}] Coordinates mismatch",
    )

    for metric in ("min", "max", "mean"):
        npt.assert_allclose(
            baseline[metric], current[metric],
            rtol=rtol,
            err_msg=f"[{mode_label}] {metric} value mismatch",
        )


def _timed_meshdata(raster_list, execution_mode, rounds):
    """Run meshdata() multiple times, return (stats, timings)."""
    timings = []
    meshdata = None

    for r in range(rounds):
        hfun = _build_hfun(raster_list, execution_mode)
        t0 = time.perf_counter()
        meshdata = hfun.meshdata()
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        if MPIExecutor.is_manager():
            print(f"  Round {r + 1}/{rounds}: {elapsed:.3f}s")

        del hfun
        gc.collect()

    stats = _extract_stats(meshdata) if meshdata is not None else None
    return stats, timings


# ─── Main ────────────────────────────────────────────────────────────

def main():
    if not _is_mpi_env_detected():
        print("ERROR: Must run under mpiexec:")
        print("  mpiexec -n 2 python tests/benchmarks/benchmark_mpi.py")
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ── Setup: create rasters on rank 0, broadcast path ──
    if rank == 0:
        tdir = Path(tempfile.mkdtemp())
        raster_list = _create_benchmark_rasters(tdir)
        print(f"\n{'=' * 60}")
        print(f"  MPI Benchmark — {size} ranks, {BENCHMARK_ROUNDS} rounds")
        print(f"{'=' * 60}\n")
    else:
        tdir = None
        raster_list = None

    tdir = comm.bcast(tdir, root=0)

    # Workers need to load rasters from the broadcast path
    if rank != 0:
        raster_list = [
            Raster(tdir / f'dem_{i}.tif') for i in range(4)
        ]

    # ── Benchmark: MPI mode ──
    if rank == 0:
        print(f"[MPI mode] ({size} ranks)")
    mpi_stats, mpi_timings = _timed_meshdata(
        raster_list, "mpi", BENCHMARK_ROUNDS
    )

    # ── Benchmark: Serial baseline (rank 0 only) ──
    if rank == 0:
        print(f"\n[Serial mode] (baseline)")
        serial_stats, serial_timings = _timed_meshdata(
            raster_list, "serial", BENCHMARK_ROUNDS
        )

    # ── Benchmark: Parallel baseline (rank 0 only) ──
    if rank == 0:
        print(f"\n[Parallel mode] (nprocs=4)")
        parallel_stats, parallel_timings = _timed_meshdata(
            raster_list, "parallel", BENCHMARK_ROUNDS
        )

    # ── Results (rank 0 only) ──
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  Results")
        print(f"{'=' * 60}\n")

        serial_mean = np.mean(serial_timings)
        parallel_mean = np.mean(parallel_timings)
        mpi_mean = np.mean(mpi_timings)

        print(f"  {'Mode':<15} {'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Speedup':<10}")
        print(f"  {'-' * 61}")
        print(f"  {'Serial':<15} {serial_mean:<12.3f} {min(serial_timings):<12.3f} {max(serial_timings):<12.3f} {'1.00x':<10}")
        print(f"  {'Parallel(4)':<15} {parallel_mean:<12.3f} {min(parallel_timings):<12.3f} {max(parallel_timings):<12.3f} {serial_mean / parallel_mean:<10.2f}x")
        print(f"  {'MPI({})'.format(size):<15} {mpi_mean:<12.3f} {min(mpi_timings):<12.3f} {max(mpi_timings):<12.3f} {serial_mean / mpi_mean:<10.2f}x")

        # ── Equivalence check ──
        print(f"\n{'=' * 60}")
        print(f"  Equivalence Check (rtol={EQUIVALENCE_RTOL})")
        print(f"{'=' * 60}\n")

        try:
            _assert_stats_equal(
                serial_stats, mpi_stats, "MPI vs Serial", EQUIVALENCE_RTOL
            )
            print(f"  ✓ MPI vs Serial: PASS")
        except AssertionError as e:
            print(f"  ✗ MPI vs Serial: FAIL — {e}")

        try:
            _assert_stats_equal(
                serial_stats, parallel_stats, "Parallel vs Serial", EQUIVALENCE_RTOL
            )
            print(f"  ✓ Parallel vs Serial: PASS")
        except AssertionError as e:
            print(f"  ✗ Parallel vs Serial: FAIL — {e}")

        # ── Summary stats ──
        print(f"\n  MPI meshdata: {mpi_stats['num_nodes']} nodes, "
              f"values [{mpi_stats['min']:.2f}, {mpi_stats['max']:.2f}], "
              f"mean={mpi_stats['mean']:.2f}")

        print()

        # ── Cleanup ──
        try:
            shutil.rmtree(tdir)
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    main()
