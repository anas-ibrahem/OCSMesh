"""Hybrid MPI + Pool benchmark — Config A (contours only).

Compares three execution strategies on contour-only workload:

1. serial_mp   — Old path: tiles processed one-by-one, shared Pool for
                 intra-tile add_feature parallelism.
2. mpi_no_pool — Pure MPI: 1 rank per tile, add_feature(nprocs=1).
                 No internal Pool, each rank uses 1 core.
3. mpi_hybrid  — NEW: 1 rank per tile, add_feature(nprocs=auto).
                 Each rank creates Pool(available_cores) for intra-tile
                 parallelism. MPI ranks are NOT daemons, so child Pools
                 are allowed.

Key differences from benchmark_contours.py (PR #250):
  # DIFF: Adds mpi_hybrid mode (MPI + internal Pool per rank)
  # DIFF: Uses os.sched_getaffinity for auto core detection
  # DIFF: Rank 0 = coordinator only, does not process tiles
  # DIFF: Uses 'forkserver' start method (set in ocsmesh.__init__)
  #       instead of 'spawn' — faster Pool worker startup

Run with SLURM (15 DEMs, rank 0 + 15 workers, 5 cores per rank = 80 cores):

    # serial_mp baseline (single node, all cores):
    python tests/benchmarks/benchmark_hybrid_A.py --mode serial_mp \\
        --tiles 15 --nprocs 80

    # Pure MPI (16 ranks = 15 tiles + 1 coordinator, 1 core each):
    srun --ntasks=16 --cpus-per-task=1 \\
        python tests/benchmarks/benchmark_hybrid_A.py --mode mpi_no_pool \\
        --tiles 15

    # Hybrid MPI + Pool (16 ranks, 5 cores each, auto-detected):
    srun --ntasks=16 --cpus-per-task=5 \\
        python tests/benchmarks/benchmark_hybrid_A.py --mode mpi_hybrid \\
        --tiles 15
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
import shutil
import tempfile

import numpy as np

# DIFF: import ocsmesh FIRST — this triggers _configure_mpi_environment()
# which now sets 'forkserver' (not 'spawn') for faster Pool startup under MPI.
import ocsmesh
from ocsmesh.hfun.raster import HfunRaster


HMIN = 200
HMAX = 5000
CONFIG = 'A'
CONFIG_DESC = 'contours only'


def available_cores():
    """Auto-detect cores available to this process via affinity mask.

    # DIFF: New utility — when SLURM sets --cpus-per-task=5, returns 5.
    # Used by mpi_hybrid to size the internal Pool automatically.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def make_tiles(out_dir, n_tiles, size):
    """Write n_tiles DEM tiles side by side. Same as PR #250."""
    paths = []
    span = 1.0
    overlap = 0.1
    for i in range(n_tiles):
        x0 = i * (span - overlap)
        x1 = x0 + span
        gx, gy = np.mgrid[x0:x1:complex(0, size), 0:1:complex(0, size)]
        z = (gy * 40.0) - 20.0 + 3.0 * np.sin(gx * 6.0)
        path = Path(out_dir) / f'dem_{i}.tif'
        ocsmesh.utils.raster_from_numpy(path, z, (gx, gy), 4326)
        paths.append(path)
    return paths


def build_and_run(tile_paths, nprocs, execution_mode):
    """Build collector with config A refinements and run contours."""

    hfun = ocsmesh.Hfun(
        [str(p) for p in tile_paths],
        hmin=HMIN, hmax=HMAX, nprocs=nprocs, method='exact')
    hfun.execution_mode = execution_mode

    # Config A: contours only
    hfun.add_contour(level=0, expansion_rate=0.005, target_size=500)

    start = time.perf_counter()
    hfun._apply_contours()
    elapsed = time.perf_counter() - start

    values = [
        np.array(h.get_values(), copy=True)
        for h in hfun._hfun_list if isinstance(h, HfunRaster)
    ]
    del hfun
    gc.collect()
    return elapsed, values


def run_serial_mp(tile_paths, nprocs):
    """Old path: serial tile loop, shared Pool(nprocs) for add_feature."""
    elapsed, values = build_and_run(tile_paths, nprocs, 'serial')
    return elapsed, values


def run_mpi_no_pool(tile_paths):
    """Pure MPI: 1 rank per tile, add_feature(nprocs=1). No internal Pool.

    # DIFF: Rank 0 = coordinator only, does not process tiles.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1  # DIFF: rank 0 excluded from tile processing

    if rank == 0:
        # DIFF: Coordinator distributes tiles, processes none itself
        elapsed, values = build_and_run(tile_paths, 1, 'parallel')
        return elapsed, values
    return None, None


def run_mpi_hybrid(tile_paths):
    """Hybrid: 1 rank per tile, each with Pool(available_cores).

    # DIFF: This is the NEW mode. MPI distributes tiles across ranks,
    # then each rank uses its allocated cores for intra-tile Pool.
    # MPI ranks are NOT daemons → creating child Pool is legal.
    # Core count auto-detected from SLURM affinity mask.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1

    # DIFF: auto-detect cores from SLURM --cpus-per-task
    cores = available_cores()

    if rank == 0:
        print(f"  Workers: {n_workers}, cores/worker: {cores}, "
              f"total: {n_workers * cores}")
        # Coordinator runs parallel mode with auto-detected nprocs
        # Each worker internally creates Pool(cores) via add_feature
        elapsed, values = build_and_run(tile_paths, cores, 'parallel')
        return elapsed, values
    return None, None


def compare_values(values_a, values_b, label_a, label_b):
    """Exact comparison — any difference is a bug."""
    if len(values_a) != len(values_b):
        return [f'{label_b}: {len(values_b)} rasters vs {len(values_a)}']
    problems = []
    for i, (a, b) in enumerate(zip(values_a, values_b)):
        if not np.array_equal(a, b, equal_nan=True):
            diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            problems.append(
                f'raster {i}: {int((diff > 0).sum())} px differ, '
                f'max diff {float(diff.max()):.6g}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',
                        choices=['serial_mp', 'mpi_no_pool', 'mpi_hybrid', 'all'],
                        default='all')
    parser.add_argument('--tiles', type=int, default=4)
    parser.add_argument('--size', type=int, default=120)
    parser.add_argument('--nprocs', type=int, default=4,
                        help='Cores for serial_mp Pool (ignored in MPI modes)')
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()

    # Determine MPI rank (0 if not under MPI)
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    tdir = Path(tempfile.mkdtemp(prefix='ocsmesh_hybridA_'))
    try:
        if rank == 0:
            print(f'=== Config {CONFIG}: {CONFIG_DESC} ===')
            print(f'Tiles: {args.tiles}, size: {args.size}x{args.size}')
            print(f'Start method: {__import__("multiprocessing").get_start_method()}')
            print(f'Available cores (this process): {available_cores()}')
            print(f'Creating tiles in {tdir} ...')

        tile_paths = make_tiles(tdir, args.tiles, args.size)
        results = {}

        # --- serial_mp ---
        if args.mode in ('serial_mp', 'all') and rank == 0:
            print(f'\n--- serial_mp (nprocs={args.nprocs}) ---')
            t, v = run_serial_mp(tile_paths, args.nprocs)
            results['serial_mp'] = {'time': t, 'values': v}
            print(f'  Time: {t:.2f}s')

        # --- mpi_no_pool ---
        if args.mode in ('mpi_no_pool', 'all'):
            if rank == 0:
                print('\n--- mpi_no_pool (nprocs=1 per rank) ---')
            t, v = run_mpi_no_pool(tile_paths)
            if rank == 0:
                results['mpi_no_pool'] = {'time': t, 'values': v}
                print(f'  Time: {t:.2f}s')

        # --- mpi_hybrid ---
        if args.mode in ('mpi_hybrid', 'all'):
            if rank == 0:
                print(f'\n--- mpi_hybrid (auto cores per rank) ---')
            t, v = run_mpi_hybrid(tile_paths)
            if rank == 0:
                results['mpi_hybrid'] = {'time': t, 'values': v}
                print(f'  Time: {t:.2f}s')

        # --- Compare ---
        if rank == 0 and len(results) > 1:
            print('\n=== Comparison ===')
            baseline_key = 'serial_mp' if 'serial_mp' in results else list(results.keys())[0]
            baseline_v = results[baseline_key]['values']
            for key, data in results.items():
                if key == baseline_key:
                    continue
                problems = compare_values(baseline_v, data['values'],
                                          baseline_key, key)
                if problems:
                    print(f'FAIL {baseline_key} vs {key}:')
                    for p in problems:
                        print(f'  {p}')
                else:
                    print(f'OK   {baseline_key} vs {key}: values identical')

                speedup = results[baseline_key]['time'] / data['time']
                print(f'     Speedup: {speedup:.2f}x')

        if rank == 0 and args.json:
            out = {k: {'time': v['time']} for k, v in results.items()}
            args.json.write_text(json.dumps({
                'config': CONFIG, 'tiles': args.tiles,
                'nprocs': args.nprocs, 'results': out}, indent=2))
            print(f'\nWrote {args.json}')

    finally:
        gc.collect()
        if rank == 0:
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
