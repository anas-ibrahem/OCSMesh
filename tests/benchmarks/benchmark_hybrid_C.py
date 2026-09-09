"""Hybrid MPI + Pool benchmark — Config C (contours + channels + flow limiter + const val).

Same structure as benchmark_hybrid_A.py. See that file for full docs.

# DIFF from Config B: adds flow limiter and constant value.
# Flow limiters and const_val are already parallelized via the
# 3-phase file-path pattern. This config tests that they work
# correctly alongside the hybrid MPI + Pool contour path.

Run with SLURM:
    srun --ntasks=16 --cpus-per-task=5 \\
        python tests/benchmarks/benchmark_hybrid_C.py --mode mpi_hybrid \\
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

import ocsmesh
from ocsmesh.hfun.raster import HfunRaster


HMIN = 200
HMAX = 5000
CONFIG = 'C'
CONFIG_DESC = 'contours + channels + flow limiter + const val'


def available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def make_tiles(out_dir, n_tiles, size):
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
    """Build collector with config C refinements and run all stages."""

    hfun = ocsmesh.Hfun(
        [str(p) for p in tile_paths],
        hmin=HMIN, hmax=HMAX, nprocs=nprocs, method='exact')
    hfun.execution_mode = execution_mode

    # Config C: contours + channels + flow limiter + const val
    hfun.add_contour(level=0, expansion_rate=0.005, target_size=500)
    hfun.add_channel(
        level=0, width=2000, target_size=500, expansion_rate=0.005)
    # DIFF from B: flow limiter and constant value added
    hfun.add_subtidal_flow_limiter(hmin=HMIN, hmax=HMAX)
    hfun.add_constant_value(value=2000, lower_bound=-20, upper_bound=-10)

    stages = {}
    start = time.perf_counter()
    t0 = start
    hfun._apply_contours()
    stages['contours'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hfun._apply_channels()
    stages['channels'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hfun._apply_flow_limiters()
    stages['flow_limiters'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hfun._apply_const_val()
    stages['const_val'] = time.perf_counter() - t0
    hfun._applied = True
    stages['total'] = time.perf_counter() - start

    values = [
        np.array(h.get_values(), copy=True)
        for h in hfun._hfun_list if isinstance(h, HfunRaster)
    ]
    del hfun
    gc.collect()
    return stages, values


def run_serial_mp(tile_paths, nprocs):
    return build_and_run(tile_paths, nprocs, 'serial')


def run_mpi_no_pool(tile_paths):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        return build_and_run(tile_paths, 1, 'parallel')
    return None, None


def run_mpi_hybrid(tile_paths):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1
    cores = available_cores()
    if rank == 0:
        print(f"  Workers: {n_workers}, cores/worker: {cores}, "
              f"total: {n_workers * cores}")
        return build_and_run(tile_paths, cores, 'parallel')
    return None, None


def compare_values(values_a, values_b, label_a, label_b):
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


def _fmt_stages(s):
    parts = [f'{k}: {v:.2f}s' for k, v in s.items()]
    return '  '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',
                        choices=['serial_mp', 'mpi_no_pool', 'mpi_hybrid', 'all'],
                        default='all')
    parser.add_argument('--tiles', type=int, default=4)
    parser.add_argument('--size', type=int, default=120)
    parser.add_argument('--nprocs', type=int, default=4)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    tdir = Path(tempfile.mkdtemp(prefix='ocsmesh_hybridC_'))
    try:
        if rank == 0:
            print(f'=== Config {CONFIG}: {CONFIG_DESC} ===')
            print(f'Tiles: {args.tiles}, size: {args.size}x{args.size}')
            print(f'Start method: {__import__("multiprocessing").get_start_method()}')
            print(f'Available cores: {available_cores()}')

        tile_paths = make_tiles(tdir, args.tiles, args.size)
        results = {}

        if args.mode in ('serial_mp', 'all') and rank == 0:
            print(f'\n--- serial_mp (nprocs={args.nprocs}) ---')
            s, v = run_serial_mp(tile_paths, args.nprocs)
            results['serial_mp'] = {'stages': s, 'values': v}
            print(f'  {_fmt_stages(s)}')

        if args.mode in ('mpi_no_pool', 'all'):
            if rank == 0:
                print('\n--- mpi_no_pool ---')
            s, v = run_mpi_no_pool(tile_paths)
            if rank == 0:
                results['mpi_no_pool'] = {'stages': s, 'values': v}
                print(f'  {_fmt_stages(s)}')

        if args.mode in ('mpi_hybrid', 'all'):
            if rank == 0:
                print(f'\n--- mpi_hybrid ---')
            s, v = run_mpi_hybrid(tile_paths)
            if rank == 0:
                results['mpi_hybrid'] = {'stages': s, 'values': v}
                print(f'  {_fmt_stages(s)}')

        if rank == 0 and len(results) > 1:
            print('\n=== Comparison ===')
            baseline_key = 'serial_mp' if 'serial_mp' in results else list(results.keys())[0]
            for key, data in results.items():
                if key == baseline_key:
                    continue
                problems = compare_values(
                    results[baseline_key]['values'], data['values'],
                    baseline_key, key)
                status = 'OK' if not problems else 'FAIL'
                speedup = results[baseline_key]['stages']['total'] / data['stages']['total']
                print(f'{status}  {baseline_key} vs {key}: speedup {speedup:.2f}x')
                for p in problems:
                    print(f'  {p}')

        if rank == 0 and args.json:
            out = {k: v['stages'] for k, v in results.items()}
            args.json.write_text(json.dumps({
                'config': CONFIG, 'tiles': args.tiles, 'results': out}, indent=2))

    finally:
        gc.collect()
        if rank == 0:
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
