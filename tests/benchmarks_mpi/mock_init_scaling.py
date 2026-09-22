"""Scalability bug: HfunCollector.__init__() clips on every rank.

Reproduces the N_ranks × N_tiles concurrent temp-file write issue and
proves the MPIExecutor.is_manager() guard fixes it.

Problem (without fix):
  Every rank runs the full clip loop in __init__, producing N_ranks × N_tiles
  concurrent raster writes to TMPDIR. On a shared filesystem (Lustre/GPFS)
  this saturates the metadata server (verified: 452 × 451 = 203,852 writes).

What this script measures:
  - tmp_files_written  — .tif files created in TMPDIR per rank during __init__
  - init_time          — wall time of HfunCollector() on each rank
  - verdict            — BUG (N×M files) or FIX (M files, rank-0 only)

Run (16 ranks = 15 tiles + 1 coordinator):
    mpiexec -n 16 python tests/benchmarks_mpi/benchmark_init_scaling.py

Serial baseline (no MPI — always triggers the bug path):
    python tests/benchmarks_mpi/benchmark_init_scaling.py --tiles 15
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import ocsmesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_util as hu  # noqa: E402


HMIN = 200
HMAX = 5000
OCSMESH_TMPDIR = Path(tempfile.gettempdir()) / 'ocsmesh'


def make_tiles(out_dir, n_tiles, size):
    """Write n_tiles DEM tiles side by side (same pattern as e2e benchmarks)."""
    paths = []
    span    = 1.0
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


def count_tif_files():
    """Count .tif files in ocsmesh's TMPDIR right now."""
    if not OCSMESH_TMPDIR.exists():
        return 0
    return sum(1 for f in OCSMESH_TMPDIR.iterdir() if f.suffix == '.tif')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tiles', type=int, default=0,
                        help='Tile count; 0 = one per worker rank (default)')
    parser.add_argument('--size',  type=int, default=120,
                        help='Tile grid resolution in pixels (default: 120)')
    parser.add_argument('--json',  type=Path, default=None)
    args = parser.parse_args()

    comm    = hu.get_comm()
    rank    = hu.comm_rank(comm)
    size    = hu.comm_size(comm)
    n_tiles = hu.resolve_tiles(comm, args.tiles)
    execution_mode = 'mpi' if size > 1 else 'serial'

    tdir = hu.shared_tmpdir(comm, 'ocsmesh_init_scaling_')
    try:
        # ── Tile creation (rank 0 only) ───────────────────────────────────
        if rank == 0:
            print(f'\n=== HfunCollector.__init__() scalability benchmark ===')
            print(f'Ranks: {size}  |  Tiles: {n_tiles}  |  '
                  f'Tile size: {args.size}×{args.size}')
            print(f'Expected tmp writes — BUG: {size} × {n_tiles} = '
                  f'{size * n_tiles}  |  FIX: {n_tiles} (rank 0 only)')
            print(f'Creating {n_tiles} tiles in {tdir} ...')
            make_tiles(tdir, n_tiles, args.size)
        else:
            tile_paths = hu.tile_paths(tdir, n_tiles)

        if size > 1:
            comm.Barrier()

        tile_paths = hu.tile_paths(tdir, n_tiles)

        # ── All ranks construct HfunCollector (collective entry point) ────
        # The bug / fix lives entirely inside __init__. We snapshot TMPDIR
        # file count before and after to count writes per rank.
        if rank == 0:
            print('\nAll ranks calling HfunCollector() ...')

        before = count_tif_files()

        with hu.CpuMeter(comm, cores=size, collective=(size > 1)) as meter:
            t0 = time.perf_counter()
            hfun = ocsmesh.Hfun(
                [str(p) for p in tile_paths],
                hmin=HMIN, hmax=HMAX, nprocs=1, method='exact')
            hfun.execution_mode = execution_mode
            init_time = time.perf_counter() - t0

        after       = count_tif_files()
        rank_writes = after - before

        # ── Gather per-rank stats to rank 0 ──────────────────────────────
        if size > 1:
            all_writes     = comm.gather(rank_writes, root=0)
            all_init_times = comm.gather(init_time,   root=0)
        else:
            all_writes     = [rank_writes]
            all_init_times = [init_time]

        # ── Report ────────────────────────────────────────────────────────
        if rank == 0:
            total_writes = sum(all_writes)
            expected_bug = size * n_tiles
            expected_fix = n_tiles

            print(f'\n--- Results ---')
            print(f'  Total tmp .tif files written (all ranks): {total_writes}')
            print(f'  Expected if BUG: {expected_bug}')
            print(f'  Expected if FIX: {expected_fix}')
            print(f'  init_time — rank 0: {all_init_times[0]:.3f}s'
                  + (f'  |  max worker: '
                     f'{max(all_init_times[1:]):.3f}s '
                     f'(should be ~0s if fix active)'
                     if size > 1 else ''))
            print(f'  {meter.format()}')

            print(f'\n  Per-rank writes:')
            for r, cnt in enumerate(all_writes):
                tag = '← coordinator' if r == 0 else f'← worker {r}'
                print(f'    rank {r:3d}: {cnt:4d} files  {tag}')

            print()
            if total_writes <= expected_fix:
                print('  ✅  FIX CONFIRMED — only rank 0 wrote clip files.')
            elif total_writes >= expected_bug:
                print('  ❌  BUG REPRODUCED — all ranks wrote clip files.')
            else:
                print(f'  ⚠️   Partial: {total_writes} writes '
                      f'(between fix={expected_fix} and bug={expected_bug})')

            if args.json:
                out = {
                    'ranks':            size,
                    'tiles':            n_tiles,
                    'execution_mode':   execution_mode,
                    'total_tmp_writes': total_writes,
                    'expected_fix':     expected_fix,
                    'expected_bug':     expected_bug,
                    'per_rank_writes':  all_writes,
                    'per_rank_init_s':  all_init_times,
                    'meter':            meter.as_dict(),
                }
                args.json.write_text(json.dumps(out, indent=2))
                print(f'Wrote {args.json}')

    finally:
        gc.collect()
        if size > 1:
            comm.Barrier()
        if rank == 0:
            import shutil
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
