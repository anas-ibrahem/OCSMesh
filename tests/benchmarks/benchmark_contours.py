"""Current shared-pool vs new per-tile worker benchmark for contours.

Same idea as the Hercules benchmark in PR #250, shrunk down so it runs on a
laptop: build N DEM tiles, add a growing set of refinements (configs A-D),
run the whole thing once per execution mode, then compare wall times and
check the numbers came out the same.

The two labels mean:

- shared-pool (current): `_apply_contours_serial()`, the old implementation.
    It loops over tiles one at a time but passes one shared process pool into
    `add_feature()` for work inside each tile.
- per-tile workers (new): `_apply_contours_parallel()`, this PR's
    implementation. It sends one raster tile per worker process, then each
    worker calls `add_feature(nprocs=1)` with no nested process pool.

`add_feature` merges with `np.minimum`, so serial and parallel must agree
*exactly*. Any difference at all is a bug, not rounding.

Run it::

    python tests/benchmarks/benchmark_contours.py
    python tests/benchmarks/benchmark_contours.py --tiles 6 --nprocs 6
    python tests/benchmarks/benchmark_contours.py --config D --profile

Nothing here needs pytest or pytest-benchmark.
"""

import argparse
import cProfile
import gc
import json
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import ocsmesh
from ocsmesh.hfun.raster import HfunRaster


# Refinement recipes, each one costs more than the last. Mirrors the
# A/B/C/D configs used in PR #250.
CONFIGS = {
    'A': 'contours only',
    'B': 'contours + channels',
    'C': 'B + flow limiter + constant value',
    'D': 'C + patch + topo-bound constraint',
}

MODES = ['serial', 'parallel']
MODE_LABELS = {
    'serial': 'shared-pool (current)',
    'parallel': 'per-tile workers (new)',
}

HMIN = 200
HMAX = 5000


def make_tiles(out_dir, n_tiles, size):
    """Write `n_tiles` DEM tiles side by side, with a bit of overlap.

    Depth goes from below sea level to above it inside every tile, so the
    zero contour always exists and there is real work to do.
    """

    paths = []
    span = 1.0
    overlap = 0.1
    for i in range(n_tiles):
        x0 = i * (span - overlap)
        x1 = x0 + span
        gx, gy = np.mgrid[x0:x1:complex(0, size), 0:1:complex(0, size)]
        # Sloping seabed plus a ripple, so contours are not straight lines.
        z = (gy * 40.0) - 20.0 + 3.0 * np.sin(gx * 6.0)
        path = Path(out_dir) / f'dem_{i}.tif'
        ocsmesh.utils.raster_from_numpy(path, z, (gx, gy), 4326)
        paths.append(path)
    return paths


def build_hfun(tile_paths, mode, nprocs, config):
    """Create the collector and register the refinements for `config`."""

    hfun = ocsmesh.Hfun(
        [str(p) for p in tile_paths],
        hmin=HMIN, hmax=HMAX, nprocs=nprocs, method='exact')
    hfun.execution_mode = mode

    hfun.add_contour(level=0, expansion_rate=0.005, target_size=500)

    if config in ('B', 'C', 'D'):
        hfun.add_channel(
            level=0, width=2000, target_size=500, expansion_rate=0.005)

    if config in ('C', 'D'):
        hfun.add_subtidal_flow_limiter(hmin=HMIN, hmax=HMAX)
        hfun.add_constant_value(value=2000, lower_bound=-20, upper_bound=-10)

    if config == 'D':
        from shapely.geometry import box
        hfun.add_patch(
            shape=box(0.2, 0.2, 0.6, 0.6),
            target_size=400, expansion_rate=0.005)
        hfun.add_topo_bound_constraint(
            value=800, upper_bound=0, lower_bound=-20, value_type='min')

    return hfun


def timed(fn):
    """Run `fn`, return (result, seconds)."""

    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def run_once(tile_paths, mode, nprocs, config, with_meshdata):
    """Run one mode end to end and report per-stage timings."""

    hfun = build_hfun(tile_paths, mode, nprocs, config)

    stages = {}
    _, stages['contours'] = timed(hfun._apply_contours)
    _, stages['flow_limiters'] = timed(hfun._apply_flow_limiters)
    _, stages['const_val'] = timed(hfun._apply_const_val)
    _, stages['linefeatures'] = timed(hfun._apply_linefeatures)
    _, stages['patch'] = timed(hfun._apply_patch)
    _, stages['channels'] = timed(hfun._apply_channels)
    _, stages['constraints'] = timed(hfun._apply_constraints)
    hfun._applied = True

    # Snapshot the raster values while the collector is still alive.
    values = [
        np.array(h.get_values(), copy=True)
        for h in hfun._hfun_list if isinstance(h, HfunRaster)
    ]

    if with_meshdata:
        _, stages['meshdata'] = timed(hfun.meshdata)

    stages['total'] = sum(stages.values())

    del hfun
    gc.collect()
    return stages, values


def compare(baseline_values, other_values, baseline_mode, other_mode):
    """Exact comparison of two runs. Returns a list of problem strings."""

    problems = []
    if len(baseline_values) != len(other_values):
        return [f'{other_mode}: got {len(other_values)} rasters, '
                f'{baseline_mode} had {len(baseline_values)}']

    for i, (a, b) in enumerate(zip(baseline_values, other_values)):
        if a.shape != b.shape:
            problems.append(f'raster {i}: shape {a.shape} vs {b.shape}')
            continue
        if not np.array_equal(a, b, equal_nan=True):
            diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            problems.append(
                f'raster {i}: {int((diff > 0).sum())} pixels differ, '
                f'max diff {float(diff.max()):.6g}')
    return problems


def _display_mode(mode):
    """Return the human label for a benchmark mode."""

    return MODE_LABELS.get(mode, mode)


def print_table(results, config, n_tiles, nprocs):
    """Print the timing table."""

    stage_names = list(results[MODES[0]].keys())
    width = max(len(s) for s in stage_names) + 2

    print()
    print(f'Config {config} — {CONFIGS[config]}')
    print(f'{n_tiles} tiles, nprocs={nprocs}')
    col_width = max(len(_display_mode(mode)) + 4 for mode in MODES)
    print('-' * (width + col_width * len(MODES) + 12))
    header = 'stage'.ljust(width)
    for mode in MODES:
        header += f'{_display_mode(mode) + " (s)":>{col_width}}'
    header += f'{"speedup":>12}'
    print(header)
    print('-' * (width + col_width * len(MODES) + 12))

    for stage in stage_names:
        row = stage.ljust(width)
        for mode in MODES:
            row += f'{results[mode][stage]:>{col_width}.2f}'
        base = results[MODES[0]][stage]
        other = results[MODES[-1]][stage]
        speedup = (base / other) if other > 0 else float('nan')
        row += f'{speedup:>11.2f}x'
        print(row)
    print('-' * (width + col_width * len(MODES) + 12))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=int, default=4,
                        help='number of DEM tiles (default: 4)')
    parser.add_argument('--size', type=int, default=120,
                        help='pixels per tile side (default: 120)')
    parser.add_argument('--nprocs', type=int, default=4,
                        help='processes for parallel mode (default: 4)')
    parser.add_argument('--config', choices=sorted(CONFIGS), default='A',
                        help='refinement recipe (default: A)')
    parser.add_argument('--meshdata', action='store_true',
                        help='also time the final meshdata() call')
    parser.add_argument('--profile', action='store_true',
                        help='cProfile each mode and print the top 25')
    parser.add_argument('--json', type=Path, default=None,
                        help='write the results to this JSON file')
    args = parser.parse_args(argv)

    tdir = Path(tempfile.mkdtemp(prefix='ocsmesh_bench_'))
    try:
        print(f'Creating {args.tiles} tiles of {args.size}x{args.size} '
              f'in {tdir} ...')
        tile_paths = make_tiles(tdir, args.tiles, args.size)

        results = {}
        values_by_mode = {}
        for mode in MODES:
            print(f'Running {_display_mode(mode)} [execution_mode={mode}] ...')
            if args.profile:
                profiler = cProfile.Profile()
                profiler.enable()
                stages, values = run_once(
                    tile_paths, mode, args.nprocs, args.config, args.meshdata)
                profiler.disable()
                print(f'\n--- cProfile: {mode} ---')
                pstats.Stats(profiler).sort_stats('cumulative').print_stats(25)
            else:
                stages, values = run_once(
                    tile_paths, mode, args.nprocs, args.config, args.meshdata)
            results[mode] = stages
            values_by_mode[mode] = values

        print_table(results, args.config, args.tiles, args.nprocs)

        baseline = MODES[0]
        all_problems = {}
        for mode in MODES[1:]:
            problems = compare(
                values_by_mode[baseline], values_by_mode[mode],
                baseline, mode)
            all_problems[mode] = problems
            if problems:
                print(f'\nFAIL {_display_mode(baseline)} vs '
                      f'{_display_mode(mode)}:')
                for p in problems:
                    print(f'  {p}')
            else:
                print(f'\nOK   {_display_mode(baseline)} vs '
                      f'{_display_mode(mode)}: '
                      f'all raster values identical')

        if args.json:
            args.json.write_text(json.dumps({
                'config': args.config,
                'description': CONFIGS[args.config],
                'tiles': args.tiles,
                'size': args.size,
                'nprocs': args.nprocs,
                'mode_labels': MODE_LABELS,
                'timings': results,
                'equivalence_problems': all_problems,
            }, indent=2))
            print(f'\nWrote {args.json}')

        return 1 if any(all_problems.values()) else 0

    finally:
        gc.collect()
        shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
