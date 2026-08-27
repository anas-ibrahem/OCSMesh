"""Tests for the parallel `_apply_contours` path and the bugs fixed with it.

The idea behind most of these tests is simple: run the OLD code path
(`execution_mode='serial'`) and the NEW one (`execution_mode='parallel'`)
on the very same input, then check the numbers came out the same.

`add_feature` merges its result with `np.minimum`, and `min` does not care
about order, so serial and parallel must agree *exactly*. We use
`assert_array_equal`, not a tolerance: a tolerance would hide a real bug.
"""

import gc
import pickle
import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as npt
from shapely.geometry import LineString, box

import ocsmesh
from ocsmesh.hfun.collector import _contours_task_worker
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.hfun.raster import HfunRaster

from .test_common import topo_2rast_1mesh


IS_WINDOWS = platform.system() == 'Windows'

HMIN = 500
HMAX = 5000
NPROCS = 2


def _values_of(hfun):
    """Read the size values out of a size function, whatever its type."""

    if isinstance(hfun, HfunRaster):
        return hfun.get_values()
    return np.asarray(hfun.mesh.meshdata.values)


def _assert_same_sizes(list_a, list_b, msg=''):
    """Compare two `_hfun_list`s entry by entry, exactly."""

    assert len(list_a) == len(list_b), f'{msg} different number of hfuns'
    for i, (a, b) in enumerate(zip(list_a, list_b)):
        assert type(a) is type(b), (  # pylint: disable=C0123
            f'{msg} entry {i} changed type: {type(a)} vs {type(b)}')
        npt.assert_array_equal(
            _values_of(a), _values_of(b),
            err_msg=f'{msg} entry {i} values differ')


class _FailingPool:
    """Stand-in for `Pool` whose `map` reports that every task failed.

    Used to reach the fail-fast branch without having to make a real
    worker process crash.
    """

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def map(func, tasks):  # pylint: disable=unused-argument
        return [{'status': 'error',
                 'original_index': t['original_index'],
                 'error': 'boom'}
                for t in tasks]

    def join(self):
        pass


class _CapturingPool:
    """Stand-in for `Pool` that keeps the tasks and runs them right here.

    Lets a test inspect the real task dictionaries (are they pickleable?)
    while still producing the correct result.
    """

    captured = []

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def map(func, tasks):
        _CapturingPool.captured.extend(tasks)
        return [func(t) for t in tasks]

    def join(self):
        pass


@unittest.skipIf(IS_WINDOWS, 'Multiprocessing file locks are flaky on Windows')
class ContoursSerialVsParallel(unittest.TestCase):
    """`_apply_contours` must give the same numbers in both modes."""

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.grd'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)

    def tearDown(self):
        # GDAL keeps file handles open; drop them before removing the dir.
        gc.collect()
        shutil.rmtree(self.tdir, ignore_errors=True)

    def _build(self, in_list, mode, method='exact'):
        hfun = ocsmesh.Hfun(
            in_list, hmin=HMIN, hmax=HMAX, nprocs=NPROCS, method=method)
        hfun.execution_mode = mode
        hfun.add_contour(level=0, expansion_rate=0.01, target_size=1000)
        return hfun

    def test_contours_serial_parallel_bitwise_equal(self):
        """Rasters only: old path and new path must match exactly."""

        in_list = [self.rast1, self.rast2]

        serial = self._build(in_list, 'serial')
        serial._apply_contours()

        parallel = self._build(in_list, 'parallel')
        parallel._apply_contours()

        _assert_same_sizes(
            serial._hfun_list, parallel._hfun_list, 'rasters only:')

    def test_contours_parallel_with_mesh_hfun(self):
        """A mesh size function first in the list must survive untouched.

        The parallel path sends rasters to workers and keeps mesh size
        functions on the coordinator. If it matched things up by position
        instead of by identity, the mesh entry here would be replaced by a
        raster result.
        """

        in_list = [self.mesh1, self.rast1, self.rast2]

        serial = self._build(in_list, 'serial')
        serial._apply_contours()

        parallel = self._build(in_list, 'parallel')
        parallel._apply_contours()

        self.assertIsInstance(parallel._hfun_list[0], HfunMesh)
        self.assertIsInstance(parallel._hfun_list[1], HfunRaster)
        self.assertIsInstance(parallel._hfun_list[2], HfunRaster)

        _assert_same_sizes(
            serial._hfun_list, parallel._hfun_list, 'mesh + rasters:')

    def test_contours_full_pipeline_serial_parallel_equal(self):
        """Same check, but through the public `meshdata()` entry point."""

        in_list = [self.rast1, self.rast2]

        serial = self._build(in_list, 'serial')
        parallel = self._build(in_list, 'parallel')

        md_serial = serial.meshdata()
        md_parallel = parallel.meshdata()

        self.assertEqual(len(md_serial.coords), len(md_parallel.coords))
        npt.assert_allclose(md_serial.coords, md_parallel.coords, rtol=1e-12)
        npt.assert_allclose(md_serial.values, md_parallel.values, rtol=1e-12)

    def test_contours_task_dicts_are_pickleable(self):
        """Whatever we hand to a worker has to survive pickling."""

        _CapturingPool.captured = []
        parallel = self._build([self.rast1, self.rast2], 'parallel')

        with mock.patch('ocsmesh.hfun.collector.Pool', _CapturingPool):
            parallel._apply_contours()

        self.assertEqual(len(_CapturingPool.captured), 2)
        for task in _CapturingPool.captured:
            pickle.loads(pickle.dumps(task))

        serial = self._build([self.rast1, self.rast2], 'serial')
        serial._apply_contours()
        _assert_same_sizes(
            serial._hfun_list, parallel._hfun_list, 'captured-pool run:')

    def test_contours_parallel_raises_when_worker_fails(self):
        """A failed worker must stop the run, not be quietly skipped."""

        parallel = self._build([self.rast1, self.rast2], 'parallel')

        with mock.patch('ocsmesh.hfun.collector.Pool', _FailingPool):
            with self.assertRaises(RuntimeError) as ctx:
                parallel._apply_contours()

        self.assertIn('contour worker(s) failed', str(ctx.exception))

    def test_contours_parallel_cleans_up_contour_dir(self):
        """The extracted contour files must not be left behind."""

        parallel = self._build([self.rast1, self.rast2], 'parallel')
        parallel._apply_contours()

        self.assertFalse(
            (Path(parallel._work_dir) / 'contours').exists())

    def test_fast_method_always_uses_serial(self):
        """`method='fast'` builds a raster that is not in `_hfun_list`.

        The parallel path writes results back by list position, so it must
        not be used here.
        """

        hfun = self._build([self.rast1, self.rast2], 'parallel', method='fast')

        with mock.patch.object(
                type(hfun), '_apply_contours_serial') as serial_mock, \
             mock.patch.object(
                type(hfun), '_apply_contours_parallel') as parallel_mock:
            hfun._apply_contours()

        serial_mock.assert_called_once()
        parallel_mock.assert_not_called()

    def test_worker_reports_error_instead_of_crashing(self):
        """A broken task returns an error dict, it does not raise."""

        result = _contours_task_worker({
            'original_index': 3,
            'hfun_input_path': str(self.tdir / 'does_not_exist.tif'),
            'topo_input_path': str(self.tdir / 'does_not_exist.tif'),
            'output_path': str(self.tdir / 'out.tif'),
            'global_hmin': HMIN,
            'global_hmax': HMAX,
            'contour_files': [],
        })

        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['original_index'], 3)
        self.assertIn('Traceback', result['error'])

    def test_contour_collector_crs_roundtrip(self):
        """Contours written to disk must come back with their CRS."""

        hfun = self._build([self.rast1, self.rast2], 'serial')
        raster_hfuns = [
            h for h in hfun._hfun_list if isinstance(h, HfunRaster)]

        out_dir = self.tdir / 'contour_out'
        hfun._contour_coll.calculate(raster_hfuns, out_dir)

        gdfs = list(hfun._contour_coll)
        self.assertGreater(len(gdfs), 0)
        self.assertEqual(len(gdfs), len(hfun._contour_coll.files))
        for gdf in gdfs:
            self.assertIsNotNone(gdf.crs)
            self.assertTrue(gdf.crs.equals(raster_hfuns[0].crs))


@unittest.skipIf(IS_WINDOWS, 'Multiprocessing file locks are flaky on Windows')
class PoolIsOptional(unittest.TestCase):
    """`nprocs=1` must mean "no pool" and still give the same answer.

    This is what lets `add_feature` run inside a worker process, where
    starting another process is forbidden.
    """

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.grd'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)
        self.line = LineString([(-0.5, -0.5), (-0.5, 0.0), (0.0, 0.0)])

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self.tdir, ignore_errors=True)

    def _raster_hfun(self):
        return ocsmesh.Hfun(
            ocsmesh.Raster(self.rast1), hmin=HMIN, hmax=HMAX)

    def _mesh_hfun(self):
        return ocsmesh.Hfun(ocsmesh.Mesh.open(self.mesh1, crs=4326))

    def test_raster_add_feature_nprocs_1(self):
        """No pool vs. a real pool: identical values."""

        seq = self._raster_hfun()
        seq.add_feature(
            feature=self.line, expansion_rate=0.01,
            target_size=1000, nprocs=1)

        par = self._raster_hfun()
        par.add_feature(
            feature=self.line, expansion_rate=0.01,
            target_size=1000, nprocs=2)

        npt.assert_array_equal(seq.get_values(), par.get_values())
        self.assertLessEqual(float(np.nanmax(seq.get_values())), HMAX)

    def test_mesh_add_feature_nprocs_1(self):
        """Same check for the mesh size function."""

        seq = self._mesh_hfun()
        seq.size_from_mesh()
        seq.add_feature(
            feature=self.line, expansion_rate=0.01,
            target_size=1000, nprocs=1)

        par = self._mesh_hfun()
        par.size_from_mesh()
        par.add_feature(
            feature=self.line, expansion_rate=0.01,
            target_size=1000, nprocs=2)

        npt.assert_array_equal(
            np.asarray(seq.mesh.meshdata.values),
            np.asarray(par.mesh.meshdata.values))

    def test_raster_add_patch_nprocs_1(self):
        """`add_patch` forwards the pool down to `add_feature`."""

        seq = self._raster_hfun()
        seq.add_patch(
            multipolygon=box(-0.6, -0.6, -0.2, -0.2),
            target_size=1000, expansion_rate=0.01, nprocs=1)

        par = self._raster_hfun()
        par.add_patch(
            multipolygon=box(-0.6, -0.6, -0.2, -0.2),
            target_size=1000, expansion_rate=0.01, nprocs=2)

        npt.assert_array_equal(seq.get_values(), par.get_values())

    def test_raster_add_channel_nprocs_1(self):
        """`add_channel` forwards the pool down to `add_patch`."""

        seq = self._raster_hfun()
        seq.add_channel(
            level=0, width=1000, target_size=1000,
            expansion_rate=0.01, nprocs=1)

        par = self._raster_hfun()
        par.add_channel(
            level=0, width=1000, target_size=1000,
            expansion_rate=0.01, nprocs=2)

        npt.assert_array_equal(seq.get_values(), par.get_values())

    def test_pool_starmap_matches_pool(self):
        """The helper returns the same thing with and without a pool."""

        from multiprocessing import Pool

        args = [(1, 2), (3, 4), (5, 6)]
        sequential = ocsmesh.utils.pool_starmap(None, _add_two, args)
        with Pool(processes=2) as p:
            pooled = ocsmesh.utils.pool_starmap(p, _add_two, args)

        self.assertEqual(sequential, pooled)
        self.assertEqual(sequential, [3, 7, 11])


def _add_two(a, b):
    """Module level so `Pool` can pickle it."""

    return a + b


@unittest.skipIf(IS_WINDOWS, 'Multiprocessing file locks are flaky on Windows')
class SourceIndexWithMixedInputs(unittest.TestCase):
    """`source_index` must pick the right hfun AND write back to the right slot.

    Before the fix the parallel paths counted rasters only but then stored
    the result at that same number in `_hfun_list`. With a mesh size
    function earlier in the list, the mesh entry got overwritten.
    """

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.grd'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)
        # Mesh FIRST on purpose: this is what exposes the bug.
        self.in_list = [self.mesh1, self.rast1, self.rast2]

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self.tdir, ignore_errors=True)

    def _build(self, mode):
        hfun = ocsmesh.Hfun(
            self.in_list, hmin=HMIN, hmax=HMAX, nprocs=NPROCS)
        hfun.execution_mode = mode
        return hfun

    def test_flow_limiter_source_index(self):
        serial = self._build('serial')
        serial.add_subtidal_flow_limiter(
            hmin=HMIN, hmax=HMAX, source_index=0)
        serial._apply_flow_limiters()

        parallel = self._build('parallel')
        parallel.add_subtidal_flow_limiter(
            hmin=HMIN, hmax=HMAX, source_index=0)
        parallel._apply_flow_limiters()

        # The mesh entry must still be a mesh, not a raster result.
        self.assertIsInstance(parallel._hfun_list[0], HfunMesh)

        # source_index=0 means the FIRST RASTER, which lives at slot 1.
        _assert_same_sizes(
            serial._hfun_list, parallel._hfun_list, 'flow limiter:')

        # And the untargeted raster must be untouched in both modes.
        untouched = self._build('serial')
        npt.assert_array_equal(
            _values_of(untouched._hfun_list[2]),
            _values_of(parallel._hfun_list[2]))

    def test_const_val_source_index(self):
        serial = self._build('serial')
        serial.add_constant_value(
            value=1500, lower_bound=-100, upper_bound=0, source_index=0)
        serial._apply_const_val()

        parallel = self._build('parallel')
        parallel.add_constant_value(
            value=1500, lower_bound=-100, upper_bound=0, source_index=0)
        parallel._apply_const_val()

        self.assertIsInstance(parallel._hfun_list[0], HfunMesh)
        _assert_same_sizes(
            serial._hfun_list, parallel._hfun_list, 'const value:')

        untouched = self._build('serial')
        npt.assert_array_equal(
            _values_of(untouched._hfun_list[2]),
            _values_of(parallel._hfun_list[2]))


if __name__ == '__main__':
    unittest.main()
