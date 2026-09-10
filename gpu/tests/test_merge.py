"""Tiny native merge regression tests; pass the lumina_lya binary as argv[1]."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cpu' / 'tests'))
from test_reference import fixture

EXECUTABLE = str(Path(sys.argv.pop(1)).resolve())


def invoke(*args):
    return subprocess.run([EXECUTABLE, *map(str, args)], text=True,
                          capture_output=True, timeout=30)


class Merge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts = ROOT / 'gpu' / 'test-artifacts'
        artifacts.mkdir(exist_ok=True)
        cls.base_temp = tempfile.TemporaryDirectory(dir=artifacts)
        cls.base = Path(cls.base_temp.name)
        lightcone = cls.base / 'input.h5'
        sources = cls.base / 'sources.txt'
        fixture(lightcone)
        sources.write_text('6.0025\n')
        cls.originals = []
        for i in range(3):
            output = cls.base / f'part{i}'
            result = invoke('run', '--input', lightcone, '--z0-file', sources,
                            '--output-dir', output, '--backend', 'cpu', '--profile', 'delta',
                            '--dv-step', 1000, '--region', f'{i},{i+1},0,3',
                            '--products', 'maps,spectra,cumulative')
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.originals.append(next(output.rglob('*.hdf5')))

    @classmethod
    def tearDownClass(cls):
        cls.base_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.base)
        self.root = Path(self.temp.name)
        self.tiles = []
        for i, source in enumerate(self.originals):
            path = self.root / f'tile{i}.hdf5'
            shutil.copy2(source, path)
            self.tiles.append(path)
        self.output = self.root / 'merged.hdf5'

    def tearDown(self):
        self.temp.cleanup()

    def command(self, *args, success=True):
        result = invoke(*args)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def merge(self, *options, success=True, tiles=None, output=None):
        return self.command('merge', '--output', output or self.output,
                            *(tiles or self.tiles), *options, success=success)

    def test_recursive_weighted_merge_and_timing(self):
        times = ('ReadSeconds', 'TransferSeconds', 'KernelSeconds', 'WriteSeconds')
        for i, path in enumerate(self.tiles):
            with h5py.File(path, 'r+') as f:
                for j, key in enumerate(times):
                    f.attrs[key] = (i + 1) * (j + 1)
        pair = self.root / 'pair.hdf5'
        self.merge(tiles=self.tiles[:2], output=pair)
        self.merge(tiles=[pair, self.tiles[2]])
        direct = self.root / 'direct.hdf5'
        self.merge(output=direct)
        with h5py.File(self.output) as a, h5py.File(direct) as b:
            for key in ('tau_band_avgs', 'taus', 'T_bands', 'T_cum_bands'):
                np.testing.assert_allclose(a[key][:], b[key][:], rtol=1e-13, atol=1e-13)
            self.assertEqual(a.attrs['PixelCount'], 9)
            self.assertEqual(a.attrs['MergedTileCount'], 2)
            for key in ('NumPixels', 'NumFreq', 'X0', 'X1', 'Y0', 'Y1', 'PixelCount', 'MergedTileCount'):
                self.assertEqual(np.asarray(a.attrs[key]).dtype.kind, 'u', key)
            config = a.attrs['Configuration']
            if isinstance(config, bytes):
                config = config.decode()
            self.assertTrue(config.endswith(';region=0,3,0,3'))
            for j, key in enumerate(times):
                self.assertEqual(a.attrs[key], 6 * (j + 1))
            np.testing.assert_array_equal(a['T_bands'][:], a['T_cum_bands'][:, -1])

    def test_resume_detects_changed_tiles(self):
        self.merge()
        self.merge('--resume', tiles=list(reversed(self.tiles)))
        with h5py.File(self.tiles[0], 'r+') as f:
            f.attrs['ReviewMarker'] = 1.
        result = self.merge('--resume', success=False)
        self.assertIn('input tile files', result.stderr)

    def test_huge_band_shape_rejected_before_read(self):
        with h5py.File(self.tiles[0], 'r+') as f:
            del f['T_bands']
            f.create_dataset('T_bands', shape=(10**9,), dtype='f8', chunks=(1,))
        result = self.merge('--dry-run', success=False)
        self.assertIn('T_bands', result.stderr)
        self.assertFalse(self.output.exists())

    def test_huge_cumulative_depth_rejected_by_budget(self):
        with h5py.File(self.tiles[0], 'r+') as f:
            del f['T_cum_bands']
            f.create_dataset('T_cum_bands', shape=(5, 2**40), dtype='f8', chunks=(1, 1))
        result = self.merge('--dry-run', success=False)
        self.assertIn('host-memory-mib', result.stderr)

    def test_frequency_and_redshift_coordinates_match(self):
        for key in ('Dvs', 'T_redshifts', 'frequency_weights'):
            with self.subTest(key=key):
                shutil.copy2(self.originals[1], self.tiles[1])
                with h5py.File(self.tiles[1], 'r+') as f:
                    if key == 'frequency_weights':
                        f[key][0, 0] += .001
                        f[key][0, 1] -= .001
                    else:
                        f[key][0] += 1e-7
                result = self.merge('--dry-run', success=False)
                self.assertIn(key, result.stderr)

    def test_nonfinite_and_nonphysical_means(self):
        for value in (float('nan'), -1., 1.1):
            with self.subTest(value=value):
                shutil.copy2(self.originals[0], self.tiles[0])
                with h5py.File(self.tiles[0], 'r+') as f:
                    f['T_bands'][0] = value
                self.assertIn('transmission', self.merge('--dry-run', success=False).stderr)

    def test_cumulative_monotonicity(self):
        with h5py.File(self.tiles[0], 'r+') as f:
            f['T_cum_bands'][2, 0] = .1
            f['T_cum_bands'][2, 1] = .5
        self.assertIn('increases', self.merge('--dry-run', success=False).stderr)

    def test_area_overflow(self):
        with h5py.File(self.tiles[0], 'r+') as f:
            f.attrs['NumPixels'] = 2**40
            f.attrs['X0'] = 0
            f.attrs['X1'] = 2**40
            f.attrs['Y0'] = 0
            f.attrs['Y1'] = 2**40
        self.assertIn('area overflow', self.merge('--dry-run', success=False).stderr)

    def test_symlink_output_cannot_replace_input(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        original = self.tiles[0].read_bytes()
        self.merge('--overwrite', output=alias / self.tiles[0].name, success=False)
        self.assertEqual(self.tiles[0].read_bytes(), original)

    def test_map_statistics_failure_is_atomic(self):
        with h5py.File(self.tiles[0], 'r+') as f:
            f['tau_band_avgs'][:] = 100.
        self.assertIn('map average', self.merge(success=False).stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + '.lock').exists())
        self.assertEqual(list(self.root.glob('merged.hdf5.partial.*')), [])

    def test_missing_timing_does_not_copy_first_tile(self):
        with h5py.File(self.tiles[1], 'r+') as f:
            del f.attrs['TransferSeconds']
        self.merge()
        with h5py.File(self.output) as f:
            self.assertNotIn('TransferSeconds', f.attrs)


if __name__ == '__main__':
    unittest.main()
