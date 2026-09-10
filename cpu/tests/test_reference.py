"""Tiny, deterministic reference tests suitable for a login node."""

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reference as ref


def fixture(path, n=3, transparent=False, dtype='f4'):
    """A tiny lightcone with three cells and explicit code-unit metadata."""
    with h5py.File(path, 'w') as f:
        h = f.create_group('Header')
        h.attrs.update(NumPixels=n, HubbleParam=.6774, Omega0=.3089,
                       OmegaBaryon=.0486, UnitLength_in_cm=3.085677581467192e21,
                       UnitMass_in_g=1.988435e43, UnitVelocity_in_cm_per_s=1e5,
                       OpeningAngle=.06283185307179587)
        f['Redshifts'] = [6.003, 6.002, 6.001, 6.]
        f['Distances'] = [1003., 1002., 1001., 1000.]
        f['Distances'].attrs['to_cgs'] = h.attrs['UnitLength_in_cm'] / h.attrs['HubbleParam']
        f.create_dataset('Temperature', data=np.full((n, n, 3), 1e4, dtype=dtype))
        density = np.linspace(.8, 1.2, n*n*3).reshape(n, n, 3) * 1e-10
        f.create_dataset('Density', data=density.astype(dtype))
        f.create_dataset('HII_Fraction', data=np.full((n, n, 3), 1. if transparent else .9, dtype=dtype))
        velocities = np.zeros((n, n, 3, 3), dtype=dtype)
        velocities[..., 2] = 20.
        f.create_dataset('Velocities', data=velocities)


class Operators(unittest.TestCase):
    def test_delta_crossings_and_boundaries(self):
        self.assertEqual(ref.comb_delta(-1., 2., 3.), 0.)
        self.assertEqual(ref.comb_delta(0., 2., 3.), 0.)
        self.assertEqual(ref.comb_delta(2., 2., 3.), 3 * math.sqrt(math.pi))
        # The downstream endpoint belongs to exactly one adjacent segment.
        self.assertEqual(ref.comb_delta(2., 2., 3.) + ref.comb_delta(0., 2., 3.),
                         3 * math.sqrt(math.pi))
        opacity = 2. * math.sqrt(math.pi)
        expected = -math.log(.5 + .5 * math.exp(-opacity))
        self.assertAlmostEqual(ref.band_delta(-1., 3., 2., 2.), expected, places=14)
        self.assertAlmostEqual(ref.band_delta(-1., 3., 2., 1e8), math.log(2.), places=14)
        self.assertEqual(ref.band_delta(0., 2., 2., 1e8), math.sqrt(math.pi) * 1e8)

    def test_gaussian_operators(self):
        lo, hi, drift, scale = -.3, 1.2, .7, 2.
        expected_comb = scale * math.sqrt(math.pi) / 2 * (erf(hi) - erf(hi - drift))
        self.assertAlmostEqual(ref.comb_voigt(hi, drift, scale, 0.), expected_comb, places=14)
        expected_band = quad(lambda x: scale * math.sqrt(math.pi) / 2 * (erf(x) - erf(x - drift)), lo, hi)[0] / (hi - lo)
        self.assertAlmostEqual(ref.band_voigt(lo, hi, drift, scale, 0.), expected_band, places=13)

    def test_full_profile_error_is_separate(self):
        a = 4.7e-4
        for lo, hi in ((-2., 3.), (1., 1.02), (8., 9.), (20., 20.01), (100., 110.)):
            expansion = ref.voigt_integral(lo, hi, a)
            full = ref.voigt_integral(lo, hi, a, full=True)
            self.assertLess(abs(expansion - full), 3e-10 * max(1., full))

    def test_quadrature_and_wing_branch(self):
        for center in (0., 5.15, 12., 20., -20., 1000.):
            lo, hi, drift, scale, a = center - .07, center + .08, .11, 4., 4.7e-4
            nested = quad(lambda x: ref.comb_voigt(x, drift, scale, a), lo, hi,
                          epsabs=1e-13, epsrel=1e-12)[0] / (hi - lo)
            actual = ref.band_voigt(lo, hi, drift, scale, a)
            self.assertAlmostEqual(actual, nested, delta=1e-12 * max(1., nested))
        left, right = ref.profile_value(12 - 1e-8, a), ref.profile_value(12 + 1e-8, a)
        self.assertLess(abs(left - right), 1e-13)

    def test_stable_band_average(self):
        grid = ref.spectral_grid(ref.Config(dv_step=1000))
        tau = np.full((2, grid.size), 1e5)
        np.testing.assert_allclose(ref.band_optical_depths(tau, grid.weights), 1e5, atol=1e-10)

    def test_grid_weights(self):
        for sampling in ('band', 'comb'):
            grid = ref.spectral_grid(ref.Config(sampling=sampling, dv_step=37.))
            np.testing.assert_allclose(grid.weights.sum(axis=1), 1., atol=3e-16)
            self.assertTrue(np.all(np.diff(grid.velocity_edges_kms) <= 37. + 1e-12))
            for edge in ref.BAND_EDGES:
                self.assertIn(edge, grid.velocity_edges_kms)
        legacy = ref.spectral_grid(ref.Config(legacy=True))
        np.testing.assert_array_equal(np.count_nonzero(legacy.weights, axis=1), [300, 80, 41, 80, 300])
        self.assertEqual(legacy.size, 801)


class InputAndPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'input.h5'
        fixture(self.input)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_geometry_partial_and_empty(self):
        info = ref.inspect_input(self.input)
        config = ref.Config()
        i, z, chi = ref.source_geometry(info, 6.0015, config)
        self.assertEqual(i, 1)
        self.assertEqual(z, 6.0015)
        self.assertAlmostEqual(chi, 1001.5)
        zm, hubble, length = next(ref.segments(info, i, z, config))
        expected = info['distance_to_cgs'] / .001 * math.log1p(.0005 / 7.001)
        self.assertAlmostEqual(length / expected, 1., places=11)
        self.assertEqual(ref.source_geometry(info, 6.002, config)[0], 1)
        self.assertEqual(ref.source_geometry(info, 6., config)[0], 3)
        self.assertEqual(ref.source_geometry(info, 6.0015, ref.Config(legacy=True))[1], 6.002)
        for bad in (5.9, 6.1, float('nan')):
            with self.assertRaises(ValueError):
                ref.source_geometry(info, bad, config)

    def test_doppler_sign(self):
        info = ref.inspect_input(self.input)
        # A red photon can be blueshifted into resonance in outward-moving gas.
        grid = ref.SpectralGrid(np.array([100.]), np.array([1/(1+100*ref.KM/ref.C)]),
                                np.array([1/(1+100*ref.KM/ref.C)]), np.array([]), np.array([]), np.ones((5, 1)))
        config = ref.Config(sampling='comb', profile='delta')
        def opacity(v):
            tau = np.zeros((1, 1))
            # Midpoint resonance requires v~100 km/s, corrected for sqrt(a) code units.
            ref._add_segment(tau, np.array([1e4]), np.array([1e-10]), np.array([.9]),
                             np.array([v * math.sqrt(7.)]), 6., 1e-15, 1e21, 6., grid, info['header'], config)
            return tau[0, 0]
        self.assertGreater(opacity(100.), 0.)
        self.assertEqual(opacity(-100.), 0.)

    def test_all_modes_tiling_and_normalization(self):
        for sampling in ('band', 'comb'):
            for profile in ('delta', 'voigt'):
                config = ref.Config(sampling=sampling, profile=profile, dv_step=1000.,
                                    tile_size=2, depth_slab=2, products=('maps', 'spectra', 'cumulative'))
                out1 = ref.run(self.input, [[6.0025, .1]], self.root / (sampling+profile+'1'), config)[0]
                out2 = ref.run(self.input, [[6.0025, .1]], self.root / (sampling+profile+'2'),
                               replace(config, tile_size=3, depth_slab=1))[0]
                with h5py.File(out1) as a, h5py.File(out2) as b:
                    for name in ('taus', 'tau_band_avgs', 'T_bands', 'T_cum_bands'):
                        np.testing.assert_allclose(a[name][:], b[name][:], rtol=1e-12, atol=1e-13)
                    np.testing.assert_array_equal(a['T_bands'][:], a['T_cum_bands'][:, -1])
                    np.testing.assert_allclose(a['T_bands'][:], np.exp(-a['tau_band_avgs'][:]).mean(axis=(1, 2)))
                    np.testing.assert_array_equal(a['T_redshifts'][:], [6.002, 6.001, 6.])
                    self.assertEqual(a.attrs['PixelCount'], 9)
                    self.assertEqual(a.attrs['Redshift'], 6.0025)
                    np.testing.assert_array_equal(a['z_spread'][:], [6.0025, .1])
                    self.assertTrue(np.all(np.diff(a['T_cum_bands'][:], axis=1) <= 1e-15))

    def test_transparent_and_observer_source(self):
        fixture(self.input, transparent=True, dtype='f8')
        config = ref.Config(dv_step=1000., products=('maps', 'spectra', 'cumulative'))
        outputs = ref.run(self.input, [6.0025, 6.], self.root / 'clear', config)
        for path in outputs:
            with h5py.File(path) as f:
                np.testing.assert_allclose(f['T_bands'][:], 1., atol=3e-16)
                np.testing.assert_array_equal(f['taus'][:], 0.)
                np.testing.assert_allclose(f['tau_band_avgs'][:], 0., atol=3e-16)
        with h5py.File(outputs[-1]) as f:
            self.assertEqual(f['T_cum_bands'].shape, (5, 0))
            self.assertEqual(f['T_redshifts'].shape, (0,))

    def test_cold_neutral_voigt_requires_full_profile(self):
        fixture(self.input, n=1)
        with h5py.File(self.input, 'r+') as f:
            f['Temperature'][:] = .01
        for legacy in (False, True):
            config = ref.Config(dv_step=1000., legacy=legacy)
            with self.assertRaisesRegex(ValueError, 'requires a full-profile solver'):
                ref.run(self.input, [6.003], self.root / f'cold_voigt_{legacy}', config)
        delta = ref.Config(profile='delta', dv_step=1000.)
        self.assertEqual(len(ref.run(self.input, [6.003], self.root / 'cold_delta', delta)), 1)
        with h5py.File(self.input, 'r+') as f:
            f['HII_Fraction'][:] = 1.
        self.assertEqual(len(ref.run(self.input, [6.003], self.root / 'cold_clear', ref.Config(dv_step=1000.))), 1)

    def test_resume_and_no_partial_publication(self):
        config = ref.Config(sampling='band', profile='delta', dv_step=1000.)
        directory = self.root / 'resume'
        outputs = ref.run(self.input, [6.002], directory, config)
        self.assertEqual(ref.run(self.input, [6.002], directory, config, resume=True), outputs)
        with self.assertRaises(FileExistsError):
            ref.run(self.input, [6.002], directory, config)
        with self.assertRaises(FileExistsError):
            ref.run(self.input, [6.002], directory, replace(config, dv_step=500), resume=True)
        with h5py.File(self.input, 'r+') as f:
            f['Temperature'][0, 0, 0] = -1.
        with self.assertRaises(ValueError):
            ref.run(self.input, [6.003], self.root / 'bad', config)
        self.assertEqual(list((self.root / 'bad').iterdir()), [])

    def test_missing_vds_source(self):
        virtual = self.root / 'virtual.h5'
        fixture(virtual)
        with h5py.File(virtual, 'r+') as f:
            del f['Density']
            layout = h5py.VirtualLayout((3, 3, 3), dtype='f4')
            layout[:] = h5py.VirtualSource('missing.h5', 'Density', shape=(3, 3, 3))
            f.create_virtual_dataset('Density', layout)
        with self.assertRaises(FileNotFoundError):
            ref.inspect_input(virtual)

    def test_truncated_vds_and_source_resume_identity(self):
        virtual = self.root / 'virtual.h5'
        fixture(virtual)
        source = self.root / 'source.h5'
        with h5py.File(source, 'w') as f:
            f['Density'] = np.full((3, 3, 3), 1e-10, dtype='f4')
        with h5py.File(virtual, 'r+') as f:
            del f['Density']
            layout = h5py.VirtualLayout((3, 3, 3), dtype='f4')
            layout[:] = h5py.VirtualSource(source.name, 'Density', shape=(3, 3, 3))
            f.create_virtual_dataset('Density', layout)
        config = ref.Config(profile='delta', dv_step=1000.)
        ref.run(virtual, [6.002], self.root / 'virtual_output', config)
        with h5py.File(source, 'r+') as f:
            f['Density'][0, 0, 0] = 2e-10
        with self.assertRaises(FileExistsError):
            ref.run(virtual, [6.002], self.root / 'virtual_output', config, resume=True)
        with h5py.File(source, 'w') as f:
            f['Density'] = np.ones((3, 3, 2), dtype='f4')
        with self.assertRaises(ValueError):
            ref.inspect_input(virtual)

    def test_memory_and_regions(self):
        config = ref.Config(profile='delta', host_memory_mb=.001)
        with self.assertRaises(ValueError):
            ref.run(self.input, [6.002], self.root / 'nomem', config)
        with self.assertRaises(ValueError):
            ref.run(self.input, [6.002], self.root / 'outside', region=(0, 4, 0, 3))


if __name__ == '__main__':
    unittest.main()
