#!/usr/bin/env python3
"""Memory-bounded scalar reference for absorption-only lightcone transmission.

The four operators follow Almualla et al. (2026), equations 16, 19–24.
Band+Voigt applies exp(-mean(tau)) separately to each segment; it is the
paper's lower-bound closure, not an exact integral of transmitted intensity.
Voigt operators integrate the second-order profile using independent SciPy
quadrature. This deliberately favors a transparent reference over speed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn, erf, erfc, logsumexp, wofz

C = 2.99792458e10
KM = 1e5
MPC = 3.085677581467192e24
KB = 1.380648813e-16
MH = 1.6735327e-24
ME = 9.109382917e-28
EE = 4.80320451e-10
NU0 = 2.466e15
F12 = 0.4162
DNUL = 9.936e7
SQRT_PI = math.sqrt(math.pi)
BAND_EDGES = np.array([-2000., -500., -100., 100., 500., 2000.])
FIELDS = ('Temperature', 'Density', 'HII_Fraction', 'Velocities')
REQUIRED_ATTRS = ('HubbleParam', 'UnitVelocity_in_cm_per_s',
                  'UnitLength_in_cm', 'UnitMass_in_g', 'OpeningAngle')


@dataclass(frozen=True)
class Config:
    sampling: str = 'band'
    profile: str = 'voigt'
    dv_step: float = 5.
    max_dln: float = 1e-3
    tile_size: int = 4
    depth_slab: int = 16
    host_memory_mb: float = 512.
    products: tuple[str, ...] = ('maps',)
    legacy: bool = False

    def validate(self):
        if self.sampling not in ('band', 'comb') or self.profile not in ('voigt', 'delta'):
            raise ValueError('sampling must be band/comb and profile must be voigt/delta')
        for name in ('dv_step', 'max_dln', 'host_memory_mb'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.tile_size < 1 or self.depth_slab < 1:
            raise ValueError('tile_size and depth_slab must be positive integers')
        if not self.products or set(self.products) - {'maps', 'spectra', 'cumulative'}:
            raise ValueError('products must contain maps, spectra and/or cumulative')


@dataclass(frozen=True)
class SpectralGrid:
    offsets_kms: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    q_edges: np.ndarray
    velocity_edges_kms: np.ndarray
    weights: np.ndarray

    @property
    def size(self):
        return len(self.q_lo)


def spectral_grid(config=Config()):
    """Return frequency ratios nu/nu0 and normalized reporting-band weights."""
    config.validate()
    if config.legacy:
        u = np.linspace(-2000., 2000., 801)
        q = 1. / (1. + u * KM / C)
        weights = np.zeros((5, 801))
        begin = 0
        for band, count in enumerate((300, 80, 41, 80, 300)):
            weights[band, begin:begin + count] = 1. / count
            begin += count
        return SpectralGrid(u, q, q, q, u, weights)
    pieces = [np.linspace(lo, hi, int(math.ceil((hi - lo) / config.dv_step)) + 1)
              for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    edges = np.concatenate([pieces[0]] + [p[1:] for p in pieces[1:]])
    qe = 1. / (1. + edges * KM / C)
    if config.sampling == 'band':
        u = .5 * (edges[:-1] + edges[1:])
        lo, hi = qe[1:], qe[:-1]
        weights = np.zeros((5, len(u)))
        for band, (a, b) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
            selected = (u > a) & (u < b)
            weights[band, selected] = hi[selected] - lo[selected]
    else:
        u, lo, hi = edges, qe, qe
        weights = np.zeros((5, len(u)))
        for band, (a, b) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
            nodes = np.flatnonzero((u >= a) & (u <= b))
            half_width = .5 * (qe[nodes[:-1]] - qe[nodes[1:]])
            weights[band, nodes[:-1]] += half_width
            weights[band, nodes[1:]] += half_width
    weights /= weights.sum(axis=1)[:, None]
    return SpectralGrid(u, lo, hi, qe, edges, weights)


def profile_value(x, a, order=2, full=False):
    """Voigt H(x), with stable wings of the paper's expansion through O(a²)."""
    if full:
        return float(wofz(complex(x, a)).real)
    xx = x * x
    if abs(x) >= 12.:
        # 2*x*Dawson(x)-1 = sum (2n-1)!!/(2*x²)^n.
        term = .5 / xx
        wing = term
        for n in range(2, 18):
            term *= (2 * n - 1) / (2 * xx)
            wing += term
    else:
        wing = 2 * x * float(dawsn(x)) - 1.
    gaussian = math.exp(-xx)
    return gaussian * (1. + (a * a * (1. - 2. * xx) if order == 2 else 0.)) + 2. * a / SQRT_PI * wing


def _split_quad(function, lo, hi, extra=()):
    if hi <= lo:
        return 0.
    splits = sorted({lo, hi, *(x for x in (-30., -12., -8., -3., -1., 0., 1., 3., 8., 12., 30., *extra)
                              if lo < x < hi)})
    return math.fsum(quad(function, a, b, epsabs=2e-14, epsrel=2e-13, limit=150)[0]
                     for a, b in zip(splits[:-1], splits[1:]))


def voigt_integral(lo, hi, a, order=2, full=False):
    """Independent scalar integral; distant differences never subtract primitives."""
    if hi <= lo:
        return 0.
    if full or (lo * hi > 0 and (min(abs(lo), abs(hi)) > 8 or hi - lo < .1)):
        return _split_quad(lambda x: profile_value(x, a, order, full), lo, hi)
    if lo >= 0:
        erf_diff = float(erfc(lo) - erfc(hi))
    elif hi <= 0:
        erf_diff = float(erfc(-hi) - erfc(-lo))
    else:
        erf_diff = float(erf(hi) - erf(lo))
    value = .5 * SQRT_PI * erf_diff - 2. * a / SQRT_PI * float(dawsn(hi) - dawsn(lo))
    if order == 2:
        value += a * a * (hi * math.exp(-hi * hi) - lo * math.exp(-lo * lo))
    return value


def comb_delta(x_entry, drift, scale):
    """Equation 21; the upstream boundary is open, the downstream boundary closed."""
    return SQRT_PI * scale if 0. < x_entry <= drift and drift > 0. else 0.


def band_delta(xlo, xhi, drift, scale):
    """Equation 24 as an effective optical depth, including opaque partial bins."""
    if drift <= 0 or scale <= 0:
        return 0.
    fraction = max(0., min(xhi, drift) - max(xlo, 0.)) / (xhi - xlo)
    fraction = min(1., fraction)
    opacity = SQRT_PI * scale
    if fraction <= 0:
        return 0.
    if fraction >= 1:
        return opacity
    return -float(np.logaddexp(math.log1p(-fraction), math.log(fraction) - opacity))


def comb_voigt(x_entry, drift, scale, a, order=2, full=False):
    if drift <= 0 or scale <= 0:
        return 0.
    return scale * voigt_integral(x_entry - drift, x_entry, a, order, full)


def band_voigt(xlo, xhi, drift, scale, a, order=2, full=False):
    """Equation 19 through an independent profile integral with overlap weights.

    Exchanging the frequency and path integrals yields a trapezoidal window.
    Quadrature avoids both the native hypergeometric fit and cancellation in
    its four-term second difference, providing a useful independent reference.
    """
    if drift <= 0 or scale <= 0:
        return 0.
    width = xhi - xlo
    if width <= 0:
        raise ValueError('band frequency interval must have positive width')
    def integrand(x):
        overlap = max(0., min(xhi, x + drift) - max(xlo, x))
        return profile_value(x, a, order, full) * (overlap / width)
    return scale * _split_quad(integrand, xlo - drift, xhi,
                              (xlo, xhi - drift))


def band_optical_depths(tau, weights):
    """Stable -log of the weighted transmitted intensity for each reporting band."""
    out = np.empty((5, tau.shape[0]))
    for b in range(5):
        use = weights[b] > 0
        out[b] = -logsumexp(-tau[:, use] + np.log(weights[b, use]), axis=1)
    return np.maximum(out, 0.)


def _validate_virtual(dataset, visited=None, identities=None):
    """Reject unavailable VDS sources before HDF5 can silently use fill values."""
    if not dataset.is_virtual:
        return
    visited = set() if visited is None else set(visited)
    key = (str(Path(dataset.file.filename).resolve()), dataset.name)
    if key in visited:
        raise ValueError(f'cyclic VDS source: {key}')
    visited.add(key)
    for mapping in dataset.virtual_sources():
        filename, name = mapping.file_name, mapping.dset_name
        if isinstance(filename, bytes):
            filename = filename.decode()
        if isinstance(name, bytes):
            name = name.decode()
        if '%' in filename or '%' in name:
            raise ValueError('printf-pattern VDS sources are not supported by the reference')
        source = Path(dataset.file.filename) if filename == '.' else Path(filename)
        if not source.is_absolute() and filename != '.':
            source = Path(dataset.file.filename).resolve().parent / source
        if not source.is_file():
            raise FileNotFoundError(f'missing VDS source: {source}')
        if identities is not None:
            stat = source.stat()
            identities.append((str(source.resolve()), stat.st_size, stat.st_mtime_ns))
        with h5py.File(source, 'r') as f:
            if name not in f:
                raise ValueError(f'missing VDS source dataset {name} in {source}')
            linked = f[name]
            if not isinstance(linked, h5py.Dataset) or linked.dtype.kind != 'f' or linked.dtype.itemsize not in (4, 8):
                raise ValueError(f'VDS source must be a float32/float64 dataset: {source}:{name}')
            selected = mapping.src_space.get_select_npoints()
            target_selected = mapping.vspace.get_select_npoints()
            if selected < 0 or target_selected < 0:
                raise ValueError('VDS selections must be finite')
            if mapping.src_space.get_select_type() == h5py.h5s.SEL_ALL:
                if linked.size != target_selected:
                    raise ValueError(f'whole VDS source extent disagrees with mapping: {source}:{name}')
            elif selected:
                if mapping.src_space.get_simple_extent_ndims() != linked.ndim or selected != target_selected:
                    raise ValueError(f'VDS source dimensions disagree with mapping: {source}:{name}')
                _, upper = mapping.src_space.get_select_bounds()
                if any(bound >= dim for bound, dim in zip(upper, linked.shape)):
                    raise ValueError(f'VDS source selection exceeds extent: {source}:{name}')
            elif target_selected:
                raise ValueError('empty VDS source selection has nonempty destination')
            _validate_virtual(linked, visited, identities)


def inspect_input(path, legacy=False):
    """Read metadata and one-dimensional geometry only; never read gas cubes."""
    with h5py.File(path, 'r') as f:
        if 'Header' not in f:
            raise ValueError('input has no Header group')
        header = {k: np.asarray(v).item() if np.asarray(v).size == 1 else np.asarray(v).tolist()
                  for k, v in f['Header'].attrs.items()}
        for name in REQUIRED_ATTRS + (('Omega0',) if legacy else ()):
            if name not in header or not math.isfinite(float(header[name])):
                raise ValueError(f'missing or nonfinite Header attribute {name}')
        for name in REQUIRED_ATTRS[:4]:
            if float(header[name]) <= 0:
                raise ValueError(f'Header {name} must be positive')
        if legacy and header['Omega0'] <= 0:
            raise ValueError('Omega0 must be positive for legacy geometry')
        virtual_identities = []
        for name in FIELDS:
            if name not in f:
                raise ValueError(f'missing dataset {name}')
            if f[name].dtype.kind != 'f' or f[name].dtype.itemsize not in (4, 8):
                raise ValueError(f'{name} must contain float32 or float64')
            _validate_virtual(f[name], identities=virtual_identities)
        shape = f['Density'].shape
        if len(shape) != 3 or any(d < 1 for d in shape) or shape[0] != shape[1]:
            raise ValueError('fields must have square spatial shape [n,n,depth]')
        if any(f[name].shape != shape for name in FIELDS[:-1]) or f['Velocities'].shape != (*shape, 3):
            raise ValueError('gas field dimensions do not match')
        if 'NumPixels' in header and int(header['NumPixels']) != shape[0]:
            raise ValueError('NumPixels does not match gas field dimensions')
        z = np.asarray(f['Redshifts'], dtype=np.float64)
        if z.shape != (shape[2] + 1,) or not np.all(np.isfinite(z)) or np.any(z <= -1) or np.any(np.diff(z) >= 0):
            raise ValueError('Redshifts must have depth+1 finite, strictly decreasing edges above -1')
        distances = None
        if 'Distances' in f:
            distances = np.asarray(f['Distances'], dtype=np.float64)
        if not legacy:
            if distances is None or distances.shape != z.shape or not np.all(np.isfinite(distances)):
                raise ValueError('physical geometry requires finite Distances at every redshift edge')
            if np.any(np.diff(distances) >= 0):
                raise ValueError('Distances must decrease toward the observer with Redshifts')
        distance_to_cgs = float(f['Distances'].attrs.get('to_cgs', header['UnitLength_in_cm'] / header['HubbleParam'])) if distances is not None else None
        if distance_to_cgs is not None and (not math.isfinite(distance_to_cgs) or distance_to_cgs <= 0):
            raise ValueError('Distances to_cgs must be finite and positive')
        return dict(shape=shape, header=header, redshifts=z, distances=distances,
                    distance_to_cgs=distance_to_cgs,
                    virtual_source_identities=sorted(set(virtual_identities)),
                    chunks={name: f[name].chunks for name in FIELDS},
                    virtual={name: f[name].is_virtual for name in FIELDS})


def source_geometry(info, requested_z, config=Config()):
    """Resolve the source cell, actual source redshift and comoving code distance."""
    z = info['redshifts']
    if not math.isfinite(requested_z) or requested_z > z[0] or requested_z < z[-1]:
        raise ValueError(f'source {requested_z} outside [{z[-1]}, {z[0]}]')
    index = int(np.searchsorted(-z, -requested_z, side='right') - 1)
    if index == len(z) - 1:
        actual = requested_z
    else:
        actual = float(z[index]) if config.legacy else requested_z
    chi = None
    if info['distances'] is not None:
        chi = float(np.interp(actual, z[::-1], info['distances'][::-1]))
    return index, actual, chi


def segments(info, cell, source_z, config):
    """Yield (midpoint z, H_eff, proper length) in propagation order."""
    z = info['redshifts']
    hi, lo = min(float(z[cell]), source_z), float(z[cell + 1])
    if hi <= lo:
        return
    header = info['header']
    if config.legacy:
        hubble = header['HubbleParam'] * 100 * KM / MPC * math.sqrt(header['Omega0']) * (1 + hi)**1.5
        yield hi, hubble, C * (hi - lo) / hubble / (1 + hi)
        return
    dchi = (info['distances'][cell] - info['distances'][cell + 1]) * info['distance_to_cgs']
    dz = z[cell] - z[cell + 1]
    hubble = C * dz / dchi
    dln = math.log1p((hi - lo) / (1 + lo))
    nsegments = int(math.ceil(dln / config.max_dln))
    loglo = math.log1p(lo)
    step = dln / nsegments
    for j in range(nsegments):
        logmid = loglo + dln - (j + .5) * step
        yield math.expm1(logmid), hubble, float(dchi / dz * step)


def _tile_shape(config, frequencies, depth):
    # Tau, temporary reductions, the six input components and Python overhead.
    budget = config.host_memory_mb * 1024**2
    slab = min(config.depth_slab, depth)
    per_pixel = 8 * (3 * frequencies + 12 * slab + 32)
    available = budget - 8 * 8 * (depth + 1) - 1024**2
    side = min(config.tile_size, int(math.sqrt(max(0., available) / per_pixel)))
    if side < 1:
        raise ValueError('host memory limit cannot hold one ray and requested spectral/depth buffers')
    return side, slab


def _projected_velocity(vel, n, angle, xs, ys):
    tx = np.tan((np.arange(xs.start, xs.stop) + .5 - .5 * n) * angle / n)
    ty = np.tan((np.arange(ys.start, ys.stop) + .5 - .5 * n) * angle / n)
    norm = np.sqrt(1 + tx[:, None]**2 + ty[None, :]**2)
    return ((vel[..., 0] * tx[:, None, None] + vel[..., 1] * ty[None, :, None] + vel[..., 2]) / norm[..., None])


def _add_segment(tau, temperature, density, hii, velocity_code, zm, hubble, length, source_z, grid, header, config):
    scale_factor = 1. / (1. + zm)
    rho_cgs = header['UnitMass_in_g'] * header['HubbleParam']**2 / (header['UnitLength_in_cm'] * scale_factor)**3
    velocity_factor = math.sqrt(scale_factor) * header['UnitVelocity_in_cm_per_s']
    for p in range(tau.shape[0]):
        if density[p] == 0 or hii[p] == 1:
            continue
        b = math.sqrt(2 * KB * temperature[p] / MH)
        a = DNUL * C / (2 * NU0 * b)
        if (config.legacy or config.profile == 'voigt') and a > .1:
            raise ValueError(f'Voigt damping a={a:.6g} exceeds 0.1; this neutral cell requires a full-profile solver')
        sigma = F12 * SQRT_PI * EE**2 / (ME * b * NU0)
        k0 = .76 * density[p] * rho_cgs * (1 - hii[p]) * sigma / MH
        scale, drift = k0 * b / hubble, hubble * length / b
        v = velocity_code[p] * velocity_factor
        if abs(v) >= C:
            raise ValueError('projected peculiar velocity exceeds the nonrelativistic model')
        if config.legacy:
            entries = -((grid.offsets_kms * KM / C + 1) * (1 + source_z) / (1 + zm) - 1) * C / b - v / b
            for j, entry in enumerate(entries):
                tau[p, j] += comb_voigt(float(entry), drift, scale, a, order=1)
        else:
            multiplier = (1 + zm) / (1 + source_z) * (1 + v / C)
            lows = (grid.q_lo * multiplier - 1) * C / b + .5 * drift
            highs = (grid.q_hi * multiplier - 1) * C / b + .5 * drift
            for j in range(grid.size):
                if config.sampling == 'comb':
                    dtau = (comb_voigt(float(lows[j]), drift, scale, a) if config.profile == 'voigt'
                            else comb_delta(float(lows[j]), drift, scale))
                else:
                    dtau = (band_voigt(float(lows[j]), float(highs[j]), drift, scale, a) if config.profile == 'voigt'
                            else band_delta(float(lows[j]), float(highs[j]), drift, scale))
                if not math.isfinite(dtau) or dtau < -1e-12:
                    raise ValueError('Voigt approximation produced invalid opacity; inspect temperature and units')
                tau[p, j] += max(0., dtau)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(input_path, sources, output_dir, config=Config(), region=None, resume=False):
    """Write one atomic HDF5 product per source using spatial tiles and depth slabs."""
    config.validate()
    info = inspect_input(input_path, config.legacy)
    grid = spectral_grid(config)
    n, _, depth = info['shape']
    region = tuple(region or (0, n, 0, n))
    if len(region) != 4 or not (0 <= region[0] < region[1] <= n and 0 <= region[2] < region[3] <= n):
        raise ValueError('region must be x0,x1,y0,y1 within input spatial bounds')
    sources = [[float(s)] if np.isscalar(s) else [float(z) for z in s] for s in sources]
    if not sources or any(not s or not all(math.isfinite(z) for z in s) for s in sources):
        raise ValueError('at least one finite source redshift is required')
    if len({s[0] for s in sources}) != len(sources):
        raise ValueError('duplicate source redshifts would collide in output filenames')
    geometries = [source_geometry(info, s[0], config) for s in sources]
    tile, slab_size = _tile_shape(config, grid.size, depth)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path).resolve()
    stat = input_path.stat()
    results = []
    with h5py.File(input_path, 'r') as f:
        for source, (i0, actual_z, source_chi) in zip(sources, geometries):
            output = output_dir / f'tau_map_{float(source[0])}.hdf5'
            settings = dict(config=asdict(config), input=str(input_path), input_size=stat.st_size,
                            input_mtime_ns=stat.st_mtime_ns, region=region, source=source,
                            virtual_source_identities=info['virtual_source_identities'],
                            actual_source_redshift=actual_z, format_version=1)
            config_json = json.dumps(settings, sort_keys=True, separators=(',', ':'))
            if output.exists():
                with h5py.File(output, 'r') as old:
                    if resume and old.attrs.get('Complete', 0) == 1 and old.attrs.get('config_json') == config_json:
                        results.append(output)
                        continue
                raise FileExistsError(f'output exists or resume configuration differs: {output}')
            fd, tmpname = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output_dir)
            os.close(fd)
            try:
                start = time.perf_counter()
                read_seconds = compute_seconds = write_seconds = 0.
                nx, ny = region[1] - region[0], region[3] - region[2]
                pixels = nx * ny
                cumulative = np.zeros((5, depth - i0)) if 'cumulative' in config.products else None
                final_sum = np.zeros(5)
                with h5py.File(tmpname, 'w') as out:
                    out.attrs['config_json'] = config_json
                    out.attrs['Complete'] = 0
                    for key in ('HubbleParam', 'Omega0', 'OmegaBaryon'):
                        if key in info['header']:
                            out.attrs[key] = info['header'][key]
                    out.attrs.update(SourceRedshiftRequested=source[0], Redshift=actual_z,
                                     SourceRedshift=actual_z, SourceCellIndex=i0,
                                     NumFreq=grid.size, PixelCount=pixels,
                                     SpatialBounds=np.array(region, dtype=np.int64),
                                     Sampling='comb' if config.legacy else config.sampling,
                                     Profile='voigt' if config.legacy else config.profile,
                                     VoigtOrder=1 if config.legacy else 2,
                                     Legacy=int(config.legacy), Dv_min=-2000., Dv_max=2000., Dv_local=0.,
                                     FrequencyConvention='red-positive wavelength offset; q=1/(1+u/c)',
                                     Closure='segment band opacity' if config.sampling == 'band' and not config.legacy else 'comb')
                    if source_chi is not None:
                        out.attrs['SourceDistanceCode'] = source_chi
                    out.create_dataset('Dvs', data=grid.offsets_kms * KM).attrs['Units'] = 'cm/s'
                    out.create_dataset('frequency_edges', data=grid.q_edges * NU0).attrs['Units'] = 'Hz at source'
                    out.create_dataset('frequency_ratio_lo', data=grid.q_lo)
                    out.create_dataset('frequency_ratio_hi', data=grid.q_hi)
                    out.create_dataset('frequency_weights', data=grid.weights)
                    out.create_dataset('velocity_edges_kms', data=grid.velocity_edges_kms)
                    edges = [-2000, -500, -100, 101, 501, 2001] if config.legacy else BAND_EDGES
                    out.create_dataset('freq_band_edges', data=edges)
                    out.create_dataset('freq_bands', data=edges)
                    if len(source) > 1:
                        out.create_dataset('z_spread', data=source).attrs['Meaning'] = 'Preserved input metadata; only first value is integrated'
                    maps = out.create_dataset('tau_band_avgs', (5, nx, ny), dtype='f8',
                                              chunks=(1, min(tile, nx), min(tile, ny))) if 'maps' in config.products else None
                    spectra = out.create_dataset('taus', (nx, ny, grid.size), dtype='f8',
                                                 chunks=(min(tile, nx), min(tile, ny), min(grid.size, 128))) if 'spectra' in config.products else None
                    if spectra is not None:
                        spectra.attrs['Meaning'] = 'effective bin optical depth' if config.sampling == 'band' and not config.legacy else 'sampled optical depth'
                    for x in range(region[0], region[1], tile):
                        for y in range(region[2], region[3], tile):
                            xs, ys = slice(x, min(x + tile, region[1])), slice(y, min(y + tile, region[3]))
                            tx, ty = xs.stop - x, ys.stop - y
                            tau = np.zeros((tx * ty, grid.size))
                            for d in range(i0, depth, slab_size):
                                ds = slice(d, min(d + slab_size, depth))
                                tick = time.perf_counter()
                                temp, rho, hii, vel = [np.asarray(f[name][xs, ys, ds], dtype=np.float64) for name in FIELDS]
                                read_seconds += time.perf_counter() - tick
                                tick = time.perf_counter()
                                if not all(np.all(np.isfinite(a)) for a in (temp, rho, hii, vel)):
                                    raise ValueError('gas fields contain nonfinite values')
                                if np.any(temp <= 0) or np.any(rho < 0) or np.any((hii < 0) | (hii > 1)):
                                    raise ValueError('gas requires T>0, density>=0, and HII fraction in [0,1]')
                                velocity = _projected_velocity(vel, n, info['header']['OpeningAngle'], xs, ys)
                                for j in range(ds.stop - d):
                                    for zm, hubble, length in segments(info, d + j, actual_z, config):
                                        _add_segment(tau, temp[:, :, j].ravel(), rho[:, :, j].ravel(),
                                                     hii[:, :, j].ravel(), velocity[:, :, j].ravel(),
                                                     zm, hubble, length, actual_z, grid, info['header'], config)
                                    if cumulative is not None:
                                        cumulative[:, d + j - i0] += np.exp(-band_optical_depths(tau, grid.weights)).sum(axis=1)
                                compute_seconds += time.perf_counter() - tick
                            tick = time.perf_counter()
                            band_tau = band_optical_depths(tau, grid.weights)
                            final_sum += np.exp(-band_tau).sum(axis=1)
                            ox, oy = slice(x - region[0], xs.stop - region[0]), slice(y - region[2], ys.stop - region[2])
                            if maps is not None:
                                maps[:, ox, oy] = band_tau.reshape(5, tx, ty)
                            if spectra is not None:
                                spectra[ox, oy, :] = tau.reshape(tx, ty, grid.size)
                            write_seconds += time.perf_counter() - tick
                    final = final_sum / pixels
                    if cumulative is not None:
                        cumulative /= pixels
                        if cumulative.shape[1]:
                            final = cumulative[:, -1].copy()
                        out.create_dataset('T_cum_bands', data=cumulative)
                        out.create_dataset('T_redshifts', data=info['redshifts'][i0 + 1:])
                    out.create_dataset('T_bands', data=final)
                    out.attrs.update(ReadSeconds=read_seconds, ComputeSeconds=compute_seconds,
                                     WriteSeconds=write_seconds, ElapsedSeconds=time.perf_counter() - start,
                                     EffectiveTileSize=tile, EffectiveDepthSlab=slab_size, Complete=1)
                    out.flush()
                # Hard-link publication is atomic and refuses concurrent overwrites.
                os.link(tmpname, output)
                os.unlink(tmpname)
            except BaseException:
                if os.path.exists(tmpname):
                    os.unlink(tmpname)
                raise
            results.append(output)
    return results


def parse_sources(filename):
    with open(filename, encoding='utf-8') as f:
        return [[float(v.strip()) for v in line.split('#', 1)[0].strip().split(',')]
                for line in f if line.split('#', 1)[0].strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='inspect metadata and validate virtual sources')
    inspect.add_argument('--input', required=True)
    inspect.add_argument('--legacy', action='store_true')
    calc = sub.add_parser('run', help='run the slow independent reference on small regions')
    calc.add_argument('--input', required=True)
    calc.add_argument('--z0-file', required=True)
    calc.add_argument('--output-dir', required=True)
    calc.add_argument('--sampling', choices=('band', 'comb'), default='band')
    calc.add_argument('--profile', choices=('voigt', 'delta'), default='voigt')
    calc.add_argument('--dv-step', type=float, default=5.)
    calc.add_argument('--max-dln', type=float, default=1e-3)
    calc.add_argument('--tile-size', type=int, default=4)
    calc.add_argument('--depth-slab', type=int, default=16)
    calc.add_argument('--host-memory-mb', type=float, default=512.)
    calc.add_argument('--region', help='x0,x1,y0,y1 with exclusive upper bounds')
    calc.add_argument('--products', default='maps')
    calc.add_argument('--legacy', action='store_true')
    calc.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        print(json.dumps(inspect_input(args.input, args.legacy), default=_json_default, indent=2))
        return
    config = Config(sampling=args.sampling, profile=args.profile, dv_step=args.dv_step,
                    max_dln=args.max_dln, tile_size=args.tile_size, depth_slab=args.depth_slab,
                    host_memory_mb=args.host_memory_mb, products=tuple(args.products.split(',')), legacy=args.legacy)
    region = tuple(map(int, args.region.split(','))) if args.region else None
    for path in run(args.input, parse_sources(args.z0_file), args.output_dir, config, region, args.resume):
        print(path)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
