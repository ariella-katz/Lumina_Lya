"""Heat map of the per-pixel transmission versus density, one panel per source redshift.

For every pixel of the map, take the band transmission T (default: red) and the gas density at the same pixel
in the SOURCE PLANE (the depth cell that contains the source, recorded in the output files as SourceCell). The
pixels are counted in a 2-D histogram (x = transmission, y = density; see --density for the variable)
and coloured by the fraction of pixels in each cell on a logarithmic scale. All panels share the same bins,
axis ranges and colour scale, so redshifts are directly comparable.

Transmission axis (--tscale): linear 0-1 for the red and ultrared bands, log10 T from --tmin-exp to 0 for the
bluer bands (as plot_T_maps.py does); override with --tscale linear|log. In log mode, pixels below the lower
limit are counted in the first bin. The density range is the pooled --dens-percentiles of all panels
(or --dens-range); pixels outside it are not drawn and their fraction is printed.

Density variable (--density):
    log-overdensity (default)  log10(rho / <rho>), <rho> = mean of the source-plane slice: compresses the heavy
                               tail and removes the overall evolution with redshift, so panels are comparable
    overdensity                rho / <rho>
    log-raw                    log10(rho) in the file's code units
    raw                        rho in the file's code units (any scaling with redshift is left in, and with a
                               linear axis the heavy tail stretches the range; --dens-range helps)

Reads   MAPS_DIR/z0=<z>/tau_map_<z>_<suffix>.hdf5   (tau_band_avgs, full map or chunk files)
and     --input (the lightcone file, for Density).

Usage:
    python plot_transmission_density_heatmap.py --maps-dir tau_maps_xcorr --input /scratch/.../All.hdf5 \
        --band red --output transmission_density.png
"""
import argparse
import math
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BANDS = {'ultrablue': 0, 'blue': 1, 'center': 2, 'red': 3, 'ultrared': 4}
LOG10_E = np.log10(np.e)


def z0_folders(data_dir):
    folders = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
               if d.startswith('z0=') and os.path.isdir(os.path.join(data_dir, d))]
    if not folders:
        raise FileNotFoundError(f'No z0=... folders in {data_dir}')
    return folders


def open_map_files(z0_dir):
    """Open only finished map files (skips .lock folders and .partial files left by a crashed run)."""
    names = sorted(f for f in os.listdir(z0_dir) if f.endswith('.hdf5'))
    if not names:
        raise FileNotFoundError(f'No .hdf5 files in {z0_dir}')
    return [h5py.File(os.path.join(z0_dir, name), 'r') for name in names]


def load_band_tau(files, band):
    """Band optical-depth map [x, y], stitched from one file or from chunk files by their pixel bounds."""
    nx = max(int(s.attrs['X1']) for s in files)
    ny = max(int(s.attrs['Y1']) for s in files)
    tau = np.full((nx, ny), np.nan)
    for s in files:
        x0, x1, y0, y1 = (int(s.attrs[a]) for a in ('X0', 'X1', 'Y0', 'Y1'))
        tau[x0:x1, y0:y1] = s['tau_band_avgs'][band]
    if np.isnan(tau).any():
        raise ValueError('Some pixels are missing: are chunk files absent?')
    return tau


class SliceReader:
    """Read single depth slices [x, y] of a cube, caching the stored depth-block (sort cells to reuse it)."""

    def __init__(self, dataset):
        self.ds = dataset
        self.depth = dataset.chunks[2] if dataset.chunks else 64
        self.start, self.block = None, None

    def __getitem__(self, k):
        if self.block is None or not (self.start <= k < self.start + self.block.shape[2]):
            self.start = (k // self.depth) * self.depth
            self.block = self.ds[:, :, self.start:self.start + self.depth]
        return self.block[:, :, k - self.start].astype(np.float64)


DENSITY_MODES = {
    'log-overdensity': r'$\log_{10}(\rho/\langle\rho\rangle)$',
    'overdensity': r'$\rho/\langle\rho\rangle$',
    'log-raw': r'$\log_{10}\rho$ [code units]',
    'raw': r'$\rho$ [code units]',
}


def transform_density(rho, mode):
    """The chosen density variable for a slice; NaN where a logarithm is undefined (rho <= 0)."""
    if mode == 'raw':
        return rho.copy()
    if mode == 'overdensity':
        return rho / rho.mean()
    base = rho / rho.mean() if mode == 'log-overdensity' else rho
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(rho > 0, np.log10(base), np.nan)


def load_pixels(maps_dir, input_path, band, only, mode='log-overdensity'):
    """[(z0, tau_map, density-variable map)] sorted by redshift (NaN where a log is undefined)."""
    entries = []
    for folder in z0_folders(maps_dir):
        files = open_map_files(folder)
        try:
            z0 = float(np.asarray(files[0].attrs['SourceRedshiftRequested']).squeeze())
            if only is None or any(abs(z0 - t) < 1e-6 for t in only):
                entries.append((z0, int(files[0].attrs['SourceCell']), load_band_tau(files, band)))
        finally:
            for f in files:
                f.close()
    if not entries:
        raise FileNotFoundError('No matching source redshifts found.')
    entries.sort(key=lambda e: e[1])                             # by cell: reuses cached depth blocks
    out = []
    with h5py.File(input_path, 'r') as f:
        dens = SliceReader(f['Density'])
        for z0, cell, tau in entries:
            rho = dens[cell]
            if rho.shape != tau.shape:
                raise ValueError(f'Map is {tau.shape} but the density slice is {rho.shape}')
            out.append((z0, tau, transform_density(rho, mode)))
    out.sort(key=lambda e: e[0])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maps-dir', required=True, help='directory with the z0=... folders (band maps)')
    p.add_argument('--input', required=True, help='lightcone file with Density (e.g. All.hdf5)')
    p.add_argument('--output', default='transmission_density.png')
    p.add_argument('--band', choices=list(BANDS), default='red')
    p.add_argument('--redshifts', type=float, nargs='+', default=None, help='subset of source redshifts (default: all)')
    p.add_argument('--tscale', choices=['auto', 'linear', 'log'], default='auto')
    p.add_argument('--tmin-exp', type=float, default=-10.0, help='lowest log10(T) bin edge in log mode')
    p.add_argument('--bins', type=int, default=60, help='bins along each axis')
    p.add_argument('--density', choices=list(DENSITY_MODES), default='log-overdensity',
                   help='density variable on the y axis (default log-overdensity)')
    p.add_argument('--dens-percentiles', type=float, nargs=2, default=[0.01, 99.99], metavar=('LO', 'HI'))
    p.add_argument('--dens-range', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='explicit range of the chosen density variable instead of the percentiles')
    p.add_argument('--cmap', default='viridis')
    args = p.parse_args()

    band = BANDS[args.band]
    log_t = args.tscale == 'log' or (args.tscale == 'auto' and band < 3)
    data = load_pixels(os.path.abspath(args.maps_dir), args.input, band, args.redshifts, args.density)

    pooled = np.concatenate([y[np.isfinite(y)] for _, _, y in data])
    ylo, yhi = args.dens_range if args.dens_range else np.percentile(pooled, args.dens_percentiles)
    ye = np.linspace(ylo, yhi, args.bins + 1)
    if log_t:
        xe = np.linspace(args.tmin_exp, 0.0, args.bins + 1)
    else:
        xe = np.linspace(0.0, 1.0, args.bins + 1)

    fracs = []
    for z0, tau, y in data:
        x = (np.clip(-tau * LOG10_E, args.tmin_exp, 0.0) if log_t else np.exp(-tau)).ravel()
        y = y.ravel()
        ok = np.isfinite(y)
        counts, _, _ = np.histogram2d(x[ok], y[ok], bins=[xe, ye])
        fracs.append(counts / ok.sum())
        print(f'z0={z0:.3f}: {ok.sum()} pixels with a defined density; {1 - counts.sum() / ok.sum():.2%} outside the '
              f'density range')
    vmax = max(f.max() for f in fracs)
    vmin = min(f[f > 0].min() for f in fracs)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    n = len(data)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1.2, 3.3 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    mesh = None
    for ax, (z0, _, _), frac in zip(axes.flat, data, fracs):
        mesh = ax.pcolormesh(xe, ye, np.ma.masked_equal(frac, 0).T, norm=norm, cmap=args.cmap)
        ax.set_title(f'$z_0$={z0:.2f}', fontsize=10)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    xlabel = (rf'$\log_{{10}}\mathcal{{T}}_{{\rm {args.band}}}$' if log_t else rf'$\mathcal{{T}}_{{\rm {args.band}}}$')
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(DENSITY_MODES[args.density])
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label='fraction of pixels')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output} ({n} redshifts)')


if __name__ == '__main__':
    main()