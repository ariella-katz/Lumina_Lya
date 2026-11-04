"""Auto- and cross-correlations of a transmission band with the HI fraction and the density.

For every source redshift z0, the 2-D map of the chosen band's transmission (default: red) is correlated
with itself and with two 2-D fields taken from the input lightcone at the SOURCE PLANE (the depth cell that
contains the source, recorded in the output files as SourceCell):
      x_HI = 1 - HII_Fraction      and      Density.
Correlations are azimuthally averaged over pixel separation (zero-padded FFTs with exact pair counting, so
the finite, non-periodic map is handled without wrap-around), after subtracting each map's mean.

One panel, coloured by source redshift (solid = autocorrelation, dashed = HI cross-correlation, dotted =
density cross-correlation; the two differences "cross - auto" use alpha = 1):
    solid  alpha .6   auto-correlation of T_band
    dashed alpha .6   cross-correlation  T_band x x_HI
    dotted alpha .6   cross-correlation  T_band x Density
    dashed alpha 1    (T x x_HI)  - (T x T)
    dotted alpha 1    (T x Density) - (T x T)

Scaling (--normalize):
    coefficient (default)  divide by the standard deviations, so every curve is a correlation coefficient and
                           the autocorrelation starts at 1; all five curves are dimensionless and comparable.
    peak                   divide each curve by its own value at the smallest separation, so every curve
                           starts at 1 and the differences show pure shape differences.
--diff-scale multiplies the two difference curves if they are too small to see.

Usage:
    python plot_xcorr.py --maps-dir tau_maps_gpu_new --input /scratch/.../All.hdf5 --output xcorr.png
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

BANDS = {'ultrablue': 0, 'blue': 1, 'center': 2, 'red': 3, 'ultrared': 4}
CM_PER_MPC = 3.0856775814913673e24


# ----------------------------------------------------------------------------- reading
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


def load_band_transmission(files, band):
    """Transmission map [x, y] of one band, stitched from one file or from chunk files by their pixel bounds."""
    nx = max(int(s.attrs['X1']) for s in files)
    ny = max(int(s.attrs['Y1']) for s in files)
    T = np.full((nx, ny), np.nan)
    for s in files:
        x0, x1, y0, y1 = (int(s.attrs[a]) for a in ('X0', 'X1', 'Y0', 'Y1'))
        T[x0:x1, y0:y1] = np.exp(-s['tau_band_avgs'][band])
    if np.isnan(T).any():
        raise ValueError('Some pixels are missing: are chunk files absent?')
    return T


class SliceReader:
    """Read single depth slices [x, y] of a cube, caching the stored depth-block so that sources in the same
    block (sorted by cell) cost one read."""

    def __init__(self, dataset):
        self.ds = dataset
        self.depth = dataset.chunks[2] if dataset.chunks else 64
        self.start, self.block = None, None

    def __getitem__(self, k):
        if self.block is None or not (self.start <= k < self.start + self.block.shape[2]):
            self.start = (k // self.depth) * self.depth
            self.block = self.ds[:, :, self.start:self.start + self.depth]
        return self.block[:, :, k - self.start].astype(np.float64)


# ----------------------------------------------------------------------------- correlations
def correlation_profile(a, b, rmax):
    """Azimuthally averaged cross-correlation <a(x) b(x+d)> as a function of rounded |d| = 0..rmax (pixels).

    Both maps are mean-subtracted. Sums over all pixel pairs (zero-padded FFT, no wrap-around) are divided by the
    exact number of pairs in each separation bin.
    """
    a = a - a.mean()
    b = b - b.mean()
    nx, ny = a.shape
    shape = (2 * nx, 2 * ny)
    S = np.fft.irfft2(np.conj(np.fft.rfft2(a, shape)) * np.fft.rfft2(b, shape), shape)       # sum_x a(x) b(x+d)
    ones = np.fft.rfft2(np.ones_like(a), shape)
    N = np.rint(np.fft.irfft2(np.conj(ones) * ones, shape))                                    # pairs per lag
    lx = np.fft.fftfreq(shape[0], 1.0 / shape[0])[:, None]
    ly = np.fft.fftfreq(shape[1], 1.0 / shape[1])[None, :]
    ir = np.rint(np.hypot(lx, ly)).astype(np.int64)
    keep = ir <= rmax
    num = np.bincount(ir[keep], weights=S[keep], minlength=rmax + 1)
    den = np.bincount(ir[keep], weights=N[keep], minlength=rmax + 1)
    return num / den


def curves(T, xhi, rho, rmax, normalize, diff_scale):
    """The five curves for one source redshift (NaN arrays if a map is constant and cannot be normalised)."""
    nan = np.full(rmax + 1, np.nan)
    auto, cx_hi, cx_rho = (correlation_profile(T, T, rmax), correlation_profile(T, xhi, rmax),
                           correlation_profile(T, rho, rmax))
    if normalize == 'coefficient':
        sT, sH, sR = T.std(), xhi.std(), rho.std()
        if min(sT, sH, sR) == 0:
            return {k: nan for k in ('auto', 'hi', 'rho', 'dhi', 'drho')}
        auto, cx_hi, cx_rho = auto / sT**2, cx_hi / (sT * sH), cx_rho / (sT * sR)
    else:
        if 0 in (auto[0], cx_hi[0], cx_rho[0]):
            return {k: nan for k in ('auto', 'hi', 'rho', 'dhi', 'drho')}
        auto, cx_hi, cx_rho = auto / auto[0], cx_hi / cx_hi[0], cx_rho / cx_rho[0]
    return {'auto': auto, 'hi': cx_hi, 'rho': cx_rho,
            'dhi': (cx_hi - auto) * diff_scale, 'drho': (cx_rho - auto) * diff_scale}


def compute(maps_dir, input_path, band, rmax_frac, normalize, diff_scale):
    """List of (z0, distance_cMpc, n_pix, curves-dict), sorted by redshift."""
    entries = []
    for folder in z0_folders(maps_dir):
        files = open_map_files(folder)
        try:
            s0 = files[0]
            entries.append((float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze()),
                            int(s0.attrs['SourceCell']),
                            float(s0.attrs['SourceDistanceComovingCm']) / CM_PER_MPC,
                            load_band_transmission(files, band)))
        finally:
            for f in files:
                f.close()
    entries.sort(key=lambda e: e[1])                                     # by cell: reuses cached depth blocks
    out = []
    with h5py.File(input_path, 'r') as f:
        hii, dens = SliceReader(f['HII_Fraction']), SliceReader(f['Density'])
        for z0, cell, dist, T in entries:
            xhi, rho = 1.0 - hii[cell], dens[cell]
            if xhi.shape != T.shape:
                raise ValueError(f'Map is {T.shape} but the input fields are {xhi.shape}')
            rmax = int(rmax_frac * min(T.shape))
            out.append((z0, dist, T.shape[0], curves(T, xhi, rho, rmax, normalize, diff_scale)))
    out.sort(key=lambda e: e[0])
    return out


# ----------------------------------------------------------------------------- plot
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maps-dir', required=True, help='directory with the z0=... folders (band maps)')
    p.add_argument('--input', required=True, help='lightcone file with HII_Fraction and Density (e.g. All.hdf5)')
    p.add_argument('--output', default='red_T_frac_dens_xcorr.png')
    p.add_argument('--band', choices=list(BANDS), default='red')
    p.add_argument('--normalize', choices=['coefficient', 'peak'], default='coefficient')
    p.add_argument('--diff-scale', type=float, default=1.0, help='multiply the two difference curves')
    p.add_argument('--rmax-frac', type=float, default=0.5, help='largest separation, as a fraction of the map size')
    p.add_argument('--fov-deg', type=float, default=3.6, help='field of view of the map in degrees')
    p.add_argument('--xunits', choices=['deg', 'cMpc'], default='deg',
                   help='separation in degrees, or comoving Mpc at the source distance')
    args = p.parse_args()

    results = compute(os.path.abspath(args.maps_dir), args.input, BANDS[args.band],
                      args.rmax_frac, args.normalize, args.diff_scale)
    zs = np.array([r[0] for r in results])
    norm = Normalize(vmin=zs.min(), vmax=zs.max())
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(9, 6))
    print('  z0    r(T,xHI)  r(T,rho)   (zero-lag values; only meaningful for --normalize coefficient)')
    for z0, dist, npix, c in results:
        r = np.arange(len(c['auto']))
        sep = r * args.fov_deg / npix                                    # degrees
        if args.xunits == 'cMpc':
            sep = np.radians(sep) * dist
        col = cmap(norm(z0))
        ax.plot(sep, c['auto'], color=col, ls='-', alpha=0.6, lw=1.2)
        ax.plot(sep, c['hi'], color=col, ls='--', alpha=0.6, lw=1.2)
        ax.plot(sep, c['rho'], color=col, ls=':', alpha=0.6, lw=1.2)
        ax.plot(sep, c['dhi'], color=col, ls='--', alpha=1.0, lw=1.2)
        ax.plot(sep, c['drho'], color=col, ls=':', alpha=1.0, lw=1.2)
        print(f'{z0:6.3f}  {c["hi"][0]:+8.3f}  {c["rho"][0]:+8.3f}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('separation [degrees]' if args.xunits == 'deg' else 'comoving separation [cMpc]')
    ax.set_ylabel('correlation coefficient' if args.normalize == 'coefficient' else 'correlation / zero-lag value')
    tag = f'$\\mathcal{{T}}_{{\\rm {args.band}}}$'
    scale = f' $\\times${args.diff_scale:g}' if args.diff_scale != 1 else ''
    handles = [Line2D([], [], color='gray', ls='-', alpha=0.6, label=f'{tag} autocorrelation'),
               Line2D([], [], color='gray', ls='--', alpha=0.6, label=f'{tag} $\\times$ $x_{{\\rm HI}}$'),
               Line2D([], [], color='gray', ls=':', alpha=0.6, label=f'{tag} $\\times$ density'),
               Line2D([], [], color='gray', ls='--', alpha=1.0, label=f'($x_{{\\rm HI}}$ xcorr) $-$ autocorr{scale}'),
               Line2D([], [], color='gray', ls=':', alpha=1.0, label=f'(density xcorr) $-$ autocorr{scale}')]
    ax.legend(handles=handles, fontsize=8, loc='best')
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'source redshift $z_0$')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output} ({len(results)} redshifts)')


if __name__ == '__main__':
    main()