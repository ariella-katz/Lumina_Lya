"""Waterfall plots of Lyman-alpha transmission spectra from lumina_lya (GPU code) output.

Layout read:  DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   (one folder per source redshift)

Each file may hold
  * T_binned [k, k, nf]  -- written by `--binning-number N` (mean exp(-tau) per spatial bin), or
  * taus     [x, y, nf]  -- written by `--products spectra` (per-pixel tau), possibly one file per chunk.
Frequency axis: Dvs (cm/s), converted to km/s. With --sampling band (the default) these are the
centres of the frequency bins (--dv-step wide); with --sampling comb they are the sample points.

Usage:
    python plot_spectra_waterfalls.py --directory tau_maps_cpu_new            # files already binned
    python plot_spectra_waterfalls.py --directory tau_maps_chunked --binning-number 100
"""
import argparse
import math
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib import colorizer

CM_PER_KM = 1e5


def open_map_files(z0_dir):
    """Open only finished map files (skips .lock folders and .partial files left by a crashed run)."""
    names = sorted(f for f in os.listdir(z0_dir) if f.endswith('.hdf5'))
    if not names:
        raise FileNotFoundError(f'No .hdf5 files in {z0_dir}')
    return [h5py.File(os.path.join(z0_dir, name), 'r') for name in names]


def bin_index(npix, k):
    """Bin of each pixel row/column. Same rule as lumina_lya --binning-number:
    bin size m = npix // k, and the last bin also takes any remainder pixels."""
    return np.minimum(np.arange(npix) // (npix // k), k - 1)


def load_spectra(z0_ss, k=None):
    """Return (z0, dv_kms, T) with T = mean transmission, shape [k, k, nf].

    z0_ss : open HDF5 files of ONE source redshift (one file, or one per chunk/region).
    k     : bins per side when averaging from per-pixel `taus`; None keeps full resolution.
            Ignored if the file already contains T_binned.
    Chunks are placed by their global pixel bounds (X0,X1,Y0,Y1), so unequal chunks are fine.
    Averaging is of exp(-tau), then done chunk by chunk, so the full [x,y,nf] array is never held.
    """
    s0 = z0_ss[0]
    z0 = float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze())
    dv_kms = s0['Dvs'][:] / CM_PER_KM

    if 'T_binned' in s0:
        if len(z0_ss) != 1:
            raise ValueError('T_binned files cannot be combined; run --binning-number on the full map.')
        return z0, dv_kms, s0['T_binned'][:]

    if 'taus' not in s0:
        raise KeyError('Neither T_binned nor taus found: run with --binning-number N or --products spectra.')

    nx = max(int(s.attrs['X1']) for s in z0_ss)
    ny = max(int(s.attrs['Y1']) for s in z0_ss)
    assert nx == ny, f'map is {nx} x {ny}, expected square'
    k = nx if k is None else k
    if not 1 <= k <= nx:
        raise ValueError(f'bins per side must be between 1 and {nx}, got {k}')
    idx = bin_index(nx, k)
    bins = np.arange(k)[:, None]

    sums, counts = None, np.zeros((k, k))
    for s in z0_ss:
        x0, x1, y0, y1 = (int(s.attrs[a]) for a in ('X0', 'X1', 'Y0', 'Y1'))
        T = np.exp(-s['taus'][:])                              # [x, y, nf] for this file
        Bx = (idx[x0:x1][None, :] == bins).astype(float)       # [k, x]  1 where pixel x is in bin i
        By = (idx[y0:y1][None, :] == bins).astype(float)       # [k, y]
        part = np.einsum('ix,jy,xyf->ijf', Bx, By, T, optimize=True)
        sums = part if sums is None else sums + part
        counts += np.outer(Bx.sum(1), By.sum(1))
    if np.any(counts == 0):
        raise ValueError(f'z0={z0}: some bins have no pixels -- are chunk files missing?')
    return z0, dv_kms, sums / counts[..., None]


def plot_waterfalls(records, outfile='spectra_waterfalls.png'):
    """records: list of (z0, dv_kms, T[k,k,nf]). One panel per z0: x = velocity offset,
    y = spectrum number (the k*k spatial bins, x-major), colour = transmission."""
    records = sorted(records, key=lambda r: r[0])
    n = len(records)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                             sharex=True, squeeze=False)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('black')                       # T == 0 pixels
    norm = LogNorm(vmin=1e-10, vmax=1.0)
    mesh = None
    vmin=np.inf
    vmax=0.0
    for ax, (z0, dv, T) in zip(axes.flat, records):
        # flat = T.reshape(-1, T.shape[-1])
        # flat = np.array(sorted(flat, key=lambda t: np.sum(T)))
        # mesh = ax.pcolormesh(dv, np.arange(flat.shape[0]), flat,
        #                      norm=norm, cmap=cmap, shading='nearest')
        Ts = T.reshape(-1, T.shape[-1])
        vmin=np.min([vmin, np.min(np.sum(Ts, axis=-1))])
        vmax=np.max([vmax, np.max(np.sum(Ts, axis=-1))])
    print(vmin, vmax)
    norm = Normalize(vmin = vmin, vmax = vmax)
    cmap = plt.colormaps["inferno"]
    cmap.set_bad('black')
    for ax, (z0, dv, T) in zip(axes.flat, records):
        Ts = T.reshape(-1, T.shape[-1])
        T_int_mean = np.mean(np.sum(Ts, axis=-1))
        for i in range(len(Ts)):
            Ti = Ts[i]
            T_int = np.sum(Ti)
            ax.plot(dv, Ti+(T_int-T_int_mean)/T_int_mean, c = cmap(norm(T_int)))

        ax.text(0.04, 0.94, f'$z_0$={z0:.1f}', transform=ax.transAxes, ha='left', va='top',
                color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', edgecolor='none', alpha=0.6))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$\Delta v$ [km/s]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\mathcal{T}+(\mathcal{T}^\text{int}-\mathcal{T}^\text{int}_\text{avg})/\mathcal{T}^\text{int}_\text{avg}$')
    for ax in axes.flat:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(colorizer.ColorizingArtist(colorizer.Colorizer(norm=norm, cmap=cmap)), ax=axes.ravel().tolist(), label=r'$\mathcal{T}^\text{int}$', aspect=75)
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f'Wrote {outfile}')


def parse_args():
    p = argparse.ArgumentParser(description='Waterfall plots of transmission spectra.')
    p.add_argument('--directory', default='tau_maps', help='Directory containing z0=... folders')
    p.add_argument('--binning-number', type=int, default=None,
                   help='Target number of spectra N (k = floor(sqrt(N)) bins per side); only needed '
                        'when files hold per-pixel taus. Same meaning as the lumina_lya option.')
    p.add_argument('--output', default='spectra_waterfalls.png')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.directory)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Could not find directory: {data_dir}')
    k = None if args.binning_number is None else math.isqrt(args.binning_number)

    records = []
    for name in sorted(os.listdir(data_dir)):
        z0_dir = os.path.join(data_dir, name)
        if not (os.path.isdir(z0_dir) and name.startswith('z0=')):
            continue
        files = open_map_files(z0_dir)
        try:
            records.append(load_spectra(files, k))
        finally:
            for f in files:
                f.close()
    plot_waterfalls(records, args.output)


if __name__ == '__main__':
    main()