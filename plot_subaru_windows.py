"""Per-frequency pixel histograms of the transmission ("sweeps"), one panel per source redshift.

For each chosen source redshift, at every frequency bin, histogram the transmission over all pixels of
the map and draw all of those histograms on the same panel, coloured by velocity offset.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   with per-pixel spectra (`taus`, shape [x, y, nf]),
i.e. runs made with `lumina_lya ... --products spectra` (full map, or one file per chunk).

Usage:
    python plot_pixel_histograms.py --directory tau_maps_pixels --redshifts 6.6 5.7 --output sweeps.png
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

CM_PER_KM = 1e5
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


def requested_redshift(f):
    return float(np.asarray(f.attrs['SourceRedshiftRequested']).squeeze())


def find_files(data_dir, target, tol=1e-6):
    """Open the files of the z0 folder whose SourceRedshiftRequested equals `target`."""
    available = []
    for folder in z0_folders(data_dir):
        files = open_map_files(folder)
        z = requested_redshift(files[0])
        if abs(z - target) < tol:
            return files
        available.append(z)
        for f in files:
            f.close()
    raise FileNotFoundError(f'z0={target} not found in {data_dir}; available: {sorted(available)}')


def pixel_histograms(files, lo, nbins):
    """Histogram of log10(T) over all pixels, separately for every frequency.

    Works from tau directly (log10 T = -tau * log10 e), so saturated pixels with tiny T do not underflow.
    Streams over slabs of pixel rows and over chunk files, so memory stays small.
    Returns (z0, dv_kms, counts[nf, nbins], n_pixels, n_below[nf]); pixels with log10 T < lo are
    counted in the first bin, and n_below says how many there were.
    """
    s0 = files[0]
    if 'taus' not in s0:
        raise KeyError("No 'taus' dataset: run lumina_lya with --products spectra for the histograms.")
    z0 = requested_redshift(s0)
    dv = s0['Dvs'][:] / CM_PER_KM
    nf = len(dv)
    width = -lo / nbins
    counts = np.zeros(nf * nbins, dtype=np.int64)
    below = np.zeros(nf, dtype=np.int64)
    n_pix = 0
    offsets = (np.arange(nf) * nbins)[None, :]
    for s in files:
        taus = s['taus']
        nx, ny, nfs = taus.shape
        assert nfs == nf, 'files disagree on the number of frequencies'
        step = max(1, 4_000_000 // (ny * nf))                          # ~4M values per slab
        for x in range(0, nx, step):
            logT = -taus[x:x + step].astype(np.float64).reshape(-1, nf) * LOG10_E      # [pixels, nf]
            idx = np.clip(np.floor((logT - lo) / width).astype(np.int64), 0, nbins - 1)
            counts += np.bincount((offsets + idx).ravel(), minlength=nf * nbins)
            below += (logT < lo).sum(axis=0)
            n_pix += logT.shape[0]
    return z0, dv, counts.reshape(nf, nbins), n_pix, below


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--redshifts', type=float, nargs='+', default=[6.6, 5.7])
    p.add_argument('--output', default='subaru_pixel_histograms_red.png')
    p.add_argument('--tmin-exp', type=float, default=-10.0, help='lowest log10(T) bin edge (default -10)')
    p.add_argument('--bins', type=int, default=60)
    p.add_argument('--linear-y', action='store_true', help='linear instead of logarithmic y axis')
    p.add_argument('--stride', type=int, default=1, help='draw every Nth frequency')
    args = p.parse_args()

    data_dir = os.path.abspath(args.directory)
    lo, nbins = float(args.tmin_exp), args.bins
    width = -lo / nbins
    centres = lo + (np.arange(nbins) + 0.5) * width

    fig, axes = plt.subplots(1, len(args.redshifts), figsize=(5.4 * len(args.redshifts) + 1.2, 4.4),
                             sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap('coolwarm')                       # blue side = negative dv, red side = positive dv
    norm = None
    for ax, target in zip(axes[0], args.redshifts):
        files = find_files(data_dir, target)
        try:
            z0, dv, counts, n_pix, below = pixel_histograms(files, lo, nbins)
        finally:
            for f in files:
                f.close()
        i1 = np.argmax(dv >= 100)
        i2 = np.argmax(dv >= 500)
        if norm is None:
            norm = Normalize(vmin=dv[i1], vmax=dv[i2])
        density = counts / (n_pix * width)                # pixel fraction per unit log10 T
        for i in range(i1, i2, args.stride):
            ax.plot(centres, density[i], color=cmap(norm(dv[i])), lw=0.6, alpha=0.65)
        ax.set_title(f'$z_0$={z0:.2f}')
        ax.set_xlabel(r'$\log_{10}\mathcal{T}$ (per pixel)')
        if not args.linear_y:
            ax.set_yscale('log')
        print(f'z0={z0:.3f}: {n_pix} pixels; max fraction below 10^{lo:g} at any frequency = '
              f'{below.max() / n_pix:.3f} (counted in the first bin)')
    axes[0, 0].set_ylabel(r'pixel fraction per unit $\log_{10}\mathcal{T}$')
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label=r'$\Delta v$ [km/s]')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()