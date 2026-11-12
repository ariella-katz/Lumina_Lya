"""Map-averaged transmission spectra as curves: one line per source redshift, coloured by redshift.

The default colormap is nipy_spectral, which is not perceptually uniform but separates neighbouring redshifts
well. Its black and light-grey ends are hard to see on white, so only the part given by --cmap-range is used.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   (one folder per source redshift) containing either
  * T_binned [k, k, nf]  -- from `lumina_lya ... --binning-number N` (N=1 gives one spectrum per source), or
  * taus     [x, y, nf]  -- from `--products spectra` (full map or chunk files; averaged here).

Usage:
    python plot_mean_spectra_curves.py --directory tau_maps_mean50 --output mean_spectra_curves.png [--ylog]
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

CM_PER_KM = 1e5


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


def mean_spectrum(files):
    """Return (z0, dv_kms, T[nf]) with T the transmission averaged over every pixel of the map."""
    s0 = files[0]
    z0 = float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze())
    dv = s0['Dvs'][:] / CM_PER_KM

    if 'T_binned' in s0:
        if len(files) != 1:
            raise ValueError('T_binned files cannot be combined; run --binning-number on the full map.')
        T = s0['T_binned'][:]                                   # [k, k, nf]
        k = T.shape[0]
        if k == 1:
            return z0, dv, T[0, 0]
        m, npix = int(s0.attrs['BinSizePixels']), int(s0.attrs['X1'])      # pixel-weighted mean of the bins
        sizes = np.full(k, float(m))
        sizes[-1] = npix - (k - 1) * m                                     # last bin holds the remainder pixels
        w = np.outer(sizes, sizes)
        return z0, dv, (T * w[..., None]).sum(axis=(0, 1)) / w.sum()

    if 'taus' in s0:
        total, n_pix = np.zeros(len(dv)), 0
        for s in files:
            taus = s['taus']
            nx, ny, nf = taus.shape
            step = max(1, 4_000_000 // (ny * nf))
            for x in range(0, nx, step):
                slab = np.exp(-taus[x:x + step].astype(np.float64))
                total += slab.reshape(-1, nf).sum(axis=0)
                n_pix += slab.shape[0] * ny
        return z0, dv, total / n_pix

    raise KeyError('Neither T_binned nor taus found: run lumina_lya with --binning-number N or --products spectra.')


def truncated_cmap(name, lo, hi):
    """The part [lo, hi] of a colormap, as a new colormap."""
    return LinearSegmentedColormap.from_list(f'{name}_{lo:g}_{hi:g}', plt.get_cmap(name)(np.linspace(lo, hi, 256)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--output', default='mean_spectra_curves.png')
    p.add_argument('--ylog', action='store_true', help='logarithmic transmission axis')
    p.add_argument('--cmap', default='nipy_spectral', help='matplotlib colormap name (default nipy_spectral)')
    p.add_argument('--cmap-range', type=float, nargs=2, default=[0.05, 0.92], metavar=('LO', 'HI'),
                   help='portion of the colormap to use (default 0.05 0.92)')
    args = p.parse_args()

    data_dir = os.path.abspath(args.directory)
    records = []
    for folder in z0_folders(data_dir):
        files = open_map_files(folder)
        try:
            records.append(mean_spectrum(files))
        finally:
            for f in files:
                f.close()
    records.sort(key=lambda r: r[0])

    zs = np.array([r[0] for r in records])
    norm = Normalize(vmin=zs.min(), vmax=zs.max())
    cmap = truncated_cmap(args.cmap, *args.cmap_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    for z0, dv, T in records:
        ax.plot(dv, T, color=cmap(norm(z0)), lw=1.1)
    ax.set_xlabel(r'$\Delta v$ [km/s]')
    ax.set_ylabel(r'map-averaged transmission $\langle\mathcal{T}\rangle$')
    if args.ylog:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, 1.02)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'source redshift $z_0$')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output} ({len(records)} redshifts)')


if __name__ == '__main__':
    main()
