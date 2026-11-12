"""Heat map of the map-averaged transmission: x = velocity offset, y = source redshift, colour = transmission.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   (one folder per source redshift) containing either
  * T_binned [k, k, nf]  -- from `lumina_lya ... --binning-number N` (N=1 gives one spectrum per source), or
  * taus     [x, y, nf]  -- from `--products spectra` (full map or chunk files; averaged here).
All source redshifts must share the same frequency grid (same --sampling and --dv-step).

Usage:
    python plot_mean_spectra.py --directory tau_maps_mean50 --output mean_spectra_heatmap.png [--log]
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

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
        # several bins: pixel-weighted mean (the last bin also holds the remainder pixels)
        m, npix = int(s0.attrs['BinSizePixels']), int(s0.attrs['X1'])
        sizes = np.full(k, float(m))
        sizes[-1] = npix - (k - 1) * m
        w = np.outer(sizes, sizes)
        return z0, dv, (T * w[..., None]).sum(axis=(0, 1)) / w.sum()

    if 'taus' in s0:
        total, n_pix = np.zeros(len(dv)), 0
        for s in files:                                         # works for one file or one per chunk
            taus = s['taus']
            nx, ny, nf = taus.shape
            step = max(1, 4_000_000 // (ny * nf))               # ~4M values per slab keeps memory small
            for x in range(0, nx, step):
                slab = np.exp(-taus[x:x + step].astype(np.float64))
                total += slab.reshape(-1, nf).sum(axis=0)
                n_pix += slab.shape[0] * ny
        return z0, dv, total / n_pix

    raise KeyError('Neither T_binned nor taus found: run lumina_lya with --binning-number N or --products spectra.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--output', default='mean_spectra_heatmap.png')
    p.add_argument('--log', action='store_true', help='logarithmic colour scale for the transmission')
    p.add_argument('--vmin', type=float, default=None,
                   help='lower colour limit (default 0 for linear; smallest positive value for --log)')
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
    if len(records) < 2:
        raise ValueError('The heat map needs at least two source redshifts.')
    records.sort(key=lambda r: r[0])

    zs = np.array([r[0] for r in records])
    dv = records[0][1]
    for z0, dv_i, _ in records:
        if dv_i.shape != dv.shape or not np.allclose(dv_i, dv):
            raise ValueError(f'z0={z0} uses a different frequency grid; rerun all sources with the same '
                             '--sampling and --dv-step.')
    T = np.array([r[2] for r in records])                        # [n_redshifts, nf]

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('black')                                        # zero transmission cannot be shown on a log scale
    if args.log:
        positive = T[T > 0]
        vmin = args.vmin if args.vmin is not None else positive.min()
        norm = LogNorm(vmin=vmin, vmax=1.0)
    else:
        norm = Normalize(vmin=0.0 if args.vmin is None else args.vmin, vmax=1.0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    mesh = ax.pcolormesh(dv, zs, T, norm=norm, cmap=cmap, shading='nearest')
    ax.set_xlabel(r'$\Delta v$ [km/s]')
    ax.set_ylabel(r'source redshift $z_0$')
    fig.colorbar(mesh, ax=ax, label=r'map-averaged transmission $\langle\mathcal{T}\rangle$')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    print(f'Wrote {args.output} ({len(records)} redshifts x {len(dv)} frequencies)')


if __name__ == '__main__':
    main()