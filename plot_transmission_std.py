"""Standard deviation of the pixel transmission versus velocity offset, one curve per source redshift.

At every frequency bin the transmission T is taken for every pixel of the map and the (population, ddof=0)
standard deviation over pixels is computed; the curves are coloured by source redshift.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   with per-pixel spectra (`taus`, shape [x, y, nf]),
i.e. runs made with `lumina_lya ... --products spectra` (full map, or one file per chunk). The statistics are
accumulated slab by slab with a numerically stable parallel update (Chan et al.), so memory stays small and
chunk files are combined exactly.

Usage:
    python plot_transmission_std.py --directory tau_maps_pixels --output transmission_std.png [--ylog]
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


def pixel_std(files):
    """Return (z0, dv_kms, mean[nf], std[nf]) of the transmission over all pixels, at every frequency."""
    s0 = files[0]
    if 'taus' not in s0:
        raise KeyError("No 'taus' dataset: run lumina_lya with --products spectra (pixel statistics need per-pixel spectra).")
    z0 = float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze())
    dv = s0['Dvs'][:] / CM_PER_KM
    nf = len(dv)
    n, mean, m2 = 0, np.zeros(nf), np.zeros(nf)
    for s in files:
        taus = s['taus']
        nx, ny, nfs = taus.shape
        assert nfs == nf, 'files disagree on the number of frequencies'
        step = max(1, 4_000_000 // (ny * nf))                       # ~4M values per slab
        for x in range(0, nx, step):
            T = np.exp(-taus[x:x + step].astype(np.float64)).reshape(-1, nf)
            nb = T.shape[0]
            mb = T.mean(axis=0)
            m2b = ((T - mb) ** 2).sum(axis=0)
            delta, tot = mb - mean, n + nb
            mean = mean + delta * nb / tot                            # merge (n, mean, m2) with the slab's statistics
            m2 = m2 + m2b + delta ** 2 * n * nb / tot
            n = tot
    return z0, dv, mean, np.sqrt(m2 / n)


def truncated_cmap(name, lo, hi):
    return LinearSegmentedColormap.from_list(f'{name}_{lo:g}_{hi:g}', plt.get_cmap(name)(np.linspace(lo, hi, 256)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--output', default='transmission_std.png')
    p.add_argument('--ylog', action='store_true', help='logarithmic y axis')
    p.add_argument('--cmap', default='nipy_spectral', help='matplotlib colormap name (default nipy_spectral)')
    p.add_argument('--cmap-range', type=float, nargs=2, default=[0.05, 0.92], metavar=('LO', 'HI'))
    args = p.parse_args()

    data_dir = os.path.abspath(args.directory)
    records = []
    for folder in z0_folders(data_dir):
        files = open_map_files(folder)
        try:
            records.append(pixel_std(files))
        finally:
            for f in files:
                f.close()
    records.sort(key=lambda r: r[0])

    zs = np.array([r[0] for r in records])
    norm = Normalize(vmin=zs.min(), vmax=zs.max())
    cmap = truncated_cmap(args.cmap, *args.cmap_range)
    fig, ax = plt.subplots(figsize=(8, 5))
    for z0, dv, mean, std in records:
        ax.plot(dv, std, color=cmap(norm(z0)), lw=1.1)
    ax.set_xlabel(r'$\Delta v$ [km/s]')
    ax.set_ylabel(r'$\sigma_{\mathcal{T}}$ (standard deviation over pixels)')
    if args.ylog:
        ax.set_yscale('log')
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'source redshift $z_0$')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output} ({len(records)} redshifts)')


if __name__ == '__main__':
    main()
