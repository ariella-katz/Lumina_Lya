"""Spatial transmission map with an n x n grid of mini-plots of transmission curves overlaid.

Each source redshift gets one figure: the band's transmission map (same orientation, colours and colour scales
as plot_T_maps.py) with, on top of every spatial bin, a small plot of that bin's mean transmission spectrum
T(dv). The selected band's velocity range is shaded in every mini-plot, so the curve can be read against the
map colour underneath. The thin white lines are the bin boundaries.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5  from a FULL-MAP run with spatial binning, e.g.
    lumina_lya run ... --sampling band --dv-step 5 --binning-number 64        # an 8 x 8 grid
which writes both `tau_band_avgs` (the maps) and `T_binned` [n, n, nf] (the spectra) to the same file.
The grid is n = floor(sqrt(binning number)) bins per side; the last bin row/column holds any remainder pixels
and is drawn at its true size.

Usage:
    python plot_spatial_map_with_spectra.py --directory tau_maps_binned --redshifts 8 13 --band red \
        --output spatial_map_spectra_z{z:.1f}.png
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

CM_PER_KM = 1e5
BAND_NAMES = ['ultrablue', 'blue', 'center', 'red', 'ultrared']
BAND_COLORS = ['#0072B2', '#009E73', '#999933', '#E69F00', '#CC3311']
BAND_EDGES = [-2000, -500, -100, 100, 500, 2000]              # km/s, default (non-legacy) band definition


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


def draw(s, band, outfile, fov, norm_mode, frac, alpha, inset_log, shade):
    """Draw the figure for one (binned, full-map) file and return the inset axes keyed by (x-bin, y-bin)."""
    if 'T_binned' not in s:
        raise KeyError("No 'T_binned' dataset: run lumina_lya with --binning-number n*n on the full map.")
    z0 = float(np.asarray(s.attrs['SourceRedshiftRequested']).squeeze())
    dv = s['Dvs'][:] / CM_PER_KM
    spectra = s['T_binned'][:]                                   # [k, k, nf]
    k = spectra.shape[0]
    npix, m = int(s.attrs['X1']), int(s.attrs['BinSizePixels'])
    Tmap = np.exp(-s['tau_band_avgs'][band])                     # [x, y]
    if Tmap.shape != (npix, npix):
        raise ValueError('The file does not hold the full map; binned runs must cover the whole map.')
    edges = np.append(np.arange(k) * m, npix)                    # bin boundaries in pixels (last bin takes the remainder)
    half = fov / 2.0
    coord = lambda pix: -half + pix * fov / npix                 # pixel index -> degrees

    cmap = LinearSegmentedColormap.from_list(f'band_{band}', ['black', BAND_COLORS[band], 'white'])
    cmap.set_bad('black')
    use_log = norm_mode == 'log' or (norm_mode == 'auto' and band < 3)       # same rule as plot_T_maps.py
    norm = LogNorm(vmin=1e-10, vmax=1.0) if use_log else Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(8.6, 8))
    # same orientation as plot_T_maps.py: first array index (x) runs up the vertical axis
    im = ax.imshow(Tmap, cmap=cmap, norm=norm, origin='lower', extent=[-half, half, -half, half])
    for e in edges[1:-1]:
        ax.axhline(coord(e), color='white', lw=0.5, alpha=0.6)
        ax.axvline(coord(e), color='white', lw=0.5, alpha=0.6)

    pad = (1.0 - frac) / 2.0
    lo, hi = BAND_EDGES[band], BAND_EDGES[band + 1]
    insets = {}
    for bx in range(k):                                          # bx: x-bin (vertical), by: y-bin (horizontal)
        for by in range(k):
            x0, x1 = coord(edges[by]), coord(edges[by + 1])      # horizontal extent in data coordinates
            y0, y1 = coord(edges[bx]), coord(edges[bx + 1])
            w, h = x1 - x0, y1 - y0
            ia = ax.inset_axes([x0 + pad * w, y0 + pad * h, frac * w, frac * h], transform=ax.transData)
            ia.plot(dv, spectra[bx, by], color='r', lw=0.7)
            if shade:
                # ia.axvspan(lo, hi, color=BAND_COLORS[band], alpha=0.30, lw=0)
                ia.axvspan(lo, hi, color='r', alpha=0.30, lw=0)
            ia.set_xlim(dv.min(), dv.max())
            if inset_log:
                ia.set_yscale('log')
                ia.set_ylim(1e-5, 1.5)
            else:
                ia.set_ylim(0, 1.05)
            ia.set_xticks([])
            ia.set_yticks([])
            ia.patch.set_facecolor((1, 1, 1, alpha))
            for spine in ia.spines.values():
                spine.set_linewidth(0.4)
            insets[(bx, by)] = ia
    ax.set_xlabel(r'$\Delta\Theta$ [degrees]')
    ax.set_ylabel(r'$\Delta\Theta$ [degrees]')
    ax.set_title(f'$z_0$={z0:.2f}, {BAND_NAMES[band]} band   '
                 f'(mini-plots: $\\mathcal{{T}}(\\Delta v)$, {dv.min():.0f} to {dv.max():.0f} km/s, per spatial bin)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=rf'$\mathcal{{T}}_{{\rm {BAND_NAMES[band]}}}$')
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return z0, insets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--redshifts', type=float, nargs='+', default=None, help='source redshifts to draw (default: all)')
    p.add_argument('--band', choices=BAND_NAMES, default='red', help='band shown as the map (default red)')
    p.add_argument('--output', default='spatial_map_spectra_z{z:.2f}.png',
                   help='output file name; {z} is replaced by the source redshift')
    p.add_argument('--fov-deg', type=float, default=3.6, help='field of view of the map in degrees')
    p.add_argument('--norm', choices=['auto', 'log', 'linear'], default='auto',
                   help='map colour scale: auto = log for the three bluest bands, linear for red/ultrared (as plot_T_maps.py)')
    p.add_argument('--inset-frac', type=float, default=1.0, help='fraction of each bin filled by its mini-plot')
    p.add_argument('--inset-alpha', type=float, default=0,
                   help='opacity of the white mini-plot background (0 = see the whole map through it)')
    p.add_argument('--inset-log', action='store_true', help='logarithmic transmission axis in the mini-plots')
    p.add_argument('--no-shade', action='store_true', help='do not shade the band range in the mini-plots')
    args = p.parse_args()

    data_dir = os.path.abspath(args.directory)
    todo = []
    for folder in z0_folders(data_dir):
        files = open_map_files(folder)
        z = float(np.asarray(files[0].attrs['SourceRedshiftRequested']).squeeze())
        if args.redshifts is None or any(abs(z - t) < 1e-6 for t in args.redshifts):
            todo.append((z, files))
        else:
            for f in files:
                f.close()
    if not todo:
        raise FileNotFoundError('None of the requested redshifts were found.')
    if len(todo) > 1 and '{' not in args.output:
        raise ValueError('Several redshifts selected: put {z} in --output so each figure gets its own file.')
    for z, files in sorted(todo, key=lambda t: t[0]):
        try:
            if len(files) != 1:
                raise ValueError('Expected one full-map file per redshift (a run with --binning-number, no --chunk).')
            out = args.output.format(z=z)
            draw(files[0], BAND_NAMES.index(args.band), out, args.fov_deg, args.norm, args.inset_frac,
                 args.inset_alpha, args.inset_log, not args.no_shade)
            print(f'Wrote {out}')
        finally:
            for f in files:
                f.close()


if __name__ == '__main__':
    main()
