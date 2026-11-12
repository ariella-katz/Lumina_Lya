"""Heat map of the map-averaged transmission (x = velocity offset, y = source redshift, colour = transmission),
with the T = 50% crossing marked on every row and, optionally, a panel aligned with the vertical axis showing
how the global neutral fraction changes with source redshift.

Reads   DIRECTORY/z0=<z>/tau_map_<z>_<suffix>.hdf5   (one folder per source redshift) containing either
  * T_binned [k, k, nf]  -- from `lumina_lya ... --binning-number N` (N=1 gives one spectrum per source), or
  * taus     [x, y, nf]  -- from `--products spectra` (full map or chunk files; averaged here).
All source redshifts must share the same frequency grid (same --sampling and --dv-step).

Crossing lines: a contour is drawn where the transmission equals --level (default 0.5); it joins the crossings
of neighbouring redshift rows by linear interpolation, so it stays correct where a row crosses the level more
than once or not at all. A dot marks the exact crossing velocity on every row (found by linear interpolation
between neighbouring frequency samples), and these values are printed.

Side panel (needs --input, the lightcone file): the global neutral fraction at each source redshift, from the
slice of the lightcone at the source plane (the depth cell that contains the source, recorded as SourceCell):
x_HI = 1 - HII_Fraction, averaged over the whole slice (volume-weighted; --weighting mass uses the density
as weights). This is one thin slice of the simulated sky, so it carries cosmic variance; --slab N also
averages the N cells on either side of the source cell.

Usage:
    python plot_mean_spectra_heatmap.py --directory tau_maps_mean50 --input /scratch/.../All.hdf5 \
        --output mean_spectra_heatmap.png [--log] [--level 0.5]
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

CM_PER_KM = 1e5


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


def mean_spectrum(files):
    """Return (z0, dv_kms, T[nf]) with T the transmission averaged over every pixel of the map."""
    s0 = files[0]
    z0 = float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze())
    dv = s0['Dvs'][:] / CM_PER_KM

    if 'T_binned' in s0:
        if len(files) != 1:
            raise ValueError('T_binned files cannot be combined; run --binning-number on the full map.')
        T = s0['T_binned'][:]
        k = T.shape[0]
        if k == 1:
            return z0, dv, T[0, 0]
        m, npix = int(s0.attrs['BinSizePixels']), int(s0.attrs['X1'])
        sizes = np.full(k, float(m))
        sizes[-1] = npix - (k - 1) * m
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


def global_neutral_fraction(input_path, cells, weighting='volume', slab=0):
    """{cell: mean neutral fraction of the slice(s) at that depth cell}."""
    out = {}
    with h5py.File(input_path, 'r') as f:
        hii = SliceReader(f['HII_Fraction'])
        dens = SliceReader(f['Density']) if weighting == 'mass' else None
        nz = f['HII_Fraction'].shape[2]
        for cell in sorted(set(cells)):                          # sorted: reuses the cached depth block
            num = den = 0.0
            for k in range(max(cell - slab, 0), min(cell + slab, nz - 1) + 1):
                x = 1.0 - hii[k]
                if dens is None:
                    num += x.sum()
                    den += x.size
                else:
                    w = dens[k]
                    num += (x * w).sum()
                    den += w.sum()
            out[cell] = num / den
    return out


# ----------------------------------------------------------------------------- crossings
def crossings(dv, T, level):
    """Velocities at which T crosses `level`, by linear interpolation between neighbouring samples."""
    above = T >= level
    idx = np.nonzero(above[:-1] != above[1:])[0]
    return dv[idx] + (level - T[idx]) * (dv[idx + 1] - dv[idx]) / (T[idx + 1] - T[idx])


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--directory', required=True, help='directory containing the z0=... folders')
    p.add_argument('--input', default=None, help='lightcone file; adds the global neutral fraction side panel')
    p.add_argument('--output', default='mean_spectra_heatmap.png')
    p.add_argument('--log', action='store_true', help='logarithmic colour scale for the transmission')
    p.add_argument('--vmin', type=float, default=None,
                   help='lower colour limit (default 0 for linear; smallest positive value for --log)')
    p.add_argument('--level', type=float, default=0.5, help='transmission level to mark (default 0.5)')
    p.add_argument('--cmap', default='viridis')
    p.add_argument('--weighting', choices=['volume', 'mass'], default='volume',
                   help='how the neutral fraction is averaged over the slice')
    p.add_argument('--slab', type=int, default=0, help='also average this many cells on each side of the source cell')
    p.add_argument('--xhi-log', action='store_true', help='logarithmic neutral-fraction axis')
    args = p.parse_args()

    data_dir = os.path.abspath(args.directory)
    records = []                                                  # (z0, cell, dv, T)
    for folder in z0_folders(data_dir):
        files = open_map_files(folder)
        try:
            z0, dv, T = mean_spectrum(files)
            records.append((z0, int(files[0].attrs['SourceCell']), dv, T))
        finally:
            for f in files:
                f.close()
    if len(records) < 2:
        raise ValueError('The heat map needs at least two source redshifts.')
    records.sort(key=lambda r: r[0])

    zs = np.array([r[0] for r in records])
    dv = records[0][2]
    for z0, _, dv_i, _ in records:
        if dv_i.shape != dv.shape or not np.allclose(dv_i, dv):
            raise ValueError(f'z0={z0} uses a different frequency grid; rerun all sources with the same '
                             '--sampling and --dv-step.')
    T = np.array([r[3] for r in records])                         # [n_redshifts, nf]

    xhi = None
    if args.input:
        by_cell = global_neutral_fraction(args.input, [r[1] for r in records], args.weighting, args.slab)
        xhi = np.array([by_cell[r[1]] for r in records])

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad('black')                                         # zero transmission cannot be shown on a log scale
    if args.log:
        vmin = args.vmin if args.vmin is not None else T[T > 0].min()
        norm = LogNorm(vmin=vmin, vmax=1.0)
    else:
        norm = Normalize(vmin=0.0 if args.vmin is None else args.vmin, vmax=1.0)

    side = xhi is not None
    fig = plt.figure(figsize=(10.5 if side else 8.5, 5.8))
    gs = GridSpec(1, 3 if side else 2, figure=fig, width_ratios=[4, 1.1, 0.12] if side else [4, 0.12],
                  wspace=0.06 if side else 0.04)
    ax = fig.add_subplot(gs[0])
    mesh = ax.pcolormesh(dv, zs, T, norm=norm, cmap=cmap, shading='nearest')

    # T = level crossings: the k-th crossing of every row is joined into one line
    cross = [crossings(dv, T[i], args.level) for i in range(len(zs))]
    stroke = [pe.withStroke(linewidth=3.4, foreground='black')]
    print('  z0     velocities where T crosses %.0f%% [km/s]%s' % (100 * args.level, '    global x_HI' if side else ''))
    for i, z0 in enumerate(zs):
        txt = ', '.join(f'{v:8.1f}' for v in cross[i]) if len(cross[i]) else '     none'
        print(f'{z0:7.3f}  {txt}' + (f'    {xhi[i]:.4g}' if side else ''))
    cs = ax.contour(dv, zs, T, levels=[args.level], colors='white', linewidths=0.8)
    try:
        cs.set_path_effects(stroke)                                # matplotlib >= 3.8
    except AttributeError:
        for c in cs.collections:
            c.set_path_effects(stroke)
    xs = [v for i in range(len(zs)) for v in cross[i]]
    ys = [zs[i] for i in range(len(zs)) for _ in cross[i]]
    # ax.plot(xs, ys, 'o', color='white', mec='black', ms=4, mew=0.8)
    ax.legend(handles=[Line2D([], [], color='white', lw=0.8, ms=4, mec='black', path_effects=stroke,
                               label=f'$\\mathcal{{T}}$ = {args.level:.0%}')],
              loc='upper left', fontsize=9, framealpha=0.85)
    ax.set_xlabel(r'$\Delta v$ [km/s]')
    ax.set_ylabel(r'source redshift $z_0$')

    if side:
        ax_side = fig.add_subplot(gs[1], sharey=ax)               # aligned with the heat map's vertical axis
        ax_side.plot(xhi, zs, 'o-', color='k', ms=4, lw=1.3)
        if args.xhi_log:
            ax_side.set_xscale('log')
        else:
            ax_side.xaxis.set_major_locator(MaxNLocator(nbins=3))      # avoids colliding labels on a narrow range
        ax_side.set_xlabel(r'global $x_{\rm HI}$')
        ax_side.grid(alpha=0.3)
        plt.setp(ax_side.get_yticklabels(), visible=False)
        ax_side.tick_params(axis='y', length=0)
    cax = fig.add_subplot(gs[-1])
    fig.colorbar(mesh, cax=cax, label=r'map-averaged transmission $\langle\mathcal{T}\rangle$')
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Wrote {args.output} ({len(records)} redshifts x {len(dv)} frequencies)')


if __name__ == '__main__':
    main()
