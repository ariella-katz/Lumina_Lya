import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator
from astropy.cosmology import Planck18
from matplotlib.cm import ScalarMappable

import plot_tau_CMB_spectra as smith

def get_full_T_grid(z0_paths):
    n_chunks = int(np.sqrt(len(z0_paths)))
    with h5py.File(z0_paths[0], 'r') as s0:
        z0 = float(np.asarray(s0.attrs['SourceRedshiftRequested']).squeeze())
        chunk_size = s0['tau_band_avgs'].shape[1]
    N = n_chunks * chunk_size

    T = np.zeros((5, N, N))   # bands: UB, B, C, R, UR
    for path in z0_paths:
        with h5py.File(path, 'r') as s:
            chunk_num = int(s.attrs['Chunk'])
            tau = s['tau_band_avgs'][:]
        x1 = chunk_size * (chunk_num // n_chunks)
        y1 = chunk_size * (chunk_num % n_chunks)
        T[:, x1:x1+chunk_size, y1:y1+chunk_size] = np.exp(-tau)

    T_ub, T_b, T_c, T_r, T_ur = T
    return z0, T_ub, T_b, T_c, T_r, T_ur

FOV_DEG = 3.6

PAIR_LABELS = [r'$\mathcal{T}_\text{int, R}\times\mathcal{T}_\text{int, R}$',
               r'$\mathcal{T}_\text{int, R}\times\mathcal{T}_\text{int, UB}$',
               r'$\mathcal{T}_\text{int, R}\times\mathcal{T}_\text{int, B}$',
               r'$\mathcal{T}_\text{int, R}\times\mathcal{T}_\text{int, C}$',
               r'$\mathcal{T}_\text{int, R}\times\mathcal{T}_\text{int, UR}$',
               r'$\mathcal{T}_\text{int, R}\times$ HI Fraction',
               r'$\mathcal{T}_\text{int, R}\times$ Density']

def load_records(ss, lightcone_file):
    """One dict per source redshift, sorted by z0, holding every map we cross-correlate."""
    records = []
    with h5py.File(lightcone_file, 'r') as cone:
        zs = cone['Redshifts'][:]
        for z0_ss in ss:
            z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(z0_ss)
            z0_i = np.argmax(zs <= z0)
            records.append(dict(
                z0=float(z0), z0_cone=float(zs[z0_i]),
                T_ub=T_ub, T_b=T_b, T_c=T_c, T_r=T_r, T_ur=T_ur,
                x_HI=1. - cone['HII_Fraction'][..., z0_i].astype(np.float64),
                density=cone['Density'][..., z0_i].astype(np.float64),
            ))
    records.sort(key=lambda r: r['z0'])
    return records

def get_map_pairs(r):
    map1 = r['T_r']
    map2s = [r['T_r'], r['T_ub'], r['T_b'], r['T_c'], r['T_ur'], r['x_HI'], r['density']]
    return map1, map2s


def plot_T_pow_1D(ss, lightcone_file, nbins=40, yscale='symlog'):
    records = load_records(ss, lightcone_file)
    n_pairs = len(PAIR_LABELS)
    z0s = [r['z0'] for r in records]
    cmap = plt.cm.inferno_r
    norm = Normalize(vmin=min(z0s), vmax=max(z0s))

    fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 2.4, 2.6), sharex=True)
    curves = [[] for _ in range(n_pairs)]

    for r in records:
        map1, map2s = get_map_pairs(r)
        color = cmap(norm(r['z0']))
        for mapi, map2 in enumerate(map2s):
            ell, Cl, *_ = smith.radial_power_spectrum(
                map1, map2=map2, theta_deg=FOV_DEG, nbins=nbins, logbins=True)
            D2 = ell * (ell + 1.) / (2. * np.pi) * Cl      # Delta^2_ell
            axes[mapi].plot(ell, D2, color=color, lw=1.5)
            curves[mapi].append(D2)

    for mapi, ax in enumerate(axes):
        ax.set_xscale('log')
        if yscale == 'symlog':
            # cross-spectra can go negative, so a pure log axis would drop those points
            ymax = np.nanmax([np.nanmax(np.abs(c)) for c in curves[mapi]])
            ax.set_yscale('symlog', linthresh=1e-3 * ymax)
        ax.set_title(PAIR_LABELS[mapi], fontsize=10)
        ax.set_xlabel(r'$\ell$', fontsize=9)
        ax.minorticks_on()
    axes[0].set_ylabel(r'$\Delta^2_\ell$', fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.015, aspect=30, fraction=0.02)
    cbar.set_label(r'$z_0$', fontsize=10)

    plt.savefig('cross_pows_1D.png', dpi=200, bbox_inches='tight')


def plot_T_pow_2D(ss, lightcone_file):
    records = load_records(ss, lightcone_file)
    n_rows, n_cols = len(records), len(PAIR_LABELS)

    fig = plt.figure(figsize=(n_cols * 2.4 + (n_cols - 1) * 0.6, n_rows * 2.4))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.6, hspace=0.25)

    for z0i, r in enumerate(records):
        map1, map2s = get_map_pairs(r)
        map2_z0s = [r['z0']] * 5 + [r['z0_cone']] * 2

        for mapi, map2 in enumerate(map2s):
            ell2d, p2d, meta = smith.power_spectrum_2d(map1, map2=map2, theta_deg=FOV_DEG)

            # FFT ordering -> zero frequency at the center
            D2 = np.fft.fftshift(ell2d * (ell2d + 1.) / (2. * np.pi) * p2d)
            npix = D2.shape[0]
            ell_ax = 2. * np.pi * np.fft.fftshift(np.fft.fftfreq(npix, d=meta['dtheta_rad']))
            extent = [ell_ax[0], ell_ax[-1], ell_ax[0], ell_ax[-1]]

            ax = fig.add_subplot(gs[z0i, mapi])
            vmax = np.percentile(np.abs(D2), 99)
            im = ax.imshow(D2, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           extent=extent, origin='lower')
            ax.text(0.05, 0.95, f'$z_0$={map2_z0s[mapi]:.1f}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='black', edgecolor='none', alpha=0.6))
            pos = ax.get_position()
            cax = fig.add_axes([pos.x1, pos.y0, 0.005, pos.height])
            cb = fig.colorbar(im, cax=cax)

            if mapi == 0:
                ax.set_ylabel(r'$\ell_y$', fontsize=9)
            else:
                ax.set_yticklabels([])
            if mapi == n_cols - 1:
                cb.set_label(r'$\Delta^2(\ell_x,\ell_y)$')
            if z0i == 0:
                ax.set_title(PAIR_LABELS[mapi])
            if z0i == n_rows - 1:
                ax.set_xlabel(r'$\ell_x$', fontsize=9)
            else:
                ax.set_xticklabels([])

    plt.savefig('cross_pows_2D.png', dpi=200, bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory of tau maps"
    )
    parser.add_argument(
        "--lightcone_file",
        type=str,
        default=None,
        help="File for density and fraction data"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dir_arg = args.directory
    data_dir = "tau_maps"
    if dir_arg is not None:
        data_dir = dir_arg

    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Could not find tau-map directory: {data_dir}")

    print(f"Reading z0 folders from: {data_dir}")

    # Group files by z0
    # Want the list of 8 z0s
    num_z0s = str(len(os.listdir(data_dir)))
    assert len(os.listdir(data_dir)) == 8, f'Must calculate from list of 8 z0s. Num z0s: {num_z0s}'
    ss = []
    for z0_name in sorted(os.listdir(data_dir)):
        z0_dir = os.path.join(data_dir, z0_name)
        if not os.path.isdir(z0_dir):
            continue
        ss.append([os.path.join(z0_dir, f) for f in sorted(os.listdir(z0_dir))])
    lightcone_arg = args.lightcone_file
    lightcone_file = 'All_40.hdf5'
    if lightcone_arg is not None:
        lightcone_file = lightcone_arg
    plot_T_pow_1D(ss, lightcone_file)
    plot_T_pow_2D(ss, lightcone_file)


if __name__ == "__main__":
    main()
