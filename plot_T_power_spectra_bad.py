import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator
from astropy.cosmology import Planck18

def get_full_T_grid(z0_ss):
    n_chunks = int(np.sqrt(len(z0_ss)))
    s0 = z0_ss[0]
    z0 = float(np.asarray(s0.attrs['Redshift']).squeeze())
    tau_band_avgs_0 = s0['tau_band_avgs'][:]
    chunk_size = tau_band_avgs_0.shape[1]
    T_ultrablue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_blue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_center = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_red = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_ultrared = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    ix1 = 0
    iy1 = 0
    x1 = ix1 * chunk_size
    y1 = iy1 * chunk_size
    T_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[0])
    T_blue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[1])
    T_center[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[2])
    T_red[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[3])
    T_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[4])
    for chunk in range(1, len(z0_ss)):
        s_chunk = z0_ss[chunk]
        chunk_num = int(s_chunk.attrs['Chunk'])
        tau_band_avgs_chunk = s_chunk['tau_band_avgs'][:]
        x1 = chunk_size * (chunk_num // n_chunks)
        y1 = chunk_size * (chunk_num % n_chunks)
        T_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[0])
        T_blue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[1])
        T_center[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[2])
        T_red[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[3])
        T_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[4])
    return z0, T_ultrablue, T_blue, T_center, T_red, T_ultrared

# def plot_T_power_spectra(ss):
#     fig, axes = plt.subplots(8, 5, figsize=(5 * 2.4, 8 * 2.4), sharex=True, sharey=True)
#     band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
#     band_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']

#     # Sort the maps by redshift
#     records = []
#     all_T_fluc_grid = []
#     for z0i in range(len(ss)):
#         z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
#         T_fluc_ub, k_ub, Delta2_ub = get_power_spectrum(z0, T_ub)
#         T_fluc_b, k_b, Delta2_b = get_power_spectrum(z0, T_b)
#         T_fluc_c, k_c, Delta2_c = get_power_spectrum(z0, T_c)
#         T_fluc_r, k_r, Delta2_r = get_power_spectrum(z0, T_r)
#         T_fluc_ur, k_ur, Delta2_ur = get_power_spectrum(z0, T_ur)
#         records.append((float(z0), T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur))
#         all_T_fluc_grid.append((T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur))
#     records.sort(key=lambda r: r[0])
#     all_T_fluc_grid = np.asarray(all_T_fluc_grid)
#     vmax = np.nanpercentile(np.abs(all_T_fluc_grid), 99)
#     vmin = -vmax

#     half_fov = 3.6 / 2
#     for row, (z0, T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur) in enumerate(records):
#         for col, (T_fluc_map, cmap) in enumerate(zip([T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur], band_cmaps)):
#             ax = axes[row, col]
#             im = ax.imshow(T_fluc_map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
#                       extent=[-half_fov, half_fov, -half_fov, half_fov])
#             ax.grid(True, color='white', alpha=0.5, linewidth=0.5, linestyle='-')
#             ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
#             ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
#             if col == 0:
#                 ax.set_ylabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
#                 ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
#                         transform=ax.transAxes, ha='left', va='top',
#                         fontsize=10, color='white',
#                         bbox=dict(boxstyle='round,pad=0.25',
#                                 facecolor='black', edgecolor='none', alpha=0.5))
#             if row == 0:
#                 ax.set_title(band_labels[col], fontsize=11)
#             if row == 7:
#                 ax.set_xlabel(r'$\Delta\Theta$ [degrees]', fontsize=9)

#     fig.subplots_adjust(top=0.90, wspace=0.05, hspace=0.05)
#     pos0 = axes[0, 0].get_position()
#     pos4 = axes[0, 4].get_position()
#     cax = fig.add_axes([pos0.x0, 0.94, pos4.x1 - pos0.x0, 0.005])
#     cb = fig.colorbar(im, ax=axes, cax=cax, orientation='horizontal',
#                       label=rf'$(\mathcal{{T}}_\text{{int}} - \langle\mathcal{{T}}_\text{{int}}\rangle)/\sigma_{{\mathcal{{T}}_\text{{int}}}}$')

#     plt.savefig('T_power_spectra.png', dpi=200, bbox_inches='tight')

def get_power_spectrum(z0, T_band_map):
    d = Planck18.comoving_transverse_distance(z0).value
    theta = np.deg2rad(3.6)
    L = d * theta
    N = T_band_map.shape[0]
    dx = L / N

    T_fluc_grid = (T_band_map - np.mean(T_band_map)) / np.std(T_band_map)
    T_fluc_grid_k = dx**2 * np.fft.fft2(T_fluc_grid)
    P = np.abs(T_fluc_grid_k)**2 / L**2

    kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # log-spaced bins so large-scale (low-k) modes get resolved individually,
    # instead of being lumped into one coarse bin near k=0
    k_fundamental = 2*np.pi / L
    k_edges = np.logspace(np.log10(k_fundamental), np.log10(k.max()), 40)
    k_edges = np.concatenate([[0], k_edges])

    Pk = np.zeros(len(k_edges)-1)
    k_center = np.zeros(len(k_edges)-1)
    for i in range(len(Pk)):
        mask = (k >= k_edges[i]) & (k < k_edges[i+1])
        Pk[i] = P[mask].mean() if mask.any() else np.nan
        k_center[i] = 0.5*(k_edges[i] + k_edges[i+1])

    Delta2 = k_center**2 * Pk / (2*np.pi)

    return T_fluc_grid, k_center, Delta2, d


def plot_T_power_spectra(ss):
    fig = plt.figure(figsize=(5 * 2.4, 8 * 2.4 * 1.4))
    outer_gs = GridSpec(8, 5, figure=fig, hspace=0.25, wspace=0.05)

    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
    cmap = 'RdBu_r'
    half_fov = 3.6 / 2
    scale_cutoff = 3.6

    # --- gather + sort by z0 ---
    records = []
    all_T_fluc_grid = []
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        results = [get_power_spectrum(z0, T) for T in (T_ub, T_b, T_c, T_r, T_ur)]
        T_flucs = [r[0] for r in results]
        k_center = results[0][1]
        Delta2s = [r[2] for r in results]
        d = results[0][3]
        records.append((float(z0), T_flucs, k_center, Delta2s, d))
        all_T_fluc_grid.append(T_flucs)

    records.sort(key=lambda r: r[0])
    all_T_fluc_grid = np.asarray(all_T_fluc_grid)
    vmax = np.nanpercentile(np.abs(all_T_fluc_grid), 99)
    vmin = -vmax

    # --- first pass: figure out what's actually going to be plotted per panel,
    #     so we can get a single global y-range for every power spectrum ---
    panel_data = []  # (row, col, z0, T_fluc, x_pos, y_pos, d)
    all_y = []
    for row, (z0, T_flucs, k_center, Delta2s, d) in enumerate(records):
        def k_to_deg(k, d=d):
            with np.errstate(divide='ignore'):
                rad = 2*np.pi / (k * d)
            return np.rad2deg(rad)

        k_nonzero = k_center[1:]
        scale_deg_row = k_to_deg(k_nonzero)
        keep = (scale_deg_row <= scale_cutoff) & np.isfinite(scale_deg_row)

        for col in range(5):
            Delta2_nonzero = Delta2s[col][1:]
            x_pos = scale_deg_row[keep]
            y_pos = Delta2_nonzero[keep]
            sort_idx = np.argsort(x_pos)
            x_pos, y_pos = x_pos[sort_idx], y_pos[sort_idx]
            panel_data.append((row, col, z0, T_flucs[col], x_pos, y_pos, d))
            all_y.append(y_pos)

    all_y = np.concatenate(all_y)
    all_y = all_y[np.isfinite(all_y) & (all_y > 0)]
    ymin, ymax = all_y.min(), all_y.max()

    # --- second pass: actually plot, now that vmin/vmax/ymin/ymax are all known ---
    im = None
    row0_map_axes = []
    for row, col, z0, T_fluc, x_pos, y_pos, d in panel_data:
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col],
                                           height_ratios=[3, 1], hspace=0)
        ax_map = fig.add_subplot(inner_gs[0])
        ax_ps = fig.add_subplot(inner_gs[1])

        # --- top: spatial fluctuation map ---
        im = ax_map.imshow(T_fluc, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                            extent=[-half_fov, half_fov, -half_fov, half_fov])
        ax_map.grid(True, color='white', alpha=0.5, linewidth=0.5, linestyle='-')
        ax_map.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_map.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_map.tick_params(labelbottom=False)
        if col == 0:
            ax_map.set_ylabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
            ax_map.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                        transform=ax_map.transAxes, ha='left', va='top',
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', edgecolor='none', alpha=0.5))
        if row == 0:
            ax_map.set_title(band_labels[col], fontsize=11)
            row0_map_axes.append(ax_map)

        # --- bottom: power spectrum ---
        ax_ps.plot(x_pos, y_pos, color='k', linewidth=1)
        ax_ps.set_xlim(x_pos.min(), scale_cutoff)
        ax_ps.set_yscale('log')
        ax_ps.set_ylim(ymin, ymax)

        if row < 7:
            ax_ps.tick_params(labelbottom=False)
        else:
            ax_ps.set_xlabel(r'$2\pi/k$ [degrees]', fontsize=9)
        if col == 0:
            ax_ps.set_ylabel(r'$\Delta^2$', fontsize=8)
        else:
            ax_ps.tick_params(labelleft=False)  # avoid repeating y-ticks across every column

        def deg_to_k(deg, d=d):
            rad = np.deg2rad(deg)
            with np.errstate(divide='ignore'):
                return 2*np.pi / (rad * d)
        def k_to_deg(k, d=d):
            with np.errstate(divide='ignore'):
                rad = 2*np.pi / (k * d)
            return np.rad2deg(rad)

        sec_ax = ax_ps.secondary_xaxis('top', functions=(deg_to_k, k_to_deg))
        sec_ax.tick_params(labelsize=6)
        if row == 0:
            sec_ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=8)

    fig.subplots_adjust(top=0.90)
    pos0 = row0_map_axes[0].get_position()
    pos4 = row0_map_axes[4].get_position()
    cax = fig.add_axes([pos0.x0, 0.94, pos4.x1 - pos0.x0, 0.005])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal',
                      label=r'$(\mathcal{T}_\text{int} - \langle\mathcal{T}_\text{int}\rangle)/\sigma_{\mathcal{T}_\text{int}}$')
    cb.ax.xaxis.set_ticks_position('top')

    plt.savefig('T_power_spectra.png', dpi=200, bbox_inches='tight')



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
        z0_ss = []
        for filename in sorted(os.listdir(z0_dir)):
            filepath = os.path.join(z0_dir, filename)
            z0_ss.append(h5py.File(filepath, 'r'))
        ss.append(z0_ss)
    plot_T_power_spectra(ss)


if __name__ == "__main__":
    main()
