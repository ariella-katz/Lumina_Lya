import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator
import colorsys

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


def light_to_dark_cmap(color, name, low_v=1.0, low_s=0.5, high_v=0.25, high_s=1.0):
    """
    Blends in HSV space to keep endpoints vivid instead of greyed-out.

    low_v, low_s:   value/saturation at the LOW-tau end (pale but still colored)
    high_v, high_s: value/saturation at the HIGH-tau end (dark but still saturated)
    """
    r, g, b = to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    low_rgb  = colorsys.hsv_to_rgb(h, low_s,  low_v)
    high_rgb = colorsys.hsv_to_rgb(h, high_s, high_v)

    return LinearSegmentedColormap.from_list(name, [high_rgb, low_rgb], N=256)

def plot_T_maps(ss):
    fig, axes = plt.subplots(8, 5, figsize=(5 * 2.4, 8 * 2.4), sharex=True, sharey=True)
    band_colors = ['#0072B2', '#009E73', '#999933', '#E69F00', '#CC3311']
    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
    band_cmaps = [light_to_dark_cmap(c, f'band_{i}') for i, c in enumerate(band_colors)]
    for band_cmap in band_cmaps:
        band_cmap.set_bad(color='black')

    # --- Step 1: gather z0 + its 5 tau maps together, and get global vmin/vmax ---
    records = []
    # max_tau, min_tau = 0, 1e7
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_data = np.asarray([T_ub, T_b, T_c, T_r, T_ur])
        # max_tau = max(max_tau, np.max(tau_data[np.isfinite(tau_data)]))
        # min_tau = min(min_tau, np.min(tau_data[np.isfinite(tau_data)]))
        records.append((float(z0), T_ub, T_b, T_c, T_r, T_ur))

    # norm = LogNorm(vmin=min_tau, vmax=max_tau)

    # --- Step 2: sort the bundled records by z0 — no axes involved yet ---
    records.sort(key=lambda r: r[0])

    # --- Step 3: plot, using position in the sorted list as the row index ---
    half_fov = 3.6 / 2
    for row, (z0, T_ub, T_b, T_c, T_r, T_ur) in enumerate(records):
        for col, (T_map, cmap) in enumerate(zip([T_ub, T_b, T_c, T_r, T_ur], band_cmaps)):
            ax = axes[row, col]
            ax.imshow(T_map, cmap=cmap, vmin=0, vmax=1, origin='lower',
                      extent=[-half_fov, half_fov, -half_fov, half_fov])
            ax.grid(True, color='white', alpha=0.5, linewidth=0.5, linestyle='-')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if col == 0:
                ax.set_ylabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
                ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='black', edgecolor='none', alpha=0.6))
            if row == 0:
                ax.set_title(band_labels[col], fontsize=11)
            if row == 7:
                ax.set_xlabel(r'$\Delta\Theta$ [degrees]', fontsize=9)

    fig.subplots_adjust(top=0.90, wspace=0.05, hspace=0.05)
    for j in range(5):
        pos = axes[0, j].get_position()
        cax = fig.add_axes([pos.x0, 0.93, pos.width, 0.005])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=band_cmaps[j]),
                          cax=cax, orientation='horizontal', label=rf'$\mathcal{{T}}_\text{{int}}$')
        cb.ax.tick_params(labelsize=7)
        cb.ax.xaxis.set_ticks_position('top')

    plt.savefig('T_maps_grid.png', dpi=200, bbox_inches='tight')



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
    plot_T_maps(ss)


if __name__ == "__main__":
    main()
