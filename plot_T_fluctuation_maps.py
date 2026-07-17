import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
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

def get_full_T_fluc_grid(T_band_map):
    T_fluc_grid = (T_band_map - np.mean(T_band_map)) / np.std(T_band_map)
    return T_fluc_grid

def plot_T_maps(ss):
    fig, axes = plt.subplots(8, 5, figsize=(5 * 2.4, 8 * 2.4), sharex=True, sharey=True)
    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
    band_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']

    # Sort the maps by redshift
    records = []
    all_T_fluc_grid = []
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur = get_full_T_fluc_grid(T_ub), get_full_T_fluc_grid(T_b), get_full_T_fluc_grid(T_c), get_full_T_fluc_grid(T_r), get_full_T_fluc_grid(T_ur)
        records.append((float(z0), T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur))
        all_T_fluc_grid.append((T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur))
    records.sort(key=lambda r: r[0])
    all_T_fluc_grid = np.asarray(all_T_fluc_grid)
    vmax = np.nanpercentile(np.abs(all_T_fluc_grid), 99)
    vmin = -vmax

    half_fov = 3.6 / 2
    for row, (z0, T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur) in enumerate(records):
        for col, (T_fluc_map, cmap) in enumerate(zip([T_fluc_ub, T_fluc_b, T_fluc_c, T_fluc_r, T_fluc_ur], band_cmaps)):
            ax = axes[row, col]
            im = ax.imshow(T_fluc_map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                      extent=[-half_fov, half_fov, -half_fov, half_fov])
            ax.grid(True, color='black', alpha=0.5, linewidth=0.5, linestyle='-')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if col == 0:
                ax.set_ylabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
                ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='black', edgecolor='none', alpha=0.5))
            if row == 0:
                ax.set_title(band_labels[col], fontsize=11)
            if row == 7:
                ax.set_xlabel(r'$\Delta\Theta$ [degrees]', fontsize=9)

    fig.subplots_adjust(top=0.90, wspace=0.05, hspace=0.05)
    pos0 = axes[0, 0].get_position()
    pos4 = axes[0, 4].get_position()
    cax = fig.add_axes([pos0.x0, 0.94, pos4.x1 - pos0.x0, 0.005])
    cb = fig.colorbar(im, ax=axes, cax=cax, orientation='horizontal',
                      label=rf'$(\mathcal{{T}}_\text{{int}} - \langle\mathcal{{T}}_\text{{int}}\rangle)/\sigma_{{\mathcal{{T}}_\text{{int}}}}$')

    plt.savefig('T_fluc_maps_grid.png', dpi=200, bbox_inches='tight')



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
