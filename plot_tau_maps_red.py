import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

def get_full_tau_grid(z0_ss):
    n_chunks = int(np.sqrt(len(z0_ss)))
    s0 = z0_ss[0]
    z0 = float(np.asarray(s0.attrs['Redshift']).squeeze())
    tau_band_avgs_0 = s0['tau_band_avgs'][:]
    chunk_size = tau_band_avgs_0.shape[1]
    tau_ultrablue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    tau_blue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    tau_center = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    tau_red = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    tau_ultrared = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    ix1 = 0
    iy1 = 0
    x1 = ix1 * chunk_size
    y1 = iy1 * chunk_size
    tau_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_0[0]
    tau_blue[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_0[1]
    tau_center[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_0[2]
    tau_red[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_0[3]
    tau_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_0[4]
    for chunk in range(1, len(z0_ss)):
        s_chunk = z0_ss[chunk]
        chunk_num = int(s_chunk.attrs['Chunk'])
        tau_band_avgs_chunk = s_chunk['tau_band_avgs'][:]
        x1 = chunk_size * (chunk_num // n_chunks)
        y1 = chunk_size * (chunk_num % n_chunks)
        tau_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_chunk[0]
        tau_blue[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_chunk[1]
        tau_center[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_chunk[2]
        tau_red[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_chunk[3]
        tau_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = tau_band_avgs_chunk[4]
    return z0, tau_ultrablue, tau_blue, tau_center, tau_red, tau_ultrared


# --- 2. Build a black -> color colormap for each band ---
def black_to_color_cmap(color, name):
    return LinearSegmentedColormap.from_list(name, ['black', color], N=256)

def plot_tau_maps(ss):
    fig, axes = plt.subplots(10, 5, figsize=(5 * 2.4, 10 * 2.4), sharex=True, sharey=True)
    
    z_source = []  # your 50 source redshifts
    max_tau = 0
    min_tau = 10e6
    for z0i in range(len(ss)):
        z0_ss = ss[z0i]
        z0, tau_ultrablue, tau_blue, tau_center, tau_red, tau_ultrared = get_full_tau_grid(z0_ss)
        z_source.append(z0)
        max_tau = max(max_tau, np.max(np.asarray([tau_ultrablue, tau_blue, tau_center, tau_red, tau_ultrared])))
        min_tau = min(min_tau, np.min(np.asarray([tau_ultrablue, tau_blue, tau_center, tau_red, tau_ultrared])))
    print(max_tau, min_tau)
    norm = LogNorm(vmin=min_tau, vmax=max_tau)
    for z0i in range(len(ss)):
        z0_ss = ss[z0i]
        z0, tau_ultrablue, tau_blue, tau_center, tau_red, tau_ultrared = get_full_tau_grid(z0_ss)
        axdim = tau_ultrablue.shape[0]
        ticks = np.linspace(0, 3.6, axdim) - 3.6/2.
        ax = axes[z0i // 10, z0i % 5]
        ax.imshow(tau_red, cmap='inferno', norm=norm,)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.text(0.95, 0.95, f'$z_0$={z_source[z0i]:.2f}',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='black', edgecolor='none', alpha=0.6))

    plt.savefig('tau_maps_red_grid.png', dpi=200, bbox_inches='tight')



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
    assert len(os.listdir(data_dir)) == 50, "Must calculate from list of 50 z0s."
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
    plot_tau_maps(ss)


if __name__ == "__main__":
    main()
