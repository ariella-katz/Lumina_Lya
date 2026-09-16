import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from matplotlib.gridspec import GridSpec

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

def plot_T_pix_hists(ss):
    fig, axes = plt.subplots(8, 5, figsize=(5 * 2.4, 8 * 2.4), sharey=False)
    band_colors = ['#0072B2', '#009E73', '#999933', '#E69F00', '#CC3311']
    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']

    records = []
    min_T = 1e7
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_data = np.asarray([T_ub, T_b, T_c, T_r, T_ur])
        # max_tau = max(max_tau, np.max(tau_data[np.isfinite(tau_data)]))
        min_T = min(min_T, np.min(T_data[T_data > 0.]))
        records.append((float(z0), T_ub, T_b, T_c, T_r, T_ur))
    records.sort(key=lambda r: r[0])

    for row, (z0, T_ub, T_b, T_c, T_r, T_ur) in enumerate(records):
        for col, T_map in enumerate([T_ub, T_b, T_c, T_r, T_ur]):
            data = T_map.flatten()
            ax = axes[row, col]
            if col<3:
                bins=np.logspace(-7, np.log(0.2), 50, endpoint=True)
                bad_mask = data < 1e-7
                ax.hist(data[~bad_mask], bins=bins, color=band_colors[col])
                ax.bar(1e-7*(0.5), data[bad_mask].sum(), width=1e-7*0.5, alpha=0.6)
                ax.set_xlim(left=1e-7*(0.5))
                ax.set_xscale('log')
            else:
                bins=np.linspace(0., 1., 50)
                ax.hist(data, bins=bins, color=band_colors[col])
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            if col == 0:
                ax.set_ylabel('Number of Pixels', fontsize=9)
            ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.25',
                            facecolor='black', edgecolor='none', alpha=0.6))
            if row == 0:
                ax.set_title(band_labels[col], fontsize=14)
            if row == 7:
                ax.set_xlabel(r'$\mathcal{T}_\text{int}$', fontsize=9)

    fig.subplots_adjust(top=0.90, wspace=0.3, hspace=0.3)
    plt.savefig('T_pix_hist.png', dpi=200, bbox_inches='tight')

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
    plot_T_pix_hists(ss)


if __name__ == "__main__":
    main()