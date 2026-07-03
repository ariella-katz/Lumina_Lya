import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import argparse

def plot_T_sig6(ss):
    z0s = []
    for s in ss:
        z0s.append(s.attrs['SourceRedshift'])
    z0s.sort()

    fig, axes = plt.subplots(figsize=(12,6))
    sig6 = 1-0.9999966
    ss_sorted = sorted(ss, key=lambda s: s.attrs['SourceRedshift'])
    z_sig6_intersects = [] #[source redshift][band]
    for i in range(len(ss)):
        s = ss_sorted[i]
        zs = np.copy(s['T_redshifts'][:])
        T_cum_bands = np.copy(s['T_cum_bands'][:])
        z_sig6_intersects_z0 =[]
        for band_num in range(5):
            T_cum_band = T_cum_bands[band_num]
            T_cum_tot = T_cum_band[-1]
            y = 1 - (1 - T_cum_band) / (1 - T_cum_tot)
            # y decreases as index increases -> reverse for searchsorted
            idx = len(y) - np.searchsorted(y[::-1], sig6)
            idx = np.clip(idx, 0, len(zs) - 1)
            z_sig6_intersects_z0.append(zs[idx])
        z_sig6_intersects.append(z_sig6_intersects_z0)
    
    z_sig6_intersects = np.array(z_sig6_intersects)  # shape (n_z0, 5)

    axes.plot(z0s, z_sig6_intersects[:, 0], color='blue', label='Ultrablue', marker='.')
    axes.plot(z0s, z_sig6_intersects[:, 1], color='green', label='Blue', marker='.')
    axes.plot(z0s, z_sig6_intersects[:, 2], color='olive', label='Center', marker='.')
    axes.plot(z0s, z_sig6_intersects[:, 3], color='orange', label='Red', marker='.')
    axes.plot(z0s, z_sig6_intersects[:, 4], color='red', label='Ultrared', marker='.')
    axes.plot(z0s, z0s, color='black', linestyle=':', label=rf'$z=z_0$')
    axes.set_xlabel("Source Redshift")
    axes.set_ylabel(rf'Redshift at which $1 - (1 - \mathcal{{T}}^\text{{int}}_{{z}})/(1 - \mathcal{{T}}^\text{{int}}_\text{{tot}})=6\sigma$')
    axes.legend()
    plt.savefig("zsig6_z0.png")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory of T maps"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dir_arg = args.directory
    dir = "T_maps"
    if dir_arg is not None:
        dir = dir_arg
    ss = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        ss.append(h5py.File(filepath, 'r'))
    plot_T_sig6(ss)

if __name__ == "__main__":
    main()