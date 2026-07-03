import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import argparse

def shades(base_color, n, v_range=(0.35, 0.95)):
    """
    Generate n shades of base_color by varying brightness (value),
    keeping hue and saturation fixed.
    """
    import colorsys
    r, g, b = mcolors.to_rgb(base_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    values = np.linspace(v_range[1], v_range[0], n)  # high z = darker, or flip as you like
    return [colorsys.hsv_to_rgb(h, s, v_i) for v_i in values]

def truncate_cmap(cmap_name, min_val=0.0, max_val=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(min_val, max_val, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{min_val:.2f},{max_val:.2f})", new_colors
    )


def plot_T_cdf(ss, stacked=True):
    z0s = []
    for s in ss:
        z0s.append(s.attrs['SourceRedshift'])
    band_colors = ["tab:blue", "tab:green", "tab:olive", "tab:orange", "tab:red"]
    band_shades = [shades(color, len(z0s)) for color in band_colors]
    bands = {"Ultrablue":"tab:blue", "Blue":"tab:green", "Center":"tab:olive", "Red":"tab:orange", "Ultrared":"tab:red"}

    if stacked:
        fig, axes = plt.subplots(figsize=(12,6))
        axes = [axes]
    else:    
        fig, axes = plt.subplots(5, 1, figsize=(6, 14), sharex=True)
        fig.subplots_adjust(hspace=0, right=0.85)
    sig5 = 1-0.99977
    sig6 = 1-0.9999966
    ss_sorted = sorted(ss, key=lambda s: s.attrs['SourceRedshift'])
    for i in range(len(ss)):
        s = ss_sorted[i]
        zs = np.copy(s['T_redshifts'][:])
        T_cum_bands = np.copy(s['T_cum_bands'][:])
        for band_num in range(5):
            T_cum_tot = T_cum_bands[band_num][-1]
            if stacked:
                ax = axes[0]
            else:
                ax = axes[band_num]
            ax.plot(zs, 1-(1-T_cum_bands[band_num])/(1-T_cum_tot), c=band_shades[band_num][i])
            if (band_num==0 and i==0):
                ax.plot(np.linspace(zs.min(), zs.max(), 50), sig5*np.ones(50), c='grey', linestyle=':', label=rf'5$\sigma$')
                ax.plot(np.linspace(zs.min(), zs.max(), 50), sig6*np.ones(50), c='grey', linestyle='--', label=rf'6$\sigma$')
            else:
                ax.plot(np.linspace(zs.min(), zs.max(), 50), sig5*np.ones(50), c='grey', linestyle=':')
                ax.plot(np.linspace(zs.min(), zs.max(), 50), sig6*np.ones(50), c='grey', linestyle='--')
            if (band_num==4 and i==0):
                ax.set_xlabel('Redshift')
    grey_cmap = truncate_cmap("Greys", min_val=0.3, max_val=1.0)    
    norm = mcolors.Normalize(vmin=np.min(z0s), vmax=np.max(z0s))
    sm = cm.ScalarMappable(cmap=grey_cmap, norm=norm)
    sm.set_array([])
    pos_top = axes[0].get_position()
    pos_bottom = axes[-1].get_position()
    cbar_top = pos_top.y1
    cbar_bottom = pos_bottom.y0
    cbar_height = cbar_top - cbar_bottom
    if stacked:
        cbar_ax = fig.add_axes([0.9, cbar_bottom, 0.01, cbar_height])  # [left, bottom, width, height]
    else:
        cbar_ax = fig.add_axes([0.85, cbar_bottom, 0.01, cbar_height])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, label="Source Redshift")
    for ax in axes:
        # ax.set_ylim((0.9997, 1.00004))
        ax.set_yscale('log')
        ax.set_ylabel(rf'$(\mathcal{{T}}^\text{{int}}_{{z}}-\mathcal{{T}}^\text{{int}}_\text{{tot}})/\mathcal{{T}}^\text{{int}}_\text{{tot}}$')
    legend1 = axes[0].legend(loc="lower right", bbox_to_anchor=(1., 1.02), ncol=2, borderaxespad=0, frameon=False)
    if stacked:
        legend_handles = [Line2D([0], [0], color=color, lw=2, label=band) for band, color in bands.items()]
        legend2 = axes[0].legend(handles=legend_handles, title="Band", ncol=5, loc="lower left", bbox_to_anchor=(0., 1.02), borderaxespad=0, frameon=False)
        axes[0].add_artist(legend1)
    else:
        panel_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
        for ax, label in zip(axes, panel_labels):
            ax.text(0.98, 0.1, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    plt.savefig('T_cdf.png')
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
    parser.add_argument(
        "--not_overlaid",
        action='store_false',
        dest='overlay',
        help="Whether to overlay all bands on the same plot"
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
    overlay_arg = args.overlay
    plot_T_cdf(ss, overlay_arg)

if __name__ == "__main__":
    main()