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


def get_power_spectrum(z0, T_band_map):
    d = Planck18.comoving_transverse_distance(z0).value
    theta = np.deg2rad(3.6)
    L = d * theta
    N = T_band_map.shape[0]
    dx = L / N

    T_fluc_grid = (T_band_map - np.mean(T_band_map)) / np.std(T_band_map)
    T_fluc_grid -= T_fluc_grid.mean()
    T_fluc_grid_k = dx**2 * np.fft.fft2(T_fluc_grid)
    P = np.abs(T_fluc_grid_k)**2 / L**2

    kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # log-spaced bins so large-scale (low-k) modes get resolved individually,
    # instead of being lumped into one coarse bin near k=0
    # k_fundamental = 2*np.pi / L
    # k_edges = np.logspace(np.log10(k_fundamental), np.log10(k.max()), 40)
    # k_edges = np.concatenate([[0], k_edges])

    k_fundamental = 2 * np.pi / L

    # logarithmically spaced bins
    nbins = 50
    k_edges = np.logspace(np.log10(k_fundamental),
                          np.log10(k.max()),
                          nbins + 1)

    k_center = np.zeros(nbins)
    Pk = np.zeros(nbins)
    Pk_err = np.zeros(nbins)

    for i in range(nbins):
        mask = (k >= k_edges[i]) & (k < k_edges[i+1])

        if np.any(mask):
            k_center[i] = np.exp(np.mean(np.log(k[mask])))
            Pk[i] = np.mean(P[mask])

            Nmodes = np.count_nonzero(mask)
            if Nmodes > 1:
                Pk_err[i] = np.std(P[mask], ddof=1) / np.sqrt(Nmodes)
            else:
                Pk_err[i] = np.nan
        else:
            k_center[i] = np.nan
            Pk[i] = np.nan
            Pk_err[i] = np.nan

    Delta2 = k_center**2 * Pk / (2*np.pi)
    Delta2_err = k_center**2 * Pk_err / (2*np.pi)

    return (
        T_fluc_grid,
        k_center,
        Delta2,
        Delta2_err,
        d,
    )


def plot_T_power_spectra(ss):
    fig, axes = plt.subplots(8, 5, figsize=(5 * 3.4, 8 * 2.4), sharex=True, sharey=True)
    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
    band_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
    band_cs = ['blue', 'green', 'olive', 'orange', 'red']

    # Sort the maps by redshift
    records = []
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_fluc_ub, k_ub, Delta2_ub, Delta2_ub_err, d = get_power_spectrum(z0, T_ub)
        T_fluc_b, k_b, Delta2_b, Delta2_b_err, d = get_power_spectrum(z0, T_b)
        T_fluc_c, k_c, Delta2_c, Delta2_c_err, d = get_power_spectrum(z0, T_c)
        T_fluc_r, k_r, Delta2_r, Delta2_r_err, d = get_power_spectrum(z0, T_r)
        T_fluc_ur, k_ur, Delta2_ur, Delta2_ur_err, d = get_power_spectrum(z0, T_ur)
        records.append((
            float(z0),
            [k_ub, Delta2_ub, Delta2_ub_err, d],
            [k_b, Delta2_b, Delta2_b_err, d],
            [k_c, Delta2_c, Delta2_c_err, d],
            [k_r, Delta2_r, Delta2_r_err, d],
            [k_ur, Delta2_ur, Delta2_ur_err, d],
        ))
    records.sort(key=lambda r: r[0])

    half_fov = 3.6 / 2
    for row, (z0, ub_set, b_set, c_set, r_set, ur_set) in enumerate(records):
        for col, (set, c) in enumerate(zip([ub_set, b_set, c_set, r_set, ur_set], band_cs)):
            ax = axes[row, col]
            k = set[0]
            D2 = set[1]
            D2_err = set[2]
            d = set[3]
            def k_to_deg(k, d=d):
                with np.errstate(divide='ignore'):
                    rad = 2*np.pi / (k * d)
                return np.rad2deg(rad)
            valid = np.isfinite(k) & np.isfinite(D2)
            ax.errorbar(k[valid], D2[valid], yerr=D2_err[valid], color=c, 
                        lw=0.9, markersize=2, fmt='.-',capsize=0,)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_xscale('log')
            ax.set_yscale('log')
            def k_to_deg(k):
                return np.rad2deg(2*np.pi/(k*d))

            def deg_to_k(theta):
                return 2*np.pi/(np.deg2rad(theta)*d)

            secax = ax.secondary_xaxis(
                'top',
                functions=(k_to_deg, deg_to_k)
            )

            if row == 0:
                secax.set_xlabel("Angular scale [deg]")
                ax.set_title(band_labels[col], fontsize=12)
            else:
                secax.set_xlabel("")
                # hide the tick labels but keep the ticks
                secax.tick_params(labeltop=False)
            if col == 0:
                ax.set_ylabel(r'$\Delta^2(k)$', fontsize=9)
                ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='black', edgecolor='none', alpha=0.5))
            if row == 7:
                ax.set_xlabel(r"$k\ [{\rm cMpc}^{-1}]$")
                # ax.set_xlabel('k [1/Mpc]', fontsize=9)

    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig('T_power_spectra_grid.png', dpi=200, bbox_inches='tight')




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
