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

def adaptive_k_bins(k, k_fundamental, k_max, min_modes=8, n_log_bins=25):
    """
    Build bin edges that guarantee at least `min_modes` grid points 
    per bin at low k, then log-spaced at high k where modes are plentiful.
    """
    k_flat = k.flatten()
    k_flat = k_flat[k_flat >= k_fundamental]
    k_sorted = np.sort(k_flat)

    edges = [k_fundamental]
    i = 0
    # Phase 1: equal-count bins until we have enough log-spaced room left
    while i < len(k_sorted):
        i_next = min(i + min_modes, len(k_sorted) - 1)
        edges.append(k_sorted[i_next])
        i = i_next
        # stop equal-count phase once bin widths are already resolving 
        # a healthy log-ratio (heuristic: after ~5-8 bins, or once 
        # remaining range can be log-spaced reasonably)
        if len(edges) > 6:
            break

    # Phase 2: log-spaced bins for the rest of the range
    remaining = np.logspace(np.log10(edges[-1]), np.log10(k_max), n_log_bins)
    edges = np.concatenate([edges, remaining[1:]])
    return np.unique(edges)

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
    # nbins = 25
    # k_edges = np.logspace(np.log10(k_fundamental),
    #                       np.log10(k.max()),
    #                       nbins + 1)
    k_edges = adaptive_k_bins(k, k_fundamental, k.max(), min_modes=100, n_log_bins=25)
    nbins = len(k_edges) - 1
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
def radial_bin_stats(x2d, y2d, edges):
    """
    Compute radial bin statistics for y(x) in bins defined by edges.

    Parameters
    ----------
    x2d, y2d : 2D arrays
    edges : 1D array
        Bin edges

    Returns
    -------
    stats : dict
        centers, mean, var, std, sem, count, edges
    """
    x = np.ravel(x2d)
    y = np.ravel(y2d)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    nbins = len(edges) - 1
    ibin = np.digitize(x, edges) - 1
    valid = (ibin >= 0) & (ibin < nbins)

    x = x[valid]
    y = y[valid]
    ibin = ibin[valid]

    counts = np.bincount(ibin, minlength=nbins).astype(np.int64)
    sumy = np.bincount(ibin, weights=y, minlength=nbins)
    sumy2 = np.bincount(ibin, weights=y*y, minlength=nbins)

    mean = np.full(nbins, np.nan, dtype=np.float64)
    var = np.full(nbins, np.nan, dtype=np.float64)

    m = counts > 0
    mean[m] = sumy[m] / counts[m]
    var[m] = sumy2[m] / counts[m] - mean[m]**2
    var[m] = np.maximum(var[m], 0.0)

    std = np.sqrt(var)

    sem = np.full(nbins, np.nan, dtype=np.float64)
    m2 = counts > 1
    sem[m2] = std[m2] / np.sqrt(counts[m2])

    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "centers": centers,
        "mean": mean,
        "var": var,
        "std": std,
        "sem": sem,
        "counts": counts,
        "edges": edges,
    }


def adaptive_radial_bin(
    x2d,
    y2d,
    nbins_target=80,
    oversample=1,
    x_min=None,
    x_max=None,
    logbins=True,
    max_bin_width=None,
    min_count=45,
    sem_frac_max=0.5,
    var_less_than_mean=False,
):
    """
    Adaptive radial binning by merging fine bins.

    Parameters
    ----------
    x2d, y2d : 2D arrays
        Radial coordinate and quantity to average
    nbins_target : int
        Desired approximate final number of bins
    oversample : int
        Start with oversample * nbins_target bins
    x_min, x_max : float or None
        Range to bin
    logbins : bool
        Use log-spaced initial bins
    max_bin_width : float or None
        Maximum allowed merged bin width in x
    min_count : int
        Minimum number of samples per merged bin
    sem_frac_max : float
        Require SEM < sem_frac_max * |mean| when possible
    var_less_than_mean : bool
        Also enforce var < |mean| if requested

    Returns
    -------
    centers, mean, var, std, sem, count, edges
    """
    x = np.ravel(x2d)
    y = np.ravel(y2d)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if x_min is None:
        x_pos = x[x > 0] if logbins else x
        x_min = np.min(x_pos)
    if x_max is None:
        x_max = np.max(x)

    n0 = oversample * nbins_target
    if logbins:
        edges0 = np.logspace(np.log10(x_min), np.log10(x_max), n0 + 1)
    else:
        edges0 = np.linspace(x_min, x_max, n0 + 1)

    stats0 = radial_bin_stats(x, y, edges0)

    fine_edges = stats0["edges"]
    fine_mean = stats0["mean"]
    fine_var = stats0["var"]
    fine_counts = stats0["counts"]

    # For stable merged-bin stats, accumulate raw sums
    nbins0 = len(fine_edges) - 1
    fine_sum = np.where(np.isfinite(fine_mean), fine_mean * fine_counts, 0.0)
    fine_sum2 = np.where(
        np.isfinite(fine_var),
        (fine_var + np.nan_to_num(fine_mean)**2) * fine_counts,
        0.0,
    )

    merged_edges = [fine_edges[0]]
    merged_mean = []
    merged_var = []
    merged_std = []
    merged_sem = []
    merged_counts = []

    i = 0
    while i < nbins0:
        j = i + 1

        c = fine_counts[i]
        s1 = fine_sum[i]
        s2 = fine_sum2[i]

        while True:
            if c > 0:
                mu = s1 / c
                var = max(s2 / c - mu**2, 0.0)
                std = np.sqrt(var)
                sem = std / np.sqrt(c) if c > 1 else np.inf
            else:
                mu = np.nan
                var = np.nan
                std = np.nan
                sem = np.inf

            width = fine_edges[j] - fine_edges[i]

            good_count = (c >= min_count)

            if np.isfinite(mu) and mu != 0.0:
                good_sem = (sem <= sem_frac_max * abs(mu))
                good_var = (var <= abs(mu)) if var_less_than_mean else True
            else:
                good_sem = False
                good_var = False if var_less_than_mean else True

            good_enough = good_count and good_sem and good_var

            hit_max_width = (
                (max_bin_width is not None) and (width >= max_bin_width)
            )

            at_end = (j >= nbins0)

            if good_enough or hit_max_width or at_end:
                merged_edges.append(fine_edges[j])
                merged_mean.append(mu)
                merged_var.append(var)
                merged_std.append(std)
                merged_sem.append(sem)
                merged_counts.append(c)
                i = j
                break

            # otherwise merge next fine bin
            c += fine_counts[j]
            s1 += fine_sum[j]
            s2 += fine_sum2[j]
            j += 1

    merged_edges = np.asarray(merged_edges)
    merged_centers = 0.5 * (merged_edges[:-1] + merged_edges[1:])

    return {
        "centers": merged_centers,
        "mean": np.asarray(merged_mean),
        "var": np.asarray(merged_var),
        "std": np.asarray(merged_std),
        "sem": np.asarray(merged_sem),
        "counts": np.asarray(merged_counts),
        "edges": merged_edges,
    }

def get_power_spectrum_adaptive(
    z0,
    T_band_map,
    nbins_target=25,
    oversample=2,
    min_count=30,
    sem_frac_max=0.3,
    max_bin_width_factor=3.0,
):
    """
    Drop-in replacement for get_power_spectrum, using adaptive_radial_bin
    for the k-binning step instead of fixed log bins.
 
    max_bin_width_factor: cap merged bin width at this many multiples of
    the fundamental k-spacing, so low-k bins can't grow unboundedly wide.
    """
    from astropy.cosmology import Planck18  # keep consistent with your existing code
 
    d = Planck18.comoving_transverse_distance(z0).value
    theta = np.deg2rad(3.6)
    L = d * theta
    N = T_band_map.shape[0]
    dx = L / N
 
    T_fluc_grid = (T_band_map - np.mean(T_band_map)) / np.std(T_band_map)
    # T_fluc_grid = T_band_map
    T_fluc_grid -= T_fluc_grid.mean()
    T_fluc_grid_k = dx**2 * np.fft.fft2(T_fluc_grid)
    P = np.abs(T_fluc_grid_k) ** 2 / L**2
 
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
 
    k_fundamental = 2 * np.pi / L
    max_bin_width = max_bin_width_factor * k_fundamental
 
    # Exclude the DC mode (k=0) before binning
    positive = k > 0
 
    stats = adaptive_radial_bin(
        k[positive],
        P[positive],
        nbins_target=nbins_target,
        oversample=oversample,
        x_min=k_fundamental,
        x_max=k.max(),
        logbins=True,
        max_bin_width=max_bin_width,
        min_count=min_count,
        sem_frac_max=sem_frac_max,
        var_less_than_mean=False,
    )
 
    k_center = stats["centers"]
    Pk = stats["mean"]
    Pk_err = stats["sem"]  # standard error of the mean, already computed properly
 
    Delta2 = k_center**2 * Pk / (2 * np.pi)
    Delta2_err = k_center**2 * Pk_err / (2 * np.pi)
 
    return (
        T_fluc_grid,
        k_center,
        Delta2,
        Delta2_err,
        d,
        # stats["counts"],  # useful to inspect how much merging happened per bin
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
                        markersize=2, fmt='.-',capsize=0,)
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
                secax.set_xlabel("Angular scale [deg]", fontsize=12)
                ax.set_title(band_labels[col], fontsize=12)
            else:
                secax.set_xlabel("")
                # hide the tick labels but keep the ticks
                secax.tick_params(labeltop=False)
            if col == 0:
                ax.set_ylabel(r'$\Delta^2(k)$', fontsize=12)
                ax.text(0.05, 0.95, f'$z_0$={z0:.1f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='black', edgecolor='none', alpha=0.5))
            if row == 7:
                ax.set_xlabel(r"$k\ [{\rm cMpc}^{-1}]$", fontsize=12)
                # ax.set_xlabel('k [1/Mpc]', fontsize=9)

    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig('T_power_spectra_grid.png', dpi=200, bbox_inches='tight')


def plot_T_power_spectra_red(ss):
    fig, axes = plt.subplots(figsize=(10,5))
    band_labels = ['Ultrablue', 'Blue', 'Center', 'Red', 'Ultrared']
    band_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
    band_cs = ['blue', 'green', 'olive', 'orange', 'red']

    # Sort the maps by redshift
    records = []
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_fluc_r, k_r, Delta2_r, Delta2_r_err, d = get_power_spectrum_adaptive(z0, T_r)
        records.append((
            float(z0),
            [k_r, Delta2_r, Delta2_r_err, d],
        ))
    records.sort(key=lambda r: r[0], reverse=True)

    half_fov = 3.6 / 2
    for row, (z0, r_set) in enumerate(records):
        k = r_set[0]
        D2 = r_set[1]
        D2_err = r_set[2]
        d = r_set[3]
        def k_to_deg(k, d=d):
            with np.errstate(divide='ignore'):
                rad = 2*np.pi / (k * d)
            return np.rad2deg(rad)
        valid = np.isfinite(k) & np.isfinite(D2)
        orange_black_cmap = mcolors.LinearSegmentedColormap.from_list(
            "orange_black", ["orange", "black"]
        )

        # Normalize redshift values to [0, 1] for the colormap
        norm = mcolors.Normalize(vmin=6., vmax=13.)

        # axes.errorbar(k[valid], D2[valid], yerr=D2_err[valid], color=orange_black_cmap(norm(z0)), 
        #             markersize=2, fmt='.-',capsize=0,)
        axes.plot(k[valid], D2[valid], color=orange_black_cmap(norm(z0)),)
        axes.fill_between(k[valid], D2[valid] + D2_err[valid], D2[valid] - D2_err[valid], color=orange_black_cmap(norm(z0)), alpha=0.2)
        axes.xaxis.set_major_locator(MaxNLocator(5))
        axes.set_xscale('log')
        axes.set_yscale('log')
        def k_to_deg(k):
            return np.rad2deg(2*np.pi/(k*d))

        def deg_to_k(theta):
            return 2*np.pi/(np.deg2rad(theta)*d)

        secax = axes.secondary_xaxis(
            'top',
            functions=(k_to_deg, deg_to_k)
        )

    secax.set_xlabel("Angular scale [deg]", fontsize=12)
    axes.set_title('Red Band Power Spectrum Evolution', fontsize=15)
    axes.set_ylabel(r'$\Delta^2(k)$', fontsize=12)
    axes.set_xlabel(r"$k\ [{\rm cMpc}^{-1}]$", fontsize=12)
    sm = ScalarMappable(norm=norm, cmap=orange_black_cmap)
    sm.set_array([])  # dummy array, needed for older matplotlib versions
    cbar = fig.colorbar(sm, ax=axes,  pad=0., aspect=50)
    cbar.set_label("Source Redshift")
    # ax.set_xlabel('k [1/Mpc]', fontsize=9)

    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig('T_power_spectra_red.png', dpi=200, bbox_inches='tight')

    newfig, newaxes = plt.subplots()
    z0s = []
    kmaxes = []
    for (z0, r_set) in records:
        z0s.append(z0)
        kmaxes.append(r_set[0][np.argmax(r_set[1])])
    newaxes.scatter(z0s, kmaxes, marker='^', color='orange')
    newaxes.set_yscale('log')
    newaxes.set_xlabel(r'$z_0$')
    newaxes.set_ylabel(r"$k\ [{\rm cMpc}^{-1}]$")
    newaxes.set_title(r"$k$ at Maximum Power")
    plt.savefig('T_power_spectra_k_minmax.png', dpi=200, bbox_inches='tight')



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
    plot_T_power_spectra_red(ss)


if __name__ == "__main__":
    main()
