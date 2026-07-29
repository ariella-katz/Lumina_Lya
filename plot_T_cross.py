import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
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

def get_cross_corr(map1, map2):
    npix = map1.shape[0]
    npad = 2 * npix

    # get data overdensities
    map1_od = (map1 / map1.mean()) - 1.
    map2_od = (map2/ map2.mean()) - 1.

    # zero-pad
    map1_pad = np.zeros((npad, npad))
    map2_pad = np.zeros((npad, npad))
    map1_pad[:npix, :npix] = map1_od
    map2_pad[:npix, :npix] = map2_od

    # get Fourier transforms
    F1 = np.fft.fft2(map1_pad)
    F2 = np.fft.fft2(map2_pad)

    # F(cross_corr(a, b)) = F(a) * conj(F(b))
    #  where F is Fourier transform 
    cross_corr = np.fft.ifft2(F1 * np.conj(F2)).real
    # Center at 0 lag
    cross_corr = np.fft.fftshift(cross_corr) / map1_od.size

    # re-crop
    start = npad // 2 - npix // 2
    end = start + npix
    cross_corr = cross_corr[start:end, start:end]
    return cross_corr

def get_cross_pow(map1, map2, z0,
                    nbins_target=25,
                    oversample=2,
                    min_count=30,
                    sem_frac_max=0.3,
                    max_bin_width_factor=3.0,):

    d = Planck18.comoving_transverse_distance(z0).value
    theta = np.deg2rad(3.6)
    L = d * theta
    npix = map1.shape[0]
    dx = L / npix

    # get data overdensities
    map1_od = (map1 / map1.mean()) - 1.
    map2_od = (map2/ map2.mean()) - 1.

    # get Fourier transforms
    F1 = np.fft.fft2(map1_od)
    F2 = np.fft.fft2(map2_od)

    # get powers
    pow1 = np.abs(F1)**2 / npix**2
    pow2 = np.abs(F2)**2 / npix**2
    cross_pow = F1 * np.conj(F2) / npix**2

    kx = 2 * np.pi * np.fft.fftfreq(npix, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(npix, d=dx)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
 
    k_fundamental = 2 * np.pi / L
    max_bin_width = max_bin_width_factor * k_fundamental
 
    # Exclude the DC mode (k=0) before binning
    positive = k > 0
 
    stats = adaptive_radial_bin(
        k[positive],
        cross_pow[positive],
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

    stats_norm = adaptive_radial_bin(
        k[positive],
        cross_pow[positive] / np.sqrt(pow1[positive]*pow2[positive]),
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

    k_center_norm = stats_norm["centers"]
    Pk_norm = stats_norm["mean"]
    Pk_err_norm = stats_norm["sem"]  # standard error of the mean, already computed properly
 
    Delta2 = k_center**2 * Pk / (2 * np.pi)
    Delta2_err = k_center**2 * Pk_err / (2 * np.pi)
 
    return (
        k_center,
        Delta2,
        Delta2_err,
        k_center_norm,
        Pk_norm,
        Pk_err_norm,
        d,
        # stats["counts"],  # useful to inspect how much merging happened per bin
    )



def plot_T_cross(ss, lightcone_file):
    # Get T maps and other maps for all z0s and bands and sort by z0
    T_records = []
    frac_records = []
    density_records = []
    for z0i in range(len(ss)):
        z0, T_ub, T_b, T_c, T_r, T_ur = get_full_T_grid(ss[z0i])
        T_records.append((float(z0), T_ub, T_b, T_c, T_r, T_ur))
        with h5py.File(lightcone_file, 'r') as cone:
            zs = cone['Redshifts'][:]
            z0_i = np.argmax(zs <= z0)
            z0_cone = zs[z0_i]
            densities = cone['Density'][..., z0_i].astype(np.float64)
            density_records.append((z0_cone, densities))
            x_HIs = 1. - cone['HII_Fraction'][..., z0_i].astype(np.float64)
            frac_records.append((z0_cone, densities))
 
    T_records.sort(key=lambda r: r[0])
    frac_records.sort(key=lambda r: r[0])
    density_records.sort(key=lambda r: r[0])

    fig, axes = plt.subplots(8, 6, figsize=(5 * 2.4, 8 * 2.4), sharex=True, sharey=True)
    half_fov = 3.6 / 4
    for z0i in range(len(T_records)):
        z0, T_ub, T_b, T_c, T_r, T_ur = T_records[z0i]
        z0_cone, x_HIs = frac_records[z0i]
        _, densities = density_records[z0i]

        map1 = T_r
        map2s = [T_ub, T_b, T_c, T_ur, x_HIs, densities]
        labels = [r'$\log\mathcal{T}_\text{int, R}\times\log\mathcal{T}_\text{int, UB}$',
                  r'$\log\mathcal{T}_\text{int, R}\times\log\mathcal{T}_\text{int, B}$',
                  r'$\log\mathcal{T}_\text{int, R}\times\log\mathcal{T}_\text{int, C}$',
                  r'$\log\mathcal{T}_\text{int, R}\times\log\mathcal{T}_\text{int, UR}$',
                  r'$\log\mathcal{T}_\text{int, R}\times$ HI Fraction',
                  r'$\log\mathcal{T}_\text{int, R}\times$ Density']
        cross_corrs = []
        # for mapi in range(len(map2s)):
        #     map2 = map2s[mapi]
        #     cross_corr = get_cross_corr(map1, map2)
        #     cross_corrs.append(cross_corr)
        # vmax = np.abs(np.asarray(cross_corrs)).max()
        for mapi in range(len(map2s)):
            cross_corr = cross_corrs[mapi]
            # cross_pow_stats = get_cross_pow(map1, map2)
            ax = axes[z0i][mapi]
            vmax = np.abs(cross_corr).max()
            im = ax.imshow(cross_corr, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      extent=[-half_fov, half_fov, -half_fov, half_fov])
            if mapi == 0:
                ax.set_ylabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
            if z0i == 0:
                ax.set_title(labels[mapi])
            if z0i == 8:
                ax.set_xlabel(r'$\Delta\Theta$ [degrees]', fontsize=9)
            fig.colorbar(im, axes=ax)

    # pos0 = axes[0, 0].get_position()
    # pos5 = axes[0, 5].get_position()
    # cax = fig.add_axes([pos0.x0, 0.93, pos5.x1 - pos0.x0, 0.005])
    # cb = fig.colorbar(im)

    plt.savefig('cross_corrs.png')






def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "--map_directory",
        type=str,
        default=None,
        help="Directory of tau maps"
    )
    parser.add_argument(
        "--lightcone_file",
        type=str,
        default=None,
        help="Lightcone data"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dir_arg = args.map_directory
    data_dir = "tau_maps"
    if dir_arg is not None:
        data_dir = dir_arg
    lightcone_arg = args.lightcone_file
    lightcone_file = 'All_40.hdf5'
    if lightcone_arg is not None:
        lightcone_file = lightcone_arg

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
    plot_T_cross(ss, lightcone_file)


if __name__ == "__main__":
    main()