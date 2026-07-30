import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patheffects as path_effects
import cmasher as cmr
import cmocean
import h5py, os
from scipy.ndimage import zoom, gaussian_filter1d
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from scipy.signal import savgol_filter
from plot_tau_CMB import shuffle
# import yt

# Set rcParams for ticks: inward direction, all sides
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# Constants
c = 2.99792458e10          # Speed of light [cm/s]
km = 1e5                   # Units: 1 km  = 1e5  cm
# au = 1.495978707e13        # Units: 1 au  = 1.5e13 cm
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 1e6 pc
# angstrom = 1e-8            # Units: 1 angstrom = 1e-8 cm
# Msun = 1.988435e33         # Solar mass [g]
# eV = 1.60217725e-12        # Electron volt: 1 eV = 1.6e-12 erg
# h = 6.626069573e-27        # Planck's constant [erg s]
# kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
mH = 1.6735327e-24         # Mass of hydrogen atom (g)
me = 9.109382917e-28       # Electron mass [g]
ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]
X  = 0.76                  # Primordial hydrogen mass fraction
r_0 = ee * ee / (me * c**2)  # Electron radius = e^2 / m c^2
sigma_T = 8. * np.pi * r_0**2 / 3.  # Thomson cross-section [cm^2]
G = 6.6725985e-8           # Gravitational constant [cm^3/g/s^2]
HE_ABUND = (1./X - 1.) / 4. # Helium abundance
ne_nH = 1. + 2. * HE_ABUND  # n_e / n_H ~ 1.158 (after HeIII reionization)
# print(f'n_e / n_H ~ {ne_nH}')

lg = [0.8, 0.8, 0.8]
percentiles = np.array([0.01, 0.1, 1., 10., 50., 90., 99., 99.9, 99.99])

def set_redshift_axis(ax):
    ax.set_xlabel(r'${\rm Redshift}$', fontsize=15)
    ax.set_xscale('log')
    ax.minorticks_on()
    xmin, xmax = 3, 15
    ax.set_xlim([xmin, xmax])
    ticks = [3,4,5,6,7,8,9,10,12,14]
    ax.set_xticks(ticks); ax.set_xticklabels([r'$%g$' % tick for tick in ticks], fontsize=12)
    ticks = np.linspace(xmin, xmax, xmax-xmin+1); ax.set_xticks(ticks, minor=True)
    ax.set_xticklabels(['']*len(ticks), minor=True)

def set_xaxis(ax, xlabel, xmin=None, xmax=None, n=None, xp=0., show_labels=True):
    if show_labels:
        ax.set_xlabel(xlabel, fontsize=15)
    if xmin is not None and xmax is not None and n is not None:
        ax.set_xlim(xmin, xmax + xp)
        ticks = np.linspace(xmin, xmax, n)
        ax.set_xticks(ticks)
        tlabs = [r'$%g$' % tick for tick in ticks] if show_labels else [''] * len(ticks)
        ax.set_xticklabels(tlabs, fontsize=12)

def set_yaxis(ax, ylabel, ymin=None, ymax=None, n=None, ym=0., yp=0., show_labels=True):
    if show_labels:
        ax.set_ylabel(ylabel, fontsize=15)
    if ymin is not None and ymax is not None and n is not None:
        ax.set_ylim(ymin - ym, ymax + yp)
        ticks = np.linspace(ymin, ymax, n)
        ticks[np.abs(ticks)<1e-12] = 0.
        ax.set_yticks(ticks)
        tlabs = [r'$%g$' % tick for tick in ticks] if show_labels else [''] * len(ticks)
        ax.set_yticklabels(tlabs, fontsize=12)

def set_log_xaxis(ax, xlabel, xmin=None, xmax=None, show_labels=True):
    if show_labels:
        ax.set_xlabel(xlabel, fontsize=15)
    ax.set_xscale('log')
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
        Lxmin,Lxmax = int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax)))
        ticks = np.linspace(Lxmin, Lxmax, Lxmax-Lxmin+1)
        ax.set_xticks(10**ticks)
        tlabs = [r'$10^{%g}$' % tick for tick in ticks] if show_labels else [''] * len(ticks)
        ax.set_xticklabels(tlabs, fontsize=12)

def set_log_yaxis(ax, ylabel, ymin=None, ymax=None, show_labels=True):
    if show_labels:
        ax.set_ylabel(ylabel, fontsize=15)
    ax.set_yscale('log')
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        Lymin,Lymax = int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax)))
        ticks = np.linspace(Lymin, Lymax, Lymax-Lymin+1)
        ax.set_yticks(10**ticks)
        tlabs = [r'$10^{%g}$' % tick for tick in ticks] if show_labels else [''] * len(ticks)
        ax.set_yticklabels(tlabs, fontsize=12)

from scipy.signal import windows
from scipy.stats import binned_statistic
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

# ============================================================
# Utilities
# ============================================================

def make_apodization_window(shape, frac=0.05, kind="tukey"):
    """
    Construct a separable 2D apodization window.

    Parameters
    ----------
    shape : tuple
        (Ny, Nx)
    frac : float
        Fractional taper width. For Tukey this is alpha.
    kind : str
        'tukey' or 'hann'

    Returns
    -------
    w2d : ndarray
        2D apodization window normalized to max=1
    """
    ny, nx = shape

    if kind == "tukey":
        wy = windows.tukey(ny, alpha=frac)
        wx = windows.tukey(nx, alpha=frac)
    elif kind == "hann":
        wy = windows.hann(ny)
        wx = windows.hann(nx)
    else:
        raise ValueError("kind must be 'tukey' or 'hann'")

    w2d = np.outer(wy, wx)
    return w2d / np.max(w2d)


def sanitize_map(m, mask=None, fill_value=0.0, subtract_mean=True, weight=None):
    """
    Apply mask/weights and remove weighted mean.

    Parameters
    ----------
    m : ndarray
        Input 2D map
    mask : ndarray or None
        Boolean or float mask, same shape as m
    fill_value : float
        Value used where mask == 0 or map is non-finite
    subtract_mean : bool
        If True, subtract weighted mean over valid pixels
    weight : ndarray or None
        Additional weight/apodization window

    Returns
    -------
    out : ndarray
        Cleaned map
    eff_mask : ndarray
        Effective float mask/weight used
    """
    m = np.asarray(m, dtype=np.float64)
    finite = np.isfinite(m)

    if mask is None:
        mask = np.ones_like(m, dtype=np.float64)
    else:
        mask = np.asarray(mask, dtype=np.float64)

    eff_mask = mask * finite.astype(np.float64)

    if weight is not None:
        eff_mask = eff_mask * np.asarray(weight, dtype=np.float64)

    out = np.where(finite, m, fill_value).copy()

    if subtract_mean:
        denom = np.sum(eff_mask)
        mean = np.sum(out * eff_mask) / denom if denom > 0 else 0.0
        out = out - mean

    out = np.where(eff_mask > 0, out, fill_value)
    return out, eff_mask


def fftfreq_2d(shape, dtheta_rad):
    """
    2D angular Fourier frequencies for flat-sky maps.

    Returns
    -------
    ellx, elly, ell2d : ndarrays
        ell = 2*pi*freq where freq is in cycles/radian
    """
    ny, nx = shape
    fx = fftfreq(nx, d=dtheta_rad)   # cycles / rad
    fy = fftfreq(ny, d=dtheta_rad)

    ellx = 2.0 * np.pi * fx[None, :]
    elly = 2.0 * np.pi * fy[:, None]
    ell2d = np.sqrt(ellx**2 + elly**2)
    return ellx, elly, ell2d


def radial_bin(x2d, y2d, bins, statistic="mean"):
    """
    Radially bin y2d as a function of x2d.
    """
    x = x2d.ravel()
    y = y2d.ravel()

    good = np.isfinite(x) & np.isfinite(y)
    vals, edges, _ = binned_statistic(x[good], y[good], statistic=statistic, bins=bins)
    counts, _, _ = binned_statistic(x[good], y[good], statistic="count", bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, vals, counts, edges


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


def adaptive_edges_from_counts(
    x2d,
    nbins_target=80,
    oversample=1,
    x_min=None,
    x_max=None,
    logbins=True,
    min_count=45,
    max_bin_width=None,
    exclude_zero=False,
):
    """
    Build common adaptive radial bin edges using only sample counts.

    Parameters
    ----------
    x2d : 2D array
        Radial coordinate grid (e.g. ell2d or r2d)
    nbins_target : int
        Approximate desired final number of bins
    oversample : int
        Initial fine binning factor
    x_min, x_max : float or None
        Bin range
    logbins : bool
        Initial bins logarithmic or linear
    min_count : int
        Minimum number of samples per merged bin
    max_bin_width : float or None
        Maximum allowed merged-bin width in x
    exclude_zero : bool
        Exclude x==0 when determining x_min

    Returns
    -------
    edges : 1D array
        Common merged bin edges
    counts : 1D array
        Counts in the merged bins
    """
    x = np.ravel(x2d)
    x = x[np.isfinite(x)]

    if exclude_zero:
        x_use = x[x > 0]
    else:
        x_use = x

    if x_min is None:
        x_min = np.min(x_use)
    if x_max is None:
        x_max = np.max(x_use)

    n0 = oversample * nbins_target
    if logbins:
        edges0 = np.logspace(np.log10(x_min), np.log10(x_max), n0 + 1)
    else:
        edges0 = np.linspace(x_min, x_max, n0 + 1)

    ibin = np.digitize(x, edges0) - 1
    valid = (ibin >= 0) & (ibin < n0)
    counts0 = np.bincount(ibin[valid], minlength=n0)

    merged_edges = [edges0[0]]
    merged_counts = []

    i = 0
    while i < n0:
        j = i + 1
        c = counts0[i]

        while True:
            width = edges0[j] - edges0[i]
            hit_count = c >= min_count
            hit_width = (max_bin_width is not None) and (width >= max_bin_width)
            at_end = (j >= n0)

            if hit_count or hit_width or at_end:
                merged_edges.append(edges0[j])
                merged_counts.append(c)
                i = j
                break

            c += counts0[j]
            j += 1

    return np.asarray(merged_edges), np.asarray(merged_counts)


def radial_bin_with_edges(x2d, y2d, edges):
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

    mean = np.full(nbins, np.nan)
    var = np.full(nbins, np.nan)

    m = counts > 0
    mean[m] = sumy[m] / counts[m]
    var[m] = np.maximum(sumy2[m] / counts[m] - mean[m]**2, 0.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    sem = np.full(nbins, np.nan)
    m2 = counts > 1
    sem[m2] = np.sqrt(var[m2]) / np.sqrt(counts[m2])

    return {
        "centers": centers,
        "mean": mean,
        "var": var,
        "sem": sem,
        "counts": counts,
        "edges": edges,
    }

# ============================================================
# Power spectra
# ============================================================

def power_spectrum_2d(
    map1,
    map2=None,
    theta_deg=3.6,
    apod_frac=0.05,
    apod_kind="tukey",
    mask1=None,
    mask2=None,
    subtract_mean=True,
    return_2d=False,
):
    """
    Flat-sky auto/cross power spectrum for a non-periodic map.

    Uses apodization to reduce ringing and divides by <W1 W2> to
    approximately correct the window suppression.

    Parameters
    ----------
    map1, map2 : ndarray
        2D maps. If map2 is None, compute auto-spectrum of map1.
    theta_deg : float
        Total map size across one dimension in degrees
    apod_frac : float
        Apodization strength
    apod_kind : str
        'tukey' or 'hann'
    mask1, mask2 : ndarray or None
        Optional masks
    subtract_mean : bool
        Remove weighted mean before FFT
    return_2d : bool
        If True, also return 2D ell-grid and 2D spectrum

    Returns
    -------
    ell2d : ndarray
        2D ell grid
    p2d : ndarray
        2D power spectrum
    meta : dict
        Metadata including pixel size and normalization
    """
    m1 = np.asarray(map1, dtype=np.float64)
    ny, nx = m1.shape
    assert ny == nx, "This implementation assumes square pixels/map."

    if map2 is None:
        map2 = map1
        mask2 = mask1

    m2 = np.asarray(map2, dtype=np.float64)
    assert m2.shape == m1.shape

    theta_rad = np.deg2rad(theta_deg)
    dtheta = theta_rad / nx
    pix_area = dtheta**2

    apod = make_apodization_window(m1.shape, frac=apod_frac, kind=apod_kind)

    f1, w1 = sanitize_map(m1, mask=mask1, subtract_mean=subtract_mean, weight=apod)
    f2, w2 = sanitize_map(m2, mask=mask2, subtract_mean=subtract_mean, weight=apod)

    F1 = fft2(f1)
    F2 = fft2(f2)

    # Approximate flat-sky power normalization:
    # integral d^2theta f(theta) e^{-i l.theta} ~ pix_area * FFT(f)
    # then P(l) ~ |F_cont|^2 / Omega, with F_cont = pix_area * FFT_discrete
    Omega = ny * nx * pix_area
    norm = pix_area**2 / Omega

    p2d = norm * (F1 * np.conjugate(F2)).real

    # Correct approximate window suppression by average overlap
    wnorm = np.mean(w1 * w2)
    if wnorm > 0:
        p2d /= wnorm

    ellx, elly, ell2d = fftfreq_2d(m1.shape, dtheta)

    meta = {
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "dtheta_rad": dtheta,
        "pix_area_sr": pix_area,
        "Omega_sr": Omega,
        "window_norm_mean": wnorm,
    }

    if return_2d:
        return ell2d, p2d, meta
    return ell2d, p2d, meta


def radial_power_spectrum(
    map1,
    map2=None,
    theta_deg=3.6,
    nbins=64,
    ell_min=None,
    ell_max=None,
    logbins=True,
    apod_frac=0.05,
    apod_kind="tukey",
    mask1=None,
    mask2=None,
    subtract_mean=True,
):
    """
    Radially binned flat-sky auto/cross power spectrum.

    Returns
    -------
    ell_centers : ndarray
    Cl : ndarray
        Radially averaged power
    counts : ndarray
        Number of Fourier pixels in each bin
    ell2d : ndarray
    p2d : ndarray
    meta : dict
    """
    ell2d, p2d, meta = power_spectrum_2d(
        map1,
        map2=map2,
        theta_deg=theta_deg,
        apod_frac=apod_frac,
        apod_kind=apod_kind,
        mask1=mask1,
        mask2=mask2,
        subtract_mean=subtract_mean,
        return_2d=True,
    )

    # Exclude DC mode from log bins
    positive = ell2d > 0
    ell_nonzero = ell2d[positive]

    if ell_min is None:
        ell_min = ell_nonzero.min()
    if ell_max is None:
        ell_max = ell_nonzero.max()

    if logbins:
        bins = np.logspace(np.log10(ell_min), np.log10(ell_max), nbins + 1)
    else:
        bins = np.linspace(ell_min, ell_max, nbins + 1)

    ellc, Cl, counts, edges = radial_bin(ell2d, p2d, bins=bins, statistic="mean")

    return ellc, Cl, counts, ell2d, p2d, meta


def radial_power_spectrum_adaptive(
    map1,
    map2=None,
    theta_deg=3.6,
    nbins=64,
    ell_min=None,
    ell_max=None,
    logbins=True,
    apod_frac=0.05,
    apod_kind="tukey",
    mask1=None,
    mask2=None,
    subtract_mean=True,
    oversample=1,
    min_count=45,
    sem_frac_max=0.5,
    max_bin_width=None,
):
    ell2d, p2d, meta = power_spectrum_2d(
        map1,
        map2=map2,
        theta_deg=theta_deg,
        apod_frac=apod_frac,
        apod_kind=apod_kind,
        mask1=mask1,
        mask2=mask2,
        subtract_mean=subtract_mean,
        return_2d=True,
    )

    positive = ell2d > 0
    ell_nonzero = ell2d[positive]

    if ell_min is None:
        ell_min = ell_nonzero.min()
    if ell_max is None:
        ell_max = ell_nonzero.max()

    stats = adaptive_radial_bin(
        ell2d,
        p2d,
        nbins_target=nbins,
        oversample=oversample,
        x_min=ell_min,
        x_max=ell_max,
        logbins=logbins,
        max_bin_width=max_bin_width,
        min_count=min_count,
        sem_frac_max=sem_frac_max,
        var_less_than_mean=False,   # better for cross-spectra
    )

    return (
        stats["centers"],
        stats["mean"],
        stats["counts"],
        stats["var"],
        stats["sem"],
        stats["edges"],
        ell2d,
        p2d,
        meta,
    )


def cross_correlation_coefficient(Cl12, Cl11, Cl22):
    """
    Fourier-space cross-correlation coefficient:
        r_ell = C12 / sqrt(C11 C22)
    """
    denom = np.sqrt(np.maximum(Cl11 * Cl22, 0.0))
    out = np.full_like(Cl12, np.nan, dtype=np.float64)
    good = denom > 0
    out[good] = Cl12[good] / denom[good]
    return out


# ============================================================
# Correlation functions
# ============================================================

def correlation_function_2d(
    map1,
    map2=None,
    theta_deg=3.6,
    apod_frac=0.05,
    apod_kind="tukey",
    mask1=None,
    mask2=None,
    subtract_mean=True,
):
    """
    2D correlation function xi(theta_x, theta_y) using FFTs.

    For non-periodic maps, computes
        xi = IFFT[ FFT(w1 f1) FFT(w2 f2)^* ] / IFFT[ FFT(w1) FFT(w2)^* ]
    which corrects for edge overlap as a function of lag.

    Returns
    -------
    dx_rad, dy_rad : ndarray
        Lag coordinates
    xi2d : ndarray
        2D correlation function
    """
    m1 = np.asarray(map1, dtype=np.float64)
    ny, nx = m1.shape
    assert ny == nx

    if map2 is None:
        map2 = map1
        mask2 = mask1

    m2 = np.asarray(map2, dtype=np.float64)
    assert m2.shape == m1.shape

    theta_rad = np.deg2rad(theta_deg)
    dtheta = theta_rad / nx

    apod = make_apodization_window(m1.shape, frac=apod_frac, kind=apod_kind)

    f1, w1 = sanitize_map(m1, mask=mask1, subtract_mean=subtract_mean, weight=apod)
    f2, w2 = sanitize_map(m2, mask=mask2, subtract_mean=subtract_mean, weight=apod)

    num = ifft2(fft2(f1) * np.conjugate(fft2(f2))).real
    den = ifft2(fft2(w1) * np.conjugate(fft2(w2))).real

    xi2d = np.full_like(num, np.nan, dtype=np.float64)
    good = den > 0
    xi2d[good] = num[good] / den[good]

    # Shift zero-lag to center
    xi2d = fftshift(xi2d)

    dx = (np.arange(nx) - nx // 2) * dtheta
    dy = (np.arange(ny) - ny // 2) * dtheta
    return dx, dy, xi2d


def radial_correlation_function(
    map1,
    map2=None,
    theta_deg=3.6,
    nbins=64,
    theta_min=None,
    theta_max=None,
    logbins=True,
    apod_frac=0.05,
    apod_kind="tukey",
    mask1=None,
    mask2=None,
    subtract_mean=True,
):
    """
    Radially averaged correlation function xi(theta).

    Returns
    -------
    theta_centers_rad : ndarray
    xi_theta : ndarray
    counts : ndarray
    dx_rad, dy_rad, xi2d : full 2D outputs
    """
    dx, dy, xi2d = correlation_function_2d(
        map1,
        map2=map2,
        theta_deg=theta_deg,
        apod_frac=apod_frac,
        apod_kind=apod_kind,
        mask1=mask1,
        mask2=mask2,
        subtract_mean=subtract_mean,
    )

    xx, yy = np.meshgrid(dx, dy)
    rr = np.sqrt(xx**2 + yy**2)

    positive = rr > 0
    r_nonzero = rr[positive]

    if theta_min is None:
        theta_min = r_nonzero.min()
    if theta_max is None:
        theta_max = rr.max()

    if logbins:
        bins = np.logspace(np.log10(theta_min), np.log10(theta_max), nbins + 1)
    else:
        bins = np.linspace(theta_min, theta_max, nbins + 1)

    rc, xi, counts, edges = radial_bin(rr, xi2d, bins=bins, statistic="mean")
    return rc, xi, counts, dx, dy, xi2d

def save_spectra(sim_dir, N=320, nbins=64, oversample=1, save_2d=False):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    # print(f'{rlc_dir}/tau_CMB_hist.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_hist.hdf5', 'r') as f, h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'w') as ef:
        header_attrs = f['Header'].attrs
        theta_deg = header_attrs['OpeningAngleDegrees']
        tau_CMB = f['tau_CMB'][:]             # Total (auto-spectrum, auto-correlation)
        tau_CMB_HII = f['tau_CMB_HII'][:]     # Hydrogen contribution
        tau_CMB_HeII = f['tau_CMB_HeII'][:]   # Hydrogen contribution
        tau_CMB_HeIII = tau_CMB - tau_CMB_HII - tau_CMB_HeII  # HeIII contribution
        if True:  # Transform to zero-mean and unit variance
            tau_CMB = (tau_CMB - tau_CMB.mean()) / tau_CMB.std()
            tau_CMB_HII = (tau_CMB_HII - tau_CMB_HII.mean()) / tau_CMB_HII.std()
            tau_CMB_HeII = (tau_CMB_HeII - tau_CMB_HeII.mean()) / tau_CMB_HeII.std()
            tau_CMB_HeIII = (tau_CMB_HeIII - tau_CMB_HeIII.mean()) / tau_CMB_HeIII.std()
        ef.create_group('Header')
        eheader_attrs = ef['Header'].attrs
        for key in header_attrs:
            eheader_attrs[key] = header_attrs[key]

        # Calculate all 2D spectra
        ell2d, p2d, meta = power_spectrum_2d(tau_CMB, theta_deg=theta_deg)
        _, p2d_HII, _ = power_spectrum_2d(tau_CMB_HII, theta_deg=theta_deg)
        _, p2d_HeII, _ = power_spectrum_2d(tau_CMB_HeII, theta_deg=theta_deg)
        _, p2d_HeIII, _ = power_spectrum_2d(tau_CMB_HeIII, theta_deg=theta_deg)
        _, p2d_HII_HeII, _ = power_spectrum_2d(tau_CMB_HII, map2=tau_CMB_HeII, theta_deg=theta_deg)
        _, p2d_HII_HeIII, _ = power_spectrum_2d(tau_CMB_HII, map2=tau_CMB_HeIII, theta_deg=theta_deg)
        _, p2d_HeII_HeIII, _ = power_spectrum_2d(tau_CMB_HeII, map2=tau_CMB_HeIII, theta_deg=theta_deg)
        # Check all quantities (ell2d, p2d, meta) Previously passed!
        # assert np.allclose(ell2d, ell2d_HII), "ell2d != ell2d_HII"
        # assert np.allclose(p2d, p2d_HII), "p2d != p2d_HII"
        # for key in meta:
        #     assert meta[key] == meta_HII[key], f"meta[{key}] != meta_HII[{key}]"

        # Build one common ell binning from the shared geometry and apply to all spectra
        positive = ell2d > 0
        edges, counts = adaptive_edges_from_counts(ell2d[positive], nbins_target=nbins, oversample=oversample,
            x_min=ell2d[positive].min(), x_max=ell2d[positive].max(), logbins=True, min_count=50, max_bin_width=None, exclude_zero=True)
        spec = radial_bin_with_edges(ell2d, p2d,  edges)
        spec_HII = radial_bin_with_edges(ell2d, p2d_HII,  edges)
        spec_HeII = radial_bin_with_edges(ell2d, p2d_HeII, edges)
        spec_HeIII = radial_bin_with_edges(ell2d, p2d_HeIII, edges)
        spec_HII_HeII = radial_bin_with_edges(ell2d, p2d_HII_HeII, edges)
        spec_HII_HeIII = radial_bin_with_edges(ell2d, p2d_HII_HeIII, edges)
        spec_HeII_HeIII = radial_bin_with_edges(ell2d, p2d_HeII_HeIII, edges)
        ell, var, sem, counts0, edges0 = spec["centers"], spec["var"], spec["sem"], spec["counts"], spec["edges"]
        C_ell, C_ell_HII, C_ell_HeII, C_ell_HeIII = spec["mean"], spec_HII["mean"], spec_HeII["mean"], spec_HeIII["mean"]
        C_ell_HII_HeII, C_ell_HII_HeIII, C_ell_HeII_HeIII = spec_HII_HeII["mean"], spec_HII_HeIII["mean"], spec_HeII_HeIII["mean"]
        # Check all quantities (ell, var, sem, count, edges) Previously passed!
        # assert np.allclose(ell, ell_HII), "ell != ell_HII"
        # assert np.allclose(counts, counts0), "counts != counts0"
        # assert np.allclose(counts, counts_HII), "counts0 != counts_HII"
        # assert np.allclose(edges, edges0), "edges != edges0"
        # assert np.allclose(edges0, edges_HII), "edges0 != edges_HII"
        # assert np.allclose(var, var_HII), "var != var_HII"
        # assert np.allclose(sem, sem_HII), "sem != sem_HII"

        # Auto and cross power spectra (non-adaptive)
        # ell, C_ell, counts, ell2d, p2d, meta = radial_power_spectrum(tau_CMB, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_H, C_ell_H, counts_H, ell2d_H, p2d_H, meta_H = radial_power_spectrum(tau_CMB_H, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_He, C_ell_He, counts_He, ell2d_He, p2d_He, meta_He = radial_power_spectrum(tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_HHe, C_ell_HHe, counts_HHe, ell2d_HHe, p2d_HHe, meta_HHe = radial_power_spectrum(tau_CMB_H, map2=tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # Auto and cross power spectra (adaptive)
        # ell, C_ell, counts, var, sem, edges, ell2d, p2d, meta = radial_power_spectrum_adaptive(tau_CMB, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_H, C_ell_H, counts_H, var_H, sem_H, edges_H, ell2d_H, p2d_H, meta_H = radial_power_spectrum_adaptive(tau_CMB_H, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_He, C_ell_He, counts_He, var_He, sem_He, edges_He, ell2d_He, p2d_He, meta_He = radial_power_spectrum_adaptive(tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # ell_HHe, C_ell_HHe, counts_HHe, var_HHe, sem_HHe, edges_HHe, ell2d_HHe, p2d_HHe, meta_HHe = radial_power_spectrum_adaptive(tau_CMB_H, map2=tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=True, apod_frac=apod_frac)
        # Check all quantities (ell, counts, ell2d, p2d, meta)
        # assert np.allclose(ell, ell_HHe), "ell != ell_HHe"
        # assert np.allclose(ell_H, ell_HHe), "ell_H != ell_HHe"
        # assert np.allclose(ell_He, ell_HHe), "ell_He != ell_HHe"
        # assert np.allclose(counts, counts_HHe), "counts != counts_HHe"
        # assert np.allclose(counts_H, counts_HHe), "counts_H != counts_HHe"
        # assert np.allclose(counts_He, counts_HHe), "counts_He != counts_HHe"
        # assert np.allclose(ell2d, ell2d_HHe), "ell2d != ell2d_HHe"
        # assert np.allclose(ell2d_H, ell2d_HHe), "ell2d_H != ell2d_HHe"
        # assert np.allclose(ell2d_He, ell2d_HHe), "ell2d_He != ell2d_HHe"
        # assert np.allclose(p2d, p2d_HHe), "p2d != p2d_HHe"
        # assert np.allclose(p2d_H, p2d_HHe), "p2d_H != p2d_HHe"
        # assert np.allclose(p2d_He, p2d_HHe), "p2d_He != p2d_HHe"
        # for key in meta:
        #     assert meta[key] == meta_HHe[key], f"meta[{key}] != meta_HHe[{key}]"
        # for key in meta_H:
        #     assert meta_H[key] == meta_HHe[key], f"meta_H[{key}] != meta_HHe[{key}]"
        # for key in meta_He:
        #     assert meta_He[key] == meta_HHe[key], f"meta_He[{key}] != meta_HHe[{key}]"
        r_ell_HII_HeII = cross_correlation_coefficient(C_ell_HII_HeII, C_ell_HII, C_ell_HeII)
        r_ell_HII_HeIII = cross_correlation_coefficient(C_ell_HII_HeIII, C_ell_HII, C_ell_HeIII)
        r_ell_HeII_HeIII = cross_correlation_coefficient(C_ell_HeII_HeIII, C_ell_HeII, C_ell_HeIII)
        g = ef.create_group('spec')
        g.create_dataset('ell', data=ell)
        g.create_dataset('C_ell', data=C_ell)
        g.create_dataset('C_ell_HII', data=C_ell_HII)
        g.create_dataset('C_ell_HeII', data=C_ell_HeII)
        g.create_dataset('C_ell_HeIII', data=C_ell_HeIII)
        g.create_dataset('C_ell_HII_HeII', data=C_ell_HII_HeII)
        g.create_dataset('C_ell_HII_HeIII', data=C_ell_HII_HeIII)
        g.create_dataset('C_ell_HeII_HeIII', data=C_ell_HeII_HeIII)
        g.create_dataset('r_ell_HII_HeII', data=r_ell_HII_HeII)
        g.create_dataset('r_ell_HII_HeIII', data=r_ell_HII_HeIII)
        g.create_dataset('r_ell_HeII_HeIII', data=r_ell_HeII_HeIII)
        g.create_dataset('var', data=var)
        g.create_dataset('sem', data=sem)
        g.create_dataset('counts', data=counts)
        g.create_dataset('edges', data=edges)
        if save_2d:
            g.create_dataset('ell2d', data=ell2d)
            g.create_dataset('p2d', data=p2d)
            g.create_dataset('p2d_HII', data=p2d_HII)
            g.create_dataset('p2d_HeII', data=p2d_HeII)
            g.create_dataset('p2d_HeIII', data=p2d_HeIII)
            g.create_dataset('p2d_HII_HeII', data=p2d_HII_HeII)
            g.create_dataset('p2d_HII_HeIII', data=p2d_HII_HeIII)
            g.create_dataset('p2d_HeII_HeIII', data=p2d_HeII_HeIII)
        for key in meta:
            g.attrs[key] = meta[key]

        # # Auto and cross correlation functions (linear binning)
        # theta, xi, counts, dx, dy, xi2d = radial_correlation_function(tau_CMB, theta_deg=theta_deg, nbins=nbins, logbins=False, apod_frac=apod_frac)
        # theta_H, xi_H, counts_H, dx_H, dy_H, xi2d_H = radial_correlation_function(tau_CMB_H, theta_deg=theta_deg, nbins=nbins, logbins=False, apod_frac=apod_frac)
        # theta_He, xi_He, counts_He, dx_He, dy_He, xi2d_He = radial_correlation_function(tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=False, apod_frac=apod_frac)
        # theta_HHe, xi_HHe, counts_HHe, dx_HHe, dy_HHe, xi2d_HHe = radial_correlation_function(tau_CMB_H, map2=tau_CMB_He, theta_deg=theta_deg, nbins=nbins, logbins=False, apod_frac=apod_frac)
        # # Check all quantities (theta, xi, counts, dx, dy, xi2d)
        # assert np.allclose(theta, theta_H), "theta != theta_H"
        # assert np.allclose(theta, theta_He), "theta != theta_He"
        # assert np.allclose(theta, theta_HHe), "theta != theta_HHe"
        # assert np.allclose(counts, counts_H), "counts != counts_H"
        # assert np.allclose(counts, counts_He), "counts != counts_He"
        # assert np.allclose(counts, counts_HHe), "counts != counts_HHe"
        # assert np.allclose(dx, dx_H), "dx != dx_H"
        # assert np.allclose(dx, dx_He), "dx != dx_He"
        # assert np.allclose(dx, dx_HHe), "dx != dx_HHe"
        # assert np.allclose(dy, dy_H), "dy != dy_H"
        # assert np.allclose(dy, dy_He), "dy != dy_He"
        # assert np.allclose(dy, dy_HHe), "dy != dy_HHe"
        # g = ef.create_group('corr_lin')
        # g.create_dataset('theta', data=theta)
        # g.create_dataset('xi', data=xi)
        # g.create_dataset('xi_H', data=xi_H)
        # g.create_dataset('xi_He', data=xi_He)
        # g.create_dataset('xi_HHe', data=xi_HHe)
        # g.create_dataset('counts', data=counts)
        # g.create_dataset('dx', data=dx)
        # g.create_dataset('dy', data=dy)
        # g.create_dataset('xi2d', data=xi2d)
        # g.create_dataset('xi2d_H', data=xi2d_H)
        # g.create_dataset('xi2d_He', data=xi2d_He)
        # g.create_dataset('xi2d_HHe', data=xi2d_HHe)

        # Auto and cross correlation functions (log binning)
        theta_min = np.deg2rad(2. * theta_deg / float(N))
        theta, xi, counts, dx, dy, xi2d = radial_correlation_function(tau_CMB, theta_deg=theta_deg, nbins=nbins)
        _, xi_HII, _, _, _, xi2d_HII = radial_correlation_function(tau_CMB_HII, theta_deg=theta_deg, nbins=nbins)
        _, xi_HeII, _, _, _, xi2d_HeII = radial_correlation_function(tau_CMB_HeII, theta_deg=theta_deg, nbins=nbins)
        _, xi_HeIII, _, _, _, xi2d_HeIII = radial_correlation_function(tau_CMB_HeIII, theta_deg=theta_deg, nbins=nbins)
        _, xi_HII_HeII, _, _, _, xi2d_HII_HeII = radial_correlation_function(tau_CMB_HII, map2=tau_CMB_HeII, theta_deg=theta_deg, nbins=nbins)
        _, xi_HII_HeIII, _, _, _, xi2d_HII_HeIII = radial_correlation_function(tau_CMB_HII, map2=tau_CMB_HeIII, theta_deg=theta_deg, nbins=nbins)
        _, xi_HeII_HeIII, _, _, _, xi2d_HeII_HeIII = radial_correlation_function(tau_CMB_HeII, map2=tau_CMB_HeIII, theta_deg=theta_deg, nbins=nbins)
        r_HII_HeII = cross_correlation_coefficient(xi_HII_HeII, xi_HII, xi_HeII)
        r_HII_HeIII = cross_correlation_coefficient(xi_HII_HeIII, xi_HII, xi_HeIII)
        r_HeII_HeIII = cross_correlation_coefficient(xi_HeII_HeIII, xi_HeII, xi_HeIII)
        # Check all quantities (theta, xi, counts, dx, dy, xi2d) Previously checked!
        # assert np.allclose(theta, theta_HII), "theta != theta_HII"
        # assert np.allclose(counts, counts_HII), "counts != counts_HII"
        # assert np.allclose(dx, dx_HII), "dx != dx_HII"
        # assert np.allclose(dy, dy_HII), "dy != dy_HII"
        g = ef.create_group('corr')
        g.create_dataset('theta', data=theta)
        g.create_dataset('xi', data=xi)
        g.create_dataset('xi_HII', data=xi_HII)
        g.create_dataset('xi_HeII', data=xi_HeII)
        g.create_dataset('xi_HeIII', data=xi_HeIII)
        g.create_dataset('xi_HII_HeII', data=xi_HII_HeII)
        g.create_dataset('xi_HII_HeIII', data=xi_HII_HeIII)
        g.create_dataset('xi_HeII_HeIII', data=xi_HeII_HeIII)
        g.create_dataset('r_HII_HeII', data=r_HII_HeII)
        g.create_dataset('r_HII_HeIII', data=r_HII_HeIII)
        g.create_dataset('r_HeII_HeIII', data=r_HeII_HeIII)
        g.create_dataset('counts', data=counts)
        if save_2d:
            g.create_dataset('dx', data=dx)
            g.create_dataset('dy', data=dy)
            g.create_dataset('xi2d', data=xi2d)
            g.create_dataset('xi2d_HII', data=xi2d_HII)
            g.create_dataset('xi2d_HeII', data=xi2d_HeII)
            g.create_dataset('xi2d_HeIII', data=xi2d_HeIII)
            g.create_dataset('xi2d_HII_HeII', data=xi2d_HII_HeII)
            g.create_dataset('xi2d_HII_HeIII', data=xi2d_HII_HeIII)
            g.create_dataset('xi2d_HeII_HeIII', data=xi2d_HeII_HeIII)

def save_spectra_z(sim_dir, N=320, nbins=64, oversample=1, n_images=8, zero_mean=True):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    # print(f'{rlc_dir}/tau_CMB_hist.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB.hdf5', 'r') as f, h5py.File(f'{rlc_dir}/tau_CMB_corr_{"z" if zero_mean else "zx"}.hdf5', 'w') as ef:
        header_attrs = f['Header'].attrs
        theta_deg = header_attrs['OpeningAngleDegrees']
        tau_CMB_below_z3 = header_attrs['tau_CMB_0']
        z_edges = f['z_edges'][:]
        tau_CMB_avg = f['tau_CMB_avg'][:]
        tau_CMB_cum = np.cumsum(tau_CMB_avg)
        tau_CMB_cum /= tau_CMB_cum[-1]
        tau_CMB_tot = np.sum(tau_CMB_avg)
        targets = np.linspace(0., 1., n_images + 1)
        indices = np.hstack([0, np.searchsorted(tau_CMB_cum, targets[1:-1]), len(tau_CMB_avg)])
        z_ranges = z_edges[indices]
        print(f'tau_CMB_tot = {tau_CMB_tot:g}')
        print(f'indices = {indices}')
        print(f'z_ranges = {z_ranges}')
        full_tau_CMB = f['tau_CMB'][:]  # Total (auto-spectrum, auto-correlation)
        full_tau_CMB_shuffled, meta = shuffle(full_tau_CMB, shift_cadence=6, symmetry_cadence=6, symmetry_mode='d4')
        tau_CMB = np.empty((n_images, N, N), dtype=np.float32)
        tau_CMB_shuffled = np.empty((n_images, N, N), dtype=np.float32)
        for i in range(n_images):
            tau_CMB[i] = np.sum(full_tau_CMB[indices[i]:indices[i+1]], axis=0)
            tau_CMB_shuffled[i] = np.sum(full_tau_CMB_shuffled[indices[i]:indices[i+1]], axis=0)
        del full_tau_CMB, full_tau_CMB_shuffled
        tau_CMB_0 = np.sum(tau_CMB, axis=0)
        tau_CMB_shuffled_0 = np.sum(tau_CMB_shuffled, axis=0)
        tau_CMB_0_avg = tau_CMB_0.mean()
        tau_CMB_shuffled_0_avg = tau_CMB_shuffled_0.mean()
        tau_CMB_avg = tau_CMB.mean(axis=0)
        tau_CMB_shuffled_avg = tau_CMB_shuffled.mean(axis=0)
        if zero_mean:  # Transform to zero-mean and unit variance
            # for i in range(n_images):
            #     tau_CMB[i] = (tau_CMB[i] - tau_CMB_avg[i]) / tau_CMB[i].std()
            # tau_CMB_0 = (tau_CMB_0 - tau_CMB_0_avg) / tau_CMB_0.std()
            for i in range(n_images):
                tau_CMB[i] = tau_CMB[i] * tau_CMB_0_avg / tau_CMB_avg[i] + tau_CMB_below_z3
                tau_CMB_shuffled[i] = tau_CMB_shuffled[i] * tau_CMB_shuffled_0_avg / tau_CMB_shuffled_avg[i] + tau_CMB_below_z3
            tau_CMB_0 += tau_CMB_below_z3
            tau_CMB_shuffled_0 += tau_CMB_below_z3
            # tau_CMB_0 = tau_CMB_0 / tau_CMB_0_avg
            # tau_CMB_shuffled_0 = tau_CMB_shuffled_0 / tau_CMB_shuffled_0_avg
            print(f'mean(tau_CMB) = {np.mean(tau_CMB_0):g}  (shuffled = {np.mean(tau_CMB_shuffled_0):g})')
            print(f'median(tau_CMB) = {np.median(tau_CMB_0):g}  (shuffled = {np.median(tau_CMB_shuffled_0):g})')
            print(f'std(tau_CMB) = {np.std(tau_CMB_0):g}  (shuffled = {np.std(tau_CMB_shuffled_0):g})')
            # 1/0
        ef.create_group('Header')
        eheader_attrs = ef['Header'].attrs
        for key in header_attrs:
            eheader_attrs[key] = header_attrs[key]
        ef.create_dataset('z_ranges', data=z_ranges)

        # Calculate all 2D spectra
        p2d = [None] * n_images
        p2d_shuffled = [None] * n_images
        ell2d, p2d[0], meta = power_spectrum_2d(tau_CMB[0], theta_deg=theta_deg, subtract_mean=True)
        _, p2d_shuffled[0], _ = power_spectrum_2d(tau_CMB_shuffled[0], theta_deg=theta_deg, subtract_mean=True)
        for i in range(1, n_images):
            _, p2d[i], _ = power_spectrum_2d(tau_CMB[i], theta_deg=theta_deg, subtract_mean=True)
            _, p2d_shuffled[i], _ = power_spectrum_2d(tau_CMB_shuffled[i], theta_deg=theta_deg, subtract_mean=True)
        _, p2d_0, _ = power_spectrum_2d(tau_CMB_0, theta_deg=theta_deg, subtract_mean=True)
        _, p2d_shuffled_0, _ = power_spectrum_2d(tau_CMB_shuffled_0, theta_deg=theta_deg, subtract_mean=True)

        # Build one common ell binning from the shared geometry and apply to all spectra
        positive = ell2d > 0
        edges, counts = adaptive_edges_from_counts(ell2d[positive], nbins_target=nbins, oversample=oversample,
            x_min=ell2d[positive].min(), x_max=ell2d[positive].max(), logbins=True, min_count=50, max_bin_width=None, exclude_zero=True)
        spec = [None] * n_images
        spec_shuffled = [None] * n_images
        for i in range(n_images):
            spec[i] = radial_bin_with_edges(ell2d, p2d[i], edges)
            spec_shuffled[i] = radial_bin_with_edges(ell2d, p2d_shuffled[i], edges)
        spec_0 = radial_bin_with_edges(ell2d, p2d_0, edges)
        spec_shuffled_0 = radial_bin_with_edges(ell2d, p2d_shuffled_0, edges)
        ell, var, sem, counts0, edges0 = spec[0]["centers"], spec[0]["var"], spec[0]["sem"], spec[0]["counts"], spec[0]["edges"]
        C_ell = [None] * n_images
        C_ell_shuffled = [None] * n_images
        for i in range(n_images):
            C_ell[i] = spec[i]["mean"]
            C_ell_shuffled[i] = spec_shuffled[i]["mean"]
        C_ell_0 = spec_0["mean"]
        C_ell_shuffled_0 = spec_shuffled_0["mean"]
        g = ef.create_group('spec')
        g.create_dataset('ell', data=ell)
        g.create_dataset('C_ell', data=np.array(C_ell))
        g.create_dataset('C_ell_0', data=C_ell_0)
        g.create_dataset('C_ell_shuffled', data=np.array(C_ell_shuffled))
        g.create_dataset('C_ell_shuffled_0', data=C_ell_shuffled_0)
        g.create_dataset('var', data=var)
        g.create_dataset('sem', data=sem)
        g.create_dataset('counts', data=counts)
        g.create_dataset('edges', data=edges)
        for key in meta:
            g.attrs[key] = meta[key]

        # Auto and cross correlation functions (log binning)
        theta_min = np.deg2rad(2. * theta_deg / float(N))
        xi = [None] * n_images
        xi_shuffled = [None] * n_images
        theta, xi[0], counts, _, _, _ = radial_correlation_function(tau_CMB[0], theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
        _, xi_shuffled[0], _, _, _, _ = radial_correlation_function(tau_CMB_shuffled[0], theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
        for i in range(1, n_images):
            _, xi[i], _, _, _, _ = radial_correlation_function(tau_CMB[i], theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
            _, xi_shuffled[i], _, _, _, _ = radial_correlation_function(tau_CMB_shuffled[i], theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
        _, xi_0, _, _, _, _ = radial_correlation_function(tau_CMB_0, theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
        _, xi_shuffled_0, _, _, _, _ = radial_correlation_function(tau_CMB_shuffled_0, theta_deg=theta_deg, nbins=nbins, subtract_mean=True)
        g = ef.create_group('corr')
        g.create_dataset('theta', data=theta)
        g.create_dataset('xi', data=np.array(xi))
        g.create_dataset('xi_0', data=xi_0)
        g.create_dataset('xi_shuffled', data=np.array(xi_shuffled))
        g.create_dataset('xi_shuffled_0', data=xi_shuffled_0)
        g.create_dataset('counts', data=counts)

def final_plot_spectra(sim_dir, N=320):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_corr.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'r') as f:
        theta_deg = f['Header'].attrs['OpeningAngleDegrees']
        l_min = 360. / theta_deg  # Lowest multipole
        g = f['spec']
        ell = g['ell'][:]
        C_ell = g['C_ell'][:]
        C_ell_HII = g['C_ell_HII'][:]
        C_ell_HeII = g['C_ell_HeII'][:]
        C_ell_HeIII = g['C_ell_HeIII'][:]
        C_ell_HII_HeII = g['C_ell_HII_HeII'][:]
        C_ell_HII_HeIII = g['C_ell_HII_HeIII'][:]
        C_ell_HeII_HeIII = g['C_ell_HeII_HeIII'][:]
        r_ell_HII_HeII = g['r_ell_HII_HeII'][:]
        r_ell_HII_HeIII = g['r_ell_HII_HeIII'][:]
        r_ell_HeII_HeIII = g['r_ell_HeII_HeIII'][:]
        # counts = g['counts'][:]
        # ell2d = g['ell2d'][:]
        # p2d = g['p2d'][:]
        # meta = dict(g.attrs)
        C_to_D2 = ell * (ell + 1.) / (2. * np.pi)
        D2_ell = C_to_D2 * C_ell
        D2_ell_HII = C_to_D2 * C_ell_HII
        D2_ell_HeII = C_to_D2 * C_ell_HeII
        D2_ell_HeIII = C_to_D2 * C_ell_HeIII
        D2_ell_HII_HeII = C_to_D2 * C_ell_HII_HeII
        D2_ell_HII_HeIII = C_to_D2 * C_ell_HII_HeIII
        D2_ell_HeII_HeIII = C_to_D2 * C_ell_HeII_HeIII
    space = 0.03
    fig = plt.figure(figsize=(4.5,2.)); ax1 = plt.axes([0,0,1,1]); ax2 = plt.axes([0,1+space,1,1])
    # ax1.axvline(x=l_min, c=[.3,.3,.3], ls='--', zorder=-100)
    # ell_arcsec_inv = ell / 206264.806247  # convert to arcsec?
    # ax1.plot(ell, C_ell, c='C0')
    # ax1.plot(ell, D2_ell, c='C0', label=r'${\rm Total}$')
    ax1.plot(ell, D2_ell_HII, ls='--', c='C2', label=r'${\rm HII}$')
    ax1.plot(ell, D2_ell_HeII, ls='--', c='C4', label=r'${\rm HeII}$')
    ax1.plot(ell, D2_ell_HeIII, ls='--', c='C6', label=r'${\rm HeIII}$')
    ax1.plot(ell, D2_ell_HII_HeII, c='C0') #, label=r'${\rm HII \times HeII}$')
    ax1.plot(ell, D2_ell_HII_HeIII, c='C1') #, label=r'${\rm HII \times HeIII}$')
    ax1.plot(ell, D2_ell_HeII_HeIII, c='C3') #, label=r'${\rm HeII \times HeIII}$')
    ax2.plot(ell, r_ell_HII_HeII, c='C0', label=r'${\rm HII \times HeII}$')
    ax2.plot(ell, r_ell_HII_HeIII, c='C1', label=r'${\rm HII \times HeIII}$')
    ax2.plot(ell, r_ell_HeII_HeIII, c='C3', label=r'${\rm HeII \times HeIII}$')
    # ax.text(0.975, 0.025, r'${\rm pixels}$', transform=ax.transAxes, va='bottom', ha='right', fontsize=12)
    # set_xaxis(ax, r'$\ell\ \,({\rm arcmin}^{-1})$') #, xmin=0, xmax=1.4, n=8, xp=0.1)
    set_log_xaxis(ax1, r'$\ell$', xmin=3e2, xmax=3e5)
    set_log_xaxis(ax2, r'$\ell$', xmin=3e2, xmax=3e5, show_labels=False)
    # set_log_yaxis(ax, r'$C_\ell$') #, ymin=4, ymax=6, n=5)
    # set_log_yaxis(ax1, r'$\Delta^2_\ell$') #, ymin=4, ymax=6, n=5)
    set_yaxis(ax1, r'$\Delta^2_\ell\ \,(\tau_{\rm CMB})$', ymin=0., ymax=0.25, n=6, ym=0.04, yp=0.04)
    set_yaxis(ax2, r'$r_\ell\ \,(\tau_{\rm CMB})$', ymin=-0.2, ymax=1, n=7, ym=0.05)
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.legend(loc='upper right', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=12, ncol=3, columnspacing=0.75) #, labelspacing=-0.3)
    ax2.legend(loc='lower right', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=12) #, labelspacing=-0.3)
    fig.savefig(f'final_spectra/tau_CMB_spectrum.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def final_plot_spectra_bias(sim_dir, N=320):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_corr.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'r') as f:
        theta_deg = f['Header'].attrs['OpeningAngleDegrees']
        l_min = 360. / theta_deg  # Lowest multipole
        g = f['spec']
        ell = g['ell'][:]
        C_ell = g['C_ell'][:]
        C_ell_HII = g['C_ell_HII'][:]
        C_ell_HeII = g['C_ell_HeII'][:]
        C_ell_HeIII = g['C_ell_HeIII'][:]
        C_ell_HII_HeII = g['C_ell_HII_HeII'][:]
        C_ell_HII_HeIII = g['C_ell_HII_HeIII'][:]
        C_ell_HeII_HeIII = g['C_ell_HeII_HeIII'][:]
        r_ell_HII_HeII = g['r_ell_HII_HeII'][:]
        r_ell_HII_HeIII = g['r_ell_HII_HeIII'][:]
        r_ell_HeII_HeIII = g['r_ell_HeII_HeIII'][:]
        # counts = g['counts'][:]
        # ell2d = g['ell2d'][:]
        # p2d = g['p2d'][:]
        # meta = dict(g.attrs)
        bias_HII_HeII = C_ell_HII_HeII / C_ell
        bias_HII_HeIII = C_ell_HII_HeIII / C_ell
        bias_HeII_HeIII = C_ell_HeII_HeIII / C_ell
        bias_HII = np.sqrt(C_ell_HII / C_ell)
        bias_HeII = np.sqrt(C_ell_HeII / C_ell)
        bias_HeIII = np.sqrt(C_ell_HeIII / C_ell)
    fig = plt.figure(figsize=(4.5,3.)); ax = plt.axes([0,0,1,1])
    # ax.axvline(x=l_min, c=[.3,.3,.3], ls='--', zorder=-100)
    # ell_arcsec_inv = ell / 206264.806247  # convert to arcsec?
    # ax.plot(ell, C_ell, c='C0')
    ax.plot(ell, bias_HII_HeII, c='C0', label=r'${\rm HII \times HeII}$')
    ax.plot(ell, bias_HII_HeIII, c='C1', label=r'${\rm HII \times HeIII}$')
    ax.plot(ell, bias_HeII_HeIII, c='C3', label=r'${\rm HeII \times HeIII}$')
    ax.plot(ell, bias_HII, ls='--', c='C2', label=r'${\rm HII}$')
    ax.plot(ell, bias_HeII, ls='--', c='C4', label=r'${\rm HeII}$')
    ax.plot(ell, bias_HeIII, ls='--', c='C6', label=r'${\rm HeIII}$')
    set_log_xaxis(ax, r'$\ell$', xmin=3e2, xmax=3e5)
    # set_log_yaxis(ax, r'$C_\ell$') #, ymin=4, ymax=6, n=5)
    # set_yaxis(ax, r'${\rm Response\ \, (C_\ell^x\,/\,C_\ell)}$', ymin=-0.25, ymax=1.25, n=7, ym=0.05, yp=0.15)
    set_yaxis(ax, r'${\rm Relative\ Response\ \, (C_\ell^x\,/\,C_\ell)}$', ymin=0., ymax=1., n=3, ym=0.3, yp=0.4)
    ax.minorticks_on()
    ax.legend(loc='lower right', frameon=False, borderaxespad=0.25, handlelength=2.5, fontsize=12, ncol=2) #, labelspacing=-0.3)
    fig.savefig(f'final_spectra/tau_CMB_spectrum_bias.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def final_plot_corr(sim_dir, N=320):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_corr.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'r') as f:
        theta_deg = f['Header'].attrs['OpeningAngleDegrees']
        g = f['corr']
        theta = np.rad2deg(g['theta'][:]) * 60.  # arcmin
        xi = g['xi'][:]
        xi_HII = g['xi_HII'][:]
        xi_HeII = g['xi_HeII'][:]
        xi_HeIII = g['xi_HeIII'][:]
        xi_HII_HeII = g['xi_HII_HeII'][:]
        xi_HII_HeIII = g['xi_HII_HeIII'][:]
        xi_HeII_HeIII = g['xi_HeII_HeIII'][:]
        r_HII_HeII = g['r_HII_HeII'][:]
        r_HII_HeIII = g['r_HII_HeIII'][:]
        r_HeII_HeIII = g['r_HeII_HeIII'][:]
        # counts = g['counts'][:]
        # dx = g['dx'][:]
        # dy = g['dy'][:]
        # xi2d = g['xi2d'][:]
        # xi2d_H = g['xi2d_H'][:]
        # xi2d_He = g['xi2d_He'][:]
        # xi2d_HHe = g['xi2d_HHe'][:]
    space = 0.03
    fig = plt.figure(figsize=(4.5,2.)); ax1 = plt.axes([0,0,1,1]); ax2 = plt.axes([0,1+space,1,1])
    # ell_arcsec_inv = ell / 206264.806247  # convert to arcsec?
    # ax1.plot(ell, C_ell, c='C0')
    # ax1.plot(theta, xi, c='C0', label=r'${\rm Total}$')
    ax1.plot(theta, xi_HII, ls='--', c='C2', label=r'${\rm HII}$')
    ax1.plot(theta, xi_HeII, ls='--', c='C4', label=r'${\rm HeII}$')
    ax1.plot(theta, xi_HeIII, ls='--', c='C6', label=r'${\rm HeIII}$')
    ax1.plot(theta, xi_HII_HeII, c='C0') #, label=r'${\rm HII \times HeII}$')
    ax1.plot(theta, xi_HII_HeIII, c='C1') #, label=r'${\rm HII \times HeIII}$')
    ax1.plot(theta, xi_HeII_HeIII, c='C3') #, label=r'${\rm HeII \times HeIII}$')
    mask = theta < 50
    ax2.plot(theta[mask], r_HII_HeII[mask], c='C0', label=r'${\rm HII \times HeII}$')
    ax2.plot(theta[mask], r_HII_HeIII[mask], c='C1', label=r'${\rm HII \times HeIII}$')
    ax2.plot(theta[mask], r_HeII_HeIII[mask], c='C3', label=r'${\rm HeII \times HeIII}$')
    set_log_xaxis(ax1, r'$\theta\ \,({\rm arcmin})$', xmin=2.5e-1, xmax=1.25e2)
    set_log_xaxis(ax2, r'$\theta\ \,({\rm arcmin})$', xmin=2.5e-1, xmax=1.25e2, show_labels=False)
    set_log_yaxis(ax1, r'$\xi_\theta\ \,(\tau_{\rm CMB})$', ymin=3e-3, ymax=2.)
    set_yaxis(ax2, r'$r_\theta\ \,(\tau_{\rm CMB})$', ymin=0, ymax=1, n=6)
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.legend(loc='lower left', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=12, ncol=3) #, labelspacing=-0.3)
    ax2.legend(loc='lower right', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=12) #, labelspacing=-0.3)
    fig.savefig(f'final_spectra/tau_CMB_corr.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def final_plot_spectra_z(sim_dir, N=5120, show_shuffled=False):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_corr_z.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_corr_z.hdf5', 'r') as f:
        theta_deg = f['Header'].attrs['OpeningAngleDegrees']
        l_min = 360. / theta_deg  # Lowest multipole
        z_ranges = f['z_ranges'][:]
        g = f['spec']
        ell = g['ell'][:]
        C_ell = g['C_ell'][:]
        C_ell_shuffled = g['C_ell_shuffled'][:]
        # counts = g['counts'][:]
        # meta = dict(g.attrs)
        C_to_D2 = ell * (ell + 1.) / (2. * np.pi)
        D2_ell = C_to_D2[None,:] * C_ell
        D2_ell_shuffled = C_to_D2[None,:] * C_ell_shuffled
        C_ell_0 = g['C_ell_0'][:]
        C_ell_shuffled_0 = g['C_ell_shuffled_0'][:]
        D2_ell_0 = C_to_D2 * C_ell_0
        D2_ell_shuffled_0 = C_to_D2 * C_ell_shuffled_0
    if show_shuffled:
        with h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'r') as f:
            g = f['spec']
            ell_ref = g['ell'][:]
            C_ell_ref = g['C_ell'][:]
            C_to_D2_ref = ell_ref * (ell_ref + 1.) / (2. * np.pi)
            D2_ell_ref = C_to_D2_ref * C_ell_ref
    fig = plt.figure(figsize=(4.5,2.75)); ax1 = plt.axes([0,0,1,1])
    # ax1.axvline(x=l_min, c=[.3,.3,.3], ls='--', zorder=-100)
    # ell_arcsec_inv = ell / 206264.806247  # convert to arcsec?
    n_ranges = len(z_ranges) - 1
    cmap = cmocean.cm.haline
    for i in range(n_ranges):
        ax1.plot(ell, D2_ell[i], c=cmap(i/n_ranges), label=r'$'+(r'z \in ' if i == 0 else '')+r'[%g, %g]$' % (round(z_ranges[i], 1), round(z_ranges[i+1], 1)))
        if show_shuffled:
            ax1.plot(ell, D2_ell_shuffled[i], c=cmap(i/n_ranges), ls='--')
    if show_shuffled:
        ax1.plot(ell, D2_ell_0, c='k', lw=2.5, label=r'$[3, 30]$')
        ax1.plot(ell_ref, D2_ell_ref*D2_ell_shuffled_0[-1]/D2_ell_ref[-1], c='r', lw=2.5) #, label=r'$\tau_{\rm CMB}$')
        ax1.plot(ell, D2_ell_shuffled_0, c='k', lw=1.5, ls='--')
    else:
        ax1.plot(ell, D2_ell_0, c=[.3,.3,.3], label=r'$[3, 30]$')
        ax1.plot(ell, D2_ell_shuffled_0, c='k', lw=2.25, label=r'${\rm Retiled}$')
        # Fit a power-law to D2_ell_shuffled_0 in the range ell = [5e3,5e4]
        mask = (ell >= 5e3) & (ell <= 5e4)
        popt, pcov = np.polyfit(np.log(ell[mask]), np.log(D2_ell_shuffled_0[mask]), 1, cov=True)
        A, gamma = np.exp(popt[1]), -popt[0]
        ax1.plot(ell[mask], 1.65 * A * ell[mask]**(-gamma), c='k', lw=1.5, ls=':')
        ax1.text(np.sqrt(5e3*5e4), 2. * A * np.sqrt(5e3*5e4)**(-gamma), f'$\\propto \\ell^{{-{gamma:.2f}}}$', fontsize=11.5, ha='center', va='bottom')
    set_log_xaxis(ax1, r'$\ell$', xmin=3e2, xmax=3e5)
    # set_yaxis(ax1, r'$\Delta^2_\ell\ \,(\tau_{\rm CMB})$') #, ymin=0., ymax=0.4, n=5, ym=0., yp=0.06)
    # set_yaxis(ax1, r'$\Delta^2_\ell\ \,(\tau_{\rm CMB})$', ymin=0., ymax=0.4, n=5, ym=0., yp=0.06)
    set_log_yaxis(ax1, r'$\Delta^2_\ell\ \,(\tau_{\rm CMB})$', ymin=5e-7, ymax=5e-5)
    ax1.minorticks_on()
    # ax1.legend(loc='upper right', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=11.5, ncol=2, columnspacing=1.75, markerfirst=False) #, labelspacing=-0.3)
    ax1.legend(loc=[-.18,1.035], frameon=False, borderaxespad=0, handlelength=1.8, fontsize=10.5, ncol=5, columnspacing=.65) #, labelspacing=-0.3)
    fig.savefig(f'final_spectra/tau_CMB_spectrum_z.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

def final_plot_corr_z(sim_dir, N=5120, show_shuffled=False):
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_corr_z.hdf5')
    with h5py.File(f'{rlc_dir}/tau_CMB_corr_z.hdf5', 'r') as f:
        theta_deg = f['Header'].attrs['OpeningAngleDegrees']
        z_ranges = f['z_ranges'][:]
        g = f['corr']
        theta = np.rad2deg(g['theta'][:]) * 60.  # arcmin
        xi = g['xi'][:]
        xi_shuffled = g['xi_shuffled'][:]
        xi_0 = g['xi_0'][:]
        xi_shuffled_0 = g['xi_shuffled_0'][:]
    if show_shuffled:
        with h5py.File(f'{rlc_dir}/tau_CMB_corr.hdf5', 'r') as f:
            g = f['corr']
            theta_ref = np.rad2deg(g['theta'][:]) * 60.  # arcmin
            xi_ref = g['xi'][:]
    fig = plt.figure(figsize=(4.5,2.75)); ax1 = plt.axes([0,0,1,1])
    n_ranges = len(z_ranges) - 1
    cmap = cmocean.cm.haline
    for i in range(n_ranges):
        if i == 5:
            mask = theta < 60.
            ax1.plot(theta[mask], xi[i][mask], c=cmap(i/n_ranges), label=r'$'+(r'z \in ' if i == 0 else '')+r'[%g, %g]$' % (round(z_ranges[i], 1), round(z_ranges[i+1], 1)))
        else:
            ax1.plot(theta, xi[i], c=cmap(i/n_ranges), label=r'$'+(r'z \in ' if i == 0 else '')+r'[%g, %g]$' % (round(z_ranges[i], 1), round(z_ranges[i+1], 1)))
        if show_shuffled:
            ax1.plot(theta, xi_shuffled[i], c=cmap(i/n_ranges), ls='--')
    if show_shuffled:
        ax1.plot(theta, xi_0, c='k', lw=2.5, label=r'$[3, 30]$')
        ax1.plot(theta_ref, xi_ref*xi_shuffled_0[-1]/xi_ref[-1], c='r', lw=2.5) #, label=r'$\tau_{\rm CMB}$')
        ax1.plot(theta, xi_shuffled_0, c='k', lw=1.5, ls='--')
    else:
        ax1.plot(theta, xi_0, c=[.3,.3,.3], label=r'$[3, 30]$')
        ax1.plot(theta, xi_shuffled_0, c='k', lw=2.25, label=r'${\rm Retiled}$')
        # Fit a power-law to xi_shuffled_0 in the range theta = [5e-1,5e1]
        # mask = (theta >= 5e-1) & (theta <= 5e1)
        # popt, pcov = np.polyfit(np.log(theta[mask]), np.log(xi_shuffled_0[mask]), 1, cov=True)
        # A, gamma = np.exp(popt[1]), -popt[0]
        # ax1.plot(theta[mask], 1.7 * A * theta[mask]**(-gamma), c='k', lw=1.5, ls=':')
        # ax1.text(np.sqrt(5e-1*5e1), 2. * A * np.sqrt(5e-1*5e1)**(-gamma), f'$\\propto \theta^{{-{gamma:.2f}}}$', fontsize=11.5, ha='center', va='bottom')
    # set_log_xaxis(ax1, r'$\theta\ \,({\rm arcmin})$', xmin=2.5e-1, xmax=1.25e2)
    set_log_xaxis(ax1, r'$\theta\ \,({\rm arcmin})$', xmin=1.6e-1, xmax=1.2e2)
    set_log_yaxis(ax1, r'$\xi_\theta\ \,(\tau_{\rm CMB})$', ymin=2e-7, ymax=1.4e-4)
    ax1.minorticks_on()
    # ax1.legend(loc='lower left', frameon=False, borderaxespad=0.5, handlelength=2.5, fontsize=12, ncol=3) #, labelspacing=-0.3)
    ax1.legend(loc=[-.18,1.035], frameon=False, borderaxespad=0, handlelength=1.8, fontsize=10.5, ncol=5, columnspacing=.65) #, labelspacing=-0.3)
    fig.savefig(f'final_spectra/tau_CMB_corr_z.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0.025)
    plt.close()

if __name__ == '__main__':
    # N = 80
    # N = 320
    # N = 1280
    N = 5120
    sim_dir = '/lustre/orion/ast211/proj-shared/arsmith/500cMpc'  # Flagship
    rlc_dir = f'{sim_dir}/postprocessing/rlc_{N}'
    print(f'{rlc_dir}/tau_CMB_hist.hdf5')
    # save_spectra(sim_dir, N)
    # save_spectra_z(sim_dir, N)
    # save_spectra_z(sim_dir, N, zero_mean=False)
    final_plot_spectra(sim_dir, N)
    final_plot_spectra_bias(sim_dir, N)
    final_plot_corr(sim_dir, N)
    final_plot_spectra_z(sim_dir, N)
    final_plot_corr_z(sim_dir, N)
