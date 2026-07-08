"""calc_ksz_new.py -- Physically-organized kSZ data product.

This is a NEW script (the original ``calc_ksz.py`` is left untouched). It produces a
re-organized kSZ HDF5 product containing:

  1. coarse redshift-bin kSZ maps (the existing product, but with a new default attenuation);
  2. exact coarse-bin source-component kSZ maps (chi*(1+delta)*v_los decomposition);
  3. coarse-bin optical-depth weighting maps for later missing-velocity work;
  4. global-mean optical-depth attenuation as the default production attenuation;
  5. NO symmetry operations of any kind inside this file.

NOTE (2026): the former fine-redshift 2D transverse power-spectrum stage (STAGE 2)
has been REMOVED. The full 3D field/momentum power spectra are now measured directly
from the 3D Cartesian renders (compute_ksz_3d_snapshot_products.py), so the per-fine-
layer 2D spectra are redundant. This script now produces only the coarse redshift-bin
maps plus cheap per-fine-layer SCALAR metadata (tau profile, plane means) under /fine.

Design choices (see the module docstrings / inline comments for the physics reasoning):

  * Default kSZ uses a *scalar, redshift-dependent* global-mean optical depth
    exp[-tau_to_obs_global(z_j)] instead of the per-pixel raw-LOS optical depth.
    Reason: the per-pixel raw-LOS optical depth is computed along an *untransformed*
    repeated-box line of sight, while the final observed map is assembled from
    *transformed* coarse shells in the downstream plotting script. The per-pixel
    attenuation is therefore formally inconsistent with the shuffled map. A global-mean
    (sky-averaged) attenuation per fine layer is shuffle-invariant and consistent.

  * The exact source decomposition chi*(1+delta)*v_los = chi_bar*v + chi_bar*delta*v +
    dchi*v + dchi*delta*v is performed at *every fine layer* and then integrated into the
    coarse bins. The mean density used for delta = rho/rho_bar - 1 is the *global plane
    mean* of the full transverse plane at each fine layer, NOT a chunk-local mean.
    Reason: a chunk-local mean would make the decomposition chunk-dependent and unphysical.

  * Shell symmetries (shifts / rotations / reflections) are applied *only later*, in the
    plotting / map-building script, where different coarse shells are combined into the
    final observed map. Nothing in this file applies them.

Single production stage:

  STAGE 1 (maps):  calc_kSZ_new_*  -> kSZ_rlc_<N>.hdf5  (coarse maps + /fine scalar metadata)

Reuses helpers and constants from calc_ksz.py without modifying it.
"""

import numpy as np
import h5py, os, sys
from multiprocessing import Pool
from tqdm import tqdm


from calc_ksz import (
    c, X, sigma_T, mH,
    CHUNK_SIZE, DEPTH_FILES, TARGET_RESOLUTION, TARGET_DEPTH, DEPTH_SIZE,
    RESULTS_ROOT,
    get_info, get_results_dir,
    get_los_unit_vectors, project_los_velocity,
    combine_kSZ_metadata,
)

# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------
ATTENUATION_MODE = "global_mean"   # default production attenuation prescription
SAVE_NO_ATTENUATION = True         # always also store the no-attenuation kSZ
SAVE_LEGACY_SPATIAL_TAU = False    # optional diagnostic: old per-pixel raw-LOS attenuation

ATTENUATION_DESCRIPTION = (
    "Default kSZ uses a scalar redshift-dependent global-mean optical depth "
    "exp[-tau_to_obs_global(z_j)], where tau_to_obs_global(z_j) = "
    "foreground_tau_global + sum_{m>=j} <dtau_m(x,y)>_xy. This sky-averaged attenuation "
    "is invariant under the downstream shell symmetry operations, unlike the old per-pixel "
    "raw-LOS optical depth which was computed along an untransformed repeated-box LOS."
)

SYMMETRY_STATUS = (
    "No shell symmetries applied in this file. "
    "Apply the existing downstream symmetry recipe before "
    "computing cross-shell, cumulative, or final-map spectra."
)

# kSZ component names (the four exact source terms)
SOURCE_COMPONENTS = ['v', 'deltav', 'chiv', 'chideltav']

# Lightcone source regions (high-z above z=4.75, low-z below), and their coarse-bin counts.
RLC_ROOT_HIGHZ = '/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/lightcone'
RLC_ROOT_LOWZ = '/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/lightcone'
N_SPLIT_HIGHZ = 60
N_SPLIT_LOWZ = 21


def get_regions(resolution):
    """(ds_highz, ds_lowz) metadata for a given transverse resolution (header reads only)."""
    ds_highz = get_info(f'{RLC_ROOT_HIGHZ}/rlc_{resolution}', n_split=N_SPLIT_HIGHZ)
    ds_lowz = get_info(f'{RLC_ROOT_LOWZ}/rlc_{resolution}', n_split=N_SPLIT_LOWZ)
    return ds_highz, ds_lowz


# ===========================================================================
# A. Global optical-depth profile (B2/B3) and plane means (D)
# ===========================================================================
def compute_global_tau_profile(tau_plane_mean, foreground_tau_global=0.0):
    """Build the scalar global-mean optical-depth profile from per-fine-layer plane means.

    The lightcone is ordered so that the *final* (largest) depth index is closest to the
    observer (this reproduces the convention in the original calc_ksz.py). The optical depth
    from fine layer ``j`` to the observer therefore includes layer ``j`` itself plus all the
    layers closer to the observer (larger index):

        tau_to_obs_global[j] = foreground_tau_global + sum_{m>=j} <dtau_m>_xy

    Returns (tau_to_obs_global, exp_tau_global, tau_total) where tau_total is the scalar sum
    of all plane means (the value to pass as ``foreground_tau_global`` to a farther region).
    """
    tau_plane_mean = np.asarray(tau_plane_mean, dtype=np.float64)
    # Reverse-cumulative sum: sum_{m>=j} tau_plane_mean[m].
    tail_sum = np.cumsum(tau_plane_mean[::-1])[::-1]
    tau_to_obs_global = foreground_tau_global + tail_sum
    exp_tau_global = np.exp(-tau_to_obs_global)
    tau_total = float(np.sum(tau_plane_mean))
    return tau_to_obs_global, exp_tau_global, tau_total


def _accumulate_plane_sums(rho, x_e):
    """Return (sum_rho, sum_chi, sum_chi_rho) summed over the transverse (x,y) axes.

    Shapes in: (nx, ny, nz). Out: (nz,) each.
    """
    sum_rho = rho.sum(axis=(0, 1))
    sum_chi = x_e.sum(axis=(0, 1))
    sum_chi_rho = (x_e * rho).sum(axis=(0, 1))
    return sum_rho, sum_chi, sum_chi_rho


def _read_xe(f, sl):
    x_e = f['HII_VolumeFraction'][sl].astype(np.float64)
    x_e += f['HeII_VolumeFraction'][sl].astype(np.float64)
    x_e += 2.0 * f['HeIII_VolumeFraction'][sl].astype(np.float64)
    return x_e


def _stream_plane_sums_chunk_range(ds, chunk_lo, chunk_hi):
    """Accumulate full-plane sums (sum_rho, sum_chi, sum_chi_rho), each (nz,), for the flat
    chunk-index range [chunk_lo, chunk_hi) of one region (chunk = ix*n_chunks + iy, matching
    the CHUNK_SIZE x CHUNK_SIZE partition used everywhere else in this file).

    This is the READ-HEAVY, chunk-parallelizable half of the plane-means computation: each
    caller only reads its own chunk range x DEPTH_FILES depth blocks, so it can be split
    across many tasks (see prep_accumulate) instead of one task streaming the whole cube
    serially. A single-task caller with chunk_lo=0, chunk_hi=n_chunks**2 reproduces the
    original whole-region streamed sum exactly.
    """
    n = int(ds['NumPixels'])
    nz = len(ds['dl'])
    n_chunks = n // CHUNK_SIZE
    n_depth = (nz + DEPTH_FILES - 1) // DEPTH_FILES

    sum_rho = np.zeros(nz, dtype=np.float64)
    sum_chi = np.zeros(nz, dtype=np.float64)
    sum_chi_rho = np.zeros(nz, dtype=np.float64)

    for chunk in range(chunk_lo, chunk_hi):
        ix, iy = chunk // n_chunks, chunk % n_chunks
        x1, x2 = ix * CHUNK_SIZE, (ix + 1) * CHUNK_SIZE
        y1, y2 = iy * CHUNK_SIZE, (iy + 1) * CHUNK_SIZE
        for iz in range(DEPTH_FILES):
            z1, z2 = iz * n_depth, min(nz, (iz + 1) * n_depth)
            if z1 >= z2:
                continue
            with h5py.File(ds['filename'], 'r') as f:
                rho = f['Density'][x1:x2, y1:y2, z1:z2].astype(np.float64)
                x_e = _read_xe(f, np.s_[x1:x2, y1:y2, z1:z2])
            sr, sc, scr = _accumulate_plane_sums(rho, x_e)
            sum_rho[z1:z2] += sr
            sum_chi[z1:z2] += sc
            sum_chi_rho[z1:z2] += scr
            del rho, x_e
    return sum_rho, sum_chi, sum_chi_rho


def finalize_plane_means(ds, sum_rho, sum_chi, sum_chi_rho, npix, foreground_tau_global=0.0):
    """Cheap tail of the plane-means computation: turn full-plane SUMS (already reduced over
    the whole region) into means, the global tau profile, and the kSZ attenuation weight.
    Splitting this out lets the expensive sums be computed however is convenient (whole-cube
    streamed, chunk-parallel accumulate+reduce, or a full in-memory load) while this part
    always runs once, in milliseconds.
    """
    rho_factor = (X * sigma_T / mH) * ds['density_to_cgs'] * ds['dl']  # (nz,)
    rho_mean = sum_rho / npix
    chi_mean = sum_chi / npix
    chi_rho_mean = sum_chi_rho / npix
    # A = chi*(1+delta) = chi*rho/rho_bar  =>  <A> = <chi*rho>/rho_bar
    A_mean = np.where(rho_mean > 0, chi_rho_mean / np.where(rho_mean > 0, rho_mean, 1.0), 0.0)
    # tau_plane_mean[j] = <dtau_j> = rho_factor[j] * <chi*rho>
    tau_plane_mean = rho_factor * chi_rho_mean

    tau_to_obs_global, exp_tau_global, tau_total = compute_global_tau_profile(
        tau_plane_mean, foreground_tau_global)

    # Per-layer kSZ prefactor (without x_e and density): rho_factor*rho_bar/c * exp_global.
    # With it, the total fine kSZ is  -weight_global[j] * A_j * v_los_j  ==  -dtau_j*exp_global[j]*v_los_j/c
    # exactly (the rho_bar cancels), which is the new default global-mean attenuation field.
    weight_global = rho_factor * rho_mean / c * exp_tau_global

    return {
        'rho_factor': rho_factor,
        'rho_mean': rho_mean,
        'chi_mean': chi_mean,
        'chi_rho_mean': chi_rho_mean,
        'A_mean': A_mean,
        'tau_plane_mean': tau_plane_mean,
        'tau_to_obs_global': tau_to_obs_global,
        'exp_tau_global': exp_tau_global,
        'weight_global': weight_global,
        'tau_total': tau_total,
        'foreground_tau_global': float(foreground_tau_global),
    }


def compute_density_plane_means(ds, foreground_tau_global=0.0):
    """Compute global per-fine-layer plane means and the global tau profile for one region.

    IMPORTANT: rho_bar_j is the GLOBAL mean of the full transverse plane at fine layer j,
    NOT a chunk-local mean. For low resolution (NumPixels < 640) the whole cube fits in
    memory and the means are exact. For high resolution we stream the full data in 64x64
    chunks (serially, in ONE task) and accumulate plane sums, giving the exact full-plane
    means. For production at high resolution prefer the chunk-PARALLEL prep_accumulate /
    prep_reduce workflow instead of calling this directly (see module docstring).

    Returns a dict of (nz,) arrays plus the per-layer prefactors used by the decomposition.
    """
    n = int(ds['NumPixels'])
    nz = len(ds['dl'])

    if n < 640:
        # Full-array path: load the whole cube.
        with h5py.File(ds['filename'], 'r') as f:
            rho = f['Density'][:].astype(np.float64)
            x_e = _read_xe(f, np.s_[:])
        sum_rho, sum_chi, sum_chi_rho = _accumulate_plane_sums(rho, x_e)
        del rho, x_e
    else:
        # Streamed path (serial): same chunk x depth-block iteration as the parallel version,
        # just covering the FULL chunk range in one task.
        n_chunks = n // CHUNK_SIZE
        sum_rho, sum_chi, sum_chi_rho = _stream_plane_sums_chunk_range(ds, 0, n_chunks**2)

    npix = float(n * n)
    return finalize_plane_means(ds, sum_rho, sum_chi, sum_chi_rho, npix,
                                foreground_tau_global=foreground_tau_global)


# ===========================================================================
# D. Exact fine-layer source decomposition + coarse binning
# ===========================================================================
def decompose_fine_source(chi, rho, v_los, rho_mean, chi_mean):
    """Exact per-fine-layer decomposition of chi*(1+delta)*v_los into four source terms.

    All inputs are (nx, ny, nz). ``rho_mean`` and ``chi_mean`` are (nz,) global plane means.

        delta      = rho / rho_bar - 1          (global mean density per plane)
        chi_bar    = <chi>_xy                    (global mean ionized fraction per plane)
        dchi       = chi - chi_bar

        source_v          = chi_bar * v_los
        source_deltav     = chi_bar * delta * v_los
        source_chiv       = dchi    * v_los
        source_chideltav  = dchi    * delta * v_los

    Their sum is exactly chi*(1+delta)*v_los = A*v_los.
    """
    delta = rho / rho_mean[None, None, :] - 1.0
    chi_bar = chi_mean[None, None, :]
    dchi = chi - chi_bar
    sources = {
        'v':         chi_bar * v_los,
        'deltav':    chi_bar * delta * v_los,
        'chiv':      dchi * v_los,
        'chideltav': dchi * delta * v_los,
    }
    return sources, delta


def _bin_coarse(ds, fine):
    """Sum a fine (nx, ny, nz) field into coarse bins -> (n_split, nx, ny)."""
    nx, ny = fine.shape[0], fine.shape[1]
    out = np.zeros((ds['n_split'], nx, ny), dtype=np.float64)
    for i in range(ds['n_split']):
        i1, i2 = ds['z_indices'][i], ds['z_indices'][i + 1]
        out[i] = np.sum(fine[:, :, i1:i2], axis=2)
    return out


def build_coarse_products(ds, dtau, v_los, chi, rho, plane_means,
                          foreground_tau_spatial=None, do_legacy=False,
                          validate=False):
    """Build all coarse-bin products for one region from fine (nx,ny,nz) arrays.

    Returns a dict of (n_split, nx, ny) maps:
        kSZ, kSZ_no_attenuation, tau_CMB, tau_weighted_global,
        kSZ_v, kSZ_deltav, kSZ_chiv, kSZ_chideltav,
      and optionally kSZ_spatial_tau_legacy, tau_weighted_spatial_legacy.
    Also returns 'tau_CMB_tot' (nx,ny) for foreground propagation and validation residuals.
    """
    rho_mean = plane_means['rho_mean']
    chi_mean = plane_means['chi_mean']
    exp_global = plane_means['exp_tau_global']
    wg = plane_means['weight_global'][None, None, :]  # = rho_factor*rho_bar/c*exp_global

    sources, _delta = decompose_fine_source(chi, rho, v_los, rho_mean, chi_mean)

    # Fine-layer kSZ component contributions: dT_a = -weight_global * source_a.
    out = {}
    fine_total = None
    for name in SOURCE_COMPONENTS:
        dT_a = -wg * sources[name]
        out[f'kSZ_{name}'] = _bin_coarse(ds, dT_a)
        fine_total = dT_a if fine_total is None else (fine_total + dT_a)
        del dT_a
    del sources

    # Default kSZ (global-mean attenuation) = sum of the four exact components.
    out['kSZ'] = _bin_coarse(ds, fine_total)

    # No-attenuation kSZ and optical-depth maps from the actual dtau planes.
    out['kSZ_no_attenuation'] = _bin_coarse(ds, -dtau * v_los / c)
    out['tau_CMB'] = _bin_coarse(ds, dtau)
    out['tau_weighted_global'] = _bin_coarse(ds, dtau * exp_global[None, None, :])
    out['tau_CMB_tot'] = np.sum(out['tau_CMB'], axis=0)

    # Optional legacy per-pixel raw-LOS attenuation (diagnostic only).
    if do_legacy:
        tau_cumsum = np.cumsum(dtau, axis=2)
        tau_tot = tau_cumsum[:, :, -1][:, :, None]
        fg = 0.0 if foreground_tau_spatial is None else foreground_tau_spatial
        tau_to_obs = fg + tau_tot - tau_cumsum + dtau
        exp_tau_spatial = np.exp(-tau_to_obs)
        out['kSZ_spatial_tau_legacy'] = _bin_coarse(ds, -dtau * exp_tau_spatial * v_los / c)
        out['tau_weighted_spatial_legacy'] = _bin_coarse(ds, dtau * exp_tau_spatial)
        del tau_cumsum, tau_tot, tau_to_obs, exp_tau_spatial

    if validate:
        # J1: fine-layer component identity (sampled). Compare exact dtau-based dT to the
        # sum of components at a few fine layers.
        nz = dtau.shape[2]
        sample = np.unique(np.linspace(0, nz - 1, min(nz, 8)).astype(int))
        max_abs, max_rel = 0.0, 0.0
        for j in sample:
            direct = -dtau[:, :, j] * exp_global[j] * v_los[:, :, j] / c
            comp = fine_total[:, :, j]
            diff = np.abs(direct - comp)
            scale = np.maximum(np.abs(direct), np.abs(comp))
            max_abs = max(max_abs, float(diff.max()))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(scale > 0, diff / scale, 0.0)
            max_rel = max(max_rel, float(rel.max()))
        out['_validate_fine_component'] = (max_abs, max_rel)

    del fine_total
    return out


# ===========================================================================
# STAGE 1 maps -- full-array (low resolution) path
# ===========================================================================
def _load_fine_arrays_all(ds):
    """Load full-cube dtau, v_los, chi (x_e), rho for one region (low-res path)."""
    nx, ny, nz_hat = get_los_unit_vectors(ds)
    rho_factor = (X * sigma_T / mH) * ds['density_to_cgs'] * ds['dl']
    with h5py.File(ds['filename'], 'r') as f:
        rho = f['Density'][:].astype(np.float64)
        x_e = _read_xe(f, np.s_[:])
        dtau = x_e * rho * rho_factor[None, None, :]
        vel = f['Velocities'][:]
        v_los = project_los_velocity(vel, ds, nx, ny, nz_hat, z1=0)
    return dtau, v_los, x_e, rho


def calc_kSZ_new_all(ds, plane_means=None, foreground_tau_global=0.0,
                     foreground_tau_spatial=None, validate=True):
    """Compute all coarse products for one low-resolution region (full array in memory)."""
    if plane_means is None:
        plane_means = compute_density_plane_means(ds, foreground_tau_global)
    dtau, v_los, chi, rho = _load_fine_arrays_all(ds)
    products = build_coarse_products(
        ds, dtau, v_los, chi, rho, plane_means,
        foreground_tau_spatial=foreground_tau_spatial,
        do_legacy=SAVE_LEGACY_SPATIAL_TAU, validate=validate)
    return products, plane_means


def calc_kSZ_new_all_single(ds_highz):
    """Single-region low-res driver."""
    products, plane_means = calc_kSZ_new_all(ds_highz)
    fine_meta = build_fine_metadata(ds_highz, plane_means, source_region=0)
    write_kSZ_new(ds_highz, products, fine_meta, source_files=[ds_highz['filename']])


def calc_kSZ_new_all_combined(ds_highz, ds_lowz):
    """Combined low-res driver: low-z first, then high-z with low-z tau as foreground."""
    if ds_highz['NumPixels'] != ds_lowz['NumPixels']:
        raise ValueError(f'NumPixels mismatch: high-z {ds_highz["NumPixels"]}, low-z {ds_lowz["NumPixels"]}')

    # H: low-z global optical depth first; its scalar total is the foreground for high-z.
    pm_lowz = compute_density_plane_means(ds_lowz, foreground_tau_global=0.0)
    lowz, _ = calc_kSZ_new_all(ds_lowz, plane_means=pm_lowz,
                               foreground_tau_spatial=None)

    pm_highz = compute_density_plane_means(ds_highz, foreground_tau_global=pm_lowz['tau_total'])
    # For the optional legacy spatial mode, the high-z foreground is the per-pixel low-z total.
    fg_spatial = lowz['tau_CMB_tot'][:, :, None] if SAVE_LEGACY_SPATIAL_TAU else None
    highz, _ = calc_kSZ_new_all(ds_highz, plane_means=pm_highz,
                                foreground_tau_spatial=fg_spatial)

    ds = combine_kSZ_metadata(ds_highz, ds_lowz)
    products = _stack_products(highz, lowz)
    fine_meta = _stack_fine_metadata(
        build_fine_metadata(ds_highz, pm_highz, source_region=0),
        build_fine_metadata(ds_lowz, pm_lowz, source_region=1),
    )
    write_kSZ_new(ds, products, fine_meta,
                  source_files=[ds_highz['filename'], ds_lowz['filename']])


def _stack_products(highz, lowz):
    """Stack high-z and low-z coarse maps in far->near (decreasing z) depth order."""
    out = {}
    for key in highz:
        if key == 'tau_CMB_tot':
            continue
        if key.startswith('_'):
            continue
        out[key] = np.vstack([highz[key], lowz[key]])
    return out


# ===========================================================================
# STAGE 1 maps -- chunked (high resolution) path
# ===========================================================================
def _load_fine_arrays_chunk(ds, chunk):
    """Load (CHUNK_SIZE, CHUNK_SIZE, nz) dtau, v_los, chi, rho for one high-res chunk."""
    n = int(ds['NumPixels'])
    assert n >= 640, f"NumPixels = {n}, must be >= 640"
    n_chunks = n // CHUNK_SIZE
    nz = len(ds['dl'])
    # Same depth blocking as the original kSZ chunk pipeline (_calc_kSZ_split_arrays).
    n_depth = (nz + DEPTH_FILES - 1) // DEPTH_FILES

    ix, iy = chunk // n_chunks, chunk % n_chunks
    x1, x2 = ix * CHUNK_SIZE, (ix + 1) * CHUNK_SIZE
    y1, y2 = iy * CHUNK_SIZE, (iy + 1) * CHUNK_SIZE
    nx, ny, nz_hat = get_los_unit_vectors(ds, x1, x2, y1, y2)
    rho_factor = (X * sigma_T / mH) * ds['density_to_cgs'] * ds['dl']

    dtau = np.zeros((CHUNK_SIZE, CHUNK_SIZE, nz), dtype=np.float64)
    v_los = np.zeros((CHUNK_SIZE, CHUNK_SIZE, nz), dtype=np.float64)
    chi = np.zeros((CHUNK_SIZE, CHUNK_SIZE, nz), dtype=np.float64)
    rho = np.zeros((CHUNK_SIZE, CHUNK_SIZE, nz), dtype=np.float64)
    for iz in range(DEPTH_FILES):
        z1, z2 = iz * n_depth, min(nz, (iz + 1) * n_depth)
        if z1 >= z2:
            continue
        with h5py.File(ds['filename'], 'r') as f:
            rho_blk = f['Density'][x1:x2, y1:y2, z1:z2].astype(np.float64)
            x_e = _read_xe(f, np.s_[x1:x2, y1:y2, z1:z2])
            rho[:, :, z1:z2] = rho_blk
            chi[:, :, z1:z2] = x_e
            dtau[:, :, z1:z2] = x_e * rho_blk * rho_factor[None, None, z1:z2]
            vel = f['Velocities'][x1:x2, y1:y2, z1:z2, :]
            v_los[:, :, z1:z2] = project_los_velocity(vel, ds, nx, ny, nz_hat, z1=z1)
    return dtau, v_los, chi, rho, (ix, iy)


def calc_kSZ_new_chunk(ds, chunk, plane_means, foreground_tau_spatial=None):
    """Compute and write one high-res chunk of all coarse products."""
    dtau, v_los, chi, rho, (ix, iy) = _load_fine_arrays_chunk(ds, chunk)
    products = build_coarse_products(
        ds, dtau, v_los, chi, rho, plane_means,
        foreground_tau_spatial=foreground_tau_spatial,
        do_legacy=SAVE_LEGACY_SPATIAL_TAU, validate=False)
    _write_chunk(ds, ix, iy, products)


def calc_kSZ_new_chunk_combined(ds_highz, ds_lowz, chunk, pm_highz, pm_lowz):
    """Compute and write one high-res chunk for the combined high-z + low-z case."""
    lowz_dtau, lowz_v, lowz_chi, lowz_rho, (ix, iy) = _load_fine_arrays_chunk(ds_lowz, chunk)
    lowz = build_coarse_products(ds_lowz, lowz_dtau, lowz_v, lowz_chi, lowz_rho, pm_lowz,
                                 do_legacy=SAVE_LEGACY_SPATIAL_TAU, validate=False)
    fg_spatial = lowz['tau_CMB_tot'][:, :, None] if SAVE_LEGACY_SPATIAL_TAU else None
    del lowz_dtau, lowz_v, lowz_chi, lowz_rho

    hz_dtau, hz_v, hz_chi, hz_rho, _ = _load_fine_arrays_chunk(ds_highz, chunk)
    highz = build_coarse_products(ds_highz, hz_dtau, hz_v, hz_chi, hz_rho, pm_highz,
                                  foreground_tau_spatial=fg_spatial,
                                  do_legacy=SAVE_LEGACY_SPATIAL_TAU, validate=False)
    del hz_dtau, hz_v, hz_chi, hz_rho

    ds = combine_kSZ_metadata(ds_highz, ds_lowz)
    products = _stack_products(highz, lowz)
    _write_chunk(ds, ix, iy, products)


def _chunk_dir(ds):
    results_dir = get_results_dir(ds['NumPixels'])
    cdir = f'{results_dir}/kSZ_new_rlc_{int(ds["NumPixels"])}_chunks'
    os.makedirs(cdir, exist_ok=True)
    return cdir


# Datasets written per chunk and re-assembled in the zip step.
_COARSE_MAP_KEYS = ['kSZ', 'kSZ_no_attenuation', 'tau_CMB', 'tau_weighted_global',
                    'kSZ_v', 'kSZ_deltav', 'kSZ_chiv', 'kSZ_chideltav']
_LEGACY_MAP_KEYS = ['kSZ_spatial_tau_legacy', 'tau_weighted_spatial_legacy']


def _write_chunk(ds, ix, iy, products):
    cdir = _chunk_dir(ds)
    with h5py.File(f'{cdir}/kSZ.{ix}.{iy}.hdf5', 'w') as f:
        for key in _COARSE_MAP_KEYS:
            f.create_dataset(key, data=products[key].astype(np.float32), dtype=np.float32)
        if SAVE_LEGACY_SPATIAL_TAU:
            for key in _LEGACY_MAP_KEYS:
                if key in products:
                    f.create_dataset(key, data=products[key].astype(np.float32), dtype=np.float32)


def _calc_chunk_wrapper(args):
    return calc_kSZ_new_chunk(*args)


def _calc_chunk_combined_wrapper(args):
    return calc_kSZ_new_chunk_combined(*args)


def _node_range(n_work, node, n_nodes):
    n_per = n_work // n_nodes
    n_rem = n_work % n_nodes
    f1, f2 = node * n_per, (node + 1) * n_per
    if node < n_rem:
        f1 += node
        f2 += node + 1
    else:
        f1 += n_rem
        f2 += n_rem
    return f1, f2


def calc_kSZs_new_chunked_combined(ds_highz, ds_lowz, pm_highz, pm_lowz,
                                   processes=1, node=0, n_nodes=1):
    """Drive the chunked combined map production across a node's share of chunks."""
    n = int(ds_highz['NumPixels'])
    n_chunks = n // CHUNK_SIZE
    assert n % CHUNK_SIZE == 0, f"n = {n}, CHUNK_SIZE = {CHUNK_SIZE}"
    f1, f2 = _node_range(n_chunks**2, node, n_nodes)
    work = [(ds_highz, ds_lowz, chunk, pm_highz, pm_lowz) for chunk in range(f1, f2)]
    if processes == 1:
        for args in tqdm(work, desc="kSZ_new chunks"):
            _calc_chunk_combined_wrapper(args)
    else:
        with Pool(processes=processes) as pool:
            for _ in tqdm(pool.imap_unordered(_calc_chunk_combined_wrapper, work),
                          total=len(work), desc="kSZ_new chunks"):
                pass


def zip_kSZs_new_chunked(ds, fine_meta, source_files=None):
    """Assemble per-chunk files into the final kSZ_rlc_<N>.hdf5 with the new layout."""
    n, n_split = int(ds['NumPixels']), ds['n_split']
    n_chunks = n // CHUNK_SIZE
    cdir = _chunk_dir(ds)

    keys = list(_COARSE_MAP_KEYS)
    if SAVE_LEGACY_SPATIAL_TAU:
        keys += _LEGACY_MAP_KEYS
    products = {key: np.zeros((n_split, n, n), dtype=np.float32) for key in keys}
    for ix in range(n_chunks):
        x1, x2 = ix * CHUNK_SIZE, (ix + 1) * CHUNK_SIZE
        for iy in range(n_chunks):
            y1, y2 = iy * CHUNK_SIZE, (iy + 1) * CHUNK_SIZE
            cf = f'{cdir}/kSZ.{ix}.{iy}.hdf5'
            if not os.path.exists(cf):
                raise FileNotFoundError(
                    f'missing chunk file {cf}: not all chunks were computed '
                    f'(expected {n_chunks**2} chunk files in {cdir}).')
            with h5py.File(cf, 'r') as f:
                for key in keys:
                    if key in f:
                        products[key][:, x1:x2, y1:y2] = f[key][:]
    return write_kSZ_new(ds, products, fine_meta, source_files=source_files)


def clean_kSZs_new(ds):
    n = int(ds['NumPixels'])
    n_chunks = n // CHUNK_SIZE
    cdir = _chunk_dir(ds)
    for ix in range(n_chunks):
        for iy in range(n_chunks):
            p = f'{cdir}/kSZ.{ix}.{iy}.hdf5'
            if os.path.exists(p):
                os.remove(p)
    # remove the (now empty) chunk directory too
    try:
        os.rmdir(cdir)
    except OSError:
        pass


# ===========================================================================
# Plane-means cache (shared by the multi-node chunk tasks and the merge step)
# ===========================================================================
# The per-fine-layer plane means / global tau profile are (nz,) arrays that are EXPENSIVE
# to compute at high resolution (they stream the whole density+ionization cube), but tiny
# to store. Computing them once in a cheap prep step and caching them here avoids every
# array task re-streaming the full cube (which would multiply the shared-FS read load by
# the number of array tasks).
_PM_ARRAY_KEYS = ['rho_factor', 'rho_mean', 'chi_mean', 'chi_rho_mean', 'A_mean',
                  'tau_plane_mean', 'tau_to_obs_global', 'exp_tau_global', 'weight_global']
_PM_SCALAR_KEYS = ['tau_total', 'foreground_tau_global']


def _pm_cache_path(resolution):
    return f'{get_results_dir(resolution)}/kSZ_new_rlc_{int(resolution)}_planemeans.hdf5'


def save_pm_cache(path, pm_highz, pm_lowz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for tag, pm in [('highz', pm_highz), ('lowz', pm_lowz)]:
            g = f.create_group(tag)
            for k in _PM_ARRAY_KEYS:
                g.create_dataset(k, data=np.asarray(pm[k], dtype=np.float64))
            for k in _PM_SCALAR_KEYS:
                g.attrs[k] = float(pm[k])
    print(f'[PREP] wrote plane-means cache -> {path}')


def load_pm_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'plane-means cache {path} not found: run the --prep step first.')
    out = {}
    with h5py.File(path, 'r') as f:
        for tag in ('highz', 'lowz'):
            g = f[tag]
            pm = {k: g[k][:] for k in _PM_ARRAY_KEYS}
            for k in _PM_SCALAR_KEYS:
                pm[k] = float(g.attrs[k])
            out[tag] = pm
    return out['highz'], out['lowz']


# ===========================================================================
# Phased chunked workflow for high resolution (>= 640):
#   PREP-ACCUMULATE : SLURM array; each task streams its share of the 100 chunks (both
#                     regions) and writes a small partial-sums file. Chunk-parallel, so it
#                     has the same footprint/wall-time character as COMPUTE, NOT one serial
#                     whole-cube pass.
#   PREP-REDUCE     : sum the partial files (cheap), finalize plane means + global tau
#                     profile, cache them, delete the partial files.
#   COMPUTE         : SLURM array; each task computes its share of the 100 chunks -> temp files
#   MERGE           : assemble temp chunk files into the final HDF5, verify, then delete temps
# ===========================================================================
def _pm_parts_dir(resolution):
    d = f'{get_results_dir(resolution)}/kSZ_new_rlc_{int(resolution)}_planemeans_parts'
    os.makedirs(d, exist_ok=True)
    return d


def prep_accumulate(resolution, node, n_nodes):
    """PREP-ACCUMULATE: this task's share of the chunk-range plane-sum reads, both regions.

    Chunk-parallelizable and read-only: reads ONLY this task's [f1,f2) chunk range x
    DEPTH_FILES depth blocks (same partition as COMPUTE), for BOTH regions (they share
    NumPixels/n_chunks), and writes the partial (sum_rho, sum_chi, sum_chi_rho) sums to a
    small temp file. No cross-task communication needed until PREP-REDUCE.
    """
    ds_highz, ds_lowz = get_regions(resolution)
    n = int(ds_highz['NumPixels'])
    if n < 640:
        raise ValueError(f'phased prep workflow is for resolution >= 640, got {n}')
    n_chunks = n // CHUNK_SIZE
    f1, f2 = _node_range(n_chunks**2, node, n_nodes)
    print(f'[PREP-ACCUMULATE] node {node}/{n_nodes}: chunks [{f1}, {f2}) of {n_chunks**2}')

    part = f'{_pm_parts_dir(resolution)}/part_{node}_of_{n_nodes}.hdf5'
    with h5py.File(part, 'w') as f:
        for tag, ds in [('highz', ds_highz), ('lowz', ds_lowz)]:
            sr, sc, scr = _stream_plane_sums_chunk_range(ds, f1, f2)
            g = f.create_group(tag)
            g.create_dataset('sum_rho', data=sr)
            g.create_dataset('sum_chi', data=sc)
            g.create_dataset('sum_chi_rho', data=scr)
    print(f'[PREP-ACCUMULATE] node {node}/{n_nodes} wrote {part}')


def prep_reduce(resolution, clean_parts=True):
    """PREP-REDUCE: sum all partial files, finalize plane means, cache, clean up."""
    ds_highz, ds_lowz = get_regions(resolution)
    n = int(ds_highz['NumPixels'])
    npix = float(n * n)
    pdir = _pm_parts_dir(resolution)

    parts = sorted(p for p in os.listdir(pdir) if p.startswith('part_') and p.endswith('.hdf5'))
    if not parts:
        raise FileNotFoundError(f'no partial-sum files found in {pdir}: run '
                                'prep_accumulate (PREP-ACCUMULATE, e.g. via the '
                                'array job) first.')
    # verify the accumulate array's coverage is complete and non-overlapping.
    n_nodes_seen = {int(p.split('_of_')[1].split('.hdf5')[0]) for p in parts}
    if len(n_nodes_seen) != 1:
        raise ValueError(f'inconsistent n_nodes across partial files: {n_nodes_seen}')
    n_nodes = n_nodes_seen.pop()
    nodes_present = sorted(int(p.split('part_')[1].split('_of_')[0]) for p in parts)
    if nodes_present != list(range(n_nodes)):
        raise ValueError(f'incomplete accumulate array: expected nodes 0..{n_nodes-1}, '
                         f'found {nodes_present}')
    print(f'[PREP-REDUCE] summing {len(parts)} partial files (n_nodes={n_nodes}) ...')

    nz_highz, nz_lowz = len(ds_highz['dl']), len(ds_lowz['dl'])
    sums = {
        'highz': [np.zeros(nz_highz), np.zeros(nz_highz), np.zeros(nz_highz)],
        'lowz': [np.zeros(nz_lowz), np.zeros(nz_lowz), np.zeros(nz_lowz)],
    }
    for p in parts:
        with h5py.File(f'{pdir}/{p}', 'r') as f:
            for tag in ('highz', 'lowz'):
                sums[tag][0] += f[tag]['sum_rho'][:]
                sums[tag][1] += f[tag]['sum_chi'][:]
                sums[tag][2] += f[tag]['sum_chi_rho'][:]

    print('[PREP-REDUCE] finalizing low-z plane means / global tau profile ...')
    pm_lowz = finalize_plane_means(ds_lowz, *sums['lowz'], npix, foreground_tau_global=0.0)
    print('[PREP-REDUCE] finalizing high-z plane means (low-z tau as foreground) ...')
    pm_highz = finalize_plane_means(ds_highz, *sums['highz'], npix,
                                    foreground_tau_global=pm_lowz['tau_total'])
    save_pm_cache(_pm_cache_path(resolution), pm_highz, pm_lowz)

    if clean_parts:
        for p in parts:
            os.remove(f'{pdir}/{p}')
        try:
            os.rmdir(pdir)
        except OSError:
            pass
        print(f'[PREP-REDUCE] deleted {len(parts)} partial-sum files in {pdir}')


def prep_plane_means(resolution):
    """Serial fallback: stream the whole cube in one task (slow at 640+; prefer the
    prep_accumulate/prep_reduce array workflow for production). Kept for < 640 or debugging.
    """
    ds_highz, ds_lowz = get_regions(resolution)
    print('[PREP] computing low-z plane means / global tau profile ...')
    pm_lowz = compute_density_plane_means(ds_lowz, foreground_tau_global=0.0)
    print('[PREP] computing high-z plane means (low-z tau as foreground) ...')
    pm_highz = compute_density_plane_means(ds_highz, foreground_tau_global=pm_lowz['tau_total'])
    save_pm_cache(_pm_cache_path(resolution), pm_highz, pm_lowz)


def compute_chunks(resolution, node, n_nodes, processes=1):
    """COMPUTE: this task's share of the chunk files (no assembly)."""
    ds_highz, ds_lowz = get_regions(resolution)
    n = int(ds_highz['NumPixels'])
    if n < 640:
        raise ValueError(f'phased chunk workflow is for resolution >= 640, got {n}')
    pm_highz, pm_lowz = load_pm_cache(_pm_cache_path(resolution))
    n_chunks = n // CHUNK_SIZE
    f1, f2 = _node_range(n_chunks**2, node, n_nodes)
    print(f'[COMPUTE] node {node}/{n_nodes}: chunks [{f1}, {f2}) of {n_chunks**2} '
          f'(processes={processes})')
    calc_kSZs_new_chunked_combined(ds_highz, ds_lowz, pm_highz, pm_lowz,
                                   processes=processes, node=node, n_nodes=n_nodes)
    print(f'[COMPUTE] node {node}/{n_nodes} done')


def _verify_merged(output_file, ds):
    """Sanity-check the assembled file before the temp chunk files are deleted."""
    n, n_split = int(ds['NumPixels']), ds['n_split']
    with h5py.File(output_file, 'r') as f:
        coarse = f['coarse']
        for key in _COARSE_MAP_KEYS:
            if key not in coarse:
                raise ValueError(f'verify: /coarse/{key} missing in {output_file}')
            shp = coarse[key].shape
            if shp != (n_split, n, n):
                raise ValueError(f'verify: /coarse/{key} shape {shp} != '
                                 f'{(n_split, n, n)}')
        # finiteness spot-check on the summed kSZ map (first + last coarse bins)
        for b in (0, n_split - 1):
            sl = coarse['kSZ'][b]
            if not np.all(np.isfinite(sl)):
                raise ValueError(f'verify: non-finite values in /coarse/kSZ bin {b}')
    print(f'[MERGE] verification OK: /coarse maps are ({n_split},{n},{n}) and finite.')
    return True


def merge_chunks(resolution, clean_parts=False):
    """MERGE: assemble all chunk files, verify, then (optionally) delete temp files."""
    ds_highz, ds_lowz = get_regions(resolution)
    pm_highz, pm_lowz = load_pm_cache(_pm_cache_path(resolution))
    ds = combine_kSZ_metadata(ds_highz, ds_lowz)
    fine_meta = _stack_fine_metadata(
        build_fine_metadata(ds_highz, pm_highz, source_region=0),
        build_fine_metadata(ds_lowz, pm_lowz, source_region=1),
    )
    cdir = _chunk_dir(ds)
    output_file = zip_kSZs_new_chunked(
        ds, fine_meta, source_files=[ds_highz['filename'], ds_lowz['filename']])
    _verify_merged(output_file, ds)
    if clean_parts:
        clean_kSZs_new(ds)
        print(f'[MERGE] deleted temporary chunk files in {cdir}')
    else:
        print(f'[MERGE] temp chunk files kept in {cdir} '
              '(pass --clean-parts to delete after a verified merge).')
    return output_file


# ===========================================================================
# Fine-shell metadata (E5) and coarse z metadata
# ===========================================================================
def get_fine_geometry(ds):
    """Return per-fine-layer (z, dz, D_comoving[cMpc]) for one region from its header."""
    with h5py.File(ds['filename'], 'r') as f:
        Redshifts = f['Redshifts'][:].astype(np.float64)
        Distances = f['Distances'][:].astype(np.float64)
    z = 0.5 * (Redshifts[1:] + Redshifts[:-1])
    dz = Redshifts[:-1] - Redshifts[1:]
    dist_cMpc = 1e-3 * Distances / ds['HubbleParam']
    D_comoving = 0.5 * (dist_cMpc[1:] + dist_cMpc[:-1])
    return z, dz, D_comoving


def build_fine_metadata(ds, plane_means, source_region=0, bin_offset=0):
    """Assemble the /fine scalar metadata for one region."""
    z, dz, D_comoving = get_fine_geometry(ds)
    nz = len(z)
    z_indices = np.asarray(ds['z_indices'])
    # coarse_bin_index[j] = coarse bin containing fine layer j.
    coarse_bin_index = np.clip(
        np.searchsorted(z_indices, np.arange(nz), side='right') - 1,
        0, ds['n_split'] - 1).astype(np.int32) + bin_offset
    return {
        'z': z.astype(np.float32),
        'dz': dz.astype(np.float32),
        'D_comoving': D_comoving.astype(np.float32),
        'coarse_bin_index': coarse_bin_index,
        'source_region': np.full(nz, source_region, dtype=np.int32),
        'chi_mean': plane_means['chi_mean'].astype(np.float32),
        'rho_mean': plane_means['rho_mean'].astype(np.float32),
        'A_mean': plane_means['A_mean'].astype(np.float32),
        'tau_plane_mean': plane_means['tau_plane_mean'].astype(np.float32),
        'tau_to_obs_global': plane_means['tau_to_obs_global'].astype(np.float32),
        'exp_tau_global': plane_means['exp_tau_global'].astype(np.float32),
    }


def _stack_fine_metadata(highz_meta, lowz_meta):
    """Stack high-z then low-z fine metadata in far->near depth order."""
    out = {}
    n_split_highz = int(highz_meta['coarse_bin_index'].max()) + 1
    lowz_meta = dict(lowz_meta)
    lowz_meta['coarse_bin_index'] = lowz_meta['coarse_bin_index'] + n_split_highz
    for key in highz_meta:
        out[key] = np.concatenate([highz_meta[key], lowz_meta[key]])
    return out


# ===========================================================================
# I. HDF5 writer
# ===========================================================================
def write_kSZ_new(ds, products, fine_meta, source_files=None):
    """Write the new physically-organized kSZ HDF5 product.

    Top-level compat datasets (kSZ, kSZ_no_attenuation, tau_CMB, z_edges, D_edges) are
    hard-linked to their /coarse counterparts so the existing downstream plotting code keeps
    working unchanged, while the default kSZ now uses the global-mean attenuation.
    """
    kSZ_tot = np.sum(products['kSZ'], axis=0)
    tau_CMB_tot = np.sum(products['tau_CMB'], axis=0)
    print(f'kSZ dimensionless mean/std/min/max = {np.mean(kSZ_tot):g} {np.std(kSZ_tot):g} {np.min(kSZ_tot):g} {np.max(kSZ_tot):g}')
    print(f'kSZ uK mean/std/min/max = {np.mean(kSZ_tot)*2.7255e6:g} {np.std(kSZ_tot)*2.7255e6:g} {np.min(kSZ_tot)*2.7255e6:g} {np.max(kSZ_tot)*2.7255e6:g}')
    print(f'tau_CMB mean/std/min/max = {np.mean(tau_CMB_tot):g} {np.std(tau_CMB_tot):g} {np.min(tau_CMB_tot):g} {np.max(tau_CMB_tot):g}')

    results_dir = get_results_dir(ds['NumPixels'])
    os.makedirs(results_dir, exist_ok=True)
    output_file = f'{results_dir}/kSZ_rlc_{int(ds["NumPixels"])}.hdf5'
    print(output_file)

    z_centers = 0.5 * (np.asarray(ds['z_edges'])[:-1] + np.asarray(ds['z_edges'])[1:])

    with h5py.File(output_file, 'w') as f:
        # ---- attributes ----
        for key in ['n_split', 'z_min', 'z_max', 'D_min', 'D_max', 'BoxSize', 'HubbleParam',
                    'NumPixels', 'Omega0', 'OmegaBaryon', 'OmegaLambda',
                    'OpeningAngleRad', 'OpeningAngleDeg',
                    'UnitLength_in_cm', 'UnitMass_in_g', 'UnitVelocity_in_cm_per_s']:
            if key in ds:
                f.attrs[key] = ds[key]
        for key in ['n_split_highz', 'n_split_lowz', 'rlc_dir_highz', 'rlc_dir_lowz',
                    'filename_highz', 'filename_lowz']:
            if key in ds:
                f.attrs[key] = ds[key]
        f.attrs['attenuation_default'] = "global_mean_tau"
        f.attrs['attenuation_description'] = ATTENUATION_DESCRIPTION
        f.attrs['symmetry_status'] = SYMMETRY_STATUS
        f.attrs['spectrum_type'] = "2D transverse plane spectra"
        f.attrs['kSZ_mean'] = np.mean(kSZ_tot)
        f.attrs['kSZ_std'] = np.std(kSZ_tot)
        f.attrs['kSZ_min'] = np.min(kSZ_tot)
        f.attrs['kSZ_max'] = np.max(kSZ_tot)
        f.attrs['tau_CMB_mean'] = np.mean(tau_CMB_tot)
        f.attrs['tau_CMB_std'] = np.std(tau_CMB_tot)
        if source_files is not None:
            f.attrs['source_files'] = '\n'.join(source_files)

        # ---- top-level geometry (compat) ----
        f.create_dataset('z_edges', data=ds['z_edges'], dtype=np.float32)
        f.create_dataset('D_edges', data=ds['D_edges'], dtype=np.float32)
        f.create_dataset('z_indices', data=np.asarray(ds['z_indices'], dtype=np.int32), dtype=np.int32)
        for key in ['z_indices_highz', 'z_indices_lowz']:
            if key in ds:
                f.create_dataset(key, data=np.asarray(ds[key], dtype=np.int32), dtype=np.int32)

        # ---- /coarse group ----
        coarse = f.create_group('coarse')
        coarse.create_dataset('z_edges', data=ds['z_edges'], dtype=np.float32)
        coarse.create_dataset('z_centers', data=z_centers, dtype=np.float32)
        coarse.create_dataset('D_edges', data=ds['D_edges'], dtype=np.float32)
        for key in _COARSE_MAP_KEYS:
            coarse.create_dataset(key, data=products[key].astype(np.float32), dtype=np.float32,
                                  compression='gzip', shuffle=True)
        if SAVE_LEGACY_SPATIAL_TAU:
            for key in _LEGACY_MAP_KEYS:
                if key in products:
                    coarse.create_dataset(key, data=products[key].astype(np.float32), dtype=np.float32,
                                          compression='gzip', shuffle=True)

        # ---- top-level compat datasets: hard links to /coarse ----
        f['kSZ'] = coarse['kSZ']
        f['kSZ_no_attenuation'] = coarse['kSZ_no_attenuation']
        f['tau_CMB'] = coarse['tau_CMB']
        # tau_CMB compat under its conventional name (was previously named tau_CMB in the old file)

        # ---- /fine group (per-fine-layer SCALAR metadata: tau profile + plane means) ----
        fine = f.create_group('fine')
        for key, val in fine_meta.items():
            fine.create_dataset(key, data=val)

    # Validation J4: global optical-depth sanity (cheap, from maps + plane means).
    tau_from_planes = float(np.sum(fine_meta['tau_plane_mean']))
    tau_from_maps = float(np.mean(tau_CMB_tot))
    print(f'[J4] mean total tau from fine planes = {tau_from_planes:g}')
    print(f'[J4] mean total tau from coarse tau_CMB maps = {tau_from_maps:g}')
    print(f'[J4] difference = {abs(tau_from_planes - tau_from_maps):g}')
    return output_file


# ===========================================================================
# Top-level drivers
# ===========================================================================
def run_combined(ds_highz, ds_lowz, processes=1, node=0, n_nodes=1):
    """Full combined high-z + low-z production: coarse maps only (STAGE 1)."""
    n = int(ds_highz['NumPixels'])

    # Plane means + global tau profile. Low-z first; its scalar total is the high-z foreground.
    print('[STAGE 1] computing plane means / global tau profile (low-z first)')
    pm_lowz = compute_density_plane_means(ds_lowz, foreground_tau_global=0.0)
    pm_highz = compute_density_plane_means(ds_highz, foreground_tau_global=pm_lowz['tau_total'])

    ds = combine_kSZ_metadata(ds_highz, ds_lowz)
    fine_meta = _stack_fine_metadata(
        build_fine_metadata(ds_highz, pm_highz, source_region=0),
        build_fine_metadata(ds_lowz, pm_lowz, source_region=1),
    )

    if n < 640:
        # Full-array (no chunking) path. Reuse the plane means already computed above so the
        # full cube is not loaded again just to recompute them.
        lowz, _ = calc_kSZ_new_all(ds_lowz, plane_means=pm_lowz)
        fg_spatial = lowz['tau_CMB_tot'][:, :, None] if SAVE_LEGACY_SPATIAL_TAU else None
        highz, _ = calc_kSZ_new_all(ds_highz, plane_means=pm_highz,
                                    foreground_tau_spatial=fg_spatial)
        products = _stack_products(highz, lowz)
        write_kSZ_new(ds, products, fine_meta,
                      source_files=[ds_highz['filename'], ds_lowz['filename']])
    else:
        calc_kSZs_new_chunked_combined(ds_highz, ds_lowz, pm_highz, pm_lowz,
                                       processes=processes, node=node, n_nodes=n_nodes)
        if n_nodes == 1:
            zip_kSZs_new_chunked(ds, fine_meta,
                                 source_files=[ds_highz['filename'], ds_lowz['filename']])
            clean_kSZs_new(ds)

    output_file = f'{get_results_dir(ds["NumPixels"])}/kSZ_rlc_{int(ds["NumPixels"])}.hdf5'
    return output_file


def run_single(ds, processes=1, node=0, n_nodes=1):
    """Single-region production (no low-z foreground)."""
    n = int(ds['NumPixels'])
    pm = compute_density_plane_means(ds, foreground_tau_global=0.0)
    fine_meta = build_fine_metadata(ds, pm, source_region=0)
    if n < 640:
        calc_kSZ_new_all_single(ds)
    else:
        n_chunks = n // CHUNK_SIZE
        f1, f2 = _node_range(n_chunks**2, node, n_nodes)
        work = [(ds, chunk, pm, None) for chunk in range(f1, f2)]
        if processes == 1:
            for args in tqdm(work, desc="kSZ_new chunks"):
                _calc_chunk_wrapper(args)
        else:
            with Pool(processes=processes) as pool:
                for _ in tqdm(pool.imap_unordered(_calc_chunk_wrapper, work),
                              total=len(work), desc="kSZ_new chunks"):
                    pass
        if n_nodes == 1:
            zip_kSZs_new_chunked(ds, fine_meta, source_files=[ds['filename']])
            clean_kSZs_new(ds)
    output_file = f'{get_results_dir(ds["NumPixels"])}/kSZ_rlc_{int(ds["NumPixels"])}.hdf5'
    return output_file


def _slurm_node_nnodes(cli_node, cli_nnodes):
    """Resolve (node, n_nodes): explicit CLI args win; else a SLURM array; else single."""
    if cli_node is not None and cli_nnodes is not None:
        return cli_node, cli_nnodes
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        node = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
            n_nodes = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        else:
            lo = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))
            hi = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0))
            n_nodes = hi - lo + 1
        return node, n_nodes
    return 0, 1


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='kSZ coarse-map production (calc_ksz_new). For resolution >= 640 use the '
                    'phased CHUNK-PARALLEL workflow: --prep-accumulate (SLURM array) then '
                    '--prep-reduce (once), then the SLURM array (COMPUTE, default mode), then '
                    '--merge. Every stage partitions the same 100 chunks, so none of them '
                    'require one task to stream the whole cube serially. --prep (serial, slow '
                    'at 640+) is kept only as a fallback/debug path. For resolution < 640 the '
                    'default runs the full single-node path.')
    ap.add_argument('--resolution', type=int, default=640)
    ap.add_argument('--prep', action='store_true',
                    help='(slow serial fallback) compute and cache plane means in one task.')
    ap.add_argument('--prep-accumulate', action='store_true',
                    help='PREP-ACCUMULATE: this array task\'s chunk-range partial plane sums.')
    ap.add_argument('--prep-reduce', action='store_true',
                    help='PREP-REDUCE: sum partial files, finalize + cache plane means.')
    ap.add_argument('--merge', action='store_true',
                    help='MERGE: assemble the per-chunk temp files into the final HDF5 and verify.')
    ap.add_argument('--clean-parts', action='store_true',
                    help='with --merge or --prep-reduce: delete temp files after verified use.')
    ap.add_argument('node', nargs='?', type=int, default=None,
                    help='(optional) explicit chunk-partition node index.')
    ap.add_argument('n_nodes', nargs='?', type=int, default=None,
                    help='(optional) explicit number of chunk-partition nodes.')
    args = ap.parse_args()

    resolution = args.resolution
    # chunk-worker pool size within one task (each worker reads its own HDF5 handles and
    # writes a private chunk file, so workers do not contend; file locking is disabled).
    processes = int(os.environ.get('KSZ_PROCESSES', '1'))
    node, n_nodes = _slurm_node_nnodes(args.node, args.n_nodes)

    if args.prep:
        prep_plane_means(resolution)
    elif args.prep_accumulate:
        prep_accumulate(resolution, node, n_nodes)
    elif args.prep_reduce:
        prep_reduce(resolution, clean_parts=args.clean_parts)
    elif args.merge:
        merge_chunks(resolution, clean_parts=args.clean_parts)
    else:
        ds_highz, ds_lowz = get_regions(resolution)
        n = int(ds_highz['NumPixels'])
        if n < 640:
            # low resolution: single-node full-array path (unchanged behavior)
            run_combined(ds_highz, ds_lowz, processes=processes, node=node, n_nodes=n_nodes)
        else:
            # high resolution: COMPUTE this task's share of the chunk temp files
            compute_chunks(resolution, node, n_nodes, processes=processes)
