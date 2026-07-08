"""compute_ksz_3d_snapshot_products.py -- 3D Cartesian snapshot extraction stage.

This is a *pure extraction* script. It reads the 3D Cartesian LUMINA grids (ren_320),
constructs the electron-momentum fields, measures all isotropically-binned 3D power
spectra needed downstream, and writes ONE complete HDF5 product. It does NOT:

    * perform the Limber integral or compute C_ell,
    * compare to the lightcone,
    * make any plots,
    * implement the disconnected Gaussian / Park-style convolution.

Those belong to the next analysis script. This script only builds the raw 3D products.

--------------------------------------------------------------------------------------
DATA CONVENTIONS (verified by inspecting the files; printed again at runtime)
--------------------------------------------------------------------------------------
Source roots (confirmed from calc_ksz_new.py + direct inspection):
    above z=4.75 : /orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/3d_cartesian_grid/ren_320
    below z=4.75 : /orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/3d_cartesian_grid/ren_320

Each field lives in its own subdirectory, one HDF5 file per snapshot, named
    <Field>/<Field>_<NNN>.hdf5
with a *global* zero-padded snapshot index. The "above" tree holds indices 000..428
(z ~ 29.4 down to 4.75) and the "below" tree holds indices 429..708 (z ~ 4.74 down to
2.99); the numbering is contiguous across the two trees.

Per-file layout:
    Density               : dataset 'Density'             shape (N,N,N)   float32
    DensityDM             : dataset 'DensityDM'           shape (N,N,N)   float32
    HII_VolumeFraction    : dataset 'HII_VolumeFraction'  shape (N,N,N)   float32
    HeII_VolumeFraction   : dataset 'HeII_VolumeFraction' shape (N,N,N)   float32
    HeIII_VolumeFraction  : dataset 'HeIII_VolumeFraction'shape (N,N,N)   float32
    Velocities            : dataset 'Velocities'          shape (N,N,N,3) float32
Every file carries a 'Header' group with attrs: BoxSize, HubbleParam, NumPixels,
Omega0, OmegaBaryon, OmegaLambda, Redshift, Time(=a), UnitLength_in_cm, UnitMass_in_g,
UnitVelocity_in_cm_per_s.

  * filename/redshift convention : <Field>_<NNN>.hdf5, redshift in Header.attrs['Redshift'];
    a = Header.attrs['Time'] = 1/(1+z). Index increases => redshift decreases.
  * array shape / dtype          : (N,N,N) float32 scalars; (N,N,N,3) float32 velocities.
  * fields at identical redshifts : YES. All field files for a given index derive from the
    same snapshot and share an identical Header (verified).
  * density convention           : 'Density' is the GAS mass density (separate DensityDM and
    DensityStars trees exist => Density is gas only, excluding DM and stars). Stored in code
    units; only delta = rho/rho_bar - 1 is used, so the absolute normalization is irrelevant.
  * velocity naming / units      : single 'Velocities' dataset, last axis = (vx,vy,vz) in
    Cartesian comoving-box coordinates. Code units convert to PHYSICAL PECULIAR velocity via
        v_phys[cm/s] = sqrt(a) * UnitVelocity_in_cm_per_s * v_code
    UnitVelocity_in_cm_per_s = 1e5 => v_phys[km/s] = sqrt(a) * v_code. We work in km/s.
    A nonzero box-wide bulk velocity is present and is removed before FFTing (k=0 only).
  * box size / metadata          : BoxSize in comoving kpc/h. Comoving box length
        L_box = BoxSize / HubbleParam / 1e3   [cMpc]   (= 500 cMpc for ren_320).
    Read from the header, never hard-coded.
  * HII / He fractions           : the *_VolumeFraction fields are NUMBER fractions per
    hydrogen (volume-weighted within each voxel). Evidence: max(HeII_VolumeFraction) and
    max(HeIII_VolumeFraction) ~ 0.0789 = HE_ABUND = (1/X - 1)/4 with X=0.76, i.e. the full
    helium-per-hydrogen abundance. Hence
        chi_e = x_HII + x_HeII + 2 x_HeIII = n_e / n_H
    is the correct electron-fraction convention (matches calc_ksz.py). A separate mass-
    weighted *_Fraction tree exists; we deliberately use the volume-weighted fields to match
    the established kSZ pipeline.
  * Density includes stars?      : NO. Gas only (see above).

--------------------------------------------------------------------------------------
FIELDS CONSTRUCTED PER SNAPSHOT
--------------------------------------------------------------------------------------
    delta(r)   = rho(r)/rho_bar - 1
    chi_e(r)   = x_HII + x_HeII + 2 x_HeIII                 (= n_e/n_H)
    chi(r)     = chi_e(r) - <chi_e>                         (mean-subtracted)
    A(r)       = chi_e(r) * (1 + delta(r))                  (electron-weighted density)
    q(r)       = A(r) * v(r)                                (electron momentum, vector)

--------------------------------------------------------------------------------------
SPECTRA  (isotropically binned in |k|; one common log-k grid; k=0 excluded)
--------------------------------------------------------------------------------------
Scalar:
    P_deltadelta   = <|delta_k|^2>
    P_chichi       = <|chi_k|^2>            (chi = chi_e - <chi_e>)
    P_chidelta     = <Re[chi_k delta_k*]>
    P_AA           = <|(A - <A>)_k|^2>
    P_AA_2pt       = chi_e_bar^2 P_deltadelta + P_chichi + 2 chi_e_bar P_chidelta
    P_AA_nonlinear = P_AA - P_AA_2pt
Velocity / momentum (v in km/s, q = A v):
    P_vv           = <|v_k|^2>              (full 3D velocity power, sum of 3 components)
    P_Av           = <Re[(A-<A>)_k  (k_hat . v_k)*]>   -> LONGITUDINAL scalar-velocity cross
                     i.e. this is P_{A v_parallel}, the cross of A with the longitudinal
                     (curl-free / line-of-sight along k_hat) velocity component.
    P_qperp_direct = <|q_k - k_hat (k_hat . q_k)|^2>   -> FULL two-component transverse power.
                     NOT divided by 2. The downstream Limber step keeps the explicit
                     P_qperp / 2 factor.
    P_qparallel    = <|k_hat . q_k|^2>      (longitudinal momentum, diagnostic)

FFT normalization (periodic box):
    F = numpy.fft.fftn(f)                       (unnormalized forward transform)
    P(k) = (L_box^3 / Ncells^2) * Re[F_a F_b*]  with Ncells = N^3
This yields the standard cosmology normalization: sum_k P(k) = V * <f^2> (Parseval), so
    P_deltadelta, P_chichi, P_AA : [cMpc]^3
    P_vv, P_qperp, P_qparallel   : [km/s]^2 [cMpc]^3
    P_Av                         : [km/s]   [cMpc]^3   (A dimensionless)

Output: ~/LUMINA_ksz/results/rlc_320/ksz_3d_snapshot_spectra_ren320.hdf5
"""

import os
import re
import sys
import glob
import time
import argparse

import numpy as np
import h5py

try:
    import hdf5plugin  # noqa: F401  (registers Blosc2 filters for the LUMINA grids)
except Exception as _e:  # pragma: no cover
    print(f"[warn] could not import hdf5plugin ({_e}); Blosc2-compressed reads may fail")

# scipy.fft is multithreaded (workers=); fall back to numpy.fft if unavailable.
try:
    import scipy.fft as _spfft
    _HAVE_SCIPY_FFT = True
except Exception:
    _HAVE_SCIPY_FFT = False

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
X = 0.76                          # primordial hydrogen mass fraction (LUMINA / calc_ksz.py)
HE_ABUND = (1.0 / X - 1.0) / 4.0  # n_He / n_H ~ 0.0789

def roots_for_resolution(res):
    """3D-Cartesian-grid source roots for a given render resolution (320 or 640).

    The above/below-z=4.75 trees share an identical directory layout at every
    resolution; only the trailing ren_<res> component changes.
    """
    base = "/orcd/data/mvogelsb/005/Lumina"
    above = f"{base}/Lumina_above_z_4p75/3d_cartesian_grid/ren_{res}"
    below = f"{base}/Lumina_below_z_4p75/3d_cartesian_grid/ren_{res}"
    return above, below


# Module-level roots (default ren_320); reassigned from --resolution in main().
DEFAULT_RESOLUTION = 320
ROOT_ABOVE, ROOT_BELOW = roots_for_resolution(DEFAULT_RESOLUTION)

REQUIRED_FIELDS = [
    "Density", "HII_VolumeFraction", "HeII_VolumeFraction",
    "HeIII_VolumeFraction", "Velocities",
]
OPTIONAL_FIELDS = ["DensityDM", "DensityStars"]

# k-binning defaults (configurable on the command line)
NK = 80
KMIN_FACTOR = 1.0            # kmin = KMIN_FACTOR * k_fund
KMAX_NYQUIST_FRACTION = 0.7  # kmax = KMAX_NYQUIST_FRACTION * k_nyquist

def default_out_for_resolution(res):
    """Canonical product path for a given resolution: results/rlc_<res>/..._ren<res>.hdf5."""
    return os.path.expanduser(
        f"~/LUMINA_ksz/results/rlc_{res}/ksz_3d_snapshot_spectra_ren{res}.hdf5")


OUT_DEFAULT = default_out_for_resolution(DEFAULT_RESOLUTION)

FFT_NORM_DESCRIPTION = (
    "Periodic-box FFT. F = numpy/scipy fftn(f) (unnormalized forward). "
    "P_ab(k) = (L_box^3 / Ncells^2) * Re[F_a conj(F_b)], Ncells = N^3, L_box in cMpc. "
    "Scalar fluctuation fields are mean-subtracted before the FFT; velocity/momentum have the "
    "box-wide mean (bulk) velocity removed. k=0 excluded. Convention gives sum_k P = V<f^2>; "
    "units: P_deltadelta/P_chichi/P_AA [cMpc^3]; P_vv/P_qperp/P_qparallel [km/s]^2 cMpc^3; "
    "P_Av [km/s] cMpc^3.")

FIELD_CONVENTION_DESCRIPTION = (
    "delta = rho/rho_bar - 1 (rho = GAS mass density, dir 'Density', stars/DM excluded). "
    "chi_e = x_HII + x_HeII + 2 x_HeIII = n_e/n_H, using the volume-weighted *_VolumeFraction "
    "fields, which are number fractions per hydrogen (max He fraction = HE_ABUND=(1/X-1)/4, "
    "X=0.76). chi = chi_e - <chi_e>. A = chi_e*(1+delta); q = A*v. v = physical peculiar "
    "velocity [km/s] = sqrt(a)*UnitVelocity_in_cm_per_s*v_code/1e5, box-mean removed. "
    "P_Av is the longitudinal cross P_{A v_parallel} = <Re[A_k (k_hat.v_k)*]>. "
    "P_qperp_direct is the FULL two-component transverse momentum power (NOT divided by 2).")


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------
def fftn(a, workers):
    if _HAVE_SCIPY_FFT:
        return _spfft.fftn(a, workers=workers)
    return np.fft.fftn(a)


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------
def discover_snapshots():
    """Return sorted list of (global_index, root, exists_dict) for all snapshots.

    Scans ONLY the Density subdirectory of each root (bounded), then checks the
    presence of every required/optional field for that index.
    """
    found = {}  # index -> root
    for root in (ROOT_ABOVE, ROOT_BELOW):
        ddir = os.path.join(root, "Density")
        if not os.path.isdir(ddir):
            print(f"[warn] missing Density dir: {ddir}")
            continue
        for p in glob.glob(os.path.join(ddir, "Density_*.hdf5")):
            m = re.search(r"Density_(\d+)\.hdf5$", os.path.basename(p))
            if m:
                found[int(m.group(1))] = root
    snaps = []
    for idx in sorted(found):
        root = found[idx]
        exists = {}
        for fld in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            fn = os.path.join(root, fld, f"{fld}_{idx:03d}.hdf5")
            exists[fld] = os.path.isfile(fn)
        snaps.append((idx, root, exists))
    return snaps


def field_path(root, fld, idx):
    return os.path.join(root, fld, f"{fld}_{idx:03d}.hdf5")


def read_header(root, idx):
    with h5py.File(field_path(root, "Density", idx), "r") as f:
        h = f["Header"]
        return {k: h.attrs[k] for k in h.attrs}


def apply_z_filter(usable, z_start, z_stop):
    """Restrict `usable` (sorted by global index, i.e. DECREASING redshift) to a
    redshift window: keep snapshots with z_stop <= z <= z_start.

    z_start is the *earliest-time* (highest-z) cut; z_stop the *latest-time*
    (lowest-z) cut. Both may be None (no cut on that side). Redshift is
    monotonically decreasing with global index (the simulation timeline), so we
    read at most the leading headers up to the z_start boundary and, when z_stop
    is set, stop scanning at the first snapshot below it -- keeping the I/O
    footprint bounded (only tiny HDF5 header reads, never the field data).
    """
    if z_start is None and z_stop is None:
        return usable
    kept = []
    started = z_start is None
    for (idx, root, exists) in usable:
        need_z = (not started) or (z_stop is not None)
        z = float(read_header(root, idx)["Redshift"]) if need_z else None
        if not started:
            if z <= z_start:
                started = True
            else:
                continue
        if z_stop is not None and z < z_stop:
            break  # monotonic: every later snapshot is at even lower z
        kept.append((idx, root, exists))
    return kept


def apply_selection(usable, args, verbose=False):
    """Apply --start/--stop (global index), the --z-start/--z-stop redshift window,
    then the coarsening --stride, returning the final ordered list of selected
    (idx, root, exists) tuples. Shared by the extraction and merge paths so the
    merge completeness check matches exactly what was requested."""
    sel = usable
    if args.start is not None:
        sel = [s for s in sel if s[0] >= args.start]
    if args.stop is not None:
        sel = [s for s in sel if s[0] <= args.stop]
    n_before_z = len(sel)
    sel = apply_z_filter(sel, args.z_start, args.z_stop)
    if verbose:
        print(f"    redshift window z in "
              f"[{args.z_stop if args.z_stop is not None else '-inf'}, "
              f"{args.z_start if args.z_start is not None else '+inf'}]: "
              f"{len(sel)} of {n_before_z} snapshots kept.")
    # Coarsen keeping every `stride`-th snapshot, ANCHORED AT THE LOW-Z END so the
    # dominant low-z / highest-kSZ-signal snapshot is always retained and only the
    # negligible high-z tail is thinned (matches fine_bin_tester.py's convention).
    if args.stride > 1:
        sel = sel[::-1][::args.stride][::-1]
    return sel


# ---------------------------------------------------------------------------
# k-grid / binning (built once; grid + box identical across all snapshots)
# ---------------------------------------------------------------------------
def build_k_grid(N, L_box, nk, kmin_factor, kmax_nyq_frac):
    """Build the isotropic |k| binning machinery for an N^3 periodic box of side L_box.

    Returns a dict with the 3D k-component grids, |k|, inverse |k|, bin assignment, and the
    per-bin centers/edges/Nmodes. k in 1/cMpc.
    """
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L_box / N)   # 1/cMpc
    kx = k1d[:, None, None]
    ky = k1d[None, :, None]
    kz = k1d[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)          # (N,N,N) f64

    k_fund = 2.0 * np.pi / L_box
    k_nyq = np.pi * N / L_box
    kmin = kmin_factor * k_fund
    kmax = kmax_nyq_frac * k_nyq
    k_edges = np.logspace(np.log10(kmin), np.log10(kmax), nk + 1)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])

    flat = kmag.ravel()
    bin_index = np.digitize(flat, k_edges) - 1           # 0..nk-1 valid
    valid = (bin_index >= 0) & (bin_index < nk) & (flat > 0)
    bidx_valid = bin_index[valid]
    Nmodes = np.bincount(bidx_valid, minlength=nk)[:nk].astype(np.int64)

    # unit-vector components (k_hat); safe at k=0 (set to 0, those modes are excluded)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_kmag = np.where(kmag > 0, 1.0 / kmag, 0.0)

    return {
        "k1d": k1d, "kx": kx, "ky": ky, "kz": kz,
        "kmag": kmag, "inv_kmag": inv_kmag,
        "k_fund": k_fund, "k_nyq": k_nyq, "kmin": kmin, "kmax": kmax,
        "k_centers": k_centers, "k_edges": k_edges,
        "valid": valid, "bidx_valid": bidx_valid, "Nmodes": Nmodes, "nk": nk,
    }


def bin_power(power3d, kg):
    """Isotropically bin a real 3D power array (units already applied) into k bins.

    Empty bins are NaN (never 0), per spec.
    """
    flat = power3d.ravel()[kg["valid"]]
    sums = np.bincount(kg["bidx_valid"], weights=flat, minlength=kg["nk"])[:kg["nk"]]
    Nm = kg["Nmodes"]
    out = np.full(kg["nk"], np.nan, dtype=np.float64)
    nz = Nm > 0
    out[nz] = sums[nz] / Nm[nz]
    return out


# ---------------------------------------------------------------------------
# Field readers
# ---------------------------------------------------------------------------
def read_scalar(root, fld, idx):
    with h5py.File(field_path(root, fld, idx), "r") as f:
        return f[fld][:].astype(np.float64)


def read_chi_e(root, idx):
    """chi_e = x_HII + x_HeII + 2 x_HeIII (= n_e/n_H)."""
    xe = read_scalar(root, "HII_VolumeFraction", idx)
    xe += read_scalar(root, "HeII_VolumeFraction", idx)
    xe += 2.0 * read_scalar(root, "HeIII_VolumeFraction", idx)
    return xe


def read_velocity_kms(root, idx, a):
    """Physical peculiar velocity [km/s] = sqrt(a)*UnitVelocity_in_cm_per_s*v_code/1e5.

    Returns three (N,N,N) f64 arrays (vx, vy, vz) with the box-wide mean removed.
    """
    with h5py.File(field_path(root, "Velocities", idx), "r") as f:
        uvel = float(f["Header"].attrs["UnitVelocity_in_cm_per_s"])
        vel = f["Velocities"][:]  # (N,N,N,3) f32
    fac = np.sqrt(a) * uvel / 1.0e5  # code -> km/s
    vx = vel[..., 0].astype(np.float64) * fac
    vy = vel[..., 1].astype(np.float64) * fac
    vz = vel[..., 2].astype(np.float64) * fac
    del vel
    vx -= vx.mean(); vy -= vy.mean(); vz -= vz.mean()
    return vx, vy, vz


# ---------------------------------------------------------------------------
# Per-snapshot spectrum computation
# ---------------------------------------------------------------------------
def process_snapshot(idx, root, kg, L_box, workers, do_parseval=False):
    """Compute all fields + spectra for one snapshot. Returns (scalars_dict, spectra_dict)."""
    Ncells = float(kg["kmag"].size)
    norm = (L_box ** 3) / (Ncells ** 2)
    N = kg["kmag"].shape[0]

    hdr = read_header(root, idx)
    z = float(hdr["Redshift"])
    a = float(hdr["Time"])

    # ---- scalar fields ----
    rho = read_scalar(root, "Density", idx)
    rho_mean = float(rho.mean())
    delta = rho / rho_mean - 1.0
    del rho

    chi_e = read_chi_e(root, idx)
    chi_e_mean = float(chi_e.mean())
    chi = chi_e - chi_e_mean                 # mean-subtracted electron fraction

    A = chi_e * (1.0 + delta)                # = chi_e * rho/rho_bar
    A_mean = float(A.mean())
    A_fluct = A - A_mean
    del chi_e

    # ---- scalar spectra ----
    Fd = fftn(delta, workers)
    Pdd = bin_power(norm * (Fd.real ** 2 + Fd.imag ** 2), kg)

    Fc = fftn(chi, workers)
    Pcc = bin_power(norm * (Fc.real ** 2 + Fc.imag ** 2), kg)
    Pcd = bin_power(norm * (Fc.real * Fd.real + Fc.imag * Fd.imag), kg)
    del Fd, Fc

    FA = fftn(A_fluct, workers)
    PAA = bin_power(norm * (FA.real ** 2 + FA.imag ** 2), kg)

    # lowest-order electron-structure approximation + nonlinear residual
    PAA2 = chi_e_mean ** 2 * Pdd + Pcc + 2.0 * chi_e_mean * Pcd
    PAA_nl = PAA - PAA2

    del delta, chi, A_fluct

    # ---- velocity ----
    vx, vy, vz = read_velocity_kms(root, idx, a)
    Fvx = fftn(vx, workers)
    Fvy = fftn(vy, workers)
    Fvz = fftn(vz, workers)
    Pvv = bin_power(norm * (Fvx.real ** 2 + Fvx.imag ** 2
                            + Fvy.real ** 2 + Fvy.imag ** 2
                            + Fvz.real ** 2 + Fvz.imag ** 2), kg)

    # longitudinal velocity in Fourier space: v_par(k) = k_hat . v_k
    v_par = (kg["kx"] * Fvx + kg["ky"] * Fvy + kg["kz"] * Fvz) * kg["inv_kmag"]
    # P_Av = longitudinal scalar-velocity cross = <Re[A_k v_par*]>
    PAv = bin_power(norm * (FA.real * v_par.real + FA.imag * v_par.imag), kg)
    del FA, Fvx, Fvy, Fvz, v_par

    # ---- momentum q = A v ----
    qx = A * vx; qy = A * vy; qz = A * vz
    del A, vx, vy, vz
    Fqx = fftn(qx, workers); del qx
    Fqy = fftn(qy, workers); del qy
    Fqz = fftn(qz, workers); del qz

    q2 = (Fqx.real ** 2 + Fqx.imag ** 2
          + Fqy.real ** 2 + Fqy.imag ** 2
          + Fqz.real ** 2 + Fqz.imag ** 2)
    q_par = (kg["kx"] * Fqx + kg["ky"] * Fqy + kg["kz"] * Fqz) * kg["inv_kmag"]
    q_par2 = q_par.real ** 2 + q_par.imag ** 2
    del Fqx, Fqy, Fqz, q_par

    Pqpar = bin_power(norm * q_par2, kg)
    Pqperp = bin_power(norm * (q2 - q_par2), kg)   # full two-component transverse (no /2)
    del q2, q_par2

    parseval = None
    if do_parseval:
        # Parseval sanity for delta: sum_k P(k) over ALL nonzero modes ?= V * <delta^2>
        # (use the full grid, not just binned range, for the identity)
        rho2 = read_scalar(root, "Density", idx)
        d2 = rho2 / rho2.mean() - 1.0
        var = float((d2 ** 2).mean())
        Fd2 = fftn(d2, workers)
        Pfull = norm * (Fd2.real ** 2 + Fd2.imag ** 2)
        Pfull.ravel()[0] = 0.0  # drop DC
        lhs = float(Pfull.sum())
        rhs = (L_box ** 3) * var
        parseval = (lhs, rhs)
        del rho2, d2, Fd2, Pfull

    scalars = {
        "z": z, "a": a, "rho_mean": rho_mean,
        "chi_e_mean": chi_e_mean, "A_mean": A_mean,
        # <A> vs <chi_e>: stored so downstream can see the covariance signal
        "A_mean_minus_chiebar": A_mean - chi_e_mean,
    }
    spectra = {
        "P_deltadelta": Pdd, "P_chichi": Pcc, "P_chidelta": Pcd,
        "P_AA": PAA, "P_AA_2pt": PAA2, "P_AA_nonlinear": PAA_nl,
        "P_vv": Pvv, "P_Av": PAv,
        "P_qperp_direct": Pqperp, "P_qparallel_direct": Pqpar,
    }
    return scalars, spectra, parseval


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SPECTRA_KEYS = [
    "P_deltadelta", "P_chichi", "P_chidelta",
    "P_AA", "P_AA_2pt", "P_AA_nonlinear",
    "P_vv", "P_Av", "P_qperp_direct", "P_qparallel_direct",
]
SCALAR_KEYS = ["z", "a", "rho_mean", "chi_e_mean", "A_mean"]
ROW_STR_KEYS = ["snapshot_density_file", "snapshot_velocity_file"]


# ---------------------------------------------------------------------------
# Snapshot-axis partitioning (job array) -- mirrors calc_ksz_new.py:_node_range
# ---------------------------------------------------------------------------
def _node_range(n_work, node, n_nodes):
    """Contiguous [f1, f2) share of n_work items for `node` of `n_nodes` (balanced)."""
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


def part_path(out, node, n_nodes):
    base, ext = os.path.splitext(out)
    return f"{base}.part_{node:03d}_of_{n_nodes:03d}{ext}"


def part_glob(out):
    base, ext = os.path.splitext(out)
    return f"{base}.part_*_of_*{ext}"


# ---------------------------------------------------------------------------
# HDF5 writer (shared by the per-task / serial path and reused for merge)
# ---------------------------------------------------------------------------
def write_product(out_path, L_box, N, cosmo, kg, args,
                  out_scalar, out_spec, global_index, dens_paths, vel_paths):
    """Write one HDF5 product (full file, a partial-block part, or the merged file).

    Every output -- partial or final -- carries the identical /metadata block and shared
    k/k_edges/Nmodes, plus the per-row /snapshots datasets for the rows it holds.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["box_size_comoving"] = L_box
        meta.attrs["box_size_comoving_unit"] = "cMpc"
        meta.attrs["grid_resolution"] = N
        meta.attrs["velocity_unit"] = (
            "physical peculiar km/s = sqrt(a)*UnitVelocity_in_cm_per_s*v_code/1e5")
        meta.attrs["density_unit"] = (
            "gas mass density, simulation code units; only delta=rho/rho_bar-1 is used")
        meta.attrs["fft_normalization_description"] = FFT_NORM_DESCRIPTION
        meta.attrs["field_convention_description"] = FIELD_CONVENTION_DESCRIPTION
        meta.attrs["k_unit"] = "1/cMpc"
        meta.attrs["NK"] = args.nk
        meta.attrs["KMIN_FACTOR"] = args.kmin_factor
        meta.attrs["KMAX_NYQUIST_FRACTION"] = args.kmax_nyq_frac
        meta.attrs["k_fund"] = kg["k_fund"]
        meta.attrs["k_nyquist"] = kg["k_nyq"]
        meta.attrs["source_root_above"] = ROOT_ABOVE
        meta.attrs["source_root_below"] = ROOT_BELOW
        meta.attrs["X_hydrogen_mass_fraction"] = X
        meta.attrs["HE_ABUND"] = HE_ABUND
        meta.attrs["P_Av_convention"] = (
            "longitudinal scalar-velocity cross P_{A v_parallel} = <Re[A_k (k_hat.v_k)*]>")
        meta.attrs["P_qperp_convention"] = (
            "full two-component transverse momentum power |q_perp|^2; NOT divided by 2")
        cosmo_grp = meta.create_group("cosmology_parameters")
        for k, v in cosmo.items():
            cosmo_grp.attrs[k] = v

        snap = f.create_group("snapshots")
        # shared k axis (common grid: identical box & resolution for all snapshots)
        snap.create_dataset("k", data=kg["k_centers"].astype(np.float64))
        snap.create_dataset("k_edges", data=kg["k_edges"].astype(np.float64))
        snap.create_dataset("Nmodes", data=kg["Nmodes"].astype(np.int64))
        # per-snapshot scalar metadata, shape (nrows,)
        for k in SCALAR_KEYS:
            snap.create_dataset(k, data=out_scalar[k].astype(np.float64))
        # spectra, shape (nrows, nk)
        for k in SPECTRA_KEYS:
            snap.create_dataset(k, data=out_spec[k].astype(np.float64),
                                compression="gzip", shuffle=True)
        snap.create_dataset("global_index", data=np.asarray(global_index, dtype=np.int64))
        snap.create_dataset("snapshot_density_file",
                            data=np.array(dens_paths, dtype=object), dtype=str_dt)
        snap.create_dataset("snapshot_velocity_file",
                            data=np.array(vel_paths, dtype=object), dtype=str_dt)


def _write_product_atomic(out_path, *rest):
    """write_product to a temp file, then atomically replace out_path.

    A checkpoint interrupted mid-write leaves the previous good part file
    untouched (os.replace is atomic on POSIX), so a time-limit or preemption
    kill can never corrupt or truncate the accumulated results.
    """
    tmp = f"{out_path}.tmp{os.getpid()}"
    try:
        write_product(tmp, *rest)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _load_part_rows(out_path):
    """Load rows from an existing (possibly interrupted) part file, for resume.

    Returns a dict of arrays/lists, or None if the file is absent or unreadable
    (in which case the task simply recomputes its whole block from scratch).
    """
    if not os.path.isfile(out_path):
        return None
    try:
        with h5py.File(out_path, "r") as f:
            s = f["snapshots"]
            return {
                "scalar": {k: s[k][:] for k in SCALAR_KEYS},
                "spec": {k: s[k][:] for k in SPECTRA_KEYS},
                "gidx": s["global_index"][:],
                "dens": list(s["snapshot_density_file"].asstr()[:]),
                "vel": list(s["snapshot_velocity_file"].asstr()[:]),
            }
    except Exception as e:
        print(f"[resume] existing part {os.path.basename(out_path)} unreadable "
              f"({e}); recomputing this task's block from scratch.")
        return None


# ---------------------------------------------------------------------------
# Merge mode: assemble part files into the single canonical product
# ---------------------------------------------------------------------------
def do_merge(args):
    parts = sorted(glob.glob(part_glob(args.out)))
    if not parts:
        print(f"ERROR: no part files match {part_glob(args.out)}")
        sys.exit(1)
    print(f"[merge] found {len(parts)} part files for {args.out}")

    row_scalar = {k: [] for k in SCALAR_KEYS}
    row_spec = {k: [] for k in SPECTRA_KEYS}
    gidx, dpaths, vpaths = [], [], []
    k_ref = kedges_ref = nmodes_ref = None
    for p in parts:
        with h5py.File(p, "r") as f:
            s = f["snapshots"]
            n_rows = s["global_index"].shape[0]
            print(f"    {os.path.basename(p)}: {n_rows} rows")
            for k in SCALAR_KEYS:
                row_scalar[k].append(s[k][:])
            for k in SPECTRA_KEYS:
                row_spec[k].append(s[k][:])
            gidx.append(s["global_index"][:])
            dpaths.append(s["snapshot_density_file"].asstr()[:])
            vpaths.append(s["snapshot_velocity_file"].asstr()[:])
            kk = s["k"][:]
            if k_ref is None:
                k_ref, kedges_ref, nmodes_ref = kk, s["k_edges"][:], s["Nmodes"][:]
            elif not np.allclose(kk, k_ref):
                print(f"ERROR: k grid in {p} differs from reference; refusing to merge.")
                sys.exit(1)

    gidx = np.concatenate(gidx)
    order = np.argsort(gidx, kind="stable")
    out_scalar = {k: np.concatenate(row_scalar[k])[order] for k in SCALAR_KEYS}
    out_spec = {k: np.concatenate(row_spec[k])[order] for k in SPECTRA_KEYS}
    dens_paths = list(np.concatenate(dpaths)[order])
    vel_paths = list(np.concatenate(vpaths)[order])
    gidx_sorted = gidx[order]

    # completeness check vs the SELECTED snapshots (same --start/--stop/--z-start/
    # --z-stop/--stride the extraction used), not the full usable set -- otherwise
    # any intentional subset run would always report the excluded snapshots as
    # "missing" and (below) block --clean-parts. (warn, do not crash)
    usable = [(idx, root, ex) for idx, root, ex in discover_snapshots()
              if all(ex[fld] for fld in REQUIRED_FIELDS)]
    selected_idx = set(idx for idx, _, _ in apply_selection(usable, args))
    merged_idx = set(int(g) for g in gidx_sorted)
    missing = sorted(selected_idx - merged_idx)
    dups = len(gidx_sorted) - len(merged_idx)
    if missing:
        print(f"[merge] WARNING: {len(missing)} usable snapshots are NOT in the merged set: "
              f"{missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if dups:
        print(f"[merge] WARNING: {dups} duplicate global indices across parts.")

    # rebuild a minimal kg + metadata from a reference part's header (no recompute needed)
    ref_idx = int(gidx_sorted[0])
    ref_root = ROOT_ABOVE if os.path.isfile(field_path(ROOT_ABOVE, "Density", ref_idx)) else ROOT_BELOW
    hdr = read_header(ref_root, ref_idx)
    N = int(hdr["NumPixels"])
    L_box = float(hdr["BoxSize"]) / float(hdr["HubbleParam"]) / 1.0e3
    cosmo = {k: float(hdr[k]) for k in
             ["HubbleParam", "Omega0", "OmegaBaryon", "OmegaLambda",
              "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s", "BoxSize"]}
    kg = {"k_centers": k_ref, "k_edges": kedges_ref, "Nmodes": nmodes_ref,
          "k_fund": 2.0 * np.pi / L_box, "k_nyq": np.pi * N / L_box}

    write_product(args.out, L_box, N, cosmo, kg, args,
                  out_scalar, out_spec, gidx_sorted, dens_paths, vel_paths)
    print(f"[merge] wrote {len(gidx_sorted)} snapshots -> {args.out}")
    if args.clean_parts and not missing and not dups:
        for p in parts:
            os.remove(p)
        print(f"[merge] removed {len(parts)} part files (--clean-parts).")


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help="3D render resolution N, e.g. 320/640/1280/2560 (sets the ren_<N> "
                         "source roots + default --out results/rlc_<N>/...; default 320)")
    ap.add_argument("--start", type=int, default=None, help="first global snapshot index")
    ap.add_argument("--stop", type=int, default=None, help="last global snapshot index (inclusive)")
    ap.add_argument("--z-start", type=float, default=15.0, dest="z_start",
                    help="max redshift to include (drop z > z_start, the high-z/early-time "
                         "tail that carries negligible D3000 signal); default 15.0")
    ap.add_argument("--z-stop", type=float, default=None, dest="z_stop",
                    help="min redshift to include (drop z < z_stop); default: no low-z cut "
                         "(process down to the last snapshot, z~2.99)")
    ap.add_argument("--stride", type=int, default=10, dest="stride",
                    help="snapshot coarsening stride: keep every Nth snapshot, anchored at the "
                         "low-z (highest-signal) end; default 10 "
                         "(<0.2%% D3000 error vs stride 1, see fine_bin_tester.py)")
    ap.add_argument("--nk", type=int, default=NK)
    ap.add_argument("--kmin-factor", type=float, default=KMIN_FACTOR)
    ap.add_argument("--kmax-nyq-frac", type=float, default=KMAX_NYQUIST_FRACTION)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                    help="FFT worker threads (default: SLURM_CPUS_PER_TASK, else all cores)")
    ap.add_argument("--node", type=int, default=None,
                    help="this task's index in a job array (default: SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--n-nodes", type=int, default=None,
                    help="total job-array size (default: SLURM_ARRAY_TASK_COUNT, else 1)")
    ap.add_argument("--merge", action="store_true",
                    help="merge mode: assemble <out>.part_*_of_*.hdf5 into the single <out>")
    ap.add_argument("--clean-parts", action="store_true",
                    help="in --merge mode, delete part files after a complete merge")
    ap.add_argument("--out", type=str, default=None,
                    help="output HDF5 path (default: results/rlc_<res>/..._ren<res>.hdf5)")
    args = ap.parse_args()

    # Resolve resolution-dependent source roots + default output before anything
    # else, so both the merge and extraction paths see the right resolution.
    global ROOT_ABOVE, ROOT_BELOW
    ROOT_ABOVE, ROOT_BELOW = roots_for_resolution(args.resolution)
    if args.out is None:
        args.out = default_out_for_resolution(args.resolution)

    print("=" * 78)
    print("compute_ksz_3d_snapshot_products.py  --  3D Cartesian snapshot extraction")
    print("=" * 78)
    print(f"resolution = ren_{args.resolution}")
    print(f"source roots:\n    above: {ROOT_ABOVE}\n    below: {ROOT_BELOW}")
    print(f"output = {args.out}")

    if args.merge:
        do_merge(args)
        return

    # resolve job-array partitioning (CLI overrides SLURM array env)
    node = args.node if args.node is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_nodes = (args.n_nodes if args.n_nodes is not None
               else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    if n_nodes < 1 or not (0 <= node < n_nodes):
        print(f"ERROR: bad partition node={node} n_nodes={n_nodes}")
        sys.exit(1)

    print(f"FFT backend: {'scipy.fft (workers=%d)' % args.workers if _HAVE_SCIPY_FFT else 'numpy.fft'}")
    print(f"Partition: node {node} of {n_nodes}")

    # --- discover snapshots ---
    snaps = discover_snapshots()
    if not snaps:
        print("ERROR: no snapshots discovered.")
        sys.exit(1)
    all_idx = [s[0] for s in snaps]
    print(f"\n[1] Discovered {len(snaps)} snapshots, global index "
          f"{min(all_idx)}..{max(all_idx)} across two redshift trees.")

    # field availability summary
    usable = []
    for idx, root, exists in snaps:
        ok = all(exists[f] for f in REQUIRED_FIELDS)
        if not ok:
            missing = [f for f in REQUIRED_FIELDS if not exists[f]]
            print(f"    snapshot {idx:03d}: MISSING {missing} -> skipped")
        else:
            usable.append((idx, root, exists))
    print(f"    {len(usable)} snapshots have all required fields "
          f"{REQUIRED_FIELDS}.")
    opt_report = {f: sum(1 for _, _, e in usable if e[f]) for f in OPTIONAL_FIELDS}
    print(f"    optional-field availability (count): {opt_report}")
    if not usable:
        print("ERROR: no usable snapshots.")
        sys.exit(1)

    # apply global-index start/stop, redshift window, and coarsening stride
    # (full usable set when everything is unset).
    sel_all = apply_selection(usable, args, verbose=True)
    if not sel_all:
        print("ERROR: selection is empty.")
        sys.exit(1)

    # --- grid / box / units from the first USABLE snapshot (partition-independent) ---
    hdr0 = read_header(usable[0][1], usable[0][0])
    N = int(hdr0["NumPixels"])
    h = float(hdr0["HubbleParam"])
    L_box = float(hdr0["BoxSize"]) / h / 1.0e3   # ckpc/h -> cMpc
    uvel = float(hdr0["UnitVelocity_in_cm_per_s"])
    cosmo = {k: float(hdr0[k]) for k in
             ["HubbleParam", "Omega0", "OmegaBaryon", "OmegaLambda",
              "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s", "BoxSize"]}

    # --- this task's contiguous block of the selected set ---
    if n_nodes > 1:
        f1, f2 = _node_range(len(sel_all), node, n_nodes)
        sel = sel_all[f1:f2]
        out_path = part_path(args.out, node, n_nodes)
    else:
        sel = sel_all
        out_path = args.out

    print(f"\n[2] Selected set: {len(sel_all)} snapshots "
          f"(start={args.start}, stop={args.stop}, z_start={args.z_start}, "
          f"z_stop={args.z_stop}, stride={args.stride}); "
          f"this task handles {len(sel)} of them.")
    if not sel:
        # legitimate empty block when n_nodes > len(sel_all): write an empty part so the
        # array task still succeeds and merge sees a (0-row) contribution.
        kg = build_k_grid(N, L_box, args.nk, args.kmin_factor, args.kmax_nyq_frac)
        empty_scalar = {k: np.zeros(0) for k in SCALAR_KEYS}
        empty_spec = {k: np.zeros((0, args.nk)) for k in SPECTRA_KEYS}
        write_product(out_path, L_box, N, cosmo, kg, args,
                      empty_scalar, empty_spec, np.zeros(0, dtype=np.int64), [], [])
        print(f"[note] empty block for node {node}; wrote 0-row part -> {out_path}")
        return
    z_first = float(read_header(sel[0][1], sel[0][0])["Redshift"])
    z_last = float(read_header(sel[-1][1], sel[-1][0])["Redshift"])
    print(f"    block global index {sel[0][0]}..{sel[-1][0]}, "
          f"z {z_first:.4f} -> {z_last:.4f}")

    print(f"\n[3] Grid / box / units:")
    print(f"    grid resolution N            = {N}")
    print(f"    BoxSize (header, ckpc/h)     = {hdr0['BoxSize']}")
    print(f"    L_box comoving               = {L_box:.4f} cMpc")
    print(f"    cell size                    = {L_box / N:.5f} cMpc")
    print(f"    velocity unit                = physical peculiar km/s "
          f"(sqrt(a)*{uvel:g}*v_code/1e5)")
    print(f"    density unit                 = gas mass density, code units (delta uses mean)")
    print(f"    cosmology                    = Om0={cosmo['Omega0']:.5f} "
          f"Ob={cosmo['OmegaBaryon']:.5f} OL={cosmo['OmegaLambda']:.5f} h={h:.4f}")

    # --- k grid ---
    kg = build_k_grid(N, L_box, args.nk, args.kmin_factor, args.kmax_nyq_frac)
    print(f"\n[5] k-binning:")
    print(f"    k_fund = 2pi/L_box           = {kg['k_fund']:.6f} 1/cMpc")
    print(f"    k_nyquist = pi*N/L_box       = {kg['k_nyq']:.6f} 1/cMpc")
    print(f"    valid k-range [kmin,kmax]    = [{kg['kmin']:.6f}, {kg['kmax']:.6f}] 1/cMpc")
    print(f"    NK = {args.nk}, empty bins -> NaN; k=0 excluded")
    print(f"    Nmodes per bin: min={kg['Nmodes'][kg['Nmodes']>0].min()}, "
          f"max={kg['Nmodes'].max()}, total={kg['Nmodes'].sum()}")

    # --- outputs: accumulated incrementally so the part file can be checkpointed
    #     after every snapshot and an interrupted task resumed without recompute ---
    nk = args.nk
    acc_scalar = {k: [] for k in SCALAR_KEYS}
    acc_spec = {k: [] for k in SPECTRA_KEYS}
    acc_gidx, dens_paths, vel_paths = [], [], []

    # resume: reuse rows already present in a previous (possibly interrupted) part.
    # Filter to this task's current assignment so a changed array size can't inject
    # stale rows from a different partitioning.
    sel_idx_set = {s[0] for s in sel}
    done_idx = set()
    if n_nodes > 1:
        prev = _load_part_rows(out_path)
        if prev is not None:
            keep = [j for j, g in enumerate(prev["gidx"]) if int(g) in sel_idx_set]
            for k in SCALAR_KEYS:
                acc_scalar[k] = [prev["scalar"][k][j] for j in keep]
            for k in SPECTRA_KEYS:
                acc_spec[k] = [prev["spec"][k][j] for j in keep]
            acc_gidx = [int(prev["gidx"][j]) for j in keep]
            dens_paths = [prev["dens"][j] for j in keep]
            vel_paths = [prev["vel"][j] for j in keep]
            done_idx = set(acc_gidx)
            print(f"[resume] existing part {os.path.basename(out_path)} has "
                  f"{len(done_idx)} completed snapshots; skipping those.")

    n_todo = sum(1 for (idx, _, _) in sel if idx not in done_idx)

    def _snapshot_arrays():
        return ({k: np.asarray(acc_scalar[k]) for k in SCALAR_KEYS},
                {k: np.asarray(acc_spec[k]) for k in SPECTRA_KEYS})

    print(f"\n[4]/[6] Per-snapshot fields, spectra, and consistency checks:")
    if done_idx:
        print(f"    {n_todo} of {len(sel)} snapshots still to compute after resume.")
    print(f"{'idx':>4} {'z':>8} {'rho_bar':>11} {'chi_e_bar':>10} {'A_bar':>10} "
          f"{'<d>~0':>10} {'P_AA~sum':>10}")

    n_new = 0
    for (idx, root, _) in sel:
        if idx in done_idx:
            continue
        first_row = (len(acc_gidx) == 0)
        scl, spc, pars = process_snapshot(
            idx, root, kg, L_box, args.workers, do_parseval=first_row)
        for k in SCALAR_KEYS:
            acc_scalar[k].append(scl[k])
        for k in SPECTRA_KEYS:
            acc_spec[k].append(spc[k])
        acc_gidx.append(idx)
        dens_paths.append(field_path(root, "Density", idx))
        vel_paths.append(field_path(root, "Velocities", idx))

        # consistency checks (on the row just appended)
        delta_mean = scl["rho_mean"] / scl["rho_mean"] - 1.0  # exactly 0 by construction
        # P_AA reconstruction residual (should be ~0: nonlinear is defined as the residual)
        with np.errstate(invalid="ignore"):
            recon = spc["P_AA_2pt"] + spc["P_AA_nonlinear"]
            denom = np.nanmax(np.abs(spc["P_AA"]))
            rel = np.nanmax(np.abs(spc["P_AA"] - recon)) / (denom if denom > 0 else 1.0)
        print(f"{idx:>4} {scl['z']:>8.4f} {scl['rho_mean']:>11.4e} "
              f"{scl['chi_e_mean']:>10.5f} {scl['A_mean']:>10.5f} "
              f"{delta_mean:>10.2e} {rel:>10.2e}")

        # checkpoint: atomically rewrite the part after every snapshot so a
        # time-limit or preemption kill loses at most the in-flight snapshot.
        if n_nodes > 1:
            cs, cp = _snapshot_arrays()
            _write_product_atomic(out_path, L_box, N, cosmo, kg, args,
                                  cs, cp, acc_gidx, dens_paths, vel_paths)

        # live progress / ETA (keeps the SLURM log informative)
        n_new += 1
        if n_new % 10 == 0 or n_new == n_todo:
            el = time.time() - t0
            rate = el / n_new
            eta = rate * (n_todo - n_new)
            print(f"      [progress] {n_new}/{n_todo} new  elapsed={el/60:.1f} min  "
                  f"rate={rate:.1f} s/snap  ETA={eta/60:.1f} min")
        sys.stdout.flush()

        if first_row:
            # sample spectrum values + Parseval sanity, only once (cheap, informative)
            j = nk // 2
            print(f"      sample @ k={kg['k_centers'][j]:.4f} 1/cMpc: "
                  f"P_dd={spc['P_deltadelta'][j]:.4e}  "
                  f"P_AA={spc['P_AA'][j]:.4e}  "
                  f"P_qperp={spc['P_qperp_direct'][j]:.4e}")
            print(f"      <A> = {scl['A_mean']:.6f},  <chi_e>*<1+delta> = {scl['chi_e_mean']:.6f}"
                  f"  (difference = chi_e-delta covariance = {scl['A_mean']-scl['chi_e_mean']:+.4e})")
            if pars is not None:
                lhs, rhs = pars
                print(f"      Parseval check (delta): sum_k P = {lhs:.5e},  V*<d^2> = {rhs:.5e}, "
                      f"ratio = {lhs/rhs:.6f}")

    # --- final write ---
    #  * serial path (n_nodes==1): accumulated in memory, write the single file once.
    #  * array-task path (n_nodes>1): the part was checkpointed after each snapshot;
    #    if nothing new was computed this run, normalize the part once (also covers
    #    a pure resume and the empty-block case).
    nsnap = len(acc_gidx)
    fs, fp = _snapshot_arrays()
    if n_nodes == 1:
        print(f"\n[7] Writing output HDF5 -> {out_path}")
        write_product(out_path, L_box, N, cosmo, kg, args,
                      fs, fp, acc_gidx, dens_paths, vel_paths)
    elif n_new == 0:
        print(f"\n[7] No new snapshots this run; finalizing part -> {out_path}")
        _write_product_atomic(out_path, L_box, N, cosmo, kg, args,
                              fs, fp, acc_gidx, dens_paths, vel_paths)
    else:
        print(f"\n[7] Output HDF5 checkpointed incrementally -> {out_path}")

    dt = time.time() - t0
    per = (dt / n_new) if n_new else float("nan")
    print(f"\n[8] Done. {nsnap} snapshots in part ({n_new} computed this run) in "
          f"{dt:.1f} s ({per:.2f} s/snapshot new). Output: {out_path}")
    if n_nodes > 1:
        print(f"    (partial part; run `--merge --out {args.out}` after all "
              f"{n_nodes} tasks finish)")


if __name__ == "__main__":
    main()
