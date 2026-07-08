import numpy as np
from scipy.special import erf, dawsn
import h5py
import sys
import os
import argparse
import shutil
from multiprocessing import Pool

#KILOMETER = 1e5             # Kilometer [cm]
#MEGAPARSEC = 3.085678e24    # Megaparsec [cm]
#CLIGHT = 2.99792458e10      # Speed of light [cm/s]
#GRAVITY = 6.6738e-8         # Gravitational constant [cgs]
BOLTZMANN = 1.38065e-16     # Boltzmann's constant [g cm^2/sec^2/k]
#PLANCK = 6.6260695e-27      # Planck's constant [erg sec]
PROTONMASS = 1.67262178e-24 # Mass of hydrogen atom [g]
#LAMBDA_0 = 1215.6e-8        # Lya wavelength (1215.6 Angstroms) [cm]
#NU_0 = CLIGHT / LAMBDA_0    # Lya frequency [Hz]
#E_0 = PLANCK * NU_0         # Lya energy [erg]
#T_0 = E_0 / BOLTZMANN       # Lya temperature [K]
HYDROGEN_MASSFRAC = 0.76    # Mass fraction of hydrogen
#HE_ABUND = (1./HYDROGEN_MASSFRAC - 1.) / 4. # Helium abundance
GAMMA = 5. / 3.             # Adiabatic index of simulated gas
GAMMA_MINUS1 = GAMMA - 1.   # For convenience

CHUNK_SIZE = 128  # Chunk size
DEPTH_SIZE = 8192  # Depth size
TARGET_RESOLUTION = 5120  # Target resolution
TARGET_DEPTH = 32081  # Target depth
DEPTH_FILES = 4  # Number of depth files
# DEPTH_FILES = 2  # Number of depth files

# Constants
Msun = 1.988435e33         # Solar mass [g]
c = 2.99792458e10          # Speed of light [cm/s]
km = 1e5                   # Units: 1 km  = 1e5  cm
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
mH = 1.6735327e-24         # Mass of hydrogen atom (g)
me = 9.109382917e-28       # Electron mass [g]
ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]
X  = 0.76                  # Primordial hydrogen mass fraction
f12 = 0.4162               # Oscillator strength
nu0 = 2.466e15             # Lya frequency [Hz]
lambda0 = 1e8 * c / nu0    # Lya wavelength [Angstroms]
DnuL = 9.936e7             # Natural line width [Hz]
kappa_dust = 7.177e4       # Lya dust opacity [cm^2/g dust]

# Thermal velocity: vth = sqrt(2 kB T / mH)
vth_div_sqrtT = np.sqrt(2. * kB / mH)
# Doppler width: DnuD = nu0 vth / c
DnuD_div_sqrtT = nu0 * vth_div_sqrtT / c
# "damping parameter": a = DnuL / 2 DnuD
a_sqrtT = 0.5 * DnuL / DnuD_div_sqrtT
# Cross section: sigma0_sqrtT = sigma0 * sqrt(T)
sigma0_sqrtT = np.sqrt(np.pi/2.) * f12 * ee**2 * np.sqrt(mH/kB) / (me * nu0)

def get_los_unit_vectors(n, opening_angle, x1=None, x2=None, y1=None, y2=None):
    """Construct LOS unit vectors for the square light-cone grid.

    The LOS direction points outward from the observer. The central ray is +z.
    """
    idx = np.arange(n, dtype=np.float64)
    if x1 is not None:
        idx_x = idx[x1:x2]
        idx_y = idx[y1:y2]
    else:
        idx_x = idx_y = idx
    theta_x = (idx_x + 0.5 - 0.5 * n) * opening_angle / n
    theta_y = (idx_y + 0.5 - 0.5 * n) * opening_angle / n

    tx = np.tan(theta_x)
    ty = np.tan(theta_y)

    nx = tx[:, None]
    ny = ty[None, :]
    nz = np.ones((len(idx_x), len(idx_y)), dtype=np.float64)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    return nx, ny, nz

def project_los_velocity(vel, s, nx, ny, nz, z1=0):
    """Project Cartesian velocities onto LOS direction and return cm/s."""
    z2 = z1 + vel.shape[2]
    vel = vel.astype(np.float64)
    vel *= s['velocity_to_cgs'][None, None, z1:z2, None]
    return (
        vel[..., 0] * nx[:, :, None]
        + vel[..., 1] * ny[:, :, None]
        + vel[..., 2] * nz[:, :, None]
    )

def calculate_tau_edges(hdf5_file, z0, dir_path, chunk):
    s = h5py.File(hdf5_file, 'r')
    header = dict(s['Header'].attrs)
    # Determine chunk boundaries
    n = header['NumPixels']  # Number of pixels
    # assert n >= 1280, f"NumPixels = {n}, must be >= 1280"
    n_chunks = np.max([n // CHUNK_SIZE, 4])  # Number of chunks in each dimension
    chunk_size = n // n_chunks # For low-res
    n_degrade = TARGET_RESOLUTION // n  # Number of times the data was degraded
    nz = TARGET_DEPTH // n_degrade  # Number of redshift slices
    n_depth = DEPTH_SIZE // n_degrade  # Number of depth slices
    assert n_degrade > 0, f"n_degrade = {n_degrade}"
    assert n_degrade * n == TARGET_RESOLUTION, f"n_degrade * n = {n_degrade * n}, TARGET_RESOLUTION = {TARGET_RESOLUTION}"
    ix = chunk // n_chunks
    iy = chunk % n_chunks
    assert 0 <= ix < n_chunks, f"ix = {ix}, n_chunks = {n_chunks}"
    assert 0 <= iy < n_chunks, f"iy = {iy}, n_chunks = {n_chunks}"
    x1, x2 = ix * chunk_size, (ix + 1) * chunk_size
    y1, y2 = iy * chunk_size, (iy + 1) * chunk_size
    # Extract necessary data and parameters from the HDF5 file
    h = header['HubbleParam']
    OmegaB = header['OmegaBaryon']
    Omega0 = header['Omega0']
    UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
    UnitLength_in_cm = header['UnitLength_in_cm']
    UnitMass_in_g = header['UnitMass_in_g']
    Ts = s['Temperature'][x1:x2,y1:y2,:].astype(np.float64) # Gas temperature [K] #3D
    zs = np.copy(s['Redshifts'])
    densities = s['Density'][x1:x2,y1:y2,:].astype(np.float64)
    x_HIs = 1. - s['HII_Fraction'][x1:x2,y1:y2,:].astype(np.float64) # Neutral hydrogen fraction [code units] #3D
    v_cells = s['Velocities'][x1:x2,y1:y2,:].astype(np.float64) # Gas velocity [code units] #3D
    # Conversions
    zmids = 0.5 * (zs[:-1] + zs[1:]) # Redshift intervals
    a_scale = 1. / (1. + zs[:-1])
    velocity_to_cgs = (np.sqrt(a_scale) * UnitVelocity_in_cm_per_s)
    length_to_cgs = a_scale * UnitLength_in_cm / h
    volume_to_cgs = length_to_cgs**3
    mass_to_cgs = UnitMass_in_g / h
    density_to_cgs = mass_to_cgs / volume_to_cgs
    # v_cells requires special care
    opening_angle = float(header['OpeningAngle'])
    nx, ny, nz = get_los_unit_vectors(n, opening_angle, x1=x1, x2=x2, y1=y1, y2=y2)
    z1 = 0
    z2 = z1 + v_cells.shape[2]
    v_cells *= velocity_to_cgs[None, None, z1:z2, None]
    v_cells = (v_cells[..., 0] * nx[:, :, None] + 
               v_cells[..., 1] * ny[:, :, None] +
               v_cells[..., 2] * nz[:, :, None])
    # Constants
    H0 = h * 100. * km / Mpc
    X = 0.76 # Hydrogen mass fraction
    G = 6.6725985e-8 # Gravitational constant [cm^3/g/s^2]
    f12 = 0.4162 # Lyman-alpha oscillator strength
    DnuL = 9.936e7 # Lyman-alpha natural line width
    nu0 = 2.466e15 # Lyman-alpha line frequency [Hz]
    n_freq = 801
    # Derived quantities
    n_H = X * densities * density_to_cgs / mH
    Hz = H0 * np.sqrt(Omega0) * (1. + zs[:-1])**(3./2.) # Hubble parameter [s^-1] 
    vth = vth_div_sqrtT * np.sqrt(Ts) #3D
    DvD = vth * nu0 / c #3D
    Ks = Hz / vth #3D
    sigma0 = f12 * np.sqrt(np.pi) * ee**2 / (me * vth * nu0) #3D
    k0 = x_HIs * n_H * sigma0 #3D #fix n_H!!!!!
    a = DnuL / (2. * DvD) #3D
    dls = c * (zs[:-1] - zs[1:]) / Hz / (1. + zs[:-1]) # Comoving line element [cm]
    # Mask so that integration begins at the source
    i0 = np.argmax(zs < z0)
    if i0 > 0:
        i0 -= 1
    zs = zs[i0:-1, None]
    z0 = zs[0]
    zmids = zmids[i0:, None]
    v_cells = v_cells[:,:,i0:, None]
    dls = dls[i0:, None]
    Ks = Ks[:,:,i0:, None]
    k0 = k0[:,:,i0:, None]
    vth = vth[:,:,i0:, None]
    a = a[:,:,i0:, None]
    # Velocity offsets
    Dv_min_kms = -2000.
    Dv_max_kms = 2000.
    Dvs = np.linspace(Dv_min_kms*km, Dv_max_kms*km, n_freq)#[None, None, None, :] # Initial frequency offset [cm/s]
    num_freq_ranges = 5
    freq_range_edges = [-2000, -500, -100, 101, 501, 2001]
    freq_range_indices = [np.argmin(np.abs(Dvs - freq_range_edges[i]*km)) + int(np.round(i/num_freq_ranges)) 
                          for i in range(len(freq_range_edges))]
    tau_band_avgs = []
    for i_bin in range(num_freq_ranges):
        i_freq_start = freq_range_indices[i_bin]
        i_freq_range = freq_range_indices[i_bin+1] - freq_range_indices[i_bin]
        # Calculate optical depths (vectorized)
        Dvs_band = Dvs[None, None, None, i_freq_start:i_freq_start+i_freq_range]
        Dv_zs = c * ((Dvs_band/c + 1) * (1 + z0)/(1 + zs) - 1)
        x = -(Dv_zs + v_cells) / vth
        dtau = (np.sqrt(np.pi) * k0 / (2 * Ks) * (erf(x) - erf(x - Ks*dls)) +
                2 * a * k0 / (np.sqrt(np.pi) * Ks) * (dawsn(x - Ks*dls) - dawsn(x))) # [x, y, z, freq]
        taus = np.sum(dtau, axis=2) # [x, y, freq]
        # Transform to transmission space
        transmissions = np.exp(-taus)
        # Take band averages
        transmission_band_avg = np.sum(transmissions, axis=-1) / i_freq_range # [x, y]
        # Back to tau space
        tau_band_avg = np.log(transmission_band_avg)
        tau_band_avgs.append(tau_band_avg) # tau_band_avgs: [band, x, y]
    # Create file
    with h5py.File(os.path.join(dir_path, f'tau_map_{z0}_{chunk}.hdf5'), 'w') as f:
        f.attrs['HubbleParam'] = h
        f.attrs['NumFreq'] = np.int32(n_freq)
        f.attrs['Dv_min'] = Dv_min_kms
        f.attrs['Dv_max'] = Dv_max_kms
        f.attrs['Dv_local'] = 0
        f.attrs['Omega0'] = Omega0
        f.attrs['OmegaBaryon'] = OmegaB
        f.attrs['Redshift'] = z0
        f.attrs['Chunk'] = chunk
        f.create_dataset('tau_band_avgs', data=taus)
        f.create_dataset('Dvs', data=Dvs)
        f.create_dataset('freq_band_edges', data=freq_range_edges)
    return Dvs, taus
# store band-average tau maps for every half-integer redshift from 3 to 12

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "chunk",
        type=int,
        help="Chunk number"
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the input lightcone HDF5 file."
    )
    parser.add_argument(
        "z0",
        type=float,
        help="Source redshift"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory in which to store tau_maps"
    )
    # parser.add_argument(
    #     "--x1", type=int, default=None,
    #     help="Start index along x-axis (pixel slice). Default: None (full range)."
    # )
    # parser.add_argument(
    #     "--x2", type=int, default=None,
    #     help="End index along x-axis (pixel slice). Default: None (full range)."
    # )
    # parser.add_argument(
    #     "--y1", type=int, default=None,
    #     help="Start index along y-axis (pixel slice). Default: None (full range)."
    # )
    # parser.add_argument(
    #     "--y2", type=int, default=None,
    #     help="End index along y-axis (pixel slice). Default: None (full range)."
    # )
    return parser.parse_args()


def main():
    args = parse_args()

    chunk = args.chunk
    hdf5_file = args.hdf5_file
    z0 = args.z0

    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file not found: {hdf5_file}", file=sys.stderr)
        sys.exit(1)
    
    dir_path = "tau_maps"
    save_dir_arg = args.save_dir
    if save_dir_arg is not None:
        dir_path = os.path.join(save_dir_arg, dir_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    # with open(args.z0_file, 'r') as f:
    #     z0_list = [[float(x) for x in line.strip().split(',')] for line in f if line.strip()]

    # with h5py.File(hdf5_file, 'r') as s:
    #     n = dict(s['Header'].attrs)['NumPixels']  # Number of pixels
    # n_chunks = np.max([n // CHUNK_SIZE, 4])  # Number of chunks in each dimension
    # with Pool(processes=64) as pool:
    #     pool.starmap(calculate_tau_edges, [(hdf5_file, z0, dir_path, chunk) for chunk in range(n_chunks*n_chunks)])

    calculate_tau_edges(hdf5_file, z0, dir_path, chunk)


if __name__ == "__main__":
    main()
