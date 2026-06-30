import numpy as np
from scipy.special import erf, dawsn
import h5py
import sys
import os
import argparse
import shutil

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

CHUNK_SIZE = 40

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

def get_los_unit_vectors(s, x1=None, x2=None, y1=None, y2=None):
    """Construct LOS unit vectors for the square light-cone grid.

    The LOS direction points outward from the observer. The central ray is +z.
    """
    header = dict(s['Header'].attrs)
    n = int(header['NumPixels'])
    opening_angle = float(header['OpeningAngle'])
    theta = (np.arange(n, dtype=np.float64) + 0.5 - 0.5 * n) * opening_angle / n

    tx = np.tan(theta)
    ty = np.tan(theta)

    nx = tx[:, None]
    ny = ty[None, :]
    nz = np.ones((n, n), dtype=np.float64)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    if x1 is None:
        return nx, ny, nz
    return nx[x1:x2, y1:y2], ny[x1:x2, y1:y2], nz[x1:x2, y1:y2]

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

def calculate_tau_edges(s, z0, x1=None, x2=None, y1=None, y2=None):
    # Extract necessary data and parameters from the HDF5 file
    header = dict(s['Header'].attrs)
    h = header['HubbleParam']
    OmegaB = header['OmegaBaryon']
    Omega0 = header['Omega0']
    UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
    UnitLength_in_cm = header['UnitLength_in_cm']
    UnitMass_in_g = header['UnitMass_in_g']
    Ts = np.copy(s['Temperature'])[x1:x2,y1:y2,:] # Gas temperature [K] #3D
    zs = np.copy(s['Redshifts'])
    densities = np.copy(s['Density'])[x1:x2,y1:y2,:]
    x_HIs = 1. - np.copy(s['HII_Fraction'])[x1:x2,y1:y2,:] # Neutral hydrogen fraction [code units] #3D
    v_cells = np.copy(s['Velocities'])[x1:x2,y1:y2,:] # Gas velocity [code units] #3D
    # Conversions
    zmids = 0.5 * (zs[:-1] + zs[1:]) # Redshift intervals
    a_scale = 1. / (1. + zs[:-1])
    xlen = Ts.shape[0]
    ylen = Ts.shape[1]
    velocity_to_cgs = (np.sqrt(a_scale) * UnitVelocity_in_cm_per_s)
    length_to_cgs = a_scale * UnitLength_in_cm / h
    volume_to_cgs = length_to_cgs**3
    mass_to_cgs = UnitMass_in_g / h
    density_to_cgs = mass_to_cgs / volume_to_cgs
    # v_cells requires special care
    nx, ny, nz = get_los_unit_vectors(s, x1=x1, x2=x2, y1=y1, y2=y2)
    z1 = 0
    z2 = z1 + v_cells.shape[2]
    v_cells = v_cells.astype(np.float64)
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
    rho_crit_0 = 3. * H0**2 / (8. * np.pi * G) # Present critical density = 3H0^2/8piG [g/cm^3]
    # n_H = X * OmegaB * rho_crit_0 * (1. + zs[:-1])**3 / mH # Physical hydrogen number density [cm^-3]
    n_H = X * densities * density_to_cgs / mH
    Hz = H0 * np.sqrt(Omega0) * (1. + zs[:-1])**(3./2.) # Hubble parameter [s^-1] 
    vth = vth_div_sqrtT * np.sqrt(Ts) #3D
    DvD = vth * nu0 / c #3D
    Ks = Hz / vth #3D
    sigma0 = f12 * np.sqrt(np.pi) * ee**2 / (me * vth * nu0) #3D
    k0 = x_HIs * n_H * sigma0 #3D #fix n_H!!!!!
    a = DnuL / (2. * DvD) #3D
    # ls = (ls[1:] - ls[:-1]) * length_to_cgs # Convert from code units to cm #250
    dls = c * (zs[:-1] - zs[1:]) / Hz / (1. + zs[:-1]) # Comoving line element [cm]
    # v_cells = v_cells * velocity_to_cgs
    # Mask so that integration begins at the source
    i0 = np.argmax(zs < z0)
    if i0 > 0:
        i0 -= 1
    zs = zs[i0:-1]
    z0 = zs[0]
    zmids = zmids[i0:]
    v_cells = v_cells[:,:,i0:]
    dls = dls[i0:]
    Ks = Ks[:,:,i0:]
    k0 = k0[:,:,i0:]
    vth = vth[:,:,i0:]
    a = a[:,:,i0:]
    # Velocity offsets
    Dv_min_kms = -2000.
    Dv_max_kms = 2000.
    Dvs = np.linspace(Dv_min_kms*km, Dv_max_kms*km, n_freq) # Initial frequency offset [cm/s]
    # Calculate optical depths
    taus = np.zeros((xlen, ylen, n_freq)) #3D
    for i in range(n_freq):
        Dv_zs = c * ((Dvs[i]/c + 1) * (1 + z0)/(1 + zs) - 1)
        x = -(Dv_zs + v_cells) / vth
        dtau = (np.sqrt(np.pi) * k0 / (2 * Ks) * (erf(x) - erf(x - Ks*dls)) +
                2 * a * k0 / (np.sqrt(np.pi) * Ks) * (dawsn(x - Ks*dls) - dawsn(x)))
        taus[:,:,i] = np.sum(dtau, axis=2) # Integrated damping wing optical depth
    # Create file
    with h5py.File(f'tau_maps/tau_map_{z0}.hdf5', 'w') as f:
        f.attrs['HubbleParam'] = h
        f.attrs['NumFreq'] = np.int32(n_freq)
        f.attrs['Dv_min'] = Dv_min_kms
        f.attrs['Dv_max'] = Dv_max_kms
        f.attrs['Dv_local'] = 0
        f.attrs['Omega0'] = Omega0
        f.attrs['OmegaBaryon'] = OmegaB
        f.attrs['Redshift'] = z0
        f.create_dataset('tau', data=taus)
        f.create_dataset('transmission', data=np.exp(-taus))
        f.create_dataset('Dvs', data=Dvs)
    return Dvs, taus

def calculate_tau_chunk(s, z0, chunk=0):
    header = dict(s['Header'].attrs)
    # Determine chunk boundaries
    n = h['NumPixels']
    n_chunks = n // CHUNK_SIZE
    ix = chunk // n_chunks
    iy = chunk % n_chunks
    x1, x2 = ix * CHUNK_SIZE, (ix + 1) * CHUNK_SIZE
    y1, y2 = iy * CHUNK_SIZE, (iy + 1) * CHUNK_SIZE
    # Extract necessary data and parameters from the HDF5 file
    h = header['HubbleParam']
    OmegaB = header['OmegaBaryon']
    Omega0 = header['Omega0']
    UnitVelocity_in_cm_per_s = header['UnitVelocity_in_cm_per_s']
    UnitLength_in_cm = header['UnitLength_in_cm']
    UnitMass_in_g = header['UnitMass_in_g']
    Ts = np.copy(s['Temperature'])[x1:x2,y1:y2,:] # Gas temperature [K] #3D
    zs = np.copy(s['Redshifts'])
    densities = np.copy(s['Density'])[x1:x2,y1:y2,:]
    x_HIs = 1. - np.copy(s['HII_Fraction'])[x1:x2,y1:y2,:] # Neutral hydrogen fraction [code units] #3D
    v_cells = np.copy(s['Velocities'])[x1:x2,y1:y2,:] # Gas velocity [code units] #3D
    # Conversions
    zmids = 0.5 * (zs[:-1] + zs[1:]) # Redshift intervals
    a_scale = 1. / (1. + zs[:-1])
    xlen = Ts.shape[0]
    ylen = Ts.shape[1]
    velocity_to_cgs = (np.sqrt(a_scale) * UnitVelocity_in_cm_per_s)
    length_to_cgs = a * UnitLength_in_cm / h
    volume_to_cgs = length_to_cgs**3
    mass_to_cgs = UnitMass_in_g / h
    density_to_cgs = mass_to_cgs / volume_to_cgs    # v_cells requires special care
    nx, ny, nz = get_los_unit_vectors(s, x1=x1, x2=x2, y1=y1, y2=y2)
    z1 = 0
    z2 = z1 + v_cells.shape[2]
    v_cells = v_cells.astype(np.float64)
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
    rho_crit_0 = 3. * H0**2 / (8. * np.pi * G) # Present critical density = 3H0^2/8piG [g/cm^3]
    # n_H = X * OmegaB * rho_crit_0 * (1. + zs[:-1])**3 / mH # Physical hydrogen number density [cm^-3]
    n_H = X * densities * density_to_cgs / mH
    Hz = H0 * np.sqrt(Omega0) * (1. + zs[:-1])**(3./2.) # Hubble parameter [s^-1] 
    vth = vth_div_sqrtT * np.sqrt(Ts) #3D
    DvD = vth * nu0 / c #3D
    Ks = Hz / vth #3D
    sigma0 = f12 * np.sqrt(np.pi) * ee**2 / (me * vth * nu0) #3D
    k0 = x_HIs * n_H * sigma0 #3D
    a = DnuL / (2. * DvD) #3D
    # ls = (ls[1:] - ls[:-1]) * length_to_cgs # Convert from code units to cm #250
    dls = c * (zs[:-1] - zs[1:]) / Hz / (1. + zs[:-1]) # Comoving line element [cm]
    # v_cells = v_cells * velocity_to_cgs
    # Mask so that integration begins at the source
    i0 = np.argmax(zs < z0)
    if i0 > 0:
        i0 -= 1
    zs = zs[i0:-1]
    z0 = zs[0]
    zmids = zmids[i0:]
    v_cells = v_cells[:,:,i0:]
    dls = dls[i0:]
    Ks = Ks[:,:,i0:]
    k0 = k0[:,:,i0:]
    vth = vth[:,:,i0:]
    a = a[:,:,i0:]
    # Velocity offsets
    Dv_min_kms = -2000.
    Dv_max_kms = 2000.
    Dvs = np.linspace(Dv_min_kms*km, Dv_max_kms*km, n_freq) # Initial frequency offset [cm/s]
    # Calculate optical depths
    taus = np.zeros((xlen, ylen, n_freq)) #3D
    for i in range(n_freq):
        Dv_zs = c * ((Dvs[i]/c + 1) * (1 + z0)/(1 + zs) - 1)
        x = -(Dv_zs + v_cells) / vth
        dtau = (np.sqrt(np.pi) * k0 / (2 * Ks) * (erf(x) - erf(x - Ks*dls)) +
                2 * a * k0 / (np.sqrt(np.pi) * Ks) * (dawsn(x - Ks*dls) - dawsn(x)))
        taus[:,:,i] = np.sum(dtau, axis=2) # Integrated damping wing optical depth
    # Create file
    with h5py.File(f'tau_maps/tau_map_{z0}.hdf5', 'w') as f:
        f.attrs['HubbleParam'] = h
        f.attrs['NumFreq'] = np.int32(n_freq)
        f.attrs['Dv_min'] = Dv_min_kms
        f.attrs['Dv_max'] = Dv_max_kms
        f.attrs['Dv_local'] = 0
        f.attrs['Omega0'] = Omega0
        f.attrs['OmegaBaryon'] = OmegaB
        f.attrs['Redshift'] = z0
        f.create_dataset('tau', data=taus)
        f.create_dataset('transmission', data=np.exp(-taus))
    return Dvs, taus

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the input lightcone HDF5 file."
    )
    parser.add_argument(
        "--z0_file",
        type=str,
        required=True,
        help="Path to a text file with one source z0 per line."
    )
    parser.add_argument(
        "--x1", type=int, default=None,
        help="Start index along x-axis (pixel slice). Default: None (full range)."
    )
    parser.add_argument(
        "--x2", type=int, default=None,
        help="End index along x-axis (pixel slice). Default: None (full range)."
    )
    parser.add_argument(
        "--y1", type=int, default=None,
        help="Start index along y-axis (pixel slice). Default: None (full range)."
    )
    parser.add_argument(
        "--y2", type=int, default=None,
        help="End index along y-axis (pixel slice). Default: None (full range)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file not found: {args.hdf5_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists("tau_maps"):
        shutil.rmtree("tau_maps")
    os.makedirs("tau_maps")

    # print(f"Opening: {args.hdf5_file}")
    # print(f"Source redshift z0 = {args.z0}")
    # if any(v is not None for v in [args.x1, args.x2, args.y1, args.y2]):
    #     print(f"Pixel slice: x=[{args.x1}:{args.x2}], y=[{args.y1}:{args.y2}]")
    # else:
    #     print("Pixel slice: full grid")

    with open(args.z0_file, 'r') as f:
        z0_list = [float(line.strip()) for line in f if line.strip()]

    with h5py.File(args.hdf5_file, 'r') as s:
        for z0 in z0_list:
            calculate_tau_edges(
            s,
            z0=z0,
            x1=args.x1,
            x2=args.x2,
            y1=args.y1,
            y2=args.y2,
        )

    # print(f"Done. tau shape: {taus.shape}")
    # print(f"Output written to: tau_maps/tau_map_{args.z0}.hdf5")


if __name__ == "__main__":
    main()