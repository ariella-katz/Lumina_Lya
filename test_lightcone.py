import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize

import plot_tau_CMB_spectra as smith

def main():
    lightcone_filename = os.path.expanduser("~/scratch/katz_All.hdf5")
    with h5py.File(lightcone_filename, 'r') as cone:
        zs = cone['Redshifts'][:]
        z0_i = np.argmax(zs <= 13)
        z0_cone = float(zs[z0_i])
        densities = cone['HII_Fraction'][..., z0_i].astype(np.float64)
        plt.imshow(densities)
        plt.savefig("density_test.png", dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    main()
