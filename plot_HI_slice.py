import numpy as np
from scipy.special import erf, dawsn
import h5py
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import quad_vec
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

with h5py.File('/titan/asmith/Lumina/500cMpc/postprocessing/rlc_640/All.hdf5', 'r') as f:
    zs = f['Redshifts'][:]
    x = 1.- f['HII_Fraction'][:]
    if len(x.shape) == 4:
        x = np.sqrt(np.sum(x**2, axis=-1))
    idx = np.argmax(zs < 6.0) + 1
    sl = x[:,:,idx]
    half_fov = 3.6 / 2
    im = plt.imshow(sl, origin='lower', cmap='viridis_r', extent=[-half_fov, half_fov, -half_fov, half_fov])
    plt.title(rf"HI Fraction at z$\approx$6", size='x-large')
    plt.ylabel(r'$\Delta\Theta$ [degrees]', size='medium')
    plt.xlabel(r'$\Delta\Theta$ [degrees]', size='medium')
    cbar = plt.colorbar(aspect=30, pad=0, label='HI Fraction')
    plt.savefig('HI_slice.png')

