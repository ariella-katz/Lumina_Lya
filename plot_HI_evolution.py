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
    x = 1.- f['HII_Fraction'][:]
    if len(x.shape) == 4:
        x = np.sqrt(np.sum(x**2, axis=-1))
    midptx = x.shape[0]//2
    midpty = x.shape[1]//2
    z = f['Redshifts'][:]
    z = 0.5 * (z[:-1] + z[1:])
    mask = (z < 13.0) & (z > 6.0)
    z = z[mask]
    x = x[:,:,mask]
    x0 = x[midptx,midpty,:]
    x1 = x[midptx,(int)(midpty*1.5),:]
    x2 = x[(int)(midptx*1.5),midpty,:]
    x3 = x[(int)(midptx*0.5),midpty,:]
    x4 = x[midptx,(int)(midpty*0.5),:]
    plt.plot(z, x0, alpha=0.5, linewidth=0.9, label='4 Random Sightlines')
    plt.plot(z, x1, alpha=0.5, linewidth=0.9)
    plt.plot(z, x2, alpha=0.5, linewidth=0.9)
    plt.plot(z, x3, alpha=0.5, linewidth=0.9)
    plt.plot(z, x4, alpha=0.5, linewidth=0.9)
    plt.plot(z, np.mean(x, axis=(0,1)), label='Mean', color='black')
    plt.xlabel('z', size='large')
    plt.ylabel('HI Fraction', size='large')
    plt.title(f'Evolution of Neutral H Fraction', size='x-large')
    plt.legend()
    plt.savefig('HI_evolution.png')

