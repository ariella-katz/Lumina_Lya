import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse

def transmission_integrated_z0_old(s):
    "Takes in a file created by calculate_tau.py, which includes z0, taus, transmission, Dvs"
    z0 = s.attrs['Redshift']
    taus = s['taus'][:]
    transmission = np.exp(-taus)
    i1 = (int)(1500/4000.*800.)
    i2 = (int)(1900/4000.*800.)
    i3 = (int)(2100/4000.*800.)
    i4 = (int)(2500/4000.*800.)
    T_int_ultrablue = np.sum(transmission[:,:,:i1]*(4000./800.)/1500., axis=2)
    T_int_blue = np.sum(transmission[:,:,i1:i2]*(4000./800.)/400., axis=2)
    T_int_center = np.sum(transmission[:,:,i2:i3]*(4000./800.)/200., axis=2)
    T_int_red = np.sum(transmission[:,:,i3:i4]*(4000./800.)/400., axis=2)
    T_int_ultrared = np.sum(transmission[:,:,i4:]*(4000./800.)/1500., axis=2)
    return z0, T_int_ultrablue, T_int_blue, T_int_center, T_int_red, T_int_ultrared

def transmission_integrated_z0_banded(z0_ss):
    "Takes in a file created by calculate_tau.py, which includes z0, taus, transmission, Dvs"
    n_chunks = len(z0_ss)
    s0 = z0_ss[0]
    z0 = s0.attrs['Redshift']
    tau_band_avgs_0 = s0['tau_band_avgs'][:]
    chunk_size = tau_band_avgs_0.shape[1]
    T_int_ultrablue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_blue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_center = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_red = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_ultrared = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    ix1 = 0
    iy1 = 0
    x1 = ix1 * chunk_size
    y1 = iy1 * chunk_size
    T_int_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[0])
    T_int_blue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[1])
    T_int_center[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[2])
    T_int_red[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[3])
    T_int_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_0[4])
    for chunk in range(n_chunks-1)+1:
        s_chunk = z0_ss[chunk]
        chunk_num = s_chunk.attrs['Chunk']
        tau_band_avgs_chunk = s_chunk['tau_band_avgs'][:]
        x1 = chunk_size * (chunk_num % int(np.sqrt(n_chunks)))
        y1 = chunk_size * (chunk_num // int(np.sqrt(n_chunks)))
        T_int_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[0])
        T_int_blue[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[1])
        T_int_center[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[2])
        T_int_red[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[3])
        T_int_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = np.exp(-tau_band_avgs_chunk[4])
    return z0, T_int_ultrablue, T_int_blue, T_int_center, T_int_red, T_int_ultrared

def transmission_integrated_z0(z0_ss):
    "Takes in a file created by calculate_tau.py, which includes z0, taus, transmission, Dvs"
    n_chunks = int(np.sqrt(len(z0_ss)))
    s0 = z0_ss[0]
    z0 = float(np.asarray(s0.attrs['Redshift']).squeeze())
    chunk_size = s0.attrs['ChunkSize']
    T_int_ultrablue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_blue = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_center = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_red = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    T_int_ultrared = np.zeros((n_chunks*chunk_size, n_chunks*chunk_size))
    ix1 = 0
    iy1 = 0
    x1 = ix1 * chunk_size
    y1 = iy1 * chunk_size
    _, T_int_ultrablue_0, T_int_blue_0, T_int_center_0, T_int_red_0, T_int_ultrared_0 = transmission_integrated_z0_old(z0_ss[0])
    T_int_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_ultrablue_0
    T_int_blue[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_blue_0
    T_int_center[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_center_0
    T_int_red[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_red_0
    T_int_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_ultrared_0
    for chunk in range(1, n_chunks):
        s_chunk = z0_ss[chunk]
        _, T_int_ultrablue_chunk, T_int_blue_chunk, T_int_center_chunk, T_int_red_chunk, T_int_ultrared_chunk = transmission_integrated_z0_old(s_chunk)
        chunk_num = s_chunk.attrs['Chunk']
        x1 = chunk_size * (chunk_num // n_chunks)
        y1 = chunk_size * (chunk_num % n_chunks)
        T_int_ultrablue[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_ultrablue_chunk
        T_int_blue[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_blue_chunk
        T_int_center[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_center_chunk
        T_int_red[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_red_chunk
        T_int_ultrared[x1:x1+chunk_size,y1:y1+chunk_size] = T_int_ultrared_chunk
    return z0, T_int_ultrablue, T_int_blue, T_int_center, T_int_red, T_int_ultrared

def plot_Tint(ss):
    z0s = []
    T_ints_ultrablue = []
    T_ints_blue = []
    T_ints_center = []
    T_ints_red = []
    T_ints_ultrared = []
    for z0_ss in ss:
        z0, T_int_ultrablue, T_int_blue, T_int_center, T_int_red, T_int_ultrared = transmission_integrated_z0(z0_ss)
        z0s.append(z0)
        T_ints_ultrablue.append(T_int_ultrablue)
        T_ints_blue.append(T_int_blue)
        T_ints_center.append(T_int_center)
        T_ints_red.append(T_int_red)
        T_ints_ultrared.append(T_int_ultrared)
    sort_idx = np.argsort(z0s)
    z0s = np.asarray(z0s)[sort_idx]
    T_ints_ultrablue = np.asarray(T_ints_ultrablue)[sort_idx]
    T_ints_blue = np.asarray(T_ints_blue)[sort_idx]
    T_ints_center = np.asarray(T_ints_center)[sort_idx]
    T_ints_red = np.asarray(T_ints_red)[sort_idx]
    T_ints_ultrared = np.asarray(T_ints_ultrared)[sort_idx]
    n = len(z0s)
    med_ultrablue = np.median(T_ints_ultrablue, axis=(1,2))
    mean_ultrablue = np.mean(T_ints_ultrablue, axis=(1,2))
    sig_ultrablue = np.std(T_ints_ultrablue, axis=(1,2))
    med_blue = np.median(T_ints_blue, axis=(1,2))
    mean_blue = np.mean(T_ints_blue, axis=(1,2))
    sig_blue = np.std(T_ints_blue, axis=(1,2))
    med_center = np.median(T_ints_center, axis=(1,2))
    mean_center = np.mean(T_ints_center, axis=(1,2))
    sig_center = np.std(T_ints_center, axis=(1,2))
    med_red = np.median(T_ints_red, axis=(1,2))
    mean_red = np.mean(T_ints_red, axis=(1,2))
    sig_red = np.std(T_ints_red, axis=(1,2))
    med_ultrared = np.median(T_ints_ultrared, axis=(1,2))
    mean_ultrared = np.mean(T_ints_ultrared, axis=(1,2))
    sig_ultrared = np.std(T_ints_ultrared, axis=(1,2))
    fig, (linax, logax) = plt.subplots(2, 1, sharex = True)
    fig.subplots_adjust(hspace=0)
    linax.plot(z0s, mean_ultrablue, color='blue', label='ultrablue, mean')
    linax.fill_between(z0s, mean_ultrablue + sig_ultrablue, mean_ultrablue - sig_ultrablue, color='blue', alpha=0.2)
    linax.plot(z0s, med_ultrablue, color='blue', linestyle='dashed') #, label='ultrablue, median')
    linax.plot(z0s, mean_blue, color='green', label='blue, mean')
    linax.fill_between(z0s, mean_blue + sig_blue, mean_blue - sig_blue, color='green', alpha=0.2)
    linax.plot(z0s, med_blue, color='green', linestyle='dashed') #, label='blue, median')
    linax.plot(z0s, mean_center, color='brown', label='center, mean')
    linax.fill_between(z0s, mean_center + sig_center, mean_center - sig_center, color='brown', alpha=0.2)
    linax.plot(z0s, med_center, color='brown', linestyle='dashed') #, label='center, median')
    linax.plot(z0s, mean_red, color='orange', label='red, mean')
    linax.fill_between(z0s, mean_red + sig_red, mean_red - sig_red, color='orange', alpha=0.2)
    linax.plot(z0s, med_red, color='orange', linestyle='dashed') #, label='red, median')
    linax.plot(z0s, mean_ultrared, color='red', label='ultrared, mean')
    linax.fill_between(z0s, mean_ultrared + sig_ultrared, mean_ultrared - sig_ultrared, color='red', alpha=0.2)
    linax.plot(z0s, med_ultrared, color='red', linestyle='dashed', label='median') #, label='ultrared, median')
    linax.set_ylim(0.1,1)
    linax.set_xlim(6,13)
    linax.set_ylabel(rf'$\mathcal{{T}}^\text{{int}}$')
    linax.legend(ncols=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    logax.plot(z0s, np.log10(10e-12 + mean_ultrablue), color='blue', label='ultrablue, mean')
    logax.fill_between(z0s, np.log10(10e-12 + mean_ultrablue + sig_ultrablue), np.log10(np.max([np.zeros(n)+10e-12, mean_ultrablue - sig_ultrablue], axis=0)), color='blue', alpha=0.2)
    logax.plot(z0s, np.log10(10e-12 + med_ultrablue), color='blue', linestyle='dashed', label='ultrablue, median')
    logax.plot(z0s, np.log10(10e-12 + mean_blue), color='green', label='blue, mean')
    logax.fill_between(z0s, np.log10(10e-12 + mean_blue + sig_blue), np.log10(np.max([np.zeros(n)+10e-12, mean_blue - sig_blue], axis=0)), color='green', alpha=0.2)
    logax.plot(z0s, np.log10(10e-12 + med_blue), color='green', linestyle='dashed', label='blue, median')
    logax.plot(z0s, np.log10(10e-12 + mean_center), color='brown', label='center, mean')
    logax.fill_between(z0s, np.log10(10e-12 + mean_center + sig_center), np.log10(np.max([np.zeros(n)+10e-12, mean_center - sig_center], axis=0)), color='brown', alpha=0.2)
    logax.plot(z0s, np.log10(10e-12 + med_center), color='brown', linestyle='dashed', label='center, median')
    logax.plot(z0s, np.log10(10e-12 + mean_red), color='orange', label='red, mean')
    logax.fill_between(z0s, np.log10(10e-12 + mean_red + sig_red), np.log10(np.max([np.zeros(n)+10e-12, mean_red - sig_red], axis=0)), color='orange', alpha=0.2)
    logax.plot(z0s, np.log10(10e-12 + med_red), color='orange', linestyle='dashed', label='red, median')
    logax.plot(z0s, np.log10(10e-12 + mean_ultrared), color='red', label='ultrared, mean')
    logax.fill_between(z0s, np.log10(10e-12 + mean_ultrared + sig_ultrared), np.log10(np.max([np.zeros(n)+10e-12, mean_ultrared - sig_ultrared], axis=0)), color='red', alpha=0.2)
    logax.plot(z0s, np.log10(10e-12 + med_ultrared), color='red', linestyle='dashed', label='ultrared, median')
    logax.set_ylim(-7, -1)
    logax.set_ylabel(rf'$\log\mathcal{{T}}^\text{{int}}$')
    logax.set_xlabel(rf'$z_0$')
    plt.savefig('Tint_z0.png')

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
    plot_Tint(ss)


if __name__ == "__main__":
    main()
