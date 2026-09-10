"""Run a deliberately small convergence study inside a GPU allocation."""
import argparse
from pathlib import Path
import subprocess

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', type=float, required=True)
    parser.add_argument('--region', default='0,4,0,4')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dv-step', type=float, default=5.)
    parser.add_argument('--max-dln', type=float, default=1e-3)
    args = parser.parse_args()
    bounds = [int(x) for x in args.region.split(',')]
    if len(bounds) != 4 or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        parser.error('region must contain increasing X0,X1,Y0,Y1 bounds')
    if (bounds[1]-bounds[0])*(bounds[3]-bounds[2]) > 256:
        parser.error('this convergence helper is limited to 256 rays; use native run for larger studies')
    if args.dv_step <= 0 or args.max_dln <= 0:
        parser.error('spectral spacing and subdivision must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = args.output_dir / 'source.txt'
    text = f'{args.source:.17g}\n'
    if sources.exists() and sources.read_text() != text:
        parser.error('output directory contains a different source; choose another directory')
    sources.write_text(text)
    cases = [('baseline', 'band', args.dv_step, args.max_dln),
             ('frequency_half', 'band', args.dv_step/2, args.max_dln),
             ('geometry_half', 'band', args.dv_step, args.max_dln/2),
             ('comb_refined', 'comb', args.dv_step/2, args.max_dln/2)]
    maps = {}
    for name, sampling, spacing, subdivision in cases:
        out = args.output_dir / name
        subprocess.run([str(args.executable.resolve()), 'run', '--input', str(args.input),
                        '--z0-file', str(sources), '--output-dir', str(out),
                        '--region', args.region, '--sampling', sampling, '--profile', 'voigt',
                        '--dv-step', str(spacing), '--max-dln', str(subdivision),
                        '--backend', 'cuda', '--resume'], check=True)
        products = list(out.rglob('*.hdf5'))
        if len(products) != 1:
            raise RuntimeError(f'{out}: expected exactly one source product')
        with h5py.File(products[0]) as f:
            maps[name] = np.exp(-f['tau_band_avgs'][:])
    base = maps['baseline']
    for name, _, _, _ in cases[1:]:
        delta = maps[name] - base
        print(f'{name}: max |delta T|={np.max(np.abs(delta)):.6g}; '
              f'per-band mean delta={delta.mean(axis=(1, 2))}')


if __name__ == '__main__':
    main()
