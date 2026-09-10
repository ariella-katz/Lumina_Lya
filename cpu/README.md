# CPU reference

`reference.py` is a corrected, memory-bounded Python reference for the expensive
transmission calculation. The original scripts are unchanged. The GPU executable
under `gpu/` is intended for production; this independent SciPy implementation
is intended for small regions and validation.

```bash
python3 cpu/reference.py inspect --input /path/to/All.hdf5
python3 cpu/reference.py run --input /path/to/All.hdf5 \
  --z0-file z0_list.txt --output-dir cpu/results \
  --region 0,2,0,2 --products maps,spectra,cumulative
python3 -m unittest discover -s cpu/tests -v
```

Do not run the second command on a login node with a production lightcone.
Even a small spatial region can span many thousands of depth cells. The tests
construct only three-cell fixtures in temporary directories.

`Config` controls `sampling='band'|'comb'`, `profile='voigt'|'delta'`, spectral
spacing, maximum segment width in ln(1+z), tile/slab sizes, and memory limits.
`run(input_path, sources, output_dir, config, region=None, resume=False)` writes
one HDF5 file per source. Each source can be a scalar or a list whose additional
values are preserved as `z_spread` metadata. Only the first value is integrated.
The output file contains a stable configuration record and is published atomically
after completion. Resume skips matching completed products; mismatches are errors.

The four absorption operators implement equations 16 and 19–24 of
[Almualla et al. (2026)](https://academic.oup.com/mnras/article/549/1/stag871/8674676).
Band+Voigt is the default because it includes wings and resolves the spectral
intervals. Its exp(-mean(tau)) closure is a lower bound for a flat incident
spectrum within each segment's bin. It does not equal the exact frequency
average of exp(-tau). Band+Delta instead uses the segment's exact fractional
resonance overlap and attenuation. Both apply their scalar bin operators
successively, as in the paper. Finer spectral bins reduce the effect of this
within-bin closure. Voigt uses the paper's expansion through second order in
the damping parameter, not the full Voigt function. Optional `full=True` scalar
operators support independent comparisons with SciPy's Faddeeva function.
Neutral cells with damping parameter `a > 0.1` are rejected in Voigt and legacy
runs because they require a full-profile solver. Delta runs and transparent
cells are unaffected. This guard establishes a supported approximation domain;
it does not imply uniform accuracy throughout that domain.

The reference integrates the Voigt profile using SciPy quadrature for band
operators and protected primitive differences for comb operators. It does not
reuse the CUDA hypergeometric approximation. Its distant-wing series avoids
subtracting nearly equal Dawson terms. Reporting bands average transmitted
intensity with frequency-width weights. Effective optical depths use log-sum-exp
so saturated bins retain finite optical depths when representable.

Input gas fields are `[x,y,depth]`, with a final three-component axis for
`Velocities`; float32 and float64 inputs are accepted. `Redshifts` and `Distances`
contain descending edges of length `depth+1`. `Distances.to_cgs`, if supplied,
converts the comoving code coordinate to cm; otherwise `UnitLength_in_cm/h` is
used. Density and velocity conversions follow the original Arepo conventions,
evaluated at each subsegment's midpoint. `OpeningAngle` is in radians.
Missing virtual source files and datasets are rejected before reading gas.

Source positions are interpolated within the distance–redshift table. Cells are
split into equal steps of ln(1+z) no larger than `max_dln`. Photon frequencies
at segment midpoints retain their exact cosmological and first-order Doppler
transformations, while within-segment drift follows the paper's linear operator.
Outward gas velocity blueshifts photons traveling toward the observer.
For a delta-profile resonance exactly at a segment boundary, the upstream
boundary is open and the downstream boundary is closed to count it once.

`--legacy` forces the original 801-point Comb+Voigt calculation, first-order
Voigt expression, redshift snapping, velocity convention, and equal sample
weights with counts `[300,80,41,80,300]`. It retains corrected spatial coverage,
normalization, validation, bounded input reads, and stable accumulation.

The default product `tau_band_avgs[5,x,y]` contains effective band optical-depth
maps. Optional `taus[x,y,frequency]` contains effective bin optical depths for
Band sampling or sampled optical depths for Comb sampling. Every file contains
the pixel-averaged final `T_bands[5]`. With `--products cumulative`,
`T_cum_bands[5,depth]` is evaluated at downstream input-cell boundaries, whose
redshifts are stored in `T_redshifts`. Its final column equals `T_bands` exactly.
A source on the observer edge has empty cumulative arrays and unit transmission.
Frequency edges, weights, source positions, spatial bounds, and physics settings
are recorded alongside these datasets. The native `merge` command handles
native spatial products; this CPU reference writes one region per file.
