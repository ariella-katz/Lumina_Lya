# CUDA Lyman-alpha transmission

`lumina_lya` reads Lumina HDF5 lightcones, streams bounded tiles to a GPU, and writes transmission products. Only the expensive lightcone absorption calculation is accelerated. Existing plotting and power-spectrum scripts remain unchanged. A corrected, independent Python implementation is available in [`cpu/reference.py`](../cpu/reference.py).

The default is **Band+Voigt**, with a maximum spectral spacing of 5 km/s. This retains damping wings, but its finite-bin transmission closure is a lower-limit approximation. Use spectral convergence and a resolved Comb+Voigt comparison to choose a cheaper production resolution. Delta modes omit damping wings. No production speedup or full-resolution runtime has been measured on this login node.

## Build on Juno

Run these commands from the repository root. They compile code without starting a GPU calculation. Two build workers limit login-node CPU use.

```bash
/opt/ohpc/pub/utils/cmake/4.0.0/bin/cmake -S gpu -B gpu/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/ohpc/pub/compiler/gcc/13.2.0/bin/gcc \
  -DCMAKE_CXX_COMPILER=/opt/ohpc/pub/compiler/gcc/13.2.0/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/opt/ohpc/pub/apps/cuda/12.6/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/opt/ohpc/pub/compiler/gcc/13.2.0/bin/g++ \
  '-DCMAKE_CUDA_ARCHITECTURES=80;90' \
  -DHDF5_ROOT=/opt/ohpc/pub/libs/gnu14/hdf5/1.14.6
/opt/ohpc/pub/utils/cmake/4.0.0/bin/cmake --build gpu/build -j 2
```

CUDA 12.6 and GCC 13 were checked locally. Architectures 80 and 90 cover the inspected A30 and Hopper partitions. CMake uses the HDF5 C API and records library search paths; Python is not needed for the native solver. Tests require NumPy, SciPy, and h5py. On another system, replace these paths or use the corresponding modules. A host-only validation build is available with `-DLYA_ENABLE_CUDA=OFF`.

## Inspect and run

Inspection reads metadata and the one-dimensional geometry arrays, without reading gas fields or initializing CUDA:

```bash
gpu/build/lumina_lya inspect \
  --input /titan/asmith/Lumina/500cMpc/postprocessing/rlc_640/All.hdf5 \
  --z0-file z0_list.txt
```

Run GPU work inside a Slurm allocation. The example below submits a job only when **you execute it**; no jobs are submitted by building or testing this project.

```bash
sbatch gpu/scripts/run.slurm \
  /titan/asmith/Lumina/500cMpc/postprocessing/rlc_640/All.hdf5 \
  z0_list.txt gpu/results/band-voigt \
  --host-memory-mib 12288 --device-memory-mib 16384
```

The example requests one GPU in Juno's `h100` partition and 16 GiB host RAM. Select your authorized partition and allocation limits as appropriate. The explicit buffer budgets must leave room for HDF5 caches, the CUDA context, runtime libraries, and other process overhead.

The same invocation inside an existing GPU allocation is:

```bash
gpu/build/lumina_lya run --input LIGHTCONE --z0-file SOURCES \
  --output-dir gpu/results/run --sampling band --profile voigt \
  --products maps,spectra,cumulative --host-memory-mib 12288
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--sampling band\|comb` | Finite frequency bins or monochromatic samples. |
| `--profile voigt\|delta` | Paper's Voigt expansion or resonance-only absorption. |
| `--dv-step 5` | Maximum source wavelength-offset spacing, in km/s. |
| `--max-dln 0.001` | Maximum subsegment width in ln(1+z). |
| `--region X0,X1,Y0,Y1` | Half-open global pixel bounds. |
| `--chunk INDEX` | Historical chunk numbering; remainder pixels are covered. |
| `--tile-size 128 --depth-slab 128` | Maximum internal tile/slab sizes, reduced to fit budgets. |
| `--host-memory-mib 1024` | Estimated working-buffer budget, excluding library overhead. |
| `--device-memory-mib N` | Device-buffer budget, capped at 80% of free GPU memory. Default: 80% free. |
| `--device 0` | Device index relative to Slurm's visible devices. |
| `--products maps,spectra,cumulative` | Maps are always written; other products are optional. |
| `--resume` | Skip matching completed sources; reject changed configurations or inputs. |
| `--overwrite` | Replace only the selected output files after successful completion. |
| `--dry-run` | Plan without opening a GPU context or creating output files. |
| `--backend cpu` | Native scalar backend for tiny validation cases. |
| `--legacy` | Historical working Python physics and sampling; see below. |

Source files accept one redshift per line and optional comments. Additional comma-separated values are retained as `z_spread` metadata; only the first value is the source redshift, as in the original calculation. They do not define a source distribution. Duplicate or out-of-range source redshifts are rejected. The calculation ends at the last supplied lightcone edge and does not extrapolate beyond it.

For parallel jobs, assign disjoint `--chunk` values or regions and use the same output directory. Internal memory tiles do not affect output chunk numbering. Merge compatible spatial products with:

```bash
gpu/build/lumina_lya merge --output gpu/results/merged.hdf5 \
  gpu/results/run/z0=6/tau_map_6_*.hdf5
```

Merge requires nonoverlapping tiles that fill their bounding rectangle. It rejects missing coverage and incompatible products and weights cumulative statistics by pixel count. Merged maps carry `Chunk=0` and can be supplied as a single map file to existing consumers. Files from custom regions or irregular edge chunks should be merged before using plotting routines that assume equally sized square chunks.

## Physics and numerical conventions

The four absorption operators follow [Almualla et al. (2026), Table 1, Sections 2.3 and 2.5.1](https://academic.oup.com/mnras/article/549/1/stag871/8674676). Comb integrates individual frequencies. Band+Voigt exponentiates the mean cell optical depth; Band+Delta uses the fraction of a bin intersecting resonance. Repeated Band operations discard unresolved spectral structure. Both spectral spacing and geometric subdivision therefore need convergence checks, especially for Delta absorption.

Reporting bands have physical boundaries `[-2000, -500, -100, 100, 500, 2000]` km/s. Each is divided into equal wavelength-offset intervals no wider than `--dv-step`, keeping reporting edges exact. A red-positive offset `u` represents `lambda/lambda0 = 1+u/c`; its source frequency is `nu0/(1+u/c)`. Band weights use frequency widths. Comb uses the same edges as sampling nodes and trapezoidal frequency weights, including shared reporting boundaries.

The stored decreasing `Distances` and `Redshifts` edges define the geometry. `Distances.to_cgs`, when present, converts the comoving coordinate; otherwise the conversion is `UnitLength_in_cm/HubbleParam`. The source location is interpolated linearly in this coordinate, and the first cell is clipped at the requested redshift. For each interval:

```text
H_eff = c * (z_hi-z_lo)/(chi_hi-chi_lo)
proper_length = (chi_hi-chi_lo)/(z_hi-z_lo) * log((1+z_hi)/(1+z_lo))
1+z_mid = sqrt((1+z_hi)*(1+z_lo))
```

Intervals are subdivided uniformly in ln(1+z). Density and velocity unit conversions are evaluated at subsegment midpoints. Cartesian velocities project onto the outward ray direction, while photons move inward, giving `nu_gas = nu*(1+v_out/c)` to first order in peculiar velocity. Cosmological frequencies are evaluated at each midpoint. The analytic operator retains the paper's linear drift `K=H_eff/v_thermal` within a subsegment, anchored symmetrically about that midpoint. This is a controlled local approximation, not exact frequency evolution throughout a finite segment.

Voigt means the paper's expansion through order `a²`, not arbitrary-temperature exact Faddeeva integration. Neutral Voigt cells with damping `a > 0.1` are rejected because they require a different full-profile solver. Tests separately compare against the full Voigt profile; sampled relative profile errors range from about `2.4e-8` at `a=0.00047` to `2.4e-3` at `a=0.1`. These samples are diagnostics, not a universal error bound. `--legacy` retains the original first-order expansion.

The supplied FP64 hypergeometric coefficients are preserved in `include/lya/numerics.hpp`. Stable wing differences, short-interval quadrature, `expm1`/`log1p`, and log-sum-exp avoid cancellation and opaque-spectrum underflow. Computation and accumulation use FP64 without fast-math or fused-operation contraction. Actual transmission may underflow to zero; finite effective optical depths remain representable.

`--legacy` selects Comb+Voigt, the 801 samples from -2000 to +2000 km/s, equal-weight group counts `[300,80,41,80,300]`, upstream source snapping, matter-dominated expansion, and the historical Doppler/frequency expressions. It fixes programming errors rather than preserving them. In particular, final band transmission is normalized, invalid sources are rejected, and output directories are never recursively deleted.

## Products, validation, and scale

Output is grouped as `OUTPUT/z0=REQUESTED/tau_map_REQUESTED_CHUNK.hdf5`. Region selections use coordinate suffixes. Source strings use round-trip decimal formatting; read redshift attributes rather than relying on string formatting.

| Dataset | Shape and meaning |
| --- | --- |
| `tau_band_avgs` | `[5,x,y]`, effective optical depth from averaged transmission. |
| `taus` | Optional `[x,y,frequency]`; sampled tau for Comb, effective bin tau for Band. |
| `T_bands` | `[5]`, spatial mean of the final band transmission. |
| `T_cum_bands` | Optional `[5,depth]`, spatial mean after each observerward input cell. |
| `T_redshifts` | Downstream input-cell edges associated with cumulative values. |
| `Dvs`, `Dv_edges` | Source wavelength offsets in cm/s; the legacy edge vector is empty. |
| `freq_band_edges`, `freq_bands` | Reporting boundaries in km/s. |
| `FrequencyLowerOverNu0`, `FrequencyUpperOverNu0`, `frequency_weights` | Actual frequency coordinates and reporting weights. |

Attributes record requested and actual source redshifts, interpolated comoving source distance, global spatial bounds, input identity, geometry, profile order, precision, and configuration. A source exactly at the observer edge has zero optical depth, unit final transmission, and an empty cumulative curve. Completed files are flushed and atomically renamed. A process killed forcibly may leave a `.partial.PID` file and `.lock` directory; inspect that job before removing its stale lock and rerunning. Resume operates on completed source products, not partially computed tiles.

Run the login-safe checks with:

```bash
/opt/ohpc/pub/utils/cmake/4.0.0/bin/ctest --test-dir gpu/build --output-on-failure -j 1
```

These checks use tiny CPU fixtures even when the executable includes CUDA. They cover all modes against independent SciPy integration, legacy parity, spectral wings, source and velocity conventions, tiling, cumulative normalization, HDF5 VDS failures, merging, and resume. Numerical CUDA/CPU acceptance is `1e-8` absolute plus relative for the same model; wing-specific scalar checks use stricter absolute tolerances.

GPU runtime validation is explicit and belongs on a compute node:

```bash
sbatch gpu/scripts/validate_gpu.slurm
# Within an allocation, refine a deliberately small real region:
python gpu/scripts/convergence.py --executable gpu/build/lumina_lya \
  --input LIGHTCONE --source 7 --region 0,4,0,4 \
  --output-dir gpu/results/convergence
```

The convergence script halves spectral spacing and geometric subdivision independently, then compares against Comb+Voigt. It reports changes without treating the lower-bound closure as exact. Native timing separates HDF5 reads, CUDA transfers, kernels, and tiled writes. Timings can overlap and do not sum to wall time; HDF5 final flush and setup are included only in wall time. Product timing attributes refer to the whole region/source batch, not each source individually.

At `5120² × 32081`, there are roughly `8.4e11` cells. Five FP64 maps require about 1 GiB per source; 801-channel FP64 spectra require about 156 GiB per source. Streaming bounds working memory but does not eliminate this computation or storage. Start with a representative small region on a compute node, measure all phases, then estimate the required allocation. See [BACKEND.md](BACKEND.md) for actual buffer layouts and compilation resource statistics.
