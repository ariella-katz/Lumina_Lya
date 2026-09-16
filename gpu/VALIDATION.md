# Validation record

Implementation checks completed on the Juno login node on 2026-09-09.
No GPU calculation, Slurm submission, or production gas-field read was run.

## Compilation

The complete native executable built successfully with CUDA 12.6.85, GCC
13.2.0, CMake 4.0.0, and the HDF5 1.14.6 C library. The CUDA build includes
architectures 80 and 90. A separate host-only build also completed.

## Lightweight tests

The final CUDA-enabled build passed all six CTest suites in 19.60 seconds:

| Suite | Coverage |
| --- | --- |
| `numerical_primitives` | Shared primitive identities, quadrature, symmetry, and saturation. |
| `native_cpu_backend` | Four modes, source and frequency indexing, cumulative reductions, tiling, and invalid gas. |
| `independent_numerics` | 88 independent SciPy comparisons, including strict far-wing and Gaussian-tail checks. |
| `tiny_pipeline` | Native HDF5 pipeline versus the independent Python reference, legacy behavior, multiple frequency tiles/sources, clipping, VDS, merging, and resume. |
| `merge_validation` | Eleven tests for coordinate consistency, memory limits, corrupted statistics, recursive merging, resume fingerprints, and input protection. |
| `python_reference` | Fifteen independent reference tests, including exact source placement and the Voigt damping-domain guard. |

These suites use the CPU backend even when the executable contains CUDA.
They do not establish CUDA runtime correctness. Scalar model agreement and
the difference from the full Voigt profile are tested separately.

## Real input inspection

Metadata-only inspection succeeded for
`/titan/asmith/Lumina/500cMpc/postprocessing/rlc_640/All.hdf5`.
It identified a `640 × 640 × 4010` lightcone, decreasing redshift edges from
30 to approximately 4.75295, and `128³` underlying storage chunks. The
repository's 50-source list passed coverage checks. No gas arrays were read.

## Compute-node validation

GPU execution remains unverified. Use `scripts/validate_gpu.slurm` for the
explicit CUDA-versus-CPU tests, including the normal kernel with more than
128 frequencies and multiple nonempty source paths. Use
`scripts/convergence.py` on a small representative region to vary spectral
spacing and geometric subdivision independently.

Production throughput, transfer overlap in practice, occupancy, and feasible
5120² runtime have not been measured. Use the native timing output to assess
these on an allocated GPU before extrapolating to a full lightcone.
