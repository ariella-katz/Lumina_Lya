# Transmission backend architecture

The CUDA backend streams spatial tiles and depth slabs. It retains optical depths with shape `[source, ray, frequency]` between submissions. No allocation contains both the complete depth axis and the frequency axis for every ray. The native CPU backend implements the same interface and arithmetic for small integration tests; `cpu/` supplies the separate Python reference.

## Execution and memory

Each CUDA submission copies the caller's gas arrays and source-dependent segment list into one of two pinned host slots. The caller can release its `RawSlab` and `Batch` immediately after `submit` returns. A transfer stream per slot copies that slot to the device. The ordered compute stream waits on its transfer event, preprocesses gas properties, and updates persistent optical depths. Reusing a slot waits for its preceding compute completion before changing any host or device buffer.

Gas preprocessing converts density and velocity units, projects velocity onto the outward sightline, and computes inverse thermal speed once per loaded cell. Every source and frequency reuses those derived values. The transmission kernel gives each block one ray, one source, and up to 128 frequency lanes. Thirty-two consecutive segments and their gas cells are staged cooperatively in 2,048 bytes of shared memory. Each active frequency lane retains its optical depth in a register while traversing the staged segments. Lanes outside the frequency grid still participate in synchronization, so arbitrary spectral sizes and final partial tiles are covered.

Cumulative output uses a separate kernel. A block owns one ray and source, stages one segment and gas cell, updates that ray's frequencies, and reduces their weighted transmissions. It makes one FP64 atomic spatial addition per band at each requested downstream cell boundary. The atomic addition order can change the last few bits between runs. Sources with no segments have zero optical depth and no cumulative cell contributions; the output layer represents their zero-length path explicitly.

For `R` tile rays, `L` maximum slab depth, `S` sources, `F` spectral channels, `D` input cells, and `M` maximum segments in either slot, the principal CUDA allocations in bytes are:

| Allocation | Device | Pinned host |
| --- | ---: | ---: |
| Two raw gas slots | `96 R L` | `96 R L` |
| Two derived gas slots | `48 R L` | 0 |
| Persistent optical depths | `8 S R F` | 0 |
| Optional cumulative spatial sums | `40 S D` | 0 |
| Two segment slots | `80 M` | `80 M` |
| Two source-offset arrays | `16 (S+1)` | `16 (S+1)` |
| Frequencies and source redshifts | `64 F + 8 S` | 0 |

The caller additionally holds `48 R L` bytes of raw gas while submitting, its segment vector, downloaded products, and the cumulative sum across spatial tiles. HDF5 caches, input geometry, output buffers, CUDA context memory, allocator bookkeeping, and compiler-generated thread stacks add overhead beyond this table. Segment slots grow when needed; allocation growth can briefly serialize otherwise overlapping work. Allocation failures report the requested size and suggest reducing the tile or slab size.

## Numerical and input requirements

The shared transport routine uses FP64 primitives and accumulation. Build without fast math or floating-point contraction; the CMake configuration applies `--fmad=false` to CUDA and `-ffp-contract=off` to host calculations. Numerical agreement between backends does not establish agreement between the paper's approximate Voigt operators and a full Voigt-profile solver.

Gas inputs must have finite, nonnegative density, ionized fractions in `[0,1]`, positive finite temperature, and finite Cartesian velocities. Absorbing cells must have subluminal projected physical velocities. Segment lengths must be finite and nonnegative, with finite positive Hubble rates. Array dimensions, source offsets, and local cell bounds are checked before submission. Nonfinite unit conversions, negative or nonfinite optical-depth increments, and accumulated optical-depth overflow fail the calculation.

For neutral gas, the Voigt expansion additionally requires damping parameter `a <= 0.1`. Colder input outside that domain raises a clear error requiring a full-profile solver. Delta modes retain their temperature-independent absorption operator and do not impose this damping restriction. Fully ionized cells contribute no absorption.

## Validation and performance limits

The small native CPU test covers all four absorption modes, two sources and two frequencies, cumulative versus final transmission, depth-slab and spatial-tile invariance, empty source paths, transparent gas, invalid input, and the Voigt damping threshold. Numerical primitive tests compare against independent scalar calculations. CUDA source compilation checks were performed with CUDA 12.6 and GCC 13 for Ampere and Hopper targets. No GPU execution or production lightcone calculation was performed on the login node.

An earlier Hopper compilation of the shared-staging implementation reported 184 registers per thread for the normal kernel and 192 for the cumulative kernel, with no register spills. Both used a 128-byte compiler-generated thread stack. Recheck resource usage when profiling the final executable, since compiler choices can change with numerical safeguards. These statistics indicate substantial register pressure; they do not measure occupancy, execution time, or throughput on an allocated GPU. Compute-node validation must compare CUDA against the CPU references at the requested `1e-8` absolute-plus-relative tolerance, including the cumulative path, source-empty batches, and spectral counts that are not multiples of 128.

Backend transfer and kernel timings use CUDA events after the relevant work completes. Transfers include downloaded products; kernel timing includes preprocessing. File reading and writing are measured by the driver. Transfers and computation can overlap, so their summed durations need not equal wall time. Representative compute-node measurements are still needed to determine whether HDF5 reads, FP64 absorption calculations, cumulative reductions, or output writing dominate, and to estimate feasible production sizes such as 5120² rays.
