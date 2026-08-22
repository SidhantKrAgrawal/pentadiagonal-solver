# pentadiagonal-solver

A batched pentadiagonal linear system solver for CPUs and NVIDIA GPUs, aimed at
ADI (Alternating Direction Implicit) workloads. It solves a large batch of
independent 5 diagonal systems along each axis (x, y, z) of a 3D grid.

This README covers building, testing and running.

## Requirements

CMake 3.21 or newer is needed for the presets, GCC 9 or newer for C++17 and
AVX2, and the CUDA Toolkit 11 or newer (tested with 13.0) for the GPU builds.
OpenMP ships with GCC. MPI (OpenMPI or MPICH) is only needed for the gpu-mpi
build.

The first configure downloads project_options, Catch2 and Google Benchmark from
GitHub. On a cluster, configure on a login node; the downloads are cached in the
build directory, so compilation and all later runs work offline.

To report the toolchain the build will use, along with the GPU name and its
compute capability:

```bash
bash scripts/check_env.sh
```

## Quick start

```bash
bash scripts/build.sh gpu            # configure and compile into build/
bash scripts/run_tests.sh gpu        # run the Catch2 correctness suites
build/apps/app_cuda 256 50 double    # solve a 256^3 grid, 50 ADI iterations
```

build.sh detects the GPU's compute capability from nvidia-smi. Passing cpu or
gpu-mpi instead of gpu selects the other two configurations, which build into
build-cpu/ and build-mpi/.

## Building without the scripts

With CMake presets:

```bash
cmake --preset gpu                   # configure into build/
cmake --build --preset gpu -j        # compile

cmake --preset cpu                   # CPU only, no CUDA toolkit needed
cmake --preset gpu-mpi               # adds the distributed MPI library and apps
```

The presets set CMAKE_CUDA_ARCHITECTURES=native, which auto detects the GPU of
the machine running the configure step. On a login node with no GPU, the compute
node's architecture has to be named explicitly:

```bash
cmake --preset gpu -DCMAKE_CUDA_ARCHITECTURES=80
```

Values are 70 for V100, 80 for A100, 86 for RTX 30 series, 89 for RTX 40 series
and L40, and 90 for H100.

Without presets, on CMake older than 3.21:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_FOR_CUDA=ON -DBUILD_FOR_SN=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

The build options are BUILD_FOR_CPU, BUILD_FOR_CUDA, BUILD_FOR_MPI,
BUILD_FOR_SN (single node) and USE_PROFILING.

## Testing

```bash
bash scripts/run_tests.sh gpu        # or cpu, or gpu-mpi
```

This runs cuda_tests and tests. Both read reference data from ./files, so they
must run from the repository root, which is what the script handles. Running a
binary directly needs a cd to the root first:

```bash
cd <repo root> && build/test/cuda/cuda_tests
```

The full suite is 201,388 assertions across 10 test cases.

## Running the solver

app_cuda and adi_cpu perform ADI iterations on an N^3 grid and report per
direction timings and achieved bandwidth.

```bash
# ./app_cuda [N] [iters] [precision]     precision = double (default) or float
build/apps/app_cuda 256 50 double        # GPU
build/apps/adi_cpu  256 10 double        # CPU baseline
OMP_NUM_THREADS=4 build/apps/adi_cpu 256 10 double
```

MPI, from the gpu-mpi build:

```bash
mpirun -np 4 build-mpi/apps/adi_mpi_cpu 256 10 double
```

The 2D lid driven cavity CFD application, on the CPU solver:

```bash
build/apps/app_cpu                       # app_cpu_orig runs the unoptimised solver
```

## Choosing the algorithm

By default the solver picks a kernel per direction from the precision and the
device. Two environment variables override that choice, and the app prints the
kernel that actually ran on every line of output.

PENTA_XALGO selects the x direction (contiguous) solve:

| Value | Algorithm | What it does |
| --- | --- | --- |
| `naive` | 1 | One thread per system, uncoalesced access. The baseline. |
| `transpose` | 2 | Transposes to [elem][sys] in DRAM, then solves with the strided kernel. |
| `thomas-pcr` | 3 | Register resident hybrid Thomas-PCR with SPIKE partitioning. |
| `shared-fact` | 4 | ADI structured solver, one precomputed factorisation shared by every system. |
| `auto` | | Precision and device aware dispatch. The default. |

PENTA_YALGO and PENTA_ZALGO select the y and z (strided) solves independently.
PENTA_YZALGO sets both at once.

| Value | Algorithm | What it does |
| --- | --- | --- |
| `naive` | 1 | One thread per system, strided walk with global scratch. |
| `thomas-pcr` | 3 | Algorithm 3 staged through a shared memory tile. |
| `shared-fact` | 4 | ADI structured solver. |
| `auto` | | Precision aware dispatch. The default. |

Algorithm 2 is x only by design: y and z are already coalesced, so transposing
there is pure added cost.

```bash
PENTA_XALGO=thomas-pcr build/apps/app_cuda 256 50 float
PENTA_XALGO=shared-fact PENTA_YZALGO=shared-fact build/apps/app_cuda 256 50 double
```

shared-fact assumes every system in a direction shares the same coefficients,
which holds for constant coefficient ADI but not in general. It is never
selected automatically and is not checked at runtime.

What auto picks:

In FP32 the x direction uses Algorithm 3 when the system size is one of 128,
256, 320, 384 or 512, and Algorithm 2 otherwise. The y and z directions use
Algorithm 3.

### Which algorithm does `app_cuda 256 50 double` use?

It depends on the GPU, because the dispatch queries the device's FP64 rate at
runtime rather than hardcoding it (src/cuda/singlenode/pentad_cuda.cu:1499):

```cpp
const bool fp64_favours_algo3 =
    (sizeof(Float) == 8) && (device_fp64_ratio_denom() <= 32);
```

The x direction takes Algorithm 3 (thomas-pcr) when FP64 runs at 1/32 of FP32 or
better, and Algorithm 2 (transpose) otherwise. The y and z directions take
Algorithm 1 (naive) in every case.

| GPU | FP64 rate | x | y | z |
| --- | --- | --- | --- | --- |
| RTX 3050 (sm_86) | 1/64 | 2 transpose | 1 naive | 1 naive |
| GTX 1080 (sm_61) | 1/32 | 3 thomas-pcr | 1 naive | 1 naive |
| V100, A100, H100 | 1/2 | 3 thomas-pcr | 1 naive | 1 naive |

The FP32 auto path to Algorithm 3 is gated on `sizeof(Float) == 4`, so FP64
never takes it, and Algorithm 3's redundant arithmetic is compute bound at the
consumer FP64 rate, costing about 44 ms there.

So on an RTX 3050 that command runs Algorithm 2 on x and Algorithm 1 on y and z.
The app prints the kernel that actually ran on each line, so none of this has to
be inferred.

Other variables:

| Variable | Effect |
| --- | --- |
| `PENTA_PCR_LANES` | Lanes per system for Algorithm 3: 8, 16 or 32. |
| `PENTA_WARMUP_MS` | Warm up wall time in ms, default 1500. 0 reverts to a fixed iteration count. |
| `PENTA_PEAK_BW_GBS` | Overrides the detected peak bandwidth used in the reported percentages. |
| `PENTA_DEBUG_LAUNCH` | Set to 1 to report why an opt in kernel declined to launch, instead of falling back silently. |

## Verifying a kernel independently

Each kernel has a validator that builds a random diagonally dominant system and
checks the solution against an independent reference. All print PASS or FAIL.

| Validator | Arguments | What it checks |
| --- | --- | --- |
| `verify_scale_cuda` | N, precision | Backward residual of the GPU x solve at a grid size the fixed suite does not cover. |
| `verify_scale_cpu` | N, precision | The same check against the CPU solver. |
| `verify_shared_fact_cuda` | N, precision, solvedim, samples | Algorithm 4 against the general path, with coefficients that vary along the line so an indexing bug cannot hide. solvedim is 0, 1 or 2. |
| `verify_strided_cuda` | N, precision, dir, algo | A strided y or z kernel against the naive one over the full grid, which is what catches system mixing. dir is 1 for y, 2 for z. |
| `fp32_accuracy_cuda` | N, precision | Forward error of the FP32 solve against a double precision host reference, not just the residual. |

```bash
build/apps/verify_scale_cuda       320 float
build/apps/verify_shared_fact_cuda 256 double 0
build/apps/verify_strided_cuda     256 float 1 thomas-pcr
build/apps/fp32_accuracy_cuda      256 float
build/apps/verify_scale_cpu        320 double
```

## The scripts

| Script | Arguments | What it does |
| --- | --- | --- |
| `build.sh` | preset (gpu, gpu-mpi, cpu) | Configures and compiles. Maps the preset to the matching CMake preset and build directory, and reads the compute capability from nvidia-smi so the architecture does not have to be known in advance. CUDA_ARCH and JOBS override. |
| `check_env.sh` | none | Reports the cmake, g++, nvcc and MPI versions, the GPU name and compute capability, and the core count. Configures nothing. The first thing to run when a build fails. |
| `run_tests.sh` | preset | Runs the Catch2 suites from the repository root, which is required because the test binaries read reference data from ./files. |
| `run_benchmarks.sh` | N, iters | Runs the full ADI solve on the GPU and the CPU for one grid size and prints the two side by side. Defaults to 256 and 50. Raw output goes under results/repro_<date>/. |
| `run_algorithm_sweep.sh` | N, iters, precision, `--with-restricted` | The per direction comparison, in two phases. Phase 1 measures every algorithm that has a kernel for a direction while the other two are held on a fixed baseline, which makes the rows comparable. Phase 2 composes the per direction winners and measures true end to end wall time. `--with-restricted` adds shared-fact. |
| `run_grid_transcript.sh` | gpu iters, cpu iters | Runs the same matrix across grids 128, 256, 320 and 384 in both precisions, recording each run as the command, its verbatim output and the extracted result, so every number traces back to what produced it. Also emits cpu.csv and gpu.csv. |
| `build_grid_report.py` | stamp, out.txt | Assembles the directories run_grid_transcript.sh wrote into one text report: methodology header, run by run transcripts, summary tables, per axis winners. Excludes Algorithm 2 from the y and z rankings, since its y and z columns are the naive strided solve, and separates any row whose requested kernel was not the one that ran. |
| `run_hpc_study.sh` | none, env only | One non interactive command for an unfamiliar machine: configure, build, correctness suite with a hard stop on failure, roofline, full sweep across grids, precisions and lane counts, readable summary, tarball. Builds a fat binary for sm_70 and sm_90 so it can be launched from a login node with no GPU. Nothing is silently skipped: a failed step is recorded with its reason and the run continues. Expect 10 to 25 minutes. Knobs are CUDA_ARCH, GRIDS, GPU_ITERS, SKIP_BUILD and OUTDIR. |

roofline_probe_cuda is an app rather than a script. It measures the GPU's FP64
issue rate and the memory floor for the solver's traffic using timing alone, by
sweeping the number of injected FMAs per element, so it needs no profiler
counters.

## Layout

```
include/pentadsolver.hpp             public C API
src/cpu/                             CPU solver (OpenMP and AVX2) and MPI variant
src/cuda/singlenode/pentad_cuda.cu   all GPU kernels and the dispatch
src/cuda/mpi/                        distributed GPU solver
apps/                                applications and verifiers
benchmarks/                          Google Benchmark micro benchmarks
test/                                Catch2 suites and reference data in files/
scripts/                             build, test and measurement scripts
```

## Troubleshooting

If nvcc is not found, load the CUDA module or add $CUDA_HOME/bin to PATH, or
build the cpu preset instead.

If the configure step hangs or fails while downloading, there is no internet on
that node. Configure on a login node, then build anywhere.

If the build stops at `cuda_runtime.h: No such file or directory`, a module has
put a broken nvcc first in PATH. Point PATH at a complete CUDA installation.

If CMake cannot detect the CUDA architecture, there is no GPU on the machine
running the configure step. Pass -DCMAKE_CUDA_ARCHITECTURES with the compute
node's value.

If a run fails with `no kernel image is available for execution`, the binary was
built for the wrong architecture. Rebuild with the compute node's value.

If the tests cannot open files/..., they were not started from the repository
root. Use scripts/run_tests.sh.

If a forced PENTA_XALGO, PENTA_YALGO or PENTA_ZALGO appears to be ignored, that
kernel declined the size and the solver fell back. Set PENTA_DEBUG_LAUNCH=1 for
the reason; the app also reports the kernel that actually ran.

If the CPU stops speeding up past about 4 threads, that is expected. The solver
is memory bandwidth bound.

## Attribution

The original CPU and CUDA pentadiagonal base, covering the library structure,
the CPU OpenMP and AVX2 solver, the MPI reduced system solver, the ADI cavity
application and the correctness suite, is the work of Nidhi Shukla, University
of Warwick.

The overall design follows the Oxford scalar tridiagonal solver tridsolver, by
Endre László, Mike Giles, Gihan Mudalige and contributors, BSD 3-Clause, at
https://github.com/OP-DSL/tridsolver.

The GPU optimisation kernels, the runtime dispatch, the benchmarking, the MPI
enablement and the ADI structured solver are the work of Sidhant Kumar Agrawal,
CS908 MSc dissertation, University of Warwick, supervised by Dr Gihan Mudalige.

Build scaffolding uses project_options, tests use Catch2, and the micro
benchmarks use Google Benchmark.

## Licence

BSD 3-Clause. See LICENSE.
