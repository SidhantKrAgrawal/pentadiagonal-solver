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

PENTA_XALGO selects the x direction (contiguous) solve. Its values are naive
for Algorithm 1, a thread per system with uncoalesced access, which is the
baseline; transpose for Algorithm 2, which transposes to [elem][sys] in DRAM and
solves with the strided kernel; thomas-pcr for Algorithm 3, the register
resident hybrid Thomas-PCR with SPIKE partitioning; shared-fact for Algorithm 4,
the ADI structured solver; and auto, the default.

PENTA_YALGO and PENTA_ZALGO select the y and z (strided) solves independently,
and accept naive, thomas-pcr, shared-fact and auto. PENTA_YZALGO sets both at
once. Algorithm 2 is x only by design: y and z are already coalesced, so
transposing there is pure added cost.

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

In FP64 the y and z directions use Algorithm 1, because Algorithm 3's redundant
arithmetic is compute bound at the consumer FP64 rate and costs around 44 ms.
The x direction depends on the device's FP64 throughput ratio, which is queried
at runtime rather than assumed. On parts where FP64 runs at 1/32 of FP32 or
better, including Pascal, V100, A100 and H100, x uses Algorithm 3. On consumer
Ampere and Ada, where FP64 runs at 1/64, x uses Algorithm 2. So
`build/apps/app_cuda 256 50 double` runs Algorithm 2 on an RTX 3050 and
Algorithm 3 on a GTX 1080 or a datacentre card, with Algorithm 1 on y and z in
both cases.

Other variables: PENTA_PCR_LANES (8, 16 or 32) sets the lanes per system for
Algorithm 3; PENTA_WARMUP_MS sets the warm up wall time, default 1500;
PENTA_PEAK_BW_GBS overrides the detected peak bandwidth; and
PENTA_DEBUG_LAUNCH=1 reports why an opt in kernel declined to launch instead of
falling back silently.

## Verifying a kernel independently

Each kernel has a validator that builds a random diagonally dominant system and
checks the solution against an independent reference. All print PASS or FAIL.

```bash
build/apps/verify_scale_cuda       320 float               # residual check at a given N
build/apps/verify_shared_fact_cuda 256 double 0            # shared-fact, solvedim 0, 1 or 2
build/apps/verify_strided_cuda     256 float 1 thomas-pcr  # strided y or z; dir 1=y, 2=z
build/apps/fp32_accuracy_cuda      256 float               # FP32 against FP64 forward error
build/apps/verify_scale_cpu        320 double              # CPU equivalent
```

## The scripts

build.sh configures and compiles. It takes gpu, gpu-mpi or cpu, defaulting to
gpu, maps that to the matching CMake preset and build directory, and reads the
compute capability from nvidia-smi so the architecture does not have to be known
in advance. CUDA_ARCH and JOBS override the detected values.

check_env.sh reports the versions of cmake, g++, nvcc and MPI, the GPU name and
its compute capability, and the core count, without configuring anything. It is
the first thing to run when a build fails.

run_tests.sh runs the Catch2 suites for a given preset from the repository root,
which is required because the test binaries read reference data from ./files.

run_benchmarks.sh runs the full ADI solve on the GPU and on the CPU for one grid
size and prints the two side by side. It takes N and the iteration count,
defaulting to 256 and 50, and writes raw output under results/repro_<date>/.

run_algorithm_sweep.sh is the per direction comparison. It runs in two phases:
first it measures every algorithm that has a kernel for a direction while
holding the other two directions on a fixed baseline, which makes the rows
comparable; then it composes the per direction winners into one configuration
and measures the true end to end wall time. It takes N, the iteration count and
a precision of double, float or both, and --with-restricted adds shared-fact.

run_grid_transcript.sh runs the same matrix across grids 128, 256, 320 and 384
in both precisions, recording every run as the command, its verbatim output and
the extracted result, so each number can be traced back to what produced it. It
also emits cpu.csv and gpu.csv.

build_grid_report.py assembles the directories that run_grid_transcript.sh wrote
into a single text report: a methodology header, the run by run transcripts, the
summary tables and the per axis winners. It excludes Algorithm 2 from the y and
z rankings, since its y and z columns are the naive strided solve, and it
separates out any row whose requested kernel was not the kernel that ran.

run_hpc_study.sh is one non interactive command for a machine that has not been
used before. It configures, builds, runs the correctness suite and stops if that
fails, measures the machine's own memory and compute roofline, runs the full
algorithm sweep across grids, precisions and lane counts, writes a readable
summary and tars the result. It builds a fat binary for sm_70 and sm_90 by
default so it can be launched from a login node with no GPU; CUDA_ARCH overrides
that. GRIDS, GPU_ITERS, SKIP_BUILD and OUTDIR are the other knobs. Nothing is
silently skipped: a step that fails is recorded with its reason and the run
continues. Expect 10 to 25 minutes.

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
