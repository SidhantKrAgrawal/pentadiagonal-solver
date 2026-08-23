# pentadiagonal-solver

A batched pentadiagonal linear system solver for CPUs and NVIDIA GPUs, aimed at
ADI (Alternating Direction Implicit) workloads. It solves a large batch of
independent 5 diagonal systems along each axis (x, y, z) of a 3D grid.

This README covers building, testing and running.

## Requirements

CMake 3.21 or newer is needed for the presets, GCC 9 or newer for C++17 and
AVX2, and the CUDA Toolkit 11 or newer for the GPU builds. OpenMP ships with
GCC. MPI (OpenMPI or MPICH) is only needed for the gpu-mpi build.

Tested with CUDA 13.0 on sm_86 and CUDA 12.6 on sm_61. CUDA 13 dropped Pascal:
building for sm_61 or older needs a 12.x toolkit, and nvcc reports
`Unsupported gpu architecture 'compute_61'` otherwise.

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

The gpu and cpu presets run the single node suites, 201,388 assertions across
10 test cases each. The gpu-mpi preset additionally runs the two MPI suites
under mpirun, 16,385 and 100,696 assertions; set NP to change the rank count
from the default of 2. If mpirun is not in PATH those two are skipped with a
message rather than silently.

## Running the solver

app_cuda and adi_cpu perform ADI iterations on an N^3 grid and report per
direction timings and achieved bandwidth.

```bash
# ./app_cuda [N] [iters] [precision]     precision = double (default) or float
build/apps/app_cuda 256 50 double        # GPU
build/apps/adi_cpu  256 10 double        # CPU baseline
OMP_NUM_THREADS=4 build/apps/adi_cpu 256 10 double
```

MPI, from the gpu-mpi build. It decomposes along z and prints a global
checksum, which must agree across rank counts for the same N, iterations and
precision:

```bash
mpirun -np 4 build-mpi/apps/adi_mpi_cpu 256 10 double
```

The 2D lid driven cavity CFD application, on the CPU solver:

```bash
build/apps/app_cpu                       # app_cpu_orig runs the unoptimised solver
```

## Choosing the algorithm

There is no automatic dispatch. Each direction runs the kernel it is named, and
with nothing set the default is naive (Algorithm 1) in all three directions.

PENTA_ALGO sets all three directions at once. PENTA_XALGO, PENTA_YALGO and
PENTA_ZALGO set one direction each and override PENTA_ALGO.

The x direction (contiguous) accepts:

| Value | Algorithm | What it does |
| --- | --- | --- |
| `naive` | 1 | One thread per system, uncoalesced access. The default. |
| `transpose` | 2 | Transposes to [elem][sys] in DRAM, then solves with the strided kernel. |
| `thomas-pcr` | 3 | Register resident hybrid Thomas-PCR with SPIKE partitioning. |
| `shared-fact` | 4 | ADI structured solver, one precomputed factorisation shared by every system. |

The y and z directions (strided) accept:

| Value | Algorithm | What it does |
| --- | --- | --- |
| `naive` | 1 | One thread per system, strided walk with global scratch. The default. |
| `thomas-pcr` | 3 | Algorithm 3 staged through a shared memory tile. |
| `shared-fact` | 4 | ADI structured solver. |

Algorithm 2 is an x direction kernel only: y and z are already coalesced, so
transposing them is pure added cost. Naming it for y or z is an error.

```bash
build/apps/app_cuda 256 50 double                       # naive on x, y and z
PENTA_ALGO=thomas-pcr build/apps/app_cuda 256 50 float  # thomas-pcr everywhere
PENTA_XALGO=transpose PENTA_YALGO=naive PENTA_ZALGO=naive \
    build/apps/app_cuda 256 50 double                   # one kernel per direction
```

An unrecognised name, a name the direction does not accept, or a size the
requested kernel has no template for stops the run with a message on stderr.
Nothing is substituted quietly, so a measurement can never be attributed to a
kernel that did not produce it. Algorithm 3 is instantiated for system sizes
128, 256, 320, 384 and 512 on x, and Algorithm 4 needs a multiple of 8.

shared-fact assumes every system in a direction shares the same coefficients,
which holds for constant coefficient ADI but not in general. This is not checked
at runtime.

Which algorithm is fastest depends on the precision and on the GPU, so this
README does not name one. Two cards measured during development disagreed on
the FP64 x winner, one preferring transpose and the other thomas-pcr, which is
why nothing is chosen automatically. Measure the machine being used:

```bash
bash scripts/run_algorithm_sweep.sh 256 50 both
```

That prints every algorithm in every direction for both precisions, then the
best combination measured end to end.

Other variables:

| Variable | Effect |
| --- | --- |
| `PENTA_PCR_LANES` | Lanes per system for Algorithm 3: 8, 16 or 32. |
| `PENTA_Y_BSYS`, `PENTA_Z_BSYS` | Systems per block for Algorithm 3 on y and z: 2, 4, 8, 16 or 32. |
| `PENTA_Y_BLOCK`, `PENTA_Z_BLOCK` | Threads per block for Algorithm 1 on y and z: any multiple of 32 up to 1024. |
| `PENTA_WARMUP_MS` | Warm up wall time in ms, default 1500. 0 reverts to a fixed iteration count. |
| `PENTA_PEAK_BW_GBS` | Overrides the peak bandwidth used in the reported percentages. Normally read from the device. |
| `PENTA_DEBUG_LAUNCH` | Set to 1 to report why a kernel declined to launch. |

The four tuning variables have defaults measured on one GPU. They are not
properties of the algorithm, and the best value for each is a property of the
card, so sweep them rather than trusting the default on a new machine.

## Moving to a different GPU

Nothing in the repository is tied to a particular machine or card. On a new
GPU, with the repository checked out and a CUDA toolkit on PATH:

```bash
bash scripts/check_env.sh                          # reports this GPU and its compute capability
bash scripts/build.sh gpu                          # builds for it, capability read from nvidia-smi
bash scripts/run_tests.sh gpu                      # correctness on this card
bash scripts/run_algorithm_sweep.sh 256 50 both    # which kernel wins here
```

Or the whole campaign in one non interactive command:

```bash
bash scripts/run_hpc_study.sh
```

It builds for the compute capability of the GPU it finds, falling back to a fat
binary only when no GPU is visible, such as on a login node. Peak bandwidth is
read from the device rather than assumed. The generated report names whatever
machines and GPUs produced the data, works with one machine or several, and
carries no timing or baseline figure over from another card.

## Verifying a kernel independently

Each kernel has a validator that builds a random diagonally dominant system and
checks the solution against an independent reference. All print PASS or FAIL.

| Validator | Arguments | What it checks |
| --- | --- | --- |
| `verify_scale_cuda` | N, precision | Backward residual of the GPU x solve at a grid size the fixed suite does not cover. |
| `verify_scale_cpu` | N, precision | The same check against the CPU solver. |
| `verify_shared_fact_cuda` | N, precision, solvedim, samples | Algorithm 4 against the general path, with coefficients that vary along the line so an indexing bug cannot hide. solvedim is 0, 1 or 2. |
| `verify_strided_cuda` | N, precision, dir, algo | Algorithm 3's strided y or z kernel against the naive one over the full grid, which is what catches system mixing. dir is 1 for y, 2 for z; algo must be thomas-pcr. It refuses shared-fact, whose problem class it cannot construct. |
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
| `run_tests.sh` | preset | Runs the Catch2 suites from the repository root, which is required because the test binaries read reference data from ./files. On the gpu-mpi preset it also runs the two MPI suites under mpirun; NP sets the rank count, default 2. |
| `run_benchmarks.sh` | N, iters | Runs the full ADI solve on the GPU and the CPU for one grid size and prints the two side by side. Defaults to 256 and 50. Raw output goes under results/repro_<date>/. |
| `run_algorithm_sweep.sh` | N, iters, precision, `--with-restricted` | The per direction comparison, in two phases. Phase 1 measures every algorithm that has a kernel for a direction while the other two are held on a fixed baseline, which makes the rows comparable. Phase 2 composes the per direction winners and measures true end to end wall time. `--with-restricted` adds shared-fact. |
| `run_grid_transcript.sh` | gpu iters, cpu iters | Runs the same matrix across grids 128, 256, 320 and 384 in both precisions, recording each run as the command, its verbatim output and the extracted result, so every number traces back to what produced it. Also emits cpu.csv and gpu.csv. |
| `build_grid_report.py` | stamp, out.txt | Assembles the directories run_grid_transcript.sh wrote into one text report: methodology header, run by run transcripts, summary tables, per axis winners, and a per machine analysis. Machine names, GPU models, grids and precisions all come from the data, so it works with one machine or several. Excludes Algorithm 2 from the y and z rankings, since its y and z columns are the naive strided solve, and separates any row whose requested kernel was not the one that ran. |
| `run_hpc_study.sh` | none, env only | One non interactive command for an unfamiliar machine: configure, build, correctness suite with a hard stop on failure, roofline, full sweep across grids, precisions and lane counts, readable summary, tarball. Builds for the compute capability of the GPU it finds, and only falls back to a fat binary when no GPU is visible. Nothing is silently skipped: a failed step is recorded with its reason and the run continues. Expect 10 to 25 minutes. Knobs are CUDA_ARCH, GRIDS, GPU_ITERS, SKIP_BUILD and OUTDIR. |

roofline_probe_cuda is an app rather than a script. It measures the GPU's FP64
issue rate and the memory floor for the solver's traffic using timing alone, by
sweeping the number of injected FMAs per element, so it needs no profiler
counters.

## Layout

```
include/pentadsolver.hpp             public C API
src/cpu/                             CPU solver (OpenMP and AVX2) and MPI variant
src/cuda/singlenode/pentad_cuda.cu   all GPU kernels and the selector
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

If cmake cannot find MPI, the compiler wrappers are not in PATH. On a Red Hat
style system they live in /usr/lib64/openmpi/bin, which is not on PATH by
default; add it, or load the site's MPI module.

If a run stops with `[penta] x solve: cannot run ...`, the named kernel has no
template for that system size or the direction does not accept it. Pick a size
it supports, or another kernel. Set PENTA_DEBUG_LAUNCH=1 for the launch level
reason.

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
