# pentadiagonal-solver

A batched pentadiagonal linear system solver for CPUs and NVIDIA GPUs, aimed at
ADI (Alternating Direction Implicit) workloads. It solves a large batch of
independent 5 diagonal systems along each axis (x, y, z) of a 3D grid.

This README covers building, testing and running.

## What is needed

A machine with an NVIDIA GPU, a CUDA toolkit, GCC 9 or newer, and CMake 3.21 or
newer. Network access is needed the first time only, because the first configure
downloads project_options, Catch2 and Google Benchmark.

Everything is run on the machine that holds the GPU. On a cluster that means a
compute node, not the login node, because the build reads the card directly.

## Step 1: get the code

```bash
git clone https://github.com/SidhantKrAgrawal/pentadiagonal-solver.git
cd pentadiagonal-solver
```

## Step 2: log into the GPU machine

```bash
ssh <gpu-host>
cd <path to pentadiagonal-solver>
```

Confirm a GPU is visible:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

If that command is not found, or lists nothing, this machine has no GPU. Move to
one that does, or build the CPU-only configuration described in Step 8.

## Step 3: put CUDA on PATH

Most clusters use modules:

```bash
module avail          # lists what the site provides
module load cuda      # or a specific one, e.g. module load CUDA/12.6.2
```

If there are no modules, find the toolkit and add its bin directory:

```bash
export PATH=/path/to/cuda/bin:$PATH
```

Do not assume this worked. Step 4 checks it.

## Step 4: check the environment

```bash
bash scripts/check_env.sh
```

This is the step that decides whether a build can succeed. Expect these four
lines:

```
nvcc                   Cuda compilation tools, release 13.0, V13.0.88
nvcc path              /some/path/cuda/bin/nvcc
nvcc compiles          yes
toolkit match          yes, this nvcc can build for compute 75
```

Both `yes` lines matter, and they fail for different reasons.

**`nvcc compiles: NO`** means the CUDA installation is incomplete: the nvcc
binary exists but its headers do not sit beside it. A modulefile pointing at a
partial install is the usual cause, and a complete copy is often nested one
directory deeper. Find it and put it first on PATH:

```bash
ls -d $(dirname $(dirname $(command -v nvcc)))/*/bin/nvcc 2>/dev/null
export PATH=/the/directory/that/printed:$PATH
```

Then run check_env.sh again. Skipping this produces a page of CMake output ending
in `cuda_runtime.h: No such file or directory`.

**`toolkit match: NO`** means this CUDA is too new for this GPU. Newer toolkits
drop older cards: CUDA 13 supports compute capability 75 and above, so a V100 at
70, or anything older, needs a 12.x toolkit. Load one and re-check:

```bash
module load CUDA/12.6.2      # or whatever 12.x the site provides
```

Do not continue until both lines read `yes`.

## Step 5: build

```bash
bash scripts/build.sh gpu
```

The compute capability is read from nvidia-smi, so no architecture has to be
given. This takes a few minutes, mostly compiling CUDA templates. It ends with:

```
>> Build complete. Binaries in build/ :
```

## Step 6: check correctness

```bash
bash scripts/run_tests.sh gpu
```

Expect `All tests passed (201388 assertions in 10 test cases)` twice, once for
the CUDA suite and once for the CPU suite. Anything else is a real failure and
should be reported rather than worked around.

The test binaries read reference data from ./files, so they must be started from
the repository root. The script handles that. Running one directly needs a cd
first:

```bash
cd <path to pentadiagonal-solver> && build/test/cuda/cuda_tests
```

## Step 7: run the solver

```bash
build/apps/app_cuda 256 50 double
```

A 256^3 grid, 50 ADI iterations, double precision. It prints the kernel used for
each direction, per direction timings, achieved bandwidth, and the end to end
time per iteration. With nothing else set this runs Algorithm 1 in all three
directions, which is the baseline and the slowest. Step 9 measures the rest.

## Step 8: CPU only, if there is no GPU

```bash
bash scripts/build.sh cpu
bash scripts/run_tests.sh cpu
build/apps/adi_cpu 256 10 double
OMP_NUM_THREADS=4 build/apps/adi_cpu 256 10 double
```

No CUDA toolkit and no GPU are needed for this path.

## Step 9: measure this machine

Which kernel is fastest depends on the GPU and the precision, so it is measured
rather than assumed:

```bash
bash scripts/run_algorithm_sweep.sh 256 50 both
```

Every algorithm in every direction, both precisions, then the best combination
timed end to end. Takes roughly 10 to 20 minutes and writes to results/.

For the full campaign in one command, including a correctness gate, a measured
roofline, a grid sweep and a tarball of the output:

```bash
bash scripts/run_hpc_study.sh
```

Expect 10 to 25 minutes. It stops if the correctness suite fails, and records any
other failure with its reason rather than skipping it silently.

## Build options

BUILD_FOR_CPU, BUILD_FOR_CUDA, BUILD_FOR_MPI, BUILD_FOR_SN (single node) and
USE_PROFILING are the CMake options behind the presets, for a hand-rolled
configure.

## Other applications

A 2D lid driven cavity CFD application, on the CPU solver:

```bash
build/apps/app_cpu                   # app_cpu_orig runs the unoptimised solver
```

A probe that measures this GPU's FP64 issue rate and memory floor by timing
alone, with no profiler counters:

```bash
build/apps/roofline_probe_cuda
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
| `build.sh` | preset (gpu, cpu) | Configures and compiles. Reads the compute capability from nvidia-smi so the architecture does not have to be known in advance, and stops with an explanation if the CUDA toolkit in PATH is too new to target that GPU. CUDA_ARCH and JOBS override. |
| `check_env.sh` | none | Reports the cmake, g++ and nvcc versions, the GPU name and compute capability, whether that nvcc can actually build for that GPU, and the core count. Configures nothing. The first thing to run when a build fails. |
| `run_tests.sh` | preset | Runs the Catch2 suites from the repository root, which is required because the test binaries read reference data from ./files. |
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
src/cpu/                             CPU solver (OpenMP and AVX2)
src/cuda/singlenode/pentad_cuda.cu   all GPU kernels and the selector
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

If the build stops at `cuda_runtime.h: No such file or directory`, the CUDA on
PATH is incomplete: the nvcc binary is there but its headers are not. This is
usually a modulefile pointing at a partial install, and the complete one is
often nested a directory deeper. Find it with

```bash
ls -d $(dirname $(dirname $(command -v nvcc)))/*/bin/nvcc 2>/dev/null
```

then put that directory first on PATH. check_env.sh reports this as
`nvcc compiles: NO` before a build is attempted.

If CMake cannot detect the CUDA architecture, there is no GPU on the machine
running the configure step. Pass -DCMAKE_CUDA_ARCHITECTURES with the compute
node's value.

If a run fails with `no kernel image is available for execution`, the binary was
built for the wrong architecture. Rebuild with the compute node's value.

If the tests cannot open files/..., they were not started from the repository
root. Use scripts/run_tests.sh.

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
