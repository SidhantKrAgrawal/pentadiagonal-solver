# pentadiagonal-solver

A batched pentadiagonal linear-system solver for CPUs and NVIDIA GPUs, aimed at
ADI (Alternating-Direction-Implicit) workloads. It solves a large batch of
independent 5-diagonal systems along each axis (x, y, z) of a 3-D grid.

This README is instructions only: how to build, test, and run.

---

## Requirements

| Tool         | Version               | Needed for       |
| ------------ | --------------------- | ---------------- |
| CMake        | ≥ 3.21 (for presets)  | all builds       |
| C++ compiler | GCC ≥ 9 (C++17, AVX2) | all builds       |
| CUDA Toolkit | ≥ 11 (tested 13.0)    | GPU builds       |
| OpenMP       | bundled with GCC      | CPU parallelism  |
| MPI          | OpenMPI / MPICH       | `gpu-mpi` build  |
| Internet     | at configure time     | see below        |

The first configure downloads `project_options`, Catch2, and Google Benchmark
from GitHub. On a cluster, **configure on a login node**; the downloads are
cached in the build directory, so compilation and all later runs work offline.

Check what the build will use:

```bash
bash scripts/check_env.sh
```
Prints your cmake / g++ / nvcc / MPI versions, the GPU name, and its compute
capability. Run this first if a build fails.

---

## Quick start

```bash
bash scripts/build.sh gpu            # configure + compile into build/
bash scripts/run_tests.sh gpu        # run the Catch2 correctness suites
build/apps/app_cuda 256 50 double    # solve a 256^3 grid, 50 ADI iterations
```

`build.sh` detects your GPU's compute capability from `nvidia-smi`. Pass
`cpu` or `gpu-mpi` instead of `gpu` for the other two build configurations
(they land in `build-cpu/` and `build-mpi/`).

---

## Building from scratch, without the scripts

With CMake presets:

```bash
cmake --preset gpu                   # configure into build/
cmake --build --preset gpu -j        # compile

cmake --preset cpu                   # CPU only, no CUDA toolkit needed
cmake --preset gpu-mpi               # adds the distributed MPI library and apps
```

The presets set `CMAKE_CUDA_ARCHITECTURES=native`, which auto-detects the
**build machine's** GPU. On a login node with no GPU, name the compute node's
architecture explicitly:

```bash
cmake --preset gpu -DCMAKE_CUDA_ARCHITECTURES=80
```

| GPU                   | Value |
| --------------------- | ----- |
| V100                  | `70`  |
| A100                  | `80`  |
| RTX 30-series (3050…) | `86`  |
| RTX 40-series / L40   | `89`  |
| H100                  | `90`  |

Without presets (CMake < 3.21):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_FOR_CUDA=ON -DBUILD_FOR_SN=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Build options: `BUILD_FOR_CPU`, `BUILD_FOR_CUDA`, `BUILD_FOR_MPI`,
`BUILD_FOR_SN` (single node), `USE_PROFILING`.

---

## Testing

```bash
bash scripts/run_tests.sh gpu        # or cpu / gpu-mpi
```
Runs `cuda_tests` and `tests`. Both must run **from the repository root** —
they read reference data from `./files` — which is what the script handles.
Running the binaries directly requires `cd` to the root first:

```bash
cd <repo root> && build/test/cuda/cuda_tests
```

---

## Running the solver

`app_cuda` and `adi_cpu` perform ADI iterations on an N³ grid and report
per-direction timings and achieved bandwidth.

```bash
# ./app_cuda [N] [iters] [precision]     precision = double (default) | float
build/apps/app_cuda 256 50 double        # GPU
build/apps/adi_cpu  256 10 double        # CPU baseline
OMP_NUM_THREADS=4 build/apps/adi_cpu 256 10 double
```

MPI, from the `gpu-mpi` build:

```bash
mpirun -np 4 build-mpi/apps/adi_mpi_cpu 256 10 double
```

2-D lid-driven cavity CFD application, on the CPU solver:

```bash
build/apps/app_cpu                       # app_cpu_orig runs the unoptimised solver
```

### Choosing the algorithm

By default a precision-aware dispatch picks the kernel. Two environment
variables override it.

`PENTA_XALGO` — the x-direction (contiguous) solve:

| value         | Algorithm                                     |
| ------------- | --------------------------------------------- |
| `naive`       | 1 — thread-per-system, uncoalesced (baseline)  |
| `transpose`   | 2 — Global-Transpose via explicit DRAM transpose |
| `thomas-pcr`  | 3 — Hybrid Thomas-PCR / SPIKE, register-resident |
| `shared-fact` | 4 — Shared-Factorisation (ADI-structured)      |
| `auto`        | production dispatch (default)                  |

`PENTA_YALGO` / `PENTA_ZALGO` — the y and z (strided) solves, chosen
independently; accept `naive`, `thomas-pcr`, `shared-fact`, `auto`.
`PENTA_YZALGO` sets both at once. Algorithm 2 is x-only by design: y and z are
already coalesced, so transposing there is pure added cost.

```bash
PENTA_XALGO=thomas-pcr build/apps/app_cuda 256 50 float
PENTA_XALGO=shared-fact PENTA_YZALGO=shared-fact build/apps/app_cuda 256 50 double
```

`shared-fact` assumes every system in a direction shares the same
coefficients, which is true for constant-coefficient ADI but not in general.
It is never selected automatically and is not checked at runtime.

Other variables: `PENTA_PCR_LANES` (8/16/32) sets the lanes per system for
Algorithm 3; `PENTA_WARMUP_MS` sets the warm-up wall time (default 1500);
`PENTA_PEAK_BW_GBS` overrides the detected peak bandwidth;
`PENTA_DEBUG_LAUNCH=1` reports why an opt-in kernel declined to launch instead
of silently falling back.

---

## Verifying a kernel independently

Each kernel has a validator that builds a random diagonally-dominant system and
checks the solution against an independent reference. All print `PASS`/`FAIL`.

```bash
build/apps/verify_scale_cuda       320 float             # residual check at a given N
build/apps/verify_shared_fact_cuda 256 double 0          # shared-fact, solvedim 0|1|2
build/apps/verify_strided_cuda     256 float 1 thomas-pcr  # strided y|z; dir 1=y, 2=z
build/apps/fp32_accuracy_cuda      256 float             # FP32-vs-FP64 forward error
build/apps/verify_scale_cpu        320 double            # CPU equivalent
```

---

## Measurement scripts

```bash
bash scripts/run_benchmarks.sh 256 50
```
GPU vs CPU on one grid; raw output under `results/repro_<date>/`.

```bash
bash scripts/run_algorithm_sweep.sh 256 50 both
```
Every algorithm in every direction, then the best combination measured
end-to-end. Add `--with-restricted` to include `shared-fact`.

```bash
build/apps/roofline_probe_cuda
```
Measures this GPU's FP64 issue rate and the memory floor for the solver's
traffic, using timing only — no profiler counters.

```bash
bash scripts/run_grid_transcript.sh 50 20
python3 scripts/build_grid_report.py <stamp> report.txt
```
The same matrix across grids 128/256/320/384 and both precisions, recorded as
a transcript of command → output → result, then assembled into one report.

```bash
bash scripts/run_hpc_study.sh
```
One non-interactive command for a machine you have not used before: configure,
build, correctness suite, roofline, full sweep, summary, tarball. Builds a fat
binary for sm_70 and sm_90 so it can be launched from a GPU-less login node;
override with `CUDA_ARCH=80`. Other knobs (`GRIDS`, `GPU_ITERS`, `SKIP_BUILD`,
`OUTDIR`) are documented in the script header. Expect 10–25 minutes.

---

## Layout

```
include/pentadsolver.hpp             public C API
src/cpu/                             CPU solver (OpenMP + AVX2) and MPI variant
src/cuda/singlenode/pentad_cuda.cu   all GPU kernels and the dispatch
src/cuda/mpi/                        distributed GPU solver
apps/                                applications and verifiers
benchmarks/                          Google Benchmark micro-benchmarks
test/                                Catch2 suites + reference data in files/
scripts/                             build, test, and measurement scripts
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `nvcc: command not found` | `module load cuda`, or add `$CUDA_HOME/bin` to `PATH`, or build the `cpu` preset. |
| Configure hangs or fails downloading | No internet. Configure on a login node, then build anywhere. |
| `cuda_runtime.h: No such file or directory` | A module put a broken `nvcc` first in `PATH`. Point `PATH` at a complete CUDA install. |
| CUDA arch not detected on a login node | No GPU to detect. Pass `-DCMAKE_CUDA_ARCHITECTURES=<cc>`. |
| `no kernel image is available for execution` | Built for the wrong arch. Rebuild with the compute node's value. |
| Tests cannot open `files/...` | Run from the repository root, or use `scripts/run_tests.sh`. |
| A forced `PENTA_*ALGO` seems ignored | That kernel declined the size and fell back. Set `PENTA_DEBUG_LAUNCH=1`; the app also reports the kernel that actually ran. |
| CPU does not speed up past ~4 threads | Expected: the solver is memory-bandwidth-bound. |

---

## Attribution

- Original CPU/CUDA pentadiagonal base — library structure, CPU OpenMP/AVX2
  solver, MPI reduced-system solver, ADI cavity application, and correctness
  suite: **Nidhi Shukla**, University of Warwick.
- The overall design follows the Oxford scalar tridiagonal solver
  [tridsolver](https://github.com/OP-DSL/tridsolver) (Endre László, Mike Giles,
  Gihan Mudalige, and contributors; BSD 3-Clause, OP-DSL).
- GPU optimisation kernels, runtime dispatch, benchmarking, MPI enablement, and
  the ADI-structured solver: **Sidhant Kumar Agrawal**, CS908 MSc dissertation,
  University of Warwick, supervised by Dr Gihan Mudalige.
- Build scaffolding uses
  [project_options](https://github.com/aminya/project_options); tests use
  Catch2; micro-benchmarks use Google Benchmark.

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
