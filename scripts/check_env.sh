#!/usr/bin/env bash
# check_env.sh: report the toolchain this repo will build with, so problems
# are visible BEFORE configuring.  Run:  bash scripts/check_env.sh
set -u

say()  { printf '%-22s %s\n' "$1" "$2"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "=== pentadiagonal-solver environment check ==="

if have cmake; then say "cmake"   "$(cmake --version | head -1)"; else say "cmake" "NOT FOUND (required)"; fi
if have g++;   then say "g++"     "$(g++ --version | head -1)";   else say "g++"   "not found"; fi
if have nvcc;  then
  say "nvcc"    "$(nvcc --version | grep -i release | sed 's/^ *//')"
  say "nvcc path" "$(command -v nvcc)"
  # An nvcc that reports a version is not necessarily an nvcc that can compile.
  # A partial install (binary present, headers absent) answers --version and
  # --list-gpu-arch quite happily and then fails on the first real translation
  # unit, which surfaces as a wall of CMake output naming cuda_runtime.h.  A
  # two-line compile settles it here instead.
  CUDA_PROBE="$(mktemp -d)"
  printf '#include <cuda_runtime.h>\nint main(){return 0;}\n' > "$CUDA_PROBE/probe.cu"
  if PROBE_ERR="$(nvcc -c "$CUDA_PROBE/probe.cu" -o "$CUDA_PROBE/probe.o" 2>&1)"; then
    say "nvcc compiles" "yes"
  else
    say "nvcc compiles" "NO, this CUDA installation is incomplete"
    echo "     $(echo "$PROBE_ERR" | grep -iE 'fatal error|error:' | head -1)"
    echo "     The nvcc binary runs but cannot build anything, usually because the"
    echo "     headers are missing next to it.  Look for a complete installation,"
    echo "     often nested a level deeper, and put THAT on PATH:"
    echo "       ls -d \$(dirname \$(dirname \$(command -v nvcc)))/*/bin/nvcc 2>/dev/null"
    echo "     A module that loads a partial install is a common cause."
  fi
  rm -rf "$CUDA_PROBE"
else say "nvcc" "NOT FOUND, GPU build needs CUDA in PATH (module load cuda / add \$CUDA_HOME/bin)"; fi
if have mpicxx || have mpic++; then say "MPI"  "$( (mpicxx --version 2>/dev/null || mpic++ --version) | head -1)";
else say "MPI" "not found (only needed for the gpu-mpi / MPI presets)"; fi

echo
if have nvidia-smi; then
  say "GPU" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "${CC:-}" ]; then
    say "compute capability" "$CC  -> pass -DCMAKE_CUDA_ARCHITECTURES=${CC//./}"
    # Newer CUDA toolkits drop older GPUs, so having both nvcc and a GPU is not
    # enough: they have to match.  Say so here rather than at the end of a build.
    if have nvcc; then
      SUP="$(nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')"
      A="${CC//./}"
      if [ -n "${SUP// /}" ]; then
        if echo " $SUP " | grep -q " $A "; then
          say "toolkit match" "yes, this nvcc can build for compute $A"
        else
          say "toolkit match" "NO, this nvcc cannot build for compute $A"
          echo "     it supports: $SUP"
          echo "     load an older CUDA (12.x covers compute 50 to 90) before building."
        fi
      fi
    fi
  else
    say "compute capability" "nvidia-smi too old to report it; look up the GPU's value"
  fi
else
  say "GPU" "nvidia-smi not found, this is probably a login node with no GPU."
  echo
  echo "  If building here, the compute-node GPU's architecture MUST be passed, e.g."
  echo "     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   # A100"
  echo "  (see the table in README.md)."
fi

echo
if have nproc; then say "CPU cores" "$(nproc)"; fi
echo "Set OMP_NUM_THREADS to tune CPU runs (4 was optimal on the reference machine)."
echo "=== done ==="
