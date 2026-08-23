#!/usr/bin/env bash
# build.sh: configure + build the pentadiagonal solver with sane defaults.
#
#   bash scripts/build.sh [gpu|gpu-mpi|cpu]        (default: gpu)
#
# For the GPU builds it auto-detects the CUDA architecture from nvidia-smi so
# the GPU's compute capability does not have to be known.  Override anything:
#   CUDA_ARCH=80 bash scripts/build.sh gpu         # force A100
#   JOBS=8       bash scripts/build.sh gpu
set -euo pipefail

PRESET="${1:-gpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBS="${JOBS:-$( (nproc 2>/dev/null) || echo 4)}"

# Map preset -> binary dir (must match CMakePresets.json).
case "$PRESET" in
  gpu)     BUILD_DIR="build" ;;
  gpu-mpi) BUILD_DIR="build-mpi" ;;
  cpu)     BUILD_DIR="build-cpu" ;;
  *) echo "unknown preset '$PRESET' (use: gpu | gpu-mpi | cpu)"; exit 1 ;;
esac

echo ">> preset=$PRESET  build_dir=$BUILD_DIR  jobs=$JOBS"

EXTRA=()
if [ "$PRESET" != "cpu" ]; then
  command -v nvcc >/dev/null 2>&1 || {
    echo "!! nvcc not in PATH. Load the CUDA module (e.g. 'module load cuda') or"
    echo "   add \$CUDA_HOME/bin to PATH, then re-run.  For a CPU-only build use: cpu"; exit 1; }

  # An nvcc that answers --version is not necessarily one that can compile: a
  # partial install has the binary but not the headers, and only fails once
  # CMake tries a real translation unit, deep inside its own output.  Two lines
  # here turn that into one sentence.
  PROBE_DIR="$(mktemp -d)"
  printf '#include <cuda_runtime.h>\nint main(){return 0;}\n' > "$PROBE_DIR/probe.cu"
  if ! PROBE_ERR="$(nvcc -c "$PROBE_DIR/probe.cu" -o "$PROBE_DIR/probe.o" 2>&1)"; then
    echo "!! nvcc is on PATH but cannot compile:"
    echo "     $(command -v nvcc)"
    echo "     $(echo "$PROBE_ERR" | grep -iE 'fatal error|error:' | head -1)"
    echo "   This CUDA installation is incomplete, the headers are not beside the"
    echo "   binary.  A complete one is often nested a level deeper; find it with:"
    echo "     ls -d \$(dirname \$(dirname \$(command -v nvcc)))/*/bin/nvcc 2>/dev/null"
    echo "   then put that directory first on PATH and re-run.  Modules that load a"
    echo "   partial install are the usual cause."
    rm -rf "$PROBE_DIR"; exit 1
  fi
  rm -rf "$PROBE_DIR"
  ARCH="${CUDA_ARCH:-}"
  if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    ARCH="${CC//./}"
  fi
  if [ -n "$ARCH" ]; then
    # A CUDA toolkit only generates code for a fixed set of architectures, and
    # newer toolkits drop older GPUs (CUDA 13 starts at compute 75, so it cannot
    # build for a V100 or anything older).  Checking here turns a page of nvcc
    # and CMake output into one sentence naming the fix.
    SUPPORTED="$(nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')"
    if [ -n "${SUPPORTED// /}" ] && ! echo " $SUPPORTED " | grep -q " $ARCH "; then
      echo "!! This GPU is compute capability $ARCH, but the CUDA toolkit in PATH"
      echo "   cannot generate code for it."
      echo "     nvcc     : $(nvcc --version 2>/dev/null | grep -i release | sed 's/^ *//')"
      echo "     supports : $SUPPORTED"
      echo "   Load an older CUDA (a 12.x toolkit covers compute 50 to 90) and re-run:"
      echo "     module load CUDA/12.6.2   # or whatever 12.x this site provides"
      exit 1
    fi
    echo ">> CUDA architecture: $ARCH"
    EXTRA+=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH")
  else
    echo ">> CUDA architecture: 'native' (no nvidia-smi / CUDA_ARCH). On a"
    echo "   login node without a GPU this WILL fail, re-run as: CUDA_ARCH=<cc> bash scripts/build.sh $PRESET"
  fi
fi

# Prefer CMake presets when supported; fall back to explicit flags otherwise.
if cmake --version | head -1 | grep -qE '3\.(2[1-9]|[3-9][0-9])'; then
  cmake --preset "$PRESET" "${EXTRA[@]}"
  cmake --build --preset "$PRESET" -j "$JOBS"
else
  echo ">> CMake < 3.21: presets unavailable, using explicit flags."
  case "$PRESET" in
    gpu)     FLAGS=(-DBUILD_FOR_CUDA=ON -DBUILD_FOR_SN=ON) ;;
    gpu-mpi) FLAGS=(-DBUILD_FOR_CUDA=ON -DBUILD_FOR_SN=ON -DBUILD_FOR_MPI=ON) ;;
    cpu)     FLAGS=(-DBUILD_FOR_CUDA=OFF -DBUILD_FOR_SN=ON) ;;
  esac
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DENABLE_DEVELOPER_MODE=OFF \
        "${FLAGS[@]}" "${EXTRA[@]}"
  cmake --build "$BUILD_DIR" -j "$JOBS"
fi

echo
echo ">> Build complete. Binaries in $BUILD_DIR/ :"
echo "     $BUILD_DIR/apps/app_cuda        full ADI solver (GPU)"
echo "     $BUILD_DIR/apps/adi_cpu         full ADI solver (CPU)"
echo "     $BUILD_DIR/test/cuda/cuda_tests correctness suite"
echo ">> Next:  bash scripts/run_tests.sh $PRESET   &&   bash scripts/run_benchmarks.sh"
