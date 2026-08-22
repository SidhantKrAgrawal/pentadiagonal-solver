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
  ARCH="${CUDA_ARCH:-}"
  if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    ARCH="${CC//./}"
  fi
  if [ -n "$ARCH" ]; then
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
