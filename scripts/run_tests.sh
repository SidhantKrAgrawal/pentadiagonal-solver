#!/usr/bin/env bash
# run_tests.sh: run the correctness suites from the correct working directory
# (the test binaries load reference data from ./files, which is why they must
# run from the repository root, which is what this script handles).
#
#   bash scripts/run_tests.sh [gpu|gpu-mpi|cpu]     (default: gpu)
set -euo pipefail

PRESET="${1:-gpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"   # <-- required: tests read ./files/*

case "$PRESET" in
  gpu|gpu-mpi) BUILD_DIR=$([ "$PRESET" = gpu ] && echo build || echo build-mpi) ;;
  cpu)         BUILD_DIR="build-cpu" ;;
  *) echo "unknown preset '$PRESET'"; exit 1 ;;
esac

ran=0
if [ -x "$BUILD_DIR/test/cuda/cuda_tests" ]; then
  echo "=== CUDA correctness suite ==="
  "$BUILD_DIR/test/cuda/cuda_tests"; ran=1
fi
if [ -x "$BUILD_DIR/test/cpu/tests" ]; then
  echo "=== CPU correctness suite (OMP_NUM_THREADS=${OMP_NUM_THREADS:-all}) ==="
  "$BUILD_DIR/test/cpu/tests"; ran=1
fi
[ "$ran" = 1 ] || { echo "No test binaries found in $BUILD_DIR, build first (scripts/build.sh $PRESET)."; exit 1; }

echo
echo "Randomized validators for the new GPU kernels (independent of the suite):"
for v in verify_scale_cuda verify_shared-fact_cuda fp32_accuracy_cuda; do
  [ -x "$BUILD_DIR/apps/$v" ] && echo "  $BUILD_DIR/apps/$v   (see README 'Verifying new kernels')"
done
