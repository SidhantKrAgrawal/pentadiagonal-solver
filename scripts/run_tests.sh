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
# The MPI suites need a launcher, so they are separate binaries and are only
# built by the gpu-mpi preset.  NP defaults to 2; the reduced-system path needs
# at least 2 ranks to exercise anything.
if [ "$PRESET" = gpu-mpi ]; then
  NP="${NP:-2}"
  if command -v mpirun >/dev/null 2>&1; then
    for t in test/cpu/mpi_tests test/cuda/cuda_mpi_tests; do
      if [ -x "$BUILD_DIR/$t" ]; then
        echo
        echo "=== $(basename "$t") (mpirun -np $NP) ==="
        mpirun --oversubscribe -np "$NP" "$BUILD_DIR/$t"; ran=1
      fi
    done
  else
    echo
    echo "mpirun not in PATH, skipping the MPI suites."
  fi
fi

[ "$ran" = 1 ] || { echo "No test binaries found in $BUILD_DIR, build first (scripts/build.sh $PRESET)."; exit 1; }

echo
echo "Randomized validators for the new GPU kernels (independent of the suite):"
for v in verify_scale_cuda verify_shared_fact_cuda verify_strided_cuda fp32_accuracy_cuda; do
  [ -x "$BUILD_DIR/apps/$v" ] && echo "  $BUILD_DIR/apps/$v   (see README 'Verifying a kernel independently')"
done
