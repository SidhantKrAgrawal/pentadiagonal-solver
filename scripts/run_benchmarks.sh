#!/usr/bin/env bash
# run_benchmarks.sh — reproduce the headline results from the paper/report.
# Runs the full ADI solver (x+y+z, one time step) on the GPU and the CPU and
# prints a comparison.  Assumes the 'gpu' build exists (scripts/build.sh gpu).
#
#   bash scripts/run_benchmarks.sh [N] [ITERS]      (default: 256 50)
set -euo pipefail

N="${1:-256}"
ITERS="${2:-50}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
B="build"
APP_GPU="$B/apps/app_cuda"
APP_CPU="$B/apps/adi_cpu"
[ -x "$APP_GPU" ] || { echo "GPU app not built. Run: bash scripts/build.sh gpu"; exit 1; }

OUT="results/repro_$(date +%Y%m%d)"
mkdir -p "$OUT"
grab() { grep -E "total/iter:" | head -1 | awk '{print $2}'; }

echo "=== GPU, ${N}^3, $ITERS iters ==============================================="
echo "-- general-purpose solver (auto dispatch: FP32 Algorithm 3/thomas-pcr x + legacy y/z, FP64 Algorithm 2/transpose) --"
G64=$("$APP_GPU" "$N" "$ITERS" double | tee "$OUT/gpu_general_double.txt" | grab)
G32=$("$APP_GPU" "$N" "$ITERS" float  | tee "$OUT/gpu_general_float.txt"  | grab)
printf "   FP64 total/iter = %s ms\n   FP32 total/iter = %s ms\n" "$G64" "$G32"

echo "-- ADI-structured solver (Algorithm 4/shared-fact; all lines share coefficients) --"
S64=$(PENTA_XALGO=shared-fact PENTA_YZALGO=shared-fact "$APP_GPU" "$N" "$ITERS" double | tee "$OUT/gpu_shared-fact_double.txt" | grab)
S32=$(PENTA_XALGO=thomas-pcr  PENTA_YZALGO=shared-fact "$APP_GPU" "$N" "$ITERS" float  | tee "$OUT/gpu_shared-fact_float.txt"  | grab)
printf "   FP64 total/iter = %s ms\n   FP32 total/iter = %s ms  (Algorithm 3/thomas-pcr x + Algorithm 4/shared-fact y/z)\n" "$S64" "$S32"

if [ -x "$APP_CPU" ]; then
  echo "=== CPU (OpenMP), ${N}^3 ================================================="
  for T in 1 4; do
    C=$(OMP_NUM_THREADS=$T "$APP_CPU" "$N" 10 double | tee "$OUT/cpu_T${T}_double.txt" | grab)
    printf "   FP64 T=%s total/iter = %s ms\n" "$T" "$C"
  done
fi

echo
echo "Raw outputs saved under $OUT/"
echo "Kernel selectors: PENTA_XALGO / PENTA_YZALGO (see README 'Choosing the algorithm')."
