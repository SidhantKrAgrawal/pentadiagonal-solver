#!/usr/bin/env bash
# run_grid_sweep.sh — problem-size scaling across every algorithm.
#
#   bash scripts/run_grid_sweep.sh [GPU_ITERS] [CPU_ITERS]
#     defaults: 50 20
#   env: BUILD=build|build-panda
#
# ---------------------------------------------------------------------------
# WHAT THIS RUNS, AND THE TWO TRAPS IT AVOIDS
# ---------------------------------------------------------------------------
# Grids N in {128,256,320,384}, both precisions, on one machine:
#
#   1. CPU OpenMP        6 thread counts x 4 N x 2 prec  = 48 runs
#   2. GPU Naive                         4 N x 2 prec    =  8
#   3. Global-Transpose                  4 N x 2 prec    =  8
#   4. Hybrid Thomas-PCR    9 valid (N,lanes) x 2 prec   = 18
#   5. Shared-Factorisation              4 N x 2 prec    =  8
#                                                  total = 90 per machine
#
# TRAP 1 -- LANES ARE NOT FREELY CHOOSABLE.  pentadsolver_batch_x_algo3 only
# instantiates certain (M,L) templates.  An unsupported (N,lanes) pair does NOT
# error: it falls through to the 32-lane path, so a run labelled "L=8" would
# silently be measuring L=32.  Only the 9 honoured pairs are run:
#     N=128: 8,16,32   N=256: 8,16,32   N=320: 32 only   N=384: 16,32
# (320/16=20 and 384/8=48 are not instantiated.)
#
# TRAP 2 -- OOM AT LARGE N.  Global-Transpose needs 15 arrays' worth of memory
# (6 inputs + 9 scratch).  At N=384 FP64 that is 6.8 GB, which does not
# comfortably fit an 8 GB card alongside the driver context.  A failing run is
# LOGGED AS oom AND THE SWEEP CONTINUES -- it is never silently dropped, and it
# is never left to look like a missing measurement.
#
# Every GPU row records the kernel that actually ran, so any fallback is
# visible in the CSV rather than mislabelled as the requested algorithm.
# ---------------------------------------------------------------------------
set -uo pipefail

GPU_ITERS="${1:-50}"
CPU_ITERS="${2:-20}"
GRIDS="${GRIDS:-128 256 320 384}"   # overridable for a quick smoke test

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD:-build}"
APP="$BUILD/apps/app_cuda"
CPUAPP="$BUILD/apps/adi_cpu"
[ -x "$APP" ]    || { echo "not built: $APP"; exit 1; }
[ -x "$CPUAPP" ] || { echo "not built: $CPUAPP"; exit 1; }

HOST="$(hostname -s)"
OUT="results/grid_sweep_${HOST}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/logs"
echo ">> output: $OUT"

# Same warm-up discipline as the 2026-08-08 campaign: bounded by WALL TIME, not
# by iteration count, so a fast kernel at a small N is not timed mid-clock-ramp.
export PENTA_WARMUP_MS=3000

# ---------------------------------------------------------------------------
# Peak bandwidth, measured under load (CUDA sits in the P2 power state, which
# underclocks memory below spec on both target cards).
# ---------------------------------------------------------------------------
SPEC_BW="$("$APP" 64 1 double 2>/dev/null | grep -o 'is against [0-9.]* GB/s' | awk '{print $3}')"
MAXCLK="$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader | awk '{print $1}')"
"$APP" 256 400 double >/dev/null 2>&1 &
LOADPID=$!
sleep 5
P2CLK="$(for i in 1 2 3 4 5; do
           nvidia-smi --query-gpu=clocks.mem --format=csv,noheader | awk '{print $1}'; sleep 1
         done | sort -n | tail -1)"
kill "$LOADPID" 2>/dev/null; wait "$LOADPID" 2>/dev/null
PEAK_BW="$(awk -v s="$SPEC_BW" -v p="$P2CLK" -v m="$MAXCLK" 'BEGIN{printf "%.1f", s*p/m}')"
export PENTA_PEAK_BW_GBS="$PEAK_BW"
{
  echo "host       : $HOST"
  echo "date       : $(date -Is)"
  echo "build      : $BUILD"
  echo "grids      : $GRIDS"
  echo "iters      : gpu=$GPU_ITERS cpu=$CPU_ITERS   warmup=${PENTA_WARMUP_MS}ms wall"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  grep -m1 'model name' /proc/cpuinfo
  echo "spec peak  : $SPEC_BW GB/s @ $MAXCLK MHz"
  echo "P2 clock   : $P2CLK MHz"
  echo "PEAK USED  : $PEAK_BW GB/s"
} > "$OUT/00_environment.txt"
echo ">> peak bandwidth: $PEAK_BW GB/s (P2 $P2CLK vs spec $MAXCLK MHz)"

# ===========================================================================
# 1. CPU OpenMP — threads 1..6
# ===========================================================================
CPUCSV="$OUT/cpu.csv"
echo "N,precision,threads,x_ms,y_ms,z_ms,total_ms" > "$CPUCSV"
echo ">> [1/5] CPU OpenMP, threads 1-6"
for N in $GRIDS; do
  for prec in double float; do
    for th in 1 2 3 4 5 6; do
      log="$OUT/logs/cpu_N${N}_${prec}_t${th}.txt"
      if OMP_NUM_THREADS="$th" OMP_PROC_BIND=close OMP_PLACES=cores \
           "$CPUAPP" "$N" "$CPU_ITERS" "$prec" > "$log" 2>&1; then
        line="$(grep '^CSV,' "$log" | head -1 | cut -d, -f2-)"   # prec,N,th,x,y,z,tot
        if [ -n "$line" ]; then
          # reorder to N,precision,threads,...
          echo "$line" | awk -F, '{printf "%s,%s,%s,%s,%s,%s,%s\n",$2,$1,$3,$4,$5,$6,$7}' >> "$CPUCSV"
        fi
      else
        echo "$N,$prec,$th,fail,fail,fail,fail" >> "$CPUCSV"
        echo "   !! CPU N=$N $prec t=$th FAILED"
      fi
    done
    echo "   cpu N=$N $prec done"
  done
done

# ===========================================================================
# 2-5. GPU
# ===========================================================================
GPUCSV="$OUT/gpu.csv"
echo "algo,N,precision,lanes,x_sel,x_kernel,y_kernel,z_kernel,e2e_wall_ms,x_ms,y_ms,z_ms,status" > "$GPUCSV"

# $1=label $2=selector $3=N $4=prec $5=lanes("-" for n/a) $6=tag
gpu_run() {
  local label="$1" sel="$2" N="$3" prec="$4" lanes="$5" tag="$6"
  local log="$OUT/logs/${tag}.txt"
  local yz="$sel"
  # Global-Transpose is x-only by design; pair it with the naive strided kernel.
  [ "$sel" = "transpose" ] && yz="naive"
  local env_lanes=()
  [ "$lanes" != "-" ] && env_lanes=(PENTA_PCR_LANES="$lanes")
  if env "${env_lanes[@]}" PENTA_XALGO="$sel" PENTA_YALGO="$yz" PENTA_ZALGO="$yz" \
         "$APP" "$N" "$GPU_ITERS" "$prec" > "$log" 2>&1; then
    local c; c="$(grep '^CSV,' "$log" | head -1)"
    if [ -n "$c" ]; then
      echo "$label,$N,$prec,$lanes,$(echo "$c" | cut -d, -f2,3,5,7,10,12,13,14),ok" >> "$GPUCSV"
      echo "   $label N=$N $prec L=$lanes -> e2e $(echo "$c" | cut -d, -f10) ms"
      return 0
    fi
  fi
  # Distinguish out-of-memory from any other failure — they mean different things.
  local st="fail"
  grep -qi "out of memory\|cudaErrorMemoryAllocation" "$log" && st="oom"
  echo "$label,$N,$prec,$lanes,$sel,-,-,-,-,-,-,-,$st" >> "$GPUCSV"
  echo "   !! $label N=$N $prec L=$lanes -> $st (see $log)"
}

echo ">> [2/5] GPU Naive"
for N in $GRIDS; do for prec in double float; do
  gpu_run naive naive "$N" "$prec" - "naive_N${N}_${prec}"
done; done

echo ">> [3/5] Global-Transpose"
for N in $GRIDS; do for prec in double float; do
  gpu_run transpose transpose "$N" "$prec" - "transpose_N${N}_${prec}"
done; done

echo ">> [4/5] Hybrid Thomas-PCR, honoured (N,lanes) pairs only"
lanes_for() {   # only the templates that are actually instantiated
  case "$1" in
    128) echo "8 16 32" ;;
    256) echo "8 16 32" ;;
    320) echo "32"      ;;   # 320/16 = 20 not instantiated
    384) echo "16 32"   ;;   # 384/8  = 48 not instantiated
    *)   echo "32"      ;;
  esac
}
for N in $GRIDS; do for prec in double float; do
  for L in $(lanes_for "$N"); do
    gpu_run thomas-pcr thomas-pcr "$N" "$prec" "$L" "pcr_N${N}_${prec}_L${L}"
  done
done; done

echo ">> [5/5] Shared-Factorisation (restricted class)"
for N in $GRIDS; do for prec in double float; do
  gpu_run shared-fact shared-fact "$N" "$prec" - "sharedfact_N${N}_${prec}"
done; done

# Production dispatch, for reference: what a caller actually gets at each N
# without setting any environment variable.  Recorded separately because it is
# a CONFIGURATION, not an algorithm — its x/y/z may come from different kernels.
echo ">> [+] auto dispatch (production default)"
for N in $GRIDS; do for prec in double float; do
  gpu_run auto auto "$N" "$prec" - "auto_N${N}_${prec}"
done; done

echo
echo "=========================================================="
echo " done: $OUT"
printf " cpu rows: %s   gpu rows: %s   (oom: %s, fail: %s)\n" \
  "$(($(wc -l < "$CPUCSV") - 1))" "$(($(wc -l < "$GPUCSV") - 1))" \
  "$(grep -c ',oom$' "$GPUCSV" || true)" "$(grep -c ',fail$' "$GPUCSV" || true)"
echo "=========================================================="
