#!/usr/bin/env bash
# run_grid_transcript.sh: the grid study, recorded as a readable transcript.
#
#   bash scripts/run_grid_transcript.sh [GPU_ITERS] [CPU_ITERS]
#     defaults: 50 20
#   env: BUILD=<build dir, default build>   GRIDS="128 256 320 384"
#
# Every run is written out as
#
#     COMMAND  ->  OUTPUT (verbatim)  ->  RESULT (x / y / z / total)
#
# so each number in the summary tables can be traced back to the exact command
# that produced it.  Machine-readable cpu.csv / gpu.csv are emitted alongside.
#
# ---------------------------------------------------------------------------
# WHAT IS MEASURED, AND WHY IT IS ONE RUN PER ROW
# ---------------------------------------------------------------------------
# The per-axis x/y/z times are NOT separate executions.  A single execution
# does two passes over the same warmed-up state: Pass A times one full ADI
# iteration end-to-end (the headline total), Pass B re-runs it with CUDA events
# bracketing each direction (the breakdown).  Quoting sum(x+y+z) as the total
# would drop the inter-kernel gaps, so the total quoted is always Pass A.
#
# TRAP 1 -- LANES ARE NOT FREELY CHOOSABLE.  Algorithm 3 only instantiates
# certain (M,L) templates; an unsupported pair falls back rather than failing.
# Only the honoured pairs are run:
#     N=128: 8,16,32   N=256: 8,16,32   N=320: 32 only   N=384: 16,32
#
# TRAP 2 -- A REQUESTED KERNEL IS NOT ALWAYS THE KERNEL THAT RUNS.  The solver
# now records what it actually launched (pentadsolver_kernel_that_ran); the app
# prints it and warns on any mismatch, and this script propagates it to the CSV
# and flags the row.  Without that, a fallback reads as a real measurement of
# the requested algorithm.
#
# TRAP 3 -- OOM AT LARGE N.  Global-Transpose needs 15 arrays' worth of memory.
# A failing run is LOGGED as oom and the sweep continues; it is never silently
# dropped.
# ---------------------------------------------------------------------------
set -uo pipefail

GPU_ITERS="${1:-50}"
CPU_ITERS="${2:-20}"
GRIDS="${GRIDS:-128 256 320 384}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD:-build}"
APP="$BUILD/apps/app_cuda"
CPUAPP="$BUILD/apps/adi_cpu"
[ -x "$APP" ]    || { echo "not built: $APP"; exit 1; }
[ -x "$CPUAPP" ] || { echo "not built: $CPUAPP"; exit 1; }

HOST="$(hostname -s)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="results/grid_transcript_${HOST}_${STAMP}"
mkdir -p "$OUT"
TR="$OUT/transcript.txt"
: > "$TR"
echo ">> output: $OUT"

# Warm-up bounded by WALL TIME, not iteration count: at 8 ms/iteration a fixed
# 30-iteration warm-up is 240 ms, short of the ~1.2 s clock ramp, and the fast
# kernels get timed mid-ramp while the slow ones do not.
export PENTA_WARMUP_MS=3000

RUN_NO=0
TOTAL_RUNS=0
for N in $GRIDS; do
  TOTAL_RUNS=$((TOTAL_RUNS + 12))                     # cpu: 6 threads x 2 prec
  TOTAL_RUNS=$((TOTAL_RUNS + 2 + 2 + 2))              # naive/transpose/shared-fact
  case "$N" in
    128|256) TOTAL_RUNS=$((TOTAL_RUNS + 6)) ;;        # 3 lanes x 2 prec
    320)     TOTAL_RUNS=$((TOTAL_RUNS + 2)) ;;
    384)     TOTAL_RUNS=$((TOTAL_RUNS + 4)) ;;
  esac
done

# ---------------------------------------------------------------------------
# Peak bandwidth, measured UNDER LOAD.  CUDA sits in the P2 power state, which
# clocks memory below spec on both target cards, so the spec figure would
# understate every efficiency number.
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
  echo "cpu cores  : $(nproc)"
  echo "spec peak  : $SPEC_BW GB/s @ $MAXCLK MHz"
  echo "P2 clock   : $P2CLK MHz"
  echo "PEAK USED  : $PEAK_BW GB/s"
} > "$OUT/00_environment.txt"
echo ">> peak bandwidth: $PEAK_BW GB/s (P2 $P2CLK vs spec $MAXCLK MHz)"

{
  echo "################################################################################"
  echo "# MACHINE: $HOST"
  cat "$OUT/00_environment.txt" | sed 's/^/# /'
  echo "################################################################################"
  echo
} >> "$TR"

# banner "<title line>"
banner() {
  {
    echo
    echo "--------------------------------------------------------------------------------"
    printf 'RUN %03d/%03d  |  %s  |  %s\n' "$RUN_NO" "$TOTAL_RUNS" "$HOST" "$1"
    echo "--------------------------------------------------------------------------------"
  } >> "$TR"
}

# ===========================================================================
# 1. CPU OpenMP, threads 1..6
# ===========================================================================
CPUCSV="$OUT/cpu.csv"
echo "N,precision,threads,x_ms,y_ms,z_ms,total_ms" > "$CPUCSV"
echo ">> [1/5] CPU OpenMP, threads 1-6"
for N in $GRIDS; do
  for prec in double float; do
    for th in 1 2 3 4 5 6; do
      RUN_NO=$((RUN_NO + 1))
      pl=$([ "$prec" = double ] && echo FP64 || echo FP32)
      banner "CPU OpenMP  |  N=$N  |  $pl  |  threads=$th"
      CMD="OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores $CPUAPP $N $CPU_ITERS $prec"
      { echo; echo "COMMAND:"; echo "  $CMD"; echo; echo "OUTPUT:"; } >> "$TR"
      log="$OUT/.last.txt"
      if eval "$CMD" > "$log" 2>&1; then
        sed 's/^/  | /' "$log" >> "$TR"
        line="$(grep '^CSV,' "$log" | head -1 | cut -d, -f2-)"   # prec,N,th,x,y,z,tot
        if [ -n "$line" ]; then
          echo "$line" | awk -F, '{printf "%s,%s,%s,%s,%s,%s,%s\n",$2,$1,$3,$4,$5,$6,$7}' >> "$CPUCSV"
          echo "$line" | awk -F, '{printf "\nRESULT:\n  x = %8.3f ms    y = %8.3f ms    z = %8.3f ms    TOTAL = %9.3f ms\n",$4,$5,$6,$7}' >> "$TR"
        else
          echo "$N,$prec,$th,fail,fail,fail,fail" >> "$CPUCSV"
          echo "" >> "$TR"; echo "RESULT:  FAILED (no CSV line)" >> "$TR"
        fi
      else
        sed 's/^/  | /' "$log" >> "$TR"
        echo "$N,$prec,$th,fail,fail,fail,fail" >> "$CPUCSV"
        { echo; echo "RESULT:  FAILED (non-zero exit)"; } >> "$TR"
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
echo "algo,N,precision,lanes,x_sel,x_kernel,y_kernel,z_kernel,e2e_wall_ms,x_ms,y_ms,z_ms,status,honoured" > "$GPUCSV"

# gpu_run <label> <selector> <N> <prec> <lanes|-> <pretty title>
gpu_run() {
  local label="$1" sel="$2" N="$3" prec="$4" lanes="$5" title="$6"
  RUN_NO=$((RUN_NO + 1))
  banner "$title"

  local yz="$sel"
  # Global-Transpose is an x-direction kernel by design; y/z run the naive
  # strided solve.  Recorded here so its y/z columns are not read as its own.
  [ "$sel" = "transpose" ] && yz="naive"

  local lane_env=""
  [ "$lanes" != "-" ] && lane_env="PENTA_PCR_LANES=$lanes "
  local CMD="${lane_env}PENTA_XALGO=$sel PENTA_YALGO=$yz PENTA_ZALGO=$yz $APP $N $GPU_ITERS $prec"
  { echo; echo "COMMAND:"; echo "  $CMD"; echo; echo "OUTPUT:"; } >> "$TR"

  local log="$OUT/.last.txt"
  if eval "$CMD" > "$log" 2>&1; then
    sed 's/^/  | /' "$log" >> "$TR"
    local c; c="$(grep '^CSV,' "$log" | head -1)"
    if [ -n "$c" ]; then
      # fields: 2=req_x 3=ran_x 5=ran_y 7=ran_z 10=e2e_wall 12=x 13=y 14=z
      local ranx rany ranz
      ranx="$(echo "$c" | cut -d, -f3)"
      rany="$(echo "$c" | cut -d, -f5)"
      ranz="$(echo "$c" | cut -d, -f7)"
      # With no automatic dispatch the selector is the kernel, so this is a
      # cross-check against the solver's own record rather than a real question.
      local honoured=yes
      [ "$ranx" = "$sel" ] || honoured=no
      [ "$rany" = "$yz"  ] || honoured=no
      [ "$ranz" = "$yz"  ] || honoured=no
      echo "$label,$N,$prec,$lanes,$(echo "$c" | cut -d, -f2,3,5,7,10,12,13,14),ok,$honoured" >> "$GPUCSV"
      echo "$c" | awk -F, -v h="$honoured" -v rx="$ranx" -v ry="$rany" -v rz="$ranz" \
        '{printf "\nRESULT:\n  x = %8.3f ms    y = %8.3f ms    z = %8.3f ms    TOTAL(end-to-end wall) = %9.3f ms\n  kernels that ran: x=%s y=%s z=%s   request honoured: %s\n",$12,$13,$14,$10,rx,ry,rz,h}' >> "$TR"
      echo "   $label N=$N $prec L=$lanes -> e2e $(echo "$c" | cut -d, -f10) ms (honoured=$honoured)"
      return 0
    fi
  else
    sed 's/^/  | /' "$log" >> "$TR"
  fi
  local st="fail"
  grep -qi "out of memory\|cudaErrorMemoryAllocation" "$log" && st="oom"
  echo "$label,$N,$prec,$lanes,$sel,-,-,-,-,-,-,-,$st,-" >> "$GPUCSV"
  { echo; echo "RESULT:  $st"; } >> "$TR"
  echo "   !! $label N=$N $prec L=$lanes -> $st"
}

echo ">> [2/5] GPU Naive"
for N in $GRIDS; do for prec in double float; do
  pl=$([ "$prec" = double ] && echo FP64 || echo FP32)
  gpu_run naive naive "$N" "$prec" - "ALGO 1  GPU Naive  |  N=$N  |  $pl"
done; done

echo ">> [3/5] Global-Transpose"
for N in $GRIDS; do for prec in double float; do
  pl=$([ "$prec" = double ] && echo FP64 || echo FP32)
  gpu_run transpose transpose "$N" "$prec" - "ALGO 2  Global-Transpose (x only; y/z = naive)  |  N=$N  |  $pl"
done; done

echo ">> [4/5] Hybrid Thomas-PCR, honoured (N,lanes) pairs only"
lanes_for() {
  case "$1" in
    128) echo "8 16 32" ;;
    256) echo "8 16 32" ;;
    320) echo "32"      ;;   # 320/16 = 20, not instantiated
    384) echo "16 32"   ;;   # 384/8  = 48, not instantiated
    *)   echo "32"      ;;
  esac
}
for N in $GRIDS; do for prec in double float; do
  pl=$([ "$prec" = double ] && echo FP64 || echo FP32)
  for L in $(lanes_for "$N"); do
    gpu_run thomas-pcr thomas-pcr "$N" "$prec" "$L" "ALGO 3  Hybrid Thomas-PCR  |  N=$N  |  $pl  |  lanes=$L"
  done
done; done

echo ">> [5/5] Shared-Factorisation"
for N in $GRIDS; do for prec in double float; do
  pl=$([ "$prec" = double ] && echo FP64 || echo FP32)
  gpu_run shared-fact shared-fact "$N" "$prec" - "ALGO 4  Shared-Factorisation  |  N=$N  |  $pl"
done; done

rm -f "$OUT/.last.txt"

echo
echo "=========================================================="
echo " done: $OUT"
printf " cpu rows: %s   gpu rows: %s   (oom: %s, fail: %s, not-honoured: %s)\n" \
  "$(($(wc -l < "$CPUCSV") - 1))" "$(($(wc -l < "$GPUCSV") - 1))" \
  "$(grep -c ',oom,' "$GPUCSV" || true)" "$(grep -c ',fail,' "$GPUCSV" || true)" \
  "$(grep -c ',no$' "$GPUCSV" || true)"
echo "=========================================================="
