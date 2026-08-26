#!/usr/bin/env bash
# run_full_study.sh — the complete measurement campaign for one machine.
#
#   bash scripts/run_full_study.sh [N] [GPU_ITERS] [CPU_ITERS]
#     defaults: 256 50 20
#   env: BUILD=build|build-panda   NSYS=<path to a working nsys>
#
# ---------------------------------------------------------------------------
# WHAT THIS COLLECTS, AND WHY EACH PIECE IS NEEDED
# ---------------------------------------------------------------------------
# The goal is a table where every number is attributable and every bandwidth
# figure has an honest denominator.  Six stages:
#
#   0. ENVIRONMENT + PEAK BANDWIDTH.  Everything downstream is quoted as a
#      percentage of peak, so the peak must be measured, not assumed.  CUDA
#      compute workloads sit in the P2 power state, which clocks memory BELOW
#      the spec figure on both cards here (RTX 3050: 6801 vs 7001 MHz;
#      GTX 1080: 4513 vs 5005).  Quoting "% of spec" would understate every
#      efficiency by 3-10%.  So the script samples clocks.mem WHILE the solver
#      runs and rescales the device's spec peak by the ratio it observes.
#
#   1. CPU OpenMP THREAD SCALING.  1/2/4/6 threads, both precisions.  Gives
#      the parallel-efficiency curve and the serial baseline that every GPU
#      speedup is quoted against.
#
#   2. GPU, UNIFORM CONFIGURATION.  Each algorithm driven in all three
#      directions it has a kernel for, so total / x / y / z come from ONE run
#      and actually add up.  This is the "what does this algorithm cost" table.
#
#   3. GPU, CONTROLLED CONFIGURATION.  One direction varied, the other two
#      pinned to a fixed baseline.  Isolates a single kernel's contribution,
#      which the uniform runs cannot do (there, three kernels change at once).
#      Stages 2 and 3 answer different questions and neither replaces the other.
#
#   4. LANES SWEEP.  PENTA_PCR_LANES over {8,16,32}.  x-direction only --
#      the override lives in the x launcher, so a y/z row would be a duplicate
#      of the default and would falsely suggest the knob did nothing there.
#
#   5. ROOFLINE + PER-KERNEL BREAKDOWN.  The Empirical Roofline Tool (ERT) from
#      Berkeley Lab measures this machine's memory and compute roofs directly
#      (ncu is blocked estate-wide, ERR_NVGPUCTRPERM).  nsys supplies per-kernel
#      times, which is what splits a fused number like Global-Transpose into
#      transpose cost vs solve cost.  perf does the same job on the CPU side.
# ---------------------------------------------------------------------------
set -uo pipefail

N="${1:-256}"
GPU_ITERS="${2:-50}"
CPU_ITERS="${3:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD:-build}"
APP="$BUILD/apps/app_cuda"
CPUAPP="$BUILD/apps/adi_cpu"
[ -x "$APP" ]    || { echo "not built: $APP"; exit 1; }
[ -x "$CPUAPP" ] || { echo "not built: $CPUAPP"; exit 1; }

HOST="$(hostname -s)"
OUT="results/full_study_${HOST}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo ">> output: $OUT"

# Warm-up policy.  Set by WALL TIME, not by an iteration count: what must
# finish is a clock ramp measured in milliseconds, and a fixed count gives a
# fast configuration far less ramp than a slow one (at 8 ms/iteration, 30
# iterations is 240 ms; at 64 ms/iteration it is 1.9 s).
#
# 3000 ms is deliberately well past the measured 1.2 s knee on the slower-
# ramping of the two cards.  Warm-up by wall time also equalises THERMAL
# preconditioning across configurations, which a fixed iteration count does
# not: every row here starts from the same 3 s of sustained load regardless of
# how fast its kernel is.
export PENTA_WARMUP_MS=3000
WARMUP_FLOOR=5

# ===========================================================================
# STAGE 0 — environment and the peak-bandwidth denominator
# ===========================================================================
ENVF="$OUT/00_environment.txt"
{
  echo "host      : $HOST"
  echo "date      : $(date -Is)"
  echo "build dir : $BUILD"
  echo "grid      : ${N}^3   gpu_iters=$GPU_ITERS  cpu_iters=$CPU_ITERS"
  echo
  echo "--- CPU ---"
  grep -m1 'model name' /proc/cpuinfo
  echo "cores (nproc): $(nproc)"
  echo
  echo "--- GPU ---"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
} > "$ENVF"

# Spec peak, as the device reports it (memoryClockRate x busWidth).
SPEC_BW="$("$APP" 64 1 double 2>/dev/null | grep -o 'is against [0-9.]* GB/s' | awk '{print $3}')"
MAXCLK="$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader | awk '{print $1}')"

# P2 clock, sampled WHILE the solver runs.  Idle sampling would read the P8
# clock and understate the peak badly, so the load must be live.
"$APP" "$N" 400 double >/dev/null 2>&1 &
LOADPID=$!
sleep 5
P2CLK="$(for i in 1 2 3 4 5; do
           nvidia-smi --query-gpu=clocks.mem --format=csv,noheader | awk '{print $1}'
           sleep 1
         done | sort -n | tail -1)"
kill "$LOADPID" 2>/dev/null; wait "$LOADPID" 2>/dev/null

PEAK_BW="$(awk -v s="$SPEC_BW" -v p="$P2CLK" -v m="$MAXCLK" 'BEGIN{printf "%.1f", s*p/m}')"
export PENTA_PEAK_BW_GBS="$PEAK_BW"
{
  echo
  echo "--- peak bandwidth denominator (measured, not assumed) ---"
  echo "spec peak from device      : $SPEC_BW GB/s  @ $MAXCLK MHz"
  echo "memory clock UNDER LOAD    : $P2CLK MHz  (P2 power state)"
  echo "peak used for all % figures: $PEAK_BW GB/s"
  echo "  = $SPEC_BW x $P2CLK / $MAXCLK"
} >> "$ENVF"
echo ">> peak bandwidth: $PEAK_BW GB/s (P2 clock $P2CLK vs spec $MAXCLK MHz)"

# ===========================================================================
# STAGE 1 — CPU OpenMP thread scaling
# ===========================================================================
CPUCSV="$OUT/01_cpu_threads.csv"
echo "precision,N,threads,x_ms,y_ms,z_ms,total_ms" > "$CPUCSV"
echo ">> stage 1: CPU thread sweep"
for prec in double float; do
  for th in 1 2 4 6; do
    log="$OUT/logs/cpu_${prec}_t${th}.txt"; mkdir -p "$OUT/logs"
    OMP_NUM_THREADS="$th" OMP_PROC_BIND=close OMP_PLACES=cores \
      "$CPUAPP" "$N" "$CPU_ITERS" "$prec" > "$log" 2>&1
    line="$(grep '^CSV,' "$log" | head -1 | cut -d, -f2-)"
    [ -n "$line" ] && echo "$line" >> "$CPUCSV" && echo "   cpu $prec t=$th -> $line"
  done
done

# ===========================================================================
# STAGE 2/3 — GPU algorithm sweep
# ===========================================================================
# Selector per algorithm and direction.  "-" = this algorithm has no kernel
# for that direction, and is skipped rather than silently falling back to auto
# (a fallback would be recorded under the wrong algorithm's name).
sel_x()  { case "$1" in 1) echo naive;; 2) echo transpose;; 3) echo thomas-pcr;; 4) echo shared-fact;; esac; }
sel_yz() { case "$1" in 1) echo naive;; 2) echo -;;         3) echo thomas-pcr;; 4) echo shared-fact;; esac; }
algo_name() {
  case "$1" in
    1) echo "Naive / thread-per-system" ;;
    2) echo "Global-Transpose (x-only)" ;;
    3) echo "Hybrid Thomas-PCR (SPIKE)" ;;
    4) echo "Shared-Factorisation [restricted]" ;;
  esac
}
BASE_X="transpose"        # must be a selector the solver really recognises:
BASE_STRIDED="naive"      # an unknown value silently becomes auto dispatch.

GPUCSV="$OUT/02_gpu_sweep.csv"
echo "mode,algo,x_sel,x_kernel,y_sel,y_kernel,z_sel,z_kernel,precision,N,e2e_wall_ms,e2e_events_ms,x_ms,y_ms,z_ms,sum_xyz_ms,overhead_pct" > "$GPUCSV"

run_cfg() {   # $1=mode $2=algo $3=x $4=y $5=z $6=prec $7=tag
  local mode="$1" algo="$2" xs="$3" ys="$4" zs="$5" prec="$6" tag="$7"
  local log="$OUT/logs/${tag}.txt"; mkdir -p "$OUT/logs"
  local wu="$WARMUP_FLOOR"
  if ! env PENTA_WARMUP="$wu" PENTA_XALGO="$xs" PENTA_YALGO="$ys" PENTA_ZALGO="$zs" \
        "$APP" "$N" "$GPU_ITERS" "$prec" > "$log" 2>&1; then
    echo "   !! FAILED $tag (see $log)" >&2; return 1
  fi
  local line; line="$(grep '^CSV,' "$log" | head -1 | cut -d, -f2-)"
  [ -n "$line" ] || { echo "   !! no CSV in $tag" >&2; return 1; }
  echo "$mode,$algo,$line" >> "$GPUCSV"
  echo "   $mode a$algo $prec -> $line"
}

echo ">> stage 2: GPU uniform configurations (total and x/y/z from one run)"
for prec in double float; do
  for a in 1 2 3 4; do
    xs="$(sel_x "$a")"; ys="$(sel_yz "$a")"
    # Algorithm 2 is x-only by design (y/z are already coalesced, so a
    # transpose there would be pure added cost).  Its "uniform" run therefore
    # pairs its x kernel with the naive strided kernel, and is labelled so.
    [ "$ys" = "-" ] && ys="$BASE_STRIDED"
    run_cfg uniform "$a" "$xs" "$ys" "$ys" "$prec" "u_a${a}_${prec}"
  done
  run_cfg uniform auto auto auto auto "$prec" "u_auto_${prec}"
done

echo ">> stage 3: GPU controlled (one direction varied, others pinned)"
for prec in double float; do
  for a in 1 2 3 4; do
    xs="$(sel_x "$a")"
    [ "$xs" != "-" ] && run_cfg ctrl_x "$a" "$xs" "$BASE_STRIDED" "$BASE_STRIDED" "$prec" "c_x_a${a}_${prec}"
    ys="$(sel_yz "$a")"
    if [ "$ys" != "-" ]; then
      run_cfg ctrl_y "$a" "$BASE_X" "$ys" "$BASE_STRIDED" "$prec" "c_y_a${a}_${prec}"
      run_cfg ctrl_z "$a" "$BASE_X" "$BASE_STRIDED" "$ys" "$prec" "c_z_a${a}_${prec}"
    fi
  done
done

# ===========================================================================
# STAGE 4 — PENTA_PCR_LANES sweep (x-direction only; see header)
# ===========================================================================
echo ">> stage 4: Thomas-PCR lanes sweep"
LANECSV="$OUT/03_lanes.csv"
echo "lanes,precision,N,x_ms,e2e_wall_ms" > "$LANECSV"
for prec in double float; do
  for L in 8 16 32; do
    tag="lanes${L}_${prec}"; log="$OUT/logs/${tag}.txt"
    wu="$WARMUP_FLOOR"
    env PENTA_WARMUP="$wu" PENTA_PCR_LANES="$L" \
        PENTA_XALGO=thomas-pcr PENTA_YALGO="$BASE_STRIDED" PENTA_ZALGO="$BASE_STRIDED" \
        "$APP" "$N" "$GPU_ITERS" "$prec" > "$log" 2>&1
    line="$(grep '^CSV,' "$log" | head -1)"
    if [ -n "$line" ]; then
      xms="$(echo "$line" | cut -d, -f12)"; e2e="$(echo "$line" | cut -d, -f10)"
      echo "$L,$prec,$N,$xms,$e2e" >> "$LANECSV"
      echo "   lanes=$L $prec -> x=$xms ms"
    fi
  done
done

# ===========================================================================
# STAGE 5 — machine roofline, per-kernel breakdown, CPU profile
# ===========================================================================
# The roofline is measured with Berkeley Lab's Empirical Roofline Tool, not with
# anything built in this repository.  Skipped without failing the study if ERT
# has not been fetched; scripts/get_ert.sh installs it.
if [ -x "$ROOT/third_party/ert/Empirical_Roofline_Tool-1.1.0/ert" ]; then
  echo ">> stage 5a: machine roofline (Berkeley ERT, FP64 and FP32)"
  mkdir -p "$OUT/logs"
  bash "$ROOT/scripts/run_ert.sh" "$OUT/04_roofline_ert" > "$OUT/logs/ert.txt" 2>&1
  if [ -f "$OUT/04_roofline_ert/SUMMARY_ERT.txt" ]; then
    sed 's/^/   /' "$OUT/04_roofline_ert/SUMMARY_ERT.txt"
  else
    echo "   ERT produced no summary -- see $OUT/logs/ert.txt"
  fi
else
  echo ">> stage 5a: skipped, ERT not installed (run: bash scripts/get_ert.sh)"
fi

NSYS="${NSYS:-}"
if [ -n "$NSYS" ] && [ -x "$NSYS" ]; then
  echo ">> stage 5b: per-kernel breakdown (nsys)"
  mkdir -p "$OUT/nsys"
  # Few iterations: this asks WHERE time goes (relative kernel shares), not
  # how fast it is.  Tracing perturbs absolute timings, so the totals here are
  # deliberately not used as performance numbers.
  for prec in double float; do
    for a in 1 2 3 4; do
      xs="$(sel_x "$a")"; ys="$(sel_yz "$a")"; [ "$ys" = "-" ] && ys="$BASE_STRIDED"
      rep="$OUT/nsys/a${a}_${prec}"
      env PENTA_WARMUP=2 PENTA_XALGO="$xs" PENTA_YALGO="$ys" PENTA_ZALGO="$ys" \
        "$NSYS" profile --trace=cuda --force-overwrite=true -o "$rep" \
        "$APP" "$N" 10 "$prec" >/dev/null 2>&1
      "$NSYS" stats --report cuda_gpu_kern_sum --format table "$rep.nsys-rep" \
        > "$OUT/nsys/kern_a${a}_${prec}.txt" 2>/dev/null
      echo "   nsys a$a $prec done"
    done
  done
  rm -f "$OUT"/nsys/*.nsys-rep "$OUT"/nsys/*.sqlite
fi

if command -v perf >/dev/null 2>&1; then
  echo ">> stage 5c: CPU hot-function profile (perf)"
  mkdir -p "$OUT/perf"
  for prec in double float; do
    OMP_NUM_THREADS=6 perf record -q -g --call-graph dwarf -o "$OUT/perf/$prec.data" \
      "$CPUAPP" "$N" 3 "$prec" >/dev/null 2>&1
    perf report -i "$OUT/perf/$prec.data" --stdio --no-children --percent-limit 0.3 \
      2>/dev/null | grep -E '^ +[0-9]' > "$OUT/perf/cpu_hot_${prec}.txt"
    rm -f "$OUT/perf/$prec.data"
    echo "   perf $prec done"
  done
fi

echo
echo "=========================================================="
echo " done: $OUT"
echo "=========================================================="
ls "$OUT"
