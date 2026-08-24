#!/usr/bin/env bash
# =============================================================================
# run_hpc_study.sh -- ONE COMMAND. Builds, verifies, measures, summarises.
#
#     bash scripts/run_hpc_study.sh
#
# Written to be run by someone who has not seen this code before and does not
# want to debug it.  It configures, builds, checks correctness, measures the
# machine's own memory/compute roofline, runs the full algorithm sweep, measures
# this machine's CPU as a baseline, writes a summary readable without the raw
# data, and tars the lot.
#
# Nothing is interactive.  Nothing is silently skipped: every step that fails
# is recorded in the CSV and the summary with the reason, and the run carries
# on.  The only hard stops are "no compiler" and "correctness failed".
#
# Expected wall time: 10-25 min total on a V100 or H100 (build ~5-10 min of
# that, because the CUDA templates are instantiated for several tile sizes).
#
# -----------------------------------------------------------------------------
# IF SOMETHING GOES WRONG, IT IS ALMOST CERTAINLY ONE OF THESE THREE
# -----------------------------------------------------------------------------
#  1. No internet at CONFIGURE time.  The build fetches project_options,
#     Catch2 and Google Benchmark from GitHub once.  Configure on a LOGIN node
#     (the downloads cache into the build dir; compiling and running are then
#     fine on an offline compute node).
#  2. No nvcc.  module load cuda   (or add $CUDA_HOME/bin to PATH).
#  3. Building on a login node with no GPU.  Handled: with no GPU visible the
#     script builds a fat binary covering the common datacentre and consumer
#     architectures, so it still produces a runnable binary.  Name the target
#     explicitly with e.g.  CUDA_ARCH=80 (A100) if that is known.
#
# -----------------------------------------------------------------------------
# KNOBS (all optional)
# -----------------------------------------------------------------------------
#   CUDA_ARCH=86        compute capabilities to build for.  The default is the
#                       compute capability of whatever GPU this node has; with
#                       no GPU visible it falls back to a fat binary.
#   GRIDS="128 256 320 384"     problem sizes N (cube N^3)
#                       add 512 if the card has >=32 GB:  GRIDS="128 256 320 384 512"
#   GPU_ITERS=50        timed ADI iterations per measurement
#   CPU_ITERS=10        timed ADI iterations per CPU baseline measurement.  The
#                       CPU solver is ~50x slower per iteration, hence the
#                       smaller count.
#   OUTDIR=<path>       where results go (default: results/hpc_<host>_<date>)
#   SKIP_BUILD=1        reuse an existing build dir
#   JOBS=16             build parallelism
#
# Under Slurm, just put this line in the job script; it needs one GPU.
# =============================================================================

# NOT `set -e`: one failed measurement must not abandon the campaign.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "cannot cd to repo root"; exit 1; }

# Resolved once the visible GPU is known, unless the caller named one.
CUDA_ARCH="${CUDA_ARCH:-}"
GRIDS="${GRIDS:-128 256 320 384}"
GPU_ITERS="${GPU_ITERS:-50}"
JOBS="${JOBS:-$( (nproc 2>/dev/null) || echo 8 )}"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
BUILD="build-hpc"
OUT="${OUTDIR:-results/hpc_${HOST}_$(date +%Y%m%d_%H%M%S)}"

# Warm-up is bounded by WALL TIME, not iteration count.  A fixed warm-up count
# gives a slow kernel more clock-ramp time than a fast one, which silently
# flatters the slow kernel; that bug cost this project a wrong result once.
export PENTA_WARMUP_MS="${PENTA_WARMUP_MS:-3000}"

mkdir -p "$OUT/logs" || { echo "cannot create $OUT"; exit 1; }
LOG="$OUT/run.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
hdr() { say ""; say "==============================================================================="; say "$*"; say "==============================================================================="; }

# Nothing may hang a batch job, so every measurement runs under `timeout`.
# TMO expands to "timeout <secs>" and is placed AFTER any VAR=value prefixes
# and BEFORE the binary -- putting it first would make timeout try to exec
# "PENTA_XALGO=..." as a program.  Empty if coreutils timeout is unavailable.
if command -v timeout >/dev/null 2>&1; then TMO() { echo "timeout $1"; }
else                                       TMO() { echo ""; }; fi

hdr "pentadiagonal-solver :: HPC study :: $HOST :: $(date -Is 2>/dev/null || date)"

# -----------------------------------------------------------------------------
# 0. Preflight -- report everything, then fail only on what is genuinely fatal.
# -----------------------------------------------------------------------------
hdr "[0/6] Environment"
{
  echo "host          : $HOST"
  echo "date          : $(date -Is 2>/dev/null || date)"
  echo "slurm job     : ${SLURM_JOB_ID:-<not under slurm>}"
  echo "repo          : $ROOT"
  echo "git commit    : $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"
  echo "cmake         : $(cmake --version 2>/dev/null | head -1 || echo MISSING)"
  echo "g++           : $(g++ --version 2>/dev/null | head -1 || echo MISSING)"
  echo "nvcc          : $(nvcc --version 2>/dev/null | grep -i release | sed 's/^ *//' || echo MISSING)"
  echo "cuda arch     : $CUDA_ARCH"
  echo "cpu cores     : $(nproc 2>/dev/null || echo '?')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
               --format=csv 2>/dev/null || nvidia-smi 2>/dev/null | head -12
  else
    echo "GPU           : nvidia-smi not present (fine at build time; needed at run time)"
  fi
  echo "grids         : $GRIDS"
  echo "gpu iters     : $GPU_ITERS   warm-up ${PENTA_WARMUP_MS} ms wall"
} | tee "$OUT/00_environment.txt" | tee -a "$LOG"

command -v cmake >/dev/null 2>&1 || { say "FATAL: cmake not found."; exit 1; }
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  command -v nvcc >/dev/null 2>&1 || {
    say ""; say "FATAL: nvcc is not in PATH.  Try:  module load cuda"
    say "       (or add \$CUDA_HOME/bin to PATH), then re-run this script."; exit 1; }
fi

# -----------------------------------------------------------------------------
# 1. Build
# -----------------------------------------------------------------------------
# Drop any architecture this nvcc cannot target.  A CUDA 11.0 toolkit on a V100
# node cannot build sm_90, and would fail the whole run over an architecture
# that machine does not even have.  Filtering here is the difference between
# "it worked" and an email exchange about an nvcc error code.
# What GPU is actually visible right now?  If this node has one, its compute
# capability is folded into the build list unconditionally -- a binary that
# cannot run on the machine that built it is the likeliest way for this script
# to waste time.
VISIBLE_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | tr -d ' .' | sort -u | tr '\n' ' ')"
fi

# Default to exactly the GPU in front of us.  Building for one architecture is
# faster than a fat binary and cannot pick the wrong one.  With no GPU visible
# there is nothing to detect, so cover the usual targets instead.
if [ -z "$CUDA_ARCH" ]; then
  if [ -n "${VISIBLE_CC// /}" ]; then
    CUDA_ARCH="$(echo "$VISIBLE_CC" | tr -s ' ' | sed 's/ *$//' | tr ' ' ';')"
    say "note: building for this node's GPU, compute capability $CUDA_ARCH"
  else
    CUDA_ARCH="70;80;86;90"
    say "note: no GPU visible here, building a fat binary for $CUDA_ARCH"
    say "      (name the target with CUDA_ARCH=<cc> if it is known)"
  fi
fi

if command -v nvcc >/dev/null 2>&1 && [ "$CUDA_ARCH" != "native" ]; then
  for cc in $VISIBLE_CC; do
    case ";$CUDA_ARCH;" in
      *";$cc;"*) ;;
      *) CUDA_ARCH="$CUDA_ARCH;$cc"
         say "note: this node's GPU is sm_$cc -- added to the build list" ;;
    esac
  done

  SUPPORTED="$(nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')"
  if [ -n "${SUPPORTED// /}" ]; then
    KEPT=""; DROPPED=""
    OLDIFS="$IFS"; IFS=';'
    for a in $CUDA_ARCH; do
      case " $SUPPORTED " in
        *" $a "*) KEPT="${KEPT:+$KEPT;}$a" ;;
        *)        DROPPED="$DROPPED $a" ;;
      esac
    done
    IFS="$OLDIFS"
    [ -n "$DROPPED" ] && say "note: this nvcc cannot target compute capability$DROPPED -- dropped (it supports: $SUPPORTED)"

    # If the architecture just dropped is the architecture of the GPU in
    # THIS node, stop now and say why.  CUDA 13 removed Volta (sm_70), so a
    # V100 node with CUDA 13 loaded would otherwise build a binary that cannot
    # launch a single kernel, failing later with a runtime error that reads
    # like a bug in this code rather than a toolkit-version mismatch.
    for cc in $VISIBLE_CC; do
      case ";$KEPT;" in
        *";$cc;"*) ;;
        *) say ""
           say "FATAL: this node's GPU is compute capability $cc, but the CUDA toolkit"
           say "       on PATH cannot generate code for it."
           say "         nvcc      : $(nvcc --version 2>/dev/null | grep -i release | sed 's/^ *//')"
           say "         supports  : $SUPPORTED"
           say ""
           say "       CUDA 13 removed Volta support (sm_70).  For a V100, load a 12.x"
           say "       toolkit and re-run:     module load cuda/12.4"
           exit 1 ;;
      esac
    done

    if [ -n "$KEPT" ]; then
      CUDA_ARCH="$KEPT"
    else
      say "note: none of the requested architectures are supported by this nvcc;"
      say "      falling back to 'native', which needs a GPU visible at build time."
      CUDA_ARCH="native"
    fi
  fi
fi

# 00_environment.txt was written before the filtering above, so record what is
# ACTUALLY being built for -- otherwise the summary reports the request rather
# than the result (e.g. "70;90" when sm_70 was dropped and sm_86 added).
echo "cuda arch built: $CUDA_ARCH" >> "$OUT/00_environment.txt"

hdr "[1/6] Build  ->  $BUILD  (arch $CUDA_ARCH, -j$JOBS)"
if [ "${SKIP_BUILD:-0}" = "1" ] && [ -x "$BUILD/apps/app_cuda" ]; then
  say "SKIP_BUILD=1 and $BUILD/apps/app_cuda exists -- reusing it."
else
  # A previous attempt that failed (wrong module loaded, say) leaves a
  # CMakeCache pinned to THAT toolkit, and every later attempt then fails
  # identically even after the environment is fixed -- which looks like the
  # fix did not work.  Detect a changed nvcc and reset the cache, keeping
  # _deps/ so the one-time downloads are not repeated.
  NVCC_NOW="$(command -v nvcc 2>/dev/null)"
  if [ -f "$BUILD/CMakeCache.txt" ] && [ -n "$NVCC_NOW" ]; then
    NVCC_WAS="$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$BUILD/CMakeCache.txt" | head -1)"
    if [ -n "$NVCC_WAS" ] && [ "$NVCC_WAS" != "$NVCC_NOW" ]; then
      say "note: the CUDA toolkit changed since the last configure --"
      say "        was: $NVCC_WAS"
      say "        now: $NVCC_NOW"
      say "      resetting the cmake cache (downloads in $BUILD/_deps are kept)"
      rm -rf "$BUILD/CMakeCache.txt" "$BUILD/CMakeFiles"
    fi
  fi

  cmake -S . -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DEVELOPER_MODE=OFF \
        -DBUILD_FOR_CPU=ON -DBUILD_FOR_CUDA=ON -DBUILD_FOR_SN=ON -DBUILD_FOR_MPI=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        > "$OUT/logs/configure.txt" 2>&1
  if [ $? -ne 0 ]; then
    say "FATAL: cmake configure failed.  Full log: $OUT/logs/configure.txt"
    say "Last 25 lines:"; tail -25 "$OUT/logs/configure.txt" | tee -a "$LOG"
    # NB: do NOT grep for "fetchcontent" here -- it appears in essentially every
    # configure log, so it reported "no internet" for unrelated failures.
    if grep -qiE "could not resolve|network is unreachable|connection timed out|failed to download|download failed|couldn't connect|timeout was reached" "$OUT/logs/configure.txt"; then
      say ""
      say ">> This looks like NO INTERNET at configure time.  Configure once on a"
      say ">> LOGIN node (dependencies cache into $BUILD/), then re-run here with"
      say ">> SKIP_BUILD=1, or just re-run this whole script on the login node."
    elif grep -qiE "cuda_runtime\.h|cicc: No such file|nvcc fatal|libdevice|CMAKE_CUDA_COMPILER" "$OUT/logs/configure.txt"; then
      say ""
      say ">> This looks like a BROKEN OR INCOMPLETE CUDA TOOLKIT, not a problem with"
      say ">> this code: cmake found an nvcc that it cannot actually use."
      say ">>   nvcc on PATH: $(command -v nvcc 2>/dev/null || echo '<none>')"
      say ">> Verify the toolkit is usable, then re-run:"
      say ">>   echo 'int main(){}' > /tmp/t.cu && nvcc /tmp/t.cu -o /tmp/t && echo TOOLKIT-OK"
      say ">> If the site offers several CUDA modules, try another (12.x is safest)."
    fi
    # Leave no poisoned cache behind: a re-run after fixing the environment
    # must not reproduce this same error from stale cmake state.
    rm -rf "$BUILD/CMakeCache.txt" "$BUILD/CMakeFiles"
    say ""
    say ">> The cmake cache has been reset, so once the above is fixed simply"
    say ">> re-run this script -- it will not repeat this error from stale state."
    exit 1
  fi
  cmake --build "$BUILD" -j "$JOBS" > "$OUT/logs/build.txt" 2>&1
  if [ $? -ne 0 ]; then
    say "FATAL: build failed.  Full log: $OUT/logs/build.txt"; tail -30 "$OUT/logs/build.txt" | tee -a "$LOG"; exit 1
  fi
  say "build OK"
fi

APP="$BUILD/apps/app_cuda"
PROBE="$BUILD/apps/roofline_probe_cuda"
VERIFY="$BUILD/apps/verify_scale_cuda"
TESTS="$BUILD/test/cuda/cuda_tests"
CPUAPP="$BUILD/apps/adi_cpu"
[ -x "$APP" ] || { say "FATAL: $APP missing after a successful build."; exit 1; }

# Fail fast on an architecture mismatch.  Without this, a wrong-arch binary
# produces ~52 identical launch failures and the reason is buried in the last
# log file.  One 128^3 iteration costs about a second.
if command -v nvidia-smi >/dev/null 2>&1; then
  SMOKE="$OUT/logs/smoke.txt"
  if ! $(TMO 300) "$APP" 128 1 double > "$SMOKE" 2>&1; then
    say ""
    say "FATAL: the freshly built binary does not run on this node's GPU."
    tail -12 "$SMOKE" | sed 's/^/    /' | tee -a "$LOG"
    if grep -qiE "no kernel image|invalid device function" "$SMOKE"; then
      say ""
      say ">> 'no kernel image is available' means the binary was built for a different"
      say ">> compute capability than this GPU.  Re-run pinned to this card:"
      say ">>   CUDA_ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .' | head -1) bash scripts/run_hpc_study.sh"
    fi
    exit 1
  fi
  say "smoke test OK -- the binary runs on this node's GPU"
fi

# Which (N, lanes) pairs the Hybrid kernel actually instantiates.  An
# unsupported pair does NOT error -- it quietly runs a different lane count, so
# a row labelled L=8 would be measuring L=32.  Only honoured pairs are run.
lanes_for() {
  local N="$1" out=""
  for L in 8 16 32; do
    [ $((N % L)) -eq 0 ] || continue
    local M=$((N / L))
    case "$L:$M" in
      8:16|8:32) out="$out 8" ;;
      16:8|16:16|16:20|16:24|16:32) out="$out 16" ;;
      32:4|32:8|32:10|32:12|32:16) out="$out 32" ;;
    esac
  done
  echo "$out"
}

# -----------------------------------------------------------------------------
# 2. Correctness -- benchmarking a wrong kernel is worse than not benchmarking.
# -----------------------------------------------------------------------------
hdr "[2/6] Correctness"
CORRECT_OK=1
if [ -x "$TESTS" ]; then
  $(TMO 1800) "$TESTS" > "$OUT/logs/tests.txt" 2>&1
  if [ $? -eq 0 ]; then say "CUDA correctness suite: PASS  ($(grep -c . "$OUT/logs/tests.txt") lines logged)"
  else say "CUDA correctness suite: FAIL -- see $OUT/logs/tests.txt"; tail -25 "$OUT/logs/tests.txt" | tee -a "$LOG"; CORRECT_OK=0; fi
else
  say "note: $TESTS not built; relying on the residual validators below."
fi

# Residual check on every (N, lanes) tile the sweep will actually launch, so a
# size-specific bug cannot reach the measurements unnoticed.
if [ -x "$VERIFY" ]; then
  say ""
  say "Residual validator (||Ax-b||/||b||), per grid and per lane count:"
  for N in $GRIDS; do
    for L in $(lanes_for "$N"); do
      out="$(PENTA_XALGO=thomas-pcr PENTA_PCR_LANES=$L $(TMO 600) "$VERIFY" "$N" double 2>&1)"
      res="$(printf '%s' "$out" | grep -o 'max |residual|/|b|  = .*' | awk '{print $NF}')"
      if printf '%s' "$out" | grep -q PASS; then
        say "  N=$N L=$L  PASS   rel-residual ${res:-?}"
      else
        say "  N=$N L=$L  FAIL   rel-residual ${res:-?}"; CORRECT_OK=0
      fi
      printf '%s\n' "$out" >> "$OUT/logs/verify.txt"
    done
  done
fi
if [ "$CORRECT_OK" != "1" ]; then
  say ""; say "FATAL: correctness failed.  Stopping rather than producing numbers that mean nothing."
  say "Please send $OUT/logs/tests.txt and $OUT/logs/verify.txt for diagnosis."
  exit 2
fi
say "correctness OK"

# -----------------------------------------------------------------------------
# 3. This machine's roofline -- measured, not looked up.
#
# The whole argument of the study is where a kernel sits relative to this
# machine's balance point (peak FP64 flop/s divided by peak bandwidth).  The
# probe measures BOTH ends of that ratio directly and needs no profiler
# counters, which matters because ncu is permission-blocked on many estates.
# -----------------------------------------------------------------------------
hdr "[3/6] Measured roofline (memory floor + FP64 issue rate)"
if [ -x "$PROBE" ]; then
  $(TMO 1800) "$PROBE" 256 30 > "$OUT/01_roofline_probe.txt" 2>&1
  if [ $? -eq 0 ]; then sed 's/^/  /' "$OUT/01_roofline_probe.txt" | tee -a "$LOG"
  else say "roofline probe failed -- see $OUT/01_roofline_probe.txt"; fi
else
  say "roofline_probe_cuda not built; skipping (the sweep still runs)."
fi

# -----------------------------------------------------------------------------
# 4. The sweep
#
# Every algorithm is requested EXPLICITLY.  The solver has no automatic
# dispatch: a direction runs the kernel it is given, and the hardcoded default
# with nothing set is naive.
#
# Every row records the kernel that ACTUALLY ran.  Requests can be declined --
# an uninstantiated tile size or one that will not fit the register/shared
# budget falls back to a working kernel, which otherwise produces a perfectly
# plausible number attributed to the wrong algorithm.  `honoured=no` marks it.
# -----------------------------------------------------------------------------
hdr "[4/6] Algorithm sweep"
CSV="$OUT/gpu.csv"
echo "algo,N,precision,lanes,x_requested,x_kernel,y_kernel,z_kernel,e2e_wall_ms,x_ms,y_ms,z_ms,status,honoured" > "$CSV"
NRUN=0; NFAIL=0; NOOM=0; NUNH=0

# gpu_run <label> <x-selector> <yz-selector> <N> <prec> <lanes|->
gpu_run() {
  local label="$1" xs="$2" yz="$3" N="$4" prec="$5" lanes="$6"
  NRUN=$((NRUN+1))
  local lane_env=""; [ "$lanes" != "-" ] && lane_env="PENTA_PCR_LANES=$lanes "
  local cmd="${lane_env}PENTA_XALGO=$xs PENTA_YALGO=$yz PENTA_ZALGO=$yz $(TMO 1800) $APP $N $GPU_ITERS $prec"
  local log="$OUT/logs/$(printf '%03d' $NRUN)_${label}_${N}_${prec}_L${lanes}.txt"
  echo "\$ $cmd" > "$log"

  local rc=0
  eval "$cmd" >> "$log" 2>&1 || rc=$?
  local c; c="$(grep '^CSV,' "$log" | head -1)"

  if [ "$rc" -eq 0 ] && [ -n "$c" ]; then
    local rx ry rz honoured=yes
    rx="$(echo "$c" | cut -d, -f3)"; ry="$(echo "$c" | cut -d, -f5)"; rz="$(echo "$c" | cut -d, -f7)"
    [ "$rx" = "$xs" ] && [ "$ry" = "$yz" ] && [ "$rz" = "$yz" ] || honoured=no
    [ "$honoured" = no ] && NUNH=$((NUNH+1))
    echo "$label,$N,$prec,$lanes,$(echo "$c" | cut -d, -f2,3,5,7,10,12,13,14),ok,$honoured" >> "$CSV"
    printf '  %-12s N=%-4s %-6s L=%-3s  e2e %9s ms   ran x=%s y=%s z=%s%s\n' \
      "$label" "$N" "$prec" "$lanes" "$(echo "$c" | cut -d, -f10)" "$rx" "$ry" "$rz" \
      "$([ "$honoured" = no ] && echo '   <-- REQUEST DECLINED, fell back')" | tee -a "$LOG"
  else
    local st=fail
    grep -qiE "out of memory|cudaErrorMemoryAllocation|bad_alloc" "$log" && st=oom
    [ "$rc" -eq 124 ] && st=timeout
    [ "$st" = oom ] && NOOM=$((NOOM+1)) || NFAIL=$((NFAIL+1))
    echo "$label,$N,$prec,$lanes,$xs,-,-,-,-,-,-,-,$st,-" >> "$CSV"
    printf '  %-12s N=%-4s %-6s L=%-3s  %s  (log: %s)\n' "$label" "$N" "$prec" "$lanes" "$st" "$log" | tee -a "$LOG"
  fi
}

say ""; say "-- Algorithm 1: Naive (thread per system) --"
for N in $GRIDS; do for p in double float; do gpu_run naive naive naive "$N" "$p" -; done; done

say ""; say "-- Algorithm 2: Global-Transpose (x only; y/z run naive) --"
for N in $GRIDS; do for p in double float; do gpu_run transpose transpose naive "$N" "$p" -; done; done

say ""; say "-- Algorithm 3: Hybrid Thomas-PCR / SPIKE, all instantiated lane counts --"
say "   (lanes L = partitions per system; PCR rounds = log2(L), independent of N)"
for N in $GRIDS; do for p in double float; do
  LL="$(lanes_for "$N")"
  if [ -z "$(echo $LL)" ]; then
    say "  thomas-pcr   N=$N $p: SKIPPED -- no (M,L) tile is instantiated for this N."
    echo "thomas-pcr,$N,$p,-,thomas-pcr,-,-,-,-,-,-,-,no-tile,-" >> "$CSV"
  else
    for L in $LL; do gpu_run thomas-pcr thomas-pcr thomas-pcr "$N" "$p" "$L"; done
  fi
done; done

say ""; say "-- Algorithm 4: Shared-Factorisation (assumes ADI shared coefficients) --"
for N in $GRIDS; do for p in double float; do gpu_run shared-fact shared-fact shared-fact "$N" "$p" -; done; done

# -----------------------------------------------------------------------------
# 5. CPU baseline
#
# A GPU speedup is only meaningful against the CPU of the SAME machine.  Without
# this phase the comparison is against whatever desktop the developer owns,
# which on a node with two server sockets is not a defensible baseline.
#
# Two passes, because the full cross-product is far too slow to be worth it:
#   (a) a thread scan at one grid, which gives the parallel-scaling curve and
#       shows where the CPU saturates;
#   (b) every grid at full threads, which gives the per-grid number the speedup
#       table actually quotes.
# The CPU solver is ~50x slower per iteration than the GPU one, so CPU_ITERS is
# small by default and every run is bounded by `timeout`.
# -----------------------------------------------------------------------------
hdr "[5/6] CPU baseline"
CPUCSV="$OUT/cpu.csv"
echo "N,precision,threads,x_ms,y_ms,z_ms,total_ms,status" > "$CPUCSV"

if [ ! -x "$CPUAPP" ]; then
  say "adi_cpu not built ($CPUAPP); skipping the CPU baseline."
  say "NOTE: the GPU-vs-CPU speedup cannot be quoted for this machine."
else
  CPU_ITERS="${CPU_ITERS:-10}"
  CORES="$(nproc 2>/dev/null || echo 1)"

  # Powers of two up to the core count, plus the core count itself.  A scan of
  # every thread count would dominate the runtime on a 96-thread node and add
  # nothing: the curve is smooth.
  THREADS=""
  t=1
  while [ "$t" -lt "$CORES" ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
  THREADS="$THREADS $CORES"
  say "cpu iters     : $CPU_ITERS   cores: $CORES   thread scan:$THREADS"

  # cpu_run <N> <precision> <threads>
  cpu_run() {
    local N="$1" p="$2" th="$3"
    local tag; tag="$(printf 'cpu_%s_%s_t%s' "$N" "$p" "$th")"
    local log="$OUT/logs/${tag}.txt"
    local cmd="OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores $(TMO 3600) $CPUAPP $N $CPU_ITERS $p"
    echo "\$ $cmd" > "$log"
    if ! eval "$cmd" >> "$log" 2>&1; then
      echo "$N,$p,$th,-,-,-,-,fail" >> "$CPUCSV"
      say "  cpu  N=$N $p threads=$th   FAILED (log: $log)"
      return
    fi
    # adi_cpu prints: CSV,precision,N,threads,x,y,z,total
    local line; line="$(grep '^CSV,' "$log" | head -1)"
    if [ -z "$line" ]; then
      echo "$N,$p,$th,-,-,-,-,no-csv" >> "$CPUCSV"
      say "  cpu  N=$N $p threads=$th   no CSV line (log: $log)"
      return
    fi
    echo "$line" | awk -F, -v OFS=, '{print $3,$2,$4,$5,$6,$7,$8,"ok"}' >> "$CPUCSV"
    say "$(echo "$line" | awk -F, '{printf "  cpu  N=%-4s %-6s threads=%-4s  total %9.3f ms", $3, $2, $4, $8}')"
  }

  say ""; say "-- (a) thread scan at N=256 --"
  for p in double float; do
    for th in $THREADS; do cpu_run 256 "$p" "$th"; done
  done

  say ""; say "-- (b) every grid at $CORES threads --"
  for N in $GRIDS; do
    [ "$N" = "256" ] && continue          # already covered by the scan above
    for p in double float; do cpu_run "$N" "$p" "$CORES"; done
  done
fi

# -----------------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------------
hdr "[6/6] Summary"
SUM="$OUT/SUMMARY.txt"
python3 - "$CSV" "$OUT/00_environment.txt" "$OUT/01_roofline_probe.txt" "$CPUCSV" > "$SUM" 2>>"$LOG" <<'PY'
import csv, sys, os
csv_path, env_path, probe_path = sys.argv[1], sys.argv[2], sys.argv[3]
PASSES = {'naive':13, 'transpose':27, 'thomas-pcr':7, 'shared-fact':4}
rows = [r for r in csv.DictReader(open(csv_path))]
ok   = [r for r in rows if r['status']=='ok']

print("="*79); print("PENTADIAGONAL GPU SOLVER -- STUDY SUMMARY"); print("="*79)
for line in open(env_path):
    if line.startswith(('host','date','git','nvcc','cuda arch','slurm')) or ', NVIDIA' in line or 'name,' in line:
        print(line.rstrip())
if os.path.exists(probe_path):
    print("\n--- measured roofline (this machine) ---")
    for line in open(probe_path):
        if '-->' in line: print("   " + line.strip())

bad = [r for r in rows if r['status']!='ok']
unh = [r for r in ok if r['honoured']=='no']
print(f"\nruns: {len(rows)}   ok: {len(ok)}   failed/oom/timeout: {len(bad)}   request-declined: {len(unh)}")
for r in bad: print(f"   {r['status']:>8}  {r['algo']} N={r['N']} {r['precision']} L={r['lanes']}")
for r in unh: print(f"   declined  {r['algo']} N={r['N']} {r['precision']} L={r['lanes']} -> ran x={r['x_kernel']} y={r['y_kernel']} z={r['z_kernel']}")

DIRS = (('x','x_ms','x_kernel'), ('y','y_ms','y_kernel'), ('z','z_ms','z_kernel'))
ALGOS = ('naive','transpose','thomas-pcr','shared-fact')

def best(algo, N, prec, dcol, kcol):
    c = [(float(r[dcol]), r['lanes']) for r in ok
         if r['algo']==algo and int(r['N'])==N and r['precision']==prec and r[kcol]==algo]
    return min(c) if c else None

grids = sorted({int(r['N']) for r in ok})
for prec, tag, nb in (('double','FP64',8), ('float','FP32',4)):
    print("\n" + "="*79); print(f"PER-DIRECTION BEST TIME (ms) -- {tag}"); print("="*79)
    print(f"{'dir':>3} {'N':>5} " + "".join(f"{a:>18}" for a in ALGOS) + "   winner")
    for d, dcol, kcol in DIRS:
        for N in grids:
            cells, bst = [], (None, float('inf'))
            for a in ALGOS:
                b = best(a, N, prec, dcol, kcol)
                if b is None: cells.append(f"{'--':>18}"); continue
                t, L = b
                cells.append(f"{t:>13.3f}{(' L'+L) if L!='-' else '   ':>5}")
                if t < bst[1]: bst = (a, t)
            print(f"{d:>3} {N:>5} " + "".join(cells) + f"   {bst[0]}")

print("\n" + "="*79)
print("ACHIEVED BANDWIDTH vs NOMINAL TRAFFIC")
print("A kernel near 100% is memory-bound (it is moving data as fast as the card")
print("allows, and only fewer array passes can help it).  A kernel far below is")
print("limited by something else -- arithmetic, or uncoalesced access.")
print("="*79)
print(f"{'algo':<13}{'dir':>4}{'prec':>6}{'N':>6}{'passes':>8}{'ms':>10}{'GB':>9}{'GB/s':>9}")
for prec, nb in (('double',8), ('float',4)):
    for a in ALGOS:
        for d, dcol, kcol in DIRS:
            for N in grids:
                b = best(a, N, prec, dcol, kcol)
                if b is None: continue
                t, L = b
                GB = PASSES[a]*nb*N**3/1e9
                print(f"{a:<13}{d:>4}{('FP64' if prec=='double' else 'FP32'):>6}{N:>6}"
                      f"{PASSES[a]:>8}{t:>10.3f}{GB:>9.3f}{GB/(t/1000):>9.1f}")

print("\n" + "="*79)
print("HYBRID THOMAS-PCR: COST PER ELEMENT vs SYSTEM LENGTH  (FP64, x-direction)")
print("The reduced interface system is solved in log2(L) PCR rounds, and L is a")
print("fixed partition count, NOT a function of N.  So per-element interface cost")
print("falls as 1/N: a longer system should get CHEAPER per element, not dearer.")
print("Any rise here is register spilling (M = N/L elements held per lane), or a")
print("tile size that is not instantiated and fell back to a different L.")
print("="*79)
print(f"{'N':>6}{'L':>5}{'M=N/L':>7}{'rounds':>8}{'x_ms':>10}{'ns/element':>13}")
for r in sorted([r for r in ok if r['algo']=='thomas-pcr' and r['precision']=='double'
                 and r['x_kernel']=='thomas-pcr'], key=lambda r:(int(r['N']), int(r['lanes']))):
    N, L = int(r['N']), int(r['lanes']); t = float(r['x_ms'])
    print(f"{N:>6}{L:>5}{N//L:>7}{L.bit_length()-1:>8}{t:>10.3f}{t*1e6/N**3:>13.4f}")

# ---------------------------------------------------------------------------
# GPU against the CPU OF THIS MACHINE.
#
# Two GPU columns, because they are different claims.  "GPU general" is the
# best of Algorithms 1-3 and is like-for-like with the CPU.  "GPU ADI" is
# Algorithm 4, which exploits coefficients shared along a direction; the CPU
# baseline does not, so that column measures the value of the structural
# insight, NOT of the hardware, and must never be quoted as a hardware speedup.
# ---------------------------------------------------------------------------
cpu_path = sys.argv[4] if len(sys.argv) > 4 else None
cpu = []
if cpu_path and os.path.exists(cpu_path):
    cpu = [r for r in csv.DictReader(open(cpu_path)) if r['status'] == 'ok']

if cpu:
    print("\n" + "="*79)
    print("END-TO-END ADI ITERATION: BEST GPU vs BEST CPU (this machine)")
    print("="*79)
    print(f"{'N':>5} {'prec':>5} {'CPU ms':>10} {'thr':>4} {'GPU gen ms':>11} {'x':>7}"
          f" {'GPU ADI ms':>11} {'x':>7}")
    GEN = ('naive', 'transpose', 'thomas-pcr')
    for N in sorted({int(r['N']) for r in cpu}):
        for prec in ('double', 'float'):
            cr = [r for r in cpu if int(r['N']) == N and r['precision'] == prec]
            gr = [r for r in ok if int(r['N']) == N and r['precision'] == prec
                  and r['e2e_wall_ms'] not in ('-', '')]
            if not cr or not gr:
                continue
            best_cpu = min(cr, key=lambda r: float(r['total_ms']))
            c = float(best_cpu['total_ms'])
            gen = [float(r['e2e_wall_ms']) for r in gr if r['algo'] in GEN]
            adi = [float(r['e2e_wall_ms']) for r in gr if r['algo'] == 'shared-fact']
            g = min(gen) if gen else float('nan')
            a = min(adi) if adi else float('nan')
            print(f"{N:>5} {('FP64' if prec=='double' else 'FP32'):>5} {c:>10.2f}"
                  f" {best_cpu['threads']:>4} {g:>11.3f} {c/g:>6.1f}x"
                  f" {a:>11.3f} {c/a:>6.1f}x")
    print("\nThe 'GPU gen' speedup is the one to quote.  Read it against the ratio of")
    print("memory bandwidths: a memory-bound solver at its ceiling on both devices")
    print("should land near that ratio, and materially above it only where the GPU")
    print("kernel also moves less data than the CPU one.")
else:
    print("\nNo CPU baseline in this run, so no GPU-vs-CPU speedup can be quoted.")
    print("(adi_cpu was not built, or every CPU run failed -- see cpu.csv.)")

print("\nFiles: gpu.csv (all GPU rows), cpu.csv (CPU baseline),")
print("       01_roofline_probe.txt, logs/ (per-run stdout).")
PY
if [ -s "$SUM" ]; then cat "$SUM" | tee -a "$LOG" >/dev/null; cat "$SUM"; else say "summary generation failed (python3 missing?); raw CSV is at $CSV"; fi

TAR="pentadsolver_${HOST}_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$TAR" -C "$(dirname "$OUT")" "${OUT##*/}" 2>/dev/null \
  && say "" && say "PLEASE SEND BACK THIS ONE FILE:  $ROOT/$TAR" \
  || say "tar failed; please send the directory $OUT"

say ""
say "runs=$NRUN  failed=$NFAIL  oom=$NOOM  request-declined=$NUNH"
say "done: $OUT"
