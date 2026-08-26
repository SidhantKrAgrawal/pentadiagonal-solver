#!/usr/bin/env bash
# run_ert.sh -- measure this machine's roofline with the Empirical Roofline
# Tool (ERT) from Berkeley Lab, in both FP64 and FP32.
#
# ERT answers "what can this GPU do?": peak memory bandwidth at each level of
# the hierarchy, and peak floating-point rate.  It runs a small kernel many
# times, varying how much data it touches and how much arithmetic it does per
# element, and keeps the best time from each configuration.  The outline of
# those best times is the roofline.
#
# Both precisions are run because the central claim of this study is that the
# solver is limited by memory traffic rather than arithmetic.  If that is true,
# the memory roof should barely move between FP64 and FP32 while the compute
# roof moves by the card's FP64:FP32 ratio -- which on a consumer card is 1:32
# or 1:64, and on a data-centre card is 1:2.
#
#   bash scripts/run_ert.sh                    # results into results/ert_<host>_<date>
#   bash scripts/run_ert.sh /path/to/outdir    # results into a chosen directory
#
# Environment knobs:
#   ERT_ARCH          compute capability without the dot, e.g. 86.  Default: the
#                     GPU that nvidia-smi reports.
#   ERT_PRECISIONS    space separated.  Default: "FP64 FP32".
#   ERT_FLOPS_LIST    flops/element sweep.  Setting it forces that exact list on
#                     both precisions.  Unset, the default is
#                     1,2,4,8,16,32,64,128,256,512,1024 for FP32 and for FP64 on
#                     a full-rate card, and the same list capped at 256 for FP64
#                     on a reduced-rate consumer card -- see the FULL_RATE_FP64
#                     block below for why.  Each value costs roughly five minutes
#                     per precision, so a shorter list trades roofline resolution
#                     for wall time.
#   ERT_EXPERIMENTS   repeats of the whole sweep.  Default: 1, matching
#                     Berkeley's own GPU configurations.  ERT already repeats
#                     trials internally at each working-set size, so the clock
#                     ramp is absorbed without a second full pass; raise this
#                     only to check run-to-run spread, and expect the runtime to
#                     multiply by the same factor.
#   ERT_MEM_MAX       largest working set in bytes.  Default: 1073741824 (1 GiB).
#   ERT_RUN_CMD       launcher.  Default: "./ERT_CODE".  Use e.g.
#                     "srun -n 1 ./ERT_CODE" on a Slurm cluster.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ERT_DIR="$REPO/third_party/ert/Empirical_Roofline_Tool-1.1.0"
TEMPLATE="$REPO/scripts/ert/config.penta.gpu.in"

HOST="$(hostname -s 2>/dev/null || echo unknown)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$REPO/results/ert_${HOST}_${STAMP}}"
mkdir -p "$OUT" || { echo "FATAL: cannot create $OUT"; exit 1; }
# Must be absolute: ERT is invoked from inside its own directory, and the config
# path is handed to it there.  A relative $OUT -- which is what the study script
# passes -- would not resolve after that cd.
OUT="$(cd "$OUT" && pwd)" || { echo "FATAL: cannot resolve $OUT"; exit 1; }

PRECISIONS="${ERT_PRECISIONS:-FP64 FP32}"
FLOPS="${ERT_FLOPS_LIST:-1,2,4,8,16,32,64,128,256,512,1024}"
EXPERIMENTS="${ERT_EXPERIMENTS:-1}"
MEM_MAX="${ERT_MEM_MAX:-1073741824}"
RUN_CMD="${ERT_RUN_CMD:-./ERT_CODE}"

say() { echo "$@" | tee -a "$OUT/run.log"; }

# --- preconditions -----------------------------------------------------------
if [ ! -x "$ERT_DIR/ert" ]; then
  say "FATAL: ERT not found at $ERT_DIR"
  say "       Run:  bash scripts/get_ert.sh"
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  say "FATAL: nvcc not on PATH.  On this estate:  module load CUDA/12.6.2"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "FATAL: no nvidia-smi, so there is no GPU here to measure."
  say "       ERT must run on the GPU node itself, not a login node."
  exit 1
fi
command -v gnuplot >/dev/null 2>&1 || say "warning: gnuplot missing; ERT will write JSON but no graph."

# --- refuse to run alongside another instance --------------------------------
# ERT measures peak bandwidth and peak flop rate.  A second GPU job running at
# the same time does not make it fail; it makes it quietly report numbers that
# are too low, which is far worse.  Two runs sharing one output directory also
# interleave their config files and results.
#
# Both happened during development: a run that was believed dead kept a
# run_ert.sh alive, and the next run measured a contended GPU.  So take a lock,
# and separately refuse to start if anything else already holds the GPU.
LOCK="$OUT/.ert.lock"
if ! ( set -o noclobber; echo "$$ $(date -Is) $HOST" > "$LOCK" ) 2>/dev/null; then
  say "FATAL: another ERT run holds $LOCK"
  say "       held by: $(cat "$LOCK" 2>/dev/null)"
  say "       Measuring a contended GPU produces numbers that are too low."
  say "       If that run is genuinely dead, delete the lock and retry."
  exit 1
fi
cleanup() { rm -f "$LOCK"; }
trap cleanup EXIT INT TERM

GPU_APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)"
if [ "${GPU_APPS:-0}" -gt 0 ]; then
  say "FATAL: $GPU_APPS process(es) are already using this GPU."
  say "       ERT measures ceilings, so it needs the card to itself."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv 2>/dev/null | sed 's/^/       /' | tee -a "$OUT/run.log"
  say "       Set ERT_ALLOW_BUSY_GPU=1 to override (the numbers will be low)."
  [ "${ERT_ALLOW_BUSY_GPU:-0}" = "1" ] || exit 1
fi

# --- which GPU are we measuring? ---------------------------------------------
# Same detection the study script uses: ask the device, do not hardcode.
ARCH="${ERT_ARCH:-}"
if [ -z "$ARCH" ]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  if [ -z "$CC" ]; then
    say "FATAL: could not read compute capability from nvidia-smi."
    say "       Set it explicitly, e.g.  ERT_ARCH=86 bash scripts/run_ert.sh"
    exit 1
  fi
  ARCH="$CC"
fi

GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- how far up the arithmetic sweep is worth going --------------------------
# The top of the flops/element range exists for one reason: to push past the
# point where arithmetic overtakes memory, so that the compute roof becomes
# measurable at all.  Where that point sits is a property of the card.
#
#   Full-rate FP64 parts (P100, V100, A100, H100, B100 -- FP64 at 1:2) hide
#     arithmetic for a very long time.  An H100's crossover sits near 385 flops
#     per element, so the sweep MUST reach 1024 or its compute roof is never
#     seen; the earlier bespoke probe stopped at 256 and reported a figure 175x
#     a V100's as a result.
#   Reduced-rate parts (consumer cards, FP64 at 1:32 or 1:64) cross over almost
#     immediately.  Measured here, the FP64 compute roof converges by K=256
#     (RTX 3050: 116.5 -> 129.2 -> 132.3 GFLOP/s at K = 16, 64, 256) and K=1024
#     adds around one percent for roughly an hour of extra runtime, because the
#     card executes that arithmetic sixty-four times slower than FP32.
#
# So the ceiling is chosen from the device, exactly as the solver chooses its
# lane count.  The architecture list mirrors device_has_full_rate_fp64() in
# src/cuda/singlenode/pentad_cuda.cu; anything unlisted is treated as
# reduced-rate, which keeps the cheaper sweep as the safe default.
# FP32 is not rate-limited on any of these parts, so it always gets the full
# range.
case "$ARCH" in
  60|70|80|90|100) FULL_RATE_FP64=1 ;;
  *)               FULL_RATE_FP64=0 ;;
esac

if [ -n "${ERT_FLOPS_LIST:-}" ]; then
  FLOPS_FP64="$ERT_FLOPS_LIST"; FLOPS_FP32="$ERT_FLOPS_LIST"
elif [ "$FULL_RATE_FP64" = "1" ]; then
  FLOPS_FP64="$FLOPS"; FLOPS_FP32="$FLOPS"
else
  # Drop only the top of the range, and only for FP64.
  FLOPS_FP64="$(echo "$FLOPS" | tr ',' '\n' | awk '$1 <= 256' | paste -sd,)"
  FLOPS_FP32="$FLOPS"
fi

{
  echo "host            : $HOST"
  echo "date            : $(date -Is)"
  echo "gpu             : $GPUNAME"
  echo "compute cap     : sm_$ARCH"
  echo "nvcc            : $(nvcc --version | tail -1)"
  echo "python          : $(python3 --version 2>&1)"
  echo "gnuplot         : $(gnuplot --version 2>/dev/null || echo none)"
  echo "flops sweep FP64: $FLOPS_FP64"
  echo "flops sweep FP32: $FLOPS_FP32"
  echo "fp64 rate       : $([ "$FULL_RATE_FP64" = 1 ] && echo "full (1:2) -- full sweep" || echo "reduced (1:32 or 1:64) -- FP64 sweep capped at 256")"
  echo "experiments     : $EXPERIMENTS"
  echo "max working set : $MEM_MAX bytes"
  echo "launcher        : $RUN_CMD"
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv 2>/dev/null
} > "$OUT/00_environment.txt"

say "=============================================================="
say " ERT roofline -- $GPUNAME (sm_$ARCH) on $HOST"
say "=============================================================="
say ""
sed 's/^/  /' "$OUT/00_environment.txt" | tee -a "$OUT/run.log" >/dev/null
say "GPU        : $GPUNAME (sm_$ARCH)"
say "precisions : $PRECISIONS"
say ""

# --- run ERT once per precision ----------------------------------------------
STATUS=0
for PREC in $PRECISIONS; do
  lc="$(echo "$PREC" | tr '[:upper:]' '[:lower:]')"
  case "$PREC" in
    FP64) FLOPS_THIS="$FLOPS_FP64" ;;
    *)    FLOPS_THIS="$FLOPS_FP32" ;;
  esac
  RESULTS_NAME="Results.penta-${HOST}-${lc}"
  CFG="$OUT/config.penta.${lc}"

  sed -e "s|@RESULTS@|$RESULTS_NAME|g" \
      -e "s|@ARCH@|$ARCH|g" \
      -e "s|@FLOPS@|$FLOPS_THIS|g" \
      -e "s|@PRECISION@|$PREC|g" \
      -e "s|@EXPERIMENTS@|$EXPERIMENTS|g" \
      -e "s|@MEMORY_MAX@|$MEM_MAX|g" \
      -e "s|@RUN@|$RUN_CMD|g" \
      "$TEMPLATE" > "$CFG"

  say "--------------------------------------------------------------"
  say "[$PREC] running ERT   (config: $(basename "$CFG"))"
  say "--------------------------------------------------------------"

  # ERT resolves its driver/kernel paths relative to its own directory, and
  # writes results into a directory named by ERT_RESULTS beneath it.
  rm -rf "${ERT_DIR:?}/$RESULTS_NAME"
  ( cd "$ERT_DIR" && ./ert --verbose 1 "$CFG" ) > "$OUT/ert_${lc}.log" 2>&1
  rc=$?

  if [ $rc -ne 0 ]; then
    say "  FAILED (exit $rc).  Last 25 lines of $OUT/ert_${lc}.log:"
    tail -25 "$OUT/ert_${lc}.log" | sed 's/^/    /' | tee -a "$OUT/run.log"
    STATUS=1
    continue
  fi

  # Collect everything ERT produced for this precision.
  DEST="$OUT/$lc"
  mkdir -p "$DEST"
  if [ -d "$ERT_DIR/$RESULTS_NAME" ]; then
    cp -r "$ERT_DIR/$RESULTS_NAME/." "$DEST/" 2>/dev/null
    rm -rf "${ERT_DIR:?}/$RESULTS_NAME"
  fi

  # ERT nests its output one level deeper than the results directory: the
  # artefacts land in Run.001/ (Run.002/ and so on if a results directory is
  # reused).  Lift the newest run's files to the top so the paths this script
  # advertises are the paths that exist.
  newest_json="$(find "$DEST" -mindepth 2 -name roofline.json -printf '%T@ %p\n' 2>/dev/null \
                  | sort -rn | head -1 | cut -d' ' -f2-)"
  if [ -n "$newest_json" ]; then
    rundir="$(dirname "$newest_json")"
    for f in roofline.json roofline.ps roofline.gnu roofline.tex; do
      [ -f "$rundir/$f" ] && cp -f "$rundir/$f" "$DEST/$f"
    done
    say "  artefacts from $(basename "$rundir")/"
  fi

  # Turn the PostScript graph into a PDF for LaTeX, if a converter exists.
  if [ -f "$DEST/roofline.ps" ]; then
    if command -v ps2pdf >/dev/null 2>&1; then
      ( cd "$DEST" && ps2pdf -dEPSCrop roofline.ps roofline.pdf 2>/dev/null ) \
        || ( cd "$DEST" && ps2pdf roofline.ps roofline.pdf 2>/dev/null )
    elif command -v epstopdf >/dev/null 2>&1; then
      ( cd "$DEST" && epstopdf roofline.ps --outfile=roofline.pdf 2>/dev/null )
    fi
  fi

  # Report the two ceilings.  roofline.json is ERT's own summary: the memory
  # roof(s) in GB/s and the compute roof in GFLOP/s.
  if [ -f "$DEST/roofline.json" ]; then
    say ""
    python3 - "$DEST/roofline.json" "$PREC" <<'PY' | tee -a "$OUT/run.log"
import json, sys
path, prec = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"  could not read roofline.json: {e}"); sys.exit(0)

gf = d.get("empirical", {}).get("gflops", {})
bw = d.get("empirical", {}).get("gbytes", {})
print(f"  --- {prec} ceilings measured by ERT ---")
for label, val in (gf.get("data") or []):
    print(f"    compute roof   {label:<12} {val:10.1f} GFLOP/s")
for label, val in (bw.get("data") or []):
    print(f"    memory roof    {label:<12} {val:10.1f} GB/s")

# The ridge point is where the two roofs meet: how much arithmetic this machine
# can afford per byte moved before arithmetic, not bandwidth, is the limit.
try:
    peak_gf = max(v for _, v in gf["data"])
    dram = [v for k, v in bw["data"] if "DRAM" in k.upper()] or [v for _, v in bw["data"]]
    peak_bw = min(dram)
    print(f"    ridge point                 {peak_gf/peak_bw:10.2f} FLOP/byte")
except Exception:
    pass
PY
  else
    say "  warning: no roofline.json produced for $PREC"
    STATUS=1
  fi
  say ""
done

# --- summary -----------------------------------------------------------------
{
  echo "ERT roofline summary"
  echo "===================="
  echo "host : $HOST"
  echo "gpu  : $GPUNAME (sm_$ARCH)"
  echo "date : $(date -Is)"
  echo ""
  grep -A20 -- "--- .* ceilings measured by ERT ---" "$OUT/run.log" 2>/dev/null \
    || echo "(no ceilings recorded -- see ert_*.log)"
} > "$OUT/SUMMARY_ERT.txt"

say "=============================================================="
if [ $STATUS -eq 0 ]; then
  say " ERT completed.  Output: $OUT"
else
  say " ERT finished with problems -- see the logs in $OUT"
fi
say "   per precision : $OUT/<fp64|fp32>/roofline.{json,ps,pdf,tex}"
say "   summary       : $OUT/SUMMARY_ERT.txt"
say "=============================================================="
exit $STATUS
