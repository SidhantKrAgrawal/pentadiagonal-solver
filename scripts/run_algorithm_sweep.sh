#!/usr/bin/env bash
# run_algorithm_sweep.sh: per-direction algorithm comparison + end-to-end total.
#
#   bash scripts/run_algorithm_sweep.sh [N] [ITERS] [double|float|both] [--with-restricted]
#     defaults: 256 50 both
#
# ---------------------------------------------------------------------------
# WHAT THIS MEASURES, AND WHY IN TWO PHASES
# ---------------------------------------------------------------------------
# "Total = x + y + z" is not the cost of an ADI iteration: it drops the launch
# overhead and the gaps between the three solves.  But an end-to-end total
# also cannot be attributed to a single algorithm, because each direction runs
# its own kernel.  So:
#
#   PHASE 1: characterise each direction INDEPENDENTLY.  For every algorithm
#     with a kernel for that direction, measure its solve time and achieved
#     bandwidth while the other two directions are held on a fixed baseline.
#     Holding the others constant makes this a controlled experiment, so
#     differences between rows are attributable to the one algorithm varied.
#
#   PHASE 2: compose the per-direction winners into one configuration and
#     measure the TRUE end-to-end wall time.  That number is a property of the
#     configuration, not of any single algorithm, and is labelled as such.
#
# Algorithm 4 (Shared-Factorisation) is EXCLUDED by default.  It assumes every
# line in a direction shares the same coefficients, so it is not a
# general-purpose solver and must not set the optimisation target.  Pass
# --with-restricted to include it as a clearly-marked reference row.
#
# Only implemented combinations are run, so no row is ever a silent fallback
# mislabelled as the requested kernel.
# ---------------------------------------------------------------------------
set -euo pipefail

N="${1:-256}"
ITERS="${2:-50}"
PREC_ARG="${3:-both}"
WITH_RESTRICTED=0
for arg in "$@"; do
  case "$arg" in --with-restricted) WITH_RESTRICTED=1 ;; esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# APP is overridable so the same sweep can be run from a second build tree on a
# different machine (e.g. APP=build-panda/apps/app_cuda for a Pascal sm_61
# build) without disturbing the primary build/.
APP="${APP:-build/apps/app_cuda}"
[ -x "$APP" ] || { echo "GPU app not built ($APP). Run: bash scripts/build.sh gpu"; exit 1; }

# Results are tagged with the host so runs from different GPUs never collide.
OUT="results/algo_sweep_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CSV="$OUT/sweep.csv"
echo "x_sel,x_kernel,y_sel,y_kernel,z_sel,z_kernel,precision,N,e2e_wall_ms,e2e_events_ms,x_ms,y_ms,z_ms,sum_xyz_ms,overhead_pct" > "$CSV"

case "$PREC_ARG" in
  both)   PRECISIONS="double float" ;;
  double) PRECISIONS="double" ;;
  float)  PRECISIONS="float" ;;
  *) echo "unknown precision '$PREC_ARG' (double|float|both)"; exit 1 ;;
esac

# Algorithm number -> selector, per direction.  "-" means the algorithm has no
# kernel for that direction.
#
# Algorithm 1's strided form is the "naive" kernel: thread-per-system Thomas
# with global scratch, the SAME algorithm as the naive x kernel, just walking
# a stride.  It is labelled Algorithm 1 here rather than left nameless.
#
# Algorithm 2 is x-only BY DESIGN: it exists to repair the x-direction's
# uncoalesced access, and y/z are already naturally coalesced, so a transpose
# there would be pure added cost.
ALGO_NUMS="1 2 3"
[ "$WITH_RESTRICTED" = 1 ] && ALGO_NUMS="$ALGO_NUMS 4"

algo_name() {
  case "$1" in
    1) echo "Naive / thread-per-system" ;;
    2) echo "Global-Transpose" ;;
    3) echo "Hybrid Thomas-PCR (SPIKE)" ;;
    4) echo "Shared-Factorisation *" ;;
  esac
}
sel_x()  { case "$1" in 1) echo naive;; 2) echo transpose;; 3) echo thomas-pcr;; 4) echo shared-fact;; esac; }
sel_yz() { case "$1" in 1) echo naive;; 2) echo -;;     3) echo thomas-pcr;; 4) echo shared-fact;; esac; }

# Baseline held on the other directions during Phase 1.  This MUST be a
# selector the solver actually recognises: an unrecognised value silently falls
# through to auto dispatch, which would change every Phase-1 y and z row
# without producing any error.
BASE_X="transpose"
BASE_STRIDED="naive"

run_cfg() {   # $1=x_sel $2=y_sel $3=z_sel $4=precision $5=tag -> echoes CSV body
  local xs="$1" ys="$2" zs="$3" prec="$4" tag="$5"
  local log="$OUT/${tag}.txt"
  if ! PENTA_XALGO="$xs" PENTA_YALGO="$ys" PENTA_ZALGO="$zs" \
        "$APP" "$N" "$ITERS" "$prec" > "$log" 2>&1; then
    echo "  !! FAILED (see $log)" >&2; return 1
  fi
  local line
  line="$(grep '^CSV,' "$log" | head -1 | cut -d, -f2-)"
  [ -n "$line" ] || { echo "  !! no CSV line in $log" >&2; return 1; }
  echo "$line" >> "$CSV"
  echo "$line"
}
fld() { echo "$1" | cut -d, -f"$2"; }   # field of a CSV body (1=x_sel .. 11=x_ms,12=y_ms,13=z_ms)

for prec in $PRECISIONS; do
  echo
  echo "==============================================================================="
  echo " ${N}^3   $prec   ($ITERS timed iterations)"
  echo "==============================================================================="

  declare -A T   # T[algo,dir] = ms
  for a in $ALGO_NUMS; do
    # ---- x ----
    s="$(sel_x "$a")"
    if [ "$s" != "-" ]; then
      body="$(run_cfg "$s" "$BASE_STRIDED" "$BASE_STRIDED" "$prec" "p1_x_a${a}_${prec}")" \
        && T[$a,x]="$(fld "$body" 11)"
    fi
    # ---- y ----
    s="$(sel_yz "$a")"
    if [ "$s" != "-" ]; then
      body="$(run_cfg "$BASE_X" "$s" "$BASE_STRIDED" "$prec" "p1_y_a${a}_${prec}")" \
        && T[$a,y]="$(fld "$body" 12)"
    fi
    # ---- z ----
    if [ "$s" != "-" ]; then
      body="$(run_cfg "$BASE_X" "$BASE_STRIDED" "$s" "$prec" "p1_z_a${a}_${prec}")" \
        && T[$a,z]="$(fld "$body" 13)"
    fi
  done

  # ---------------- matrix ----------------
  echo
  echo "PHASE 1: solve time per direction (ms), each measured with the other"
  echo "          two directions held fixed"
  echo
  printf "  %-4s %-28s %10s %10s %10s\n" "Alg" "" "x" "y" "z"
  printf "  %s\n" "-------------------------------------------------------------------------"
  for a in $ALGO_NUMS; do
    printf "  %-4s %-28s %10s %10s %10s\n" "$a" "$(algo_name "$a")" \
           "${T[$a,x]:---}" "${T[$a,y]:---}" "${T[$a,z]:---}"
  done
  printf "  %s\n" "-------------------------------------------------------------------------"
  echo "  '--' = no kernel for that direction (Algorithm 2 is x-only by design)"

  # ---------------- winners (Algorithm 4 never eligible) ----------------
  declare -A BEST_T BEST_A
  for dir in x y z; do
    BEST_T[$dir]=""; BEST_A[$dir]=""
    for a in $ALGO_NUMS; do
      [ "$a" = "4" ] && continue          # restricted class, not a general solver
      v="${T[$a,$dir]:-}"
      [ -n "$v" ] || continue
      if [ -z "${BEST_T[$dir]}" ] || awk "BEGIN{exit !($v < ${BEST_T[$dir]})}"; then
        BEST_T[$dir]="$v"; BEST_A[$dir]="$a"
      fi
    done
  done
  echo
  printf "  best (general-purpose): x = Algorithm %s (%s ms)\n" "${BEST_A[x]}" "${BEST_T[x]}"
  printf "                          y = Algorithm %s (%s ms)\n" "${BEST_A[y]}" "${BEST_T[y]}"
  printf "                          z = Algorithm %s (%s ms)\n" "${BEST_A[z]}" "${BEST_T[z]}"
  [ "$WITH_RESTRICTED" = 1 ] && echo "  * Algorithm 4 is restricted to problems where all lines in a direction" \
                        && echo "    share coefficients; shown for reference, excluded from 'best'."

  # ---------------- Phase 2 ----------------
  echo
  echo "PHASE 2: true end-to-end wall time of one ADI iteration"
  echo
  printf "  %-42s %11s %11s %7s\n" "configuration" "e2e (ms)" "sum x+y+z" "gap"
  printf "  %s\n" "-------------------------------------------------------------------------"
  report() {  # $1=label $2=x $3=y $4=z $5=tag
    local body; body="$(run_cfg "$2" "$3" "$4" "$prec" "$5")" || return
    printf "  %-42s %11s %11s %6s%%\n" "$1" "$(fld "$body" 9)" "$(fld "$body" 14)" "$(fld "$body" 15)"
  }
  report "production auto dispatch" "auto" "auto" "auto" "p2_auto_${prec}"
  report "best general (Alg ${BEST_A[x]}/${BEST_A[y]}/${BEST_A[z]})" \
         "$(sel_x "${BEST_A[x]}")" "$(sel_yz "${BEST_A[y]}")" "$(sel_yz "${BEST_A[z]}")" \
         "p2_bestgen_${prec}"
  if [ "$WITH_RESTRICTED" = 1 ]; then
    report "Algorithm 4 everywhere (restricted)" "shared-fact" "shared-fact" "shared-fact" "p2_restricted_${prec}"
  fi
  printf "  %s\n" "-------------------------------------------------------------------------"
  echo "  'sum x+y+z' is the OLD, incorrect total; 'gap' is what it misses."

  unset T BEST_T BEST_A
done

echo
echo "CSV: $CSV"
echo "Logs: $OUT/"
