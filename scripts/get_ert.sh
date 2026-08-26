#!/usr/bin/env bash
# get_ert.sh -- fetch the Empirical Roofline Tool (ERT) from Berkeley Lab.
#
# ERT measures what a MACHINE can do: its peak memory bandwidth at each level of
# the hierarchy and its peak floating-point rate, and it draws the roofline
# graph.  It works by running a small kernel many times over, varying how much
# data it touches and how much arithmetic it does per element, and keeping the
# best time from each configuration.  The outline of those best times is the
# roofline.
#
# Nothing here is committed to this repository: ERT lands in third_party/,
# which .gitignore excludes.  Re-running is safe and cheap.
#
#   bash scripts/get_ert.sh          # fetch (skips if already present)
#   bash scripts/get_ert.sh --force  # re-fetch from scratch
#
# Reference: Empirical Roofline Toolkit, LBNL CRD.
#   https://bitbucket.org/berkeleylab/cs-roofline-toolkit
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/third_party/ert"
ERT_DIR="$DEST/Empirical_Roofline_Tool-1.1.0"
URL="https://bitbucket.org/berkeleylab/cs-roofline-toolkit.git"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ "$FORCE" = "1" ]; then
  echo ">> removing existing $DEST"
  rm -rf "$DEST"
fi

if [ -x "$ERT_DIR/ert" ]; then
  echo "ERT already present: $ERT_DIR"
else
  command -v git >/dev/null 2>&1 || { echo "FATAL: git not found."; exit 1; }
  mkdir -p "$(dirname "$DEST")"
  echo ">> cloning ERT from $URL"
  if ! git clone --depth 1 "$URL" "$DEST" 2>&1 | sed 's/^/   /'; then
    echo ""
    echo "FATAL: clone failed.  If this machine has no outbound network access,"
    echo "       fetch the toolkit elsewhere and copy it to:"
    echo "         $DEST"
    exit 1
  fi
fi

# --- sanity checks -----------------------------------------------------------
# The driver is Python 3 on current master.  Check rather than assume, because a
# Python 2 copy would fail deep inside a run with a confusing message.
fail=0
if [ ! -x "$ERT_DIR/ert" ]; then
  echo "FATAL: $ERT_DIR/ert missing or not executable."; fail=1
fi
# ERT compiles Drivers/<name>.cxx and Kernels/<name>.cxx -- .cxx, not .c, for
# both, even on the CUDA path (nvcc is handed -x cu).
if [ ! -f "$ERT_DIR/Kernels/kernel1.cxx" ] || [ ! -f "$ERT_DIR/Drivers/driver1.cxx" ]; then
  echo "FATAL: ERT kernel/driver sources missing."
  echo "       expected $ERT_DIR/{Drivers/driver1.cxx,Kernels/kernel1.cxx}"
  fail=1
fi
[ "$fail" = "1" ] && exit 1

if ! python3 -m py_compile "$ERT_DIR/ert" \
        "$ERT_DIR/Python/ert_core.py" "$ERT_DIR/Python/ert_utils.py" 2>/dev/null; then
  echo "FATAL: this ERT copy is not Python 3 compatible; cannot drive it with"
  echo "       $(python3 --version 2>&1)."
  exit 1
fi

echo "ERT ready:  $ERT_DIR"
echo "  driver    : $(head -1 "$ERT_DIR/ert")"
echo "  python    : $(python3 --version 2>&1)"
echo "  gnuplot   : $(command -v gnuplot >/dev/null 2>&1 && gnuplot --version || echo 'NOT FOUND -- graphs will be skipped')"
echo ""
echo "Next:  bash scripts/run_ert.sh <outdir>     (needs a GPU and nvcc)"
