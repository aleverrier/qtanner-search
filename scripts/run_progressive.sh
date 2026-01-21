#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

# Usage:
#   ./scripts/run_progressive.sh 'C2^2' 12 12
#   ./scripts/run_progressive.sh 'C2^4' 24 24
GROUP="${1:?Missing group, e.g. 'C2^4'}"
TARGET="${2:?Missing target distance, e.g. 24}"
MINC="${3:?Missing min classical distance, e.g. 24}"

# If you use dist-m4ri for distance estimation:
export DIST_M4RI="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"

# Make sure progress prints immediately
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/prog_${GROUP//[^A-Za-z0-9]/_}_t${TARGET}_minc${MINC}_${STAMP}.log"

echo "[run] group=$GROUP target_distance=$TARGET min_classical_distance=$MINC"
echo "[run] log=$LOG"

python -u scripts/search_progressive.py \
  --group "$GROUP" \
  --target-distance "$TARGET" \
  --min-classical-distance "$MINC" \
  2>&1 | tee "$LOG"
