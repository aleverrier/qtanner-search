#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

# Usage:
#   ./scripts/run_progressive.sh 'C2^2' 12 12
#   ./scripts/run_progressive.sh 'C2^4' 24 24
GROUP="${1:?Missing group, e.g. 'C2^4'}"
TARGET="${2:?Missing quantum target distance, e.g. 24}"
CLASSICAL_TARGET="${3:?Missing classical target distance, e.g. 24}"

# If you use dist-m4ri for distance estimation:
DIST_M4RI_CMD="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"

# Make sure progress prints immediately
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/prog_${GROUP//[^A-Za-z0-9]/_}_t${TARGET}_ct${CLASSICAL_TARGET}_${STAMP}.log"

echo "[run] group=$GROUP target_distance=$TARGET classical_target=$CLASSICAL_TARGET"
echo "[run] dist_m4ri_cmd=$DIST_M4RI_CMD"
echo "[run] log=$LOG"

python -u scripts/search_progressive.py \
  --group "$GROUP" \
  --target-distance "$TARGET" \
  --classical-target "$CLASSICAL_TARGET" \
  --classical-steps 100 \
  --dist-m4ri-cmd "$DIST_M4RI_CMD" \
  2>&1 | tee "$LOG"
