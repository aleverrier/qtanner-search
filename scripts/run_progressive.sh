#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

# Usage:
#   ./scripts/run_progressive.sh 'C2^2' 12 12
#   ./scripts/run_progressive.sh 'C2^4' 24 24
G="${1:?Missing group, e.g. 'C2^4'}"
D0="${2:?Missing d0, e.g. 24}"
MINC="${3:?Missing min_classical_distance, e.g. 24}"

# If you use dist-m4ri for distance estimation:
export DIST_M4RI="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"

# Make sure progress prints immediately
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/prog_${G//[^A-Za-z0-9]/_}_d${D0}_minc${MINC}_${STAMP}.log"

echo "[run] G=$G d0=$D0 min_classical_distance=$MINC"
echo "[run] log=$LOG"
python -u scripts/search_progressive.py \
  --G "$G" \
  --d0 "$D0" \
  --min_classical_distance "$MINC" \
  2>&1 | tee "$LOG"
