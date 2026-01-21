#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

# If you use dist-m4ri for distance estimation:
export DIST_M4RI="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"

# Make sure progress prints immediately
export PYTHONUNBUFFERED=1

python -u scripts/search_progressive.py \
  --G 'C2^4' \
  --d0 24 \
  --min_classical_distance 24 \
  2>&1 | tee "logs/prog_C2_4_d24_$(date +%Y%m%d_%H%M%S).log"
