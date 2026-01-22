#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DIST_M4RI_CMD="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/smoke_histogram_table_${STAMP}"

python -u scripts/search_progressive.py \
  --group "C2" \
  --target-distance 2 \
  --classical-target 2 \
  --classical-steps 50 \
  --classical-distance-backend fast \
  --classical-exhaustive-k-max 4 \
  --classical-sample-count 16 \
  --classical-jobs 1 \
  --max-quantum-evals 1 \
  --quantum-steps-fast 50 \
  --quantum-steps-slow 50 \
  --quantum-refine-chunk 50 \
  --seed 1 \
  --dist-m4ri-cmd "$DIST_M4RI_CMD" \
  --results-dir "$OUTDIR"
