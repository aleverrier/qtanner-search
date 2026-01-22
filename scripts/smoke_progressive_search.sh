#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DIST_M4RI_CMD="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/smoke_progressive_${STAMP}"

python -u scripts/search_progressive.py \
  --group "C2" \
  --target-distance 2 \
  --classical-target 2 \
  --classical-distance-backend fast \
  --classical-exhaustive-k-max 8 \
  --classical-sample-count 64 \
  --classical-jobs 1 \
  --max-quantum-evals 3 \
  --quantum-steps-fast 200 \
  --quantum-steps-slow 200 \
  --quantum-refine-chunk 200 \
  --seed 1 \
  --dist-m4ri-cmd "$DIST_M4RI_CMD" \
  --results-dir "$OUTDIR"
