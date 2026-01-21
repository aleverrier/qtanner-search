#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/progressive_c2xc2xc2_d16_${TS}"

python3 -m qtanner.search progressive \
  --group C2xC2xC2 \
  --target-distance 16 \
  --classical-steps 100 \
  --quantum-steps-fast 2000 \
  --quantum-steps-slow 50000 \
  --report-every 50 \
  --seed 1 \
  --results-dir "$OUTDIR" \
  --dist-m4ri-cmd dist_m4ri \
  "$@"
