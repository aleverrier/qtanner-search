#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/c2xc2xc2_d16_${TS}"

python -m qtanner.search \
  --groups C2xC2xC2 \
  --max-n 288 \
  --steps 5000 \
  --target-distance 16 \
  --batch-size 50 \
  --frontier-max-per-point 20 \
  --frontier-max-total 100 \
  --max-quantum 200 \
  --A-enum multiset \
  --B-enum multiset \
  --seed 1 \
  --outdir "$OUTDIR" \
  --dist-m4ri-cmd dist_m4ri \
  "$@"
