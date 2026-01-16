#!/usr/bin/env bash
set -euo pipefail

DIST_BIN="$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri"
CODEDIR="$HOME/research/qtanner-search/data/lrz_paper_mtx/633x633"
OUTROOT="$HOME/research/qtanner-search/results/dist_m4ri_633x633"
STEPS=100000
WMIN=10
SEED=1

test -x "$DIST_BIN"
test -d "$CODEDIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUTROOT/$TS"
mkdir -p "$OUTDIR"

echo "dist_m4ri: $DIST_BIN" | tee "$OUTDIR/_runinfo.txt"
echo "CODEDIR:   $CODEDIR"   | tee -a "$OUTDIR/_runinfo.txt"
echo "STEPS:     $STEPS"     | tee -a "$OUTDIR/_runinfo.txt"
echo "WMIN:      $WMIN"      | tee -a "$OUTDIR/_runinfo.txt"
echo "SEED:      $SEED"      | tee -a "$OUTDIR/_runinfo.txt"
echo "Started:   $(date)"    | tee -a "$OUTDIR/_runinfo.txt"
echo                       | tee -a "$OUTDIR/_runinfo.txt"

shopt -s nullglob

count=0
for HX in "$CODEDIR"/HX_*.mtx; do
  base="$(basename "$HX")"
  suffix="${base#HX_}"
  HZ="$CODEDIR/HZ_$suffix"

  if [ ! -f "$HZ" ]; then
    echo "Skipping (no matching HZ): $base" | tee -a "$OUTDIR/_runinfo.txt"
    continue
  fi

  name="${suffix%.mtx}"
  echo "=== $name ===" | tee -a "$OUTDIR/_runinfo.txt"

  echo "Z-type: ker(HX) \\ row(HZ)" | tee "$OUTDIR/${name}_Z.txt"
  "$DIST_BIN" debug=0 method=1 finH="$HX" finG="$HZ" steps="$STEPS" wmin="$WMIN" seed="$SEED" | tee -a "$OUTDIR/${name}_Z.txt"

  echo "X-type: ker(HZ) \\ row(HX)" | tee "$OUTDIR/${name}_X.txt"
  "$DIST_BIN" debug=0 method=1 finH="$HZ" finG="$HX" steps="$STEPS" wmin="$WMIN" seed="$SEED" | tee -a "$OUTDIR/${name}_X.txt"

  echo | tee -a "$OUTDIR/_runinfo.txt"
  count=$((count+1))
done

echo "Done. Processed $count HX/HZ pairs." | tee -a "$OUTDIR/_runinfo.txt"
echo "Finished: $(date)" | tee -a "$OUTDIR/_runinfo.txt"
