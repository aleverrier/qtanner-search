TS="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/sweep_recover_C2xC2_${TS}"
mkdir -p "$ROOT"

for SEED in $(seq 1 10); do
  OUT="$ROOT/seed${SEED}"
  mkdir -p "$OUT"
  (
    python -m qtanner.search progressive \
      --group "C2xC2" \
      --target-distance 10 \
      --classical-target 1 \
      --classical-distance-backend fast \
      --classical-exhaustive-k-max 12 \
      --classical-sample-count 1 \
      --classical-steps 500000 \
      --classical-workers 1 \
      --quantum-steps-fast 1000 \
      --quantum-steps-slow 10000 \
      --max-quantum-evals 2000000 \
      --min-distinct 2 \
      --report-every 200 \
      --seed "$SEED" \
      --results-dir "$OUT" \
      --dist-m4ri-cmd "$DIST_M4RI" \
      | tee "$OUT/run.log"
  ) &
done
wait
echo "[done] $ROOT"
