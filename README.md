for seed in 1 2 3 4 5 6; do
  RUN_DIR="results/progressive_C11_target24_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_DIR"

  python3 -u scripts/search_progressive.py \
    --group C11 \
    --target-distance 14 \
    --seed "$seed" \
    --classical-distance-backend fast \
    --classical-enum-kmax 8 \
    --quantum-steps-fast 3000 \
    --quantum-steps-slow 300000 \
    --kmax 0 \
    2>&1 | tee "$RUN_DIR/run.log"
done
