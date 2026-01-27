for seed in 1 2 3 4 5 6; do
  RUN_DIR="results/progressive_C11_target24_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_DIR"

  python3 -u scripts/search_progressive.py \
    --group C11 \
    --target-distance 14 \
    --seed "$seed" \
    --classical-distance-backend fast \
    --quantum-steps-fast 3000 \
    --quantum-steps-slow 300000 \
    2>&1 | tee "$RUN_DIR/run.log"
done

## Update best_codes (scrape + publish)

Dry run (no filesystem or git changes):
```
python3 scripts/scrape_and_publish_best_codes.py --dry-run
```

Full update (sync best_codes/, rebuild data.json/index.tsv, commit + push):
```
python3 scripts/scrape_and_publish_best_codes.py
```

Common flags:
- `--no-git` skip commit/push
- `--no-publish` skip website data updates
- `--verbose` show skipped files + actions

Note: the scraper now scans git history to recover older best codes, so a run can take about a minute.
