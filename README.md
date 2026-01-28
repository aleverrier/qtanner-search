for seed in 1 2 3 4 5 6; do
  RUN_DIR="results/progressive_C11_target24_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_DIR"

  ./scripts/py -u scripts/search_progressive.py \
    --group C11 \
    --target-distance 14 \
    --seed "$seed" \
    --classical-distance-backend fast \
    --quantum-steps-fast 3000 \
    --quantum-steps-slow 300000 \
    2>&1 | tee "$RUN_DIR/run.log"
done

By default, `scripts/search_progressive.py` now updates `best_codes/` at the end of a
successful run (including publishing website data and pushing to GitHub). Use
`--no-best-codes-update` to disable this, or pass `--no-git` / `--no-publish`
to limit the post-run steps.
If you stop a run with Ctrl-C, it will still attempt the best-codes update.

Quick smoke integration test (skips git/publish + history scan):
```
SMOKE_BEST_CODES_UPDATE=1 bash scripts/smoke_progressive_search.sh
```

## Update best_codes (scrape + publish)

Dry run (no filesystem or git changes):
```
./scripts/py scripts/scrape_and_publish_best_codes.py --dry-run
```

Full update (sync best_codes/, rebuild data.json/index.tsv, commit + push):
```
./scripts/py scripts/scrape_and_publish_best_codes.py
```

Common flags:
- `--no-git` skip commit/push
- `--no-publish` skip website data updates
- `--verbose` show skipped files + actions

Note: the scraper now scans git history to recover older best codes, so a run can take about a minute.
