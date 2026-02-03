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

## Matrix download tracking (GitHub Releases)
- Website matrix links point to GitHub release assets tagged per track:
  `best-codes-matrices-633`, `best-codes-matrices-844`, `best-codes-matrices-212`.
- Publish (or sync) release assets:
```
python3 scripts/publish_matrices_release.py --dry-run
python3 scripts/publish_matrices_release.py
```
- Report download counts:
```
python3 scripts/release_download_counts.py
```

## Workflow
See `docs/PROGRESS_LOG.md` for the current project state, canonical scripts, and
the end-to-end workflow (search -> scrape/publish -> refine).

## Two-machine setup
- Keep a dedicated maintenance clone for publishing; do not `git pull` into a
  running search clone.
- Archive untracked run artifacts into `local_results/` before syncing.
- Run website/best_codes publish steps from the maintenance clone only.

## Refine then publish
- Refinement scripts (`scripts/refine_best_codes.py` and the length/group
  pipelines) update metadata and then resync/publish best_codes by default.
- If you run custom refinement steps, follow with
  `scripts/scrape_and_publish_best_codes.py` to rebuild `best_codes/data.json`
  and `best_codes/index.tsv`.

## Local code tracks
- Default track uses local [6,3,3] codes and publishes to `best_codes/`.
- The [8,4,4] track uses `--local-a 8_4_4 --local-b 8_4_4` and publishes to
  `best_codes_844/` (with pending artifacts under `codes/pending_844/`).

## Context for new contributors / new chats
- `docs/CHAT_CONTEXT.md`
- `docs/CHAT_PROMPT_TEMPLATE.md`
- `docs/ARCHITECTURE.md`
