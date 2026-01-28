# Current workflows (audit)

This document summarizes existing scripts that already implement the requested workflows and the exact step sequence they run.

## Progressive search

### Canonical implementation
- Script: `scripts/search_progressive.py`
- Entry point: `src/qtanner/progressive_search.py` (`progressive_main`)
- Typical invocation: `./scripts/py scripts/search_progressive.py ...`
- Alternate entry point: `./scripts/py -m qtanner.search progressive ...` (dispatches to the same progressive implementation)

### Wrapper
- Script: `scripts/run_progressive.sh`
- Steps:
  1. Sets environment defaults (classical backend/steps, quantum refine chunk, dist_m4ri path).
  2. Builds argument list.
  3. Runs `./scripts/py -u scripts/search_progressive.py ...` and logs to `logs/`.

## Refine distances by length

### Core refine logic
- Script: `scripts/refine_best_codes_length.py`
- Steps:
  1. Scans `best_codes/meta/*.json` for codes with the requested length `n`.
  2. Locates `Hx/Hz` matrices for each code.
  3. Runs dist-m4ri to refine distances.
  4. Updates `best_codes/meta/*.json` and `best_codes/collected/*/meta.json`.
  5. Archives below-trials entries.

### Pipeline wrapper
- Script: `scripts/refine_best_codes_length.sh`
- Steps:
  1. (Optional) `git pull --rebase`.
  2. Runs `./scripts/py scripts/refine_best_codes_length.py --n N --trials-per-side T ...`.
  3. Runs `./scripts/py scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir best_codes`.
  4. Runs `./scripts/py scripts/sync_best_codes_matrices.py --best-dir best_codes`.
  5. Commits + pushes updates.

### Alternate pipeline
- Script: `scripts/refine_length_m4ri_pipeline.sh`
- Steps:
  1. Runs `./scripts/py scripts/refine_best_codes_m4ri_by_length.py ...`.
  2. Archives below-trials codes for the length.
  3. Rebuilds website artifacts via `scripts/rebuild_best_codes_artifacts_from_meta.py` and `scripts/ensure_best_codes_data_json_from_meta.py`.
  4. Syncs matrices.
  5. Commits + pushes.

## Publish/update best_codes + website

### Full repo publish pipeline
- Script: `scripts/update_best_codes_repo.sh`
- Steps:
  1. Runs `./scripts/py scripts/collect_best_codes.py` (collects results/**/best_codes into best_codes/).
  2. Runs `./scripts/py scripts/generate_best_codes_site.py` (regenerates best_codes/data.json).
  3. `git add` + commit + push.

### Publish for a group/order
- Script: `scripts/publish_best_codes_group.sh`
- Steps:
  1. Runs `./scripts/py scripts/publish_best_codes_group.py ...` to merge results into best_codes/.
  2. Runs `./scripts/py scripts/prune_best_codes_group.py ...` for each matching group.
  3. Runs `./scripts/py scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir best_codes`.
  4. Runs `./scripts/py scripts/sync_best_codes_matrices.py --best-dir best_codes`.
  5. Commits + pushes.

### Publish from meta only
- Script: `scripts/publish_best_codes_from_meta.sh`
- Steps:
  1. Runs `./scripts/py scripts/rebuild_best_codes_artifacts_from_meta.py`.
  2. `git add` + commit (no push).
