# Architecture (quick map)

## Top-level layout
- `src/qtanner/`
  - Core library: group handling, local codes, matrix construction, search loops.
- `scripts/`
  - CLI entrypoints, pipelines, publish/refine helpers.
- `best_codes/` and `best_codes_844/`
  - Published best-code artifacts + website data.
- `docs/`
  - Workflow docs, CLI references, context packs.
- `best_codes/index.html` and `best_codes_844/index.html`
  - Static website tables (read data.json).

## Module map (core)
- `src/qtanner/local_codes.py`
  - Local code definitions + column-permutation reps.
- `src/qtanner/lift_matrices.py`
  - Builds HX/HZ from group + local codes.
- `src/qtanner/progressive_search.py`
  - Progressive search: classical filter, fast/slow quantum distance, best_codes gating.
- `src/qtanner/search.py`
  - Non-progressive search/pilot pipeline.
- `src/qtanner/best_codes_updater.py`
  - Scrape/select/publish best codes; rebuilds data.json/index.tsv.

## Scripts (entrypoints)
- `scripts/search_progressive.py`
  - Runs progressive search and best_codes update.
- `scripts/scrape_and_publish_best_codes.py`
  - Publish best_codes for a track (`--track 633/844` or `--best-dir`).
- `scripts/refine_best_codes.py`
  - Refines distances for a length; republish.
- `scripts/rebuild_best_codes_artifacts_from_meta.py`
  - Rebuilds `data.json`, `index.tsv`, `best_by_group_k.tsv` for a track.

## Code record representation
- A “code record” lives in:
  - `best_codes*/meta/<code_id>.json`
  - `best_codes*/matrices/<code_id>__Hx.mtx` and `__Hz.mtx`
  - (optional) `best_codes*/collected/<code_id>/` with `Hx.mtx`, `Hz.mtx`, `meta.json`
- `data.json` schema (per entry):
  - `code_id`, `group`, `n`, `k`, `d_ub`, `m4ri_trials`, `meta` (full meta JSON).
- `index.tsv` provides a compact table: group, n, k, d_ub, trials, code_id.

## How to add a new track
- Choose a best_codes directory name (e.g., `best_codes_844`).
- Run searches with `--best-codes-dir <dir>` and optionally `--save-new-bests-dir`.
- Publish with `scripts/scrape_and_publish_best_codes.py --track <id>` or `--best-dir`.
- Add website pages under `<dir>/index.html` and `<dir>/simple.html` that load `<dir>/data.json`.
- Keep pending artifacts in `codes/pending_<suffix>` to avoid cross-track mixing.
