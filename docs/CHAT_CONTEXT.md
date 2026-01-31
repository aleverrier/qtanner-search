# Chat context (quickstart)

- Project: search explicit quantum Tanner codes from left–right Cayley complexes (LRZ, arXiv:2512.20532).
- Current focus: lengths ~200–500; laptop-scale compute; filter-first workflows.
- Two local-code tracks:
  - 633 track (default): local [6,3,3] shortened Hamming codes.
  - 844 track: local [8,4,4] extended Hamming codes.
- Progressive search is the main pipeline: `scripts/search_progressive.py` -> `src/qtanner/progressive_search.py`.
- Progressive search uses a fast classical filter, then dist-m4ri fast/slow quantum estimates.
- Slow trials policy: `max(50k, best_codes trials)` unless overridden.
- Early abort: refinement stops if candidate cannot beat current best_codes for (n,k).
- Base-k filter: `--min-base-k` prunes weak local-code permutation pairs.
- New best candidates are written to `codes/pending*/` (track-specific).
- Best-codes publishing scans run artifacts + `best_codes*/meta` + history.
- Website data is derived from `best_codes*/data.json` and `best_codes*/index.tsv`.
- Best-codes selection rule: per (n,k), keep max trials; break ties by higher d, then code_id.
- “Promising” codes saved: d >= sqrt(n) and k*d >= n.
- Avoid data loss: archive untracked artifacts to `local_results/` before git pull/rebase.
- Two-machine workflow: search on one clone; publish from a maintenance clone only.
- Current tracks live at:
  - 633: `best_codes/`, website: `best_codes/index.html`
  - 844: `best_codes_844/`, website: `best_codes_844/index.html`

## Key scripts
- `scripts/search_progressive.py`: main search runner; triggers best_codes update post-run.
- `scripts/scrape_and_publish_best_codes.py`: scan + select + publish best_codes (supports `--track 633/844`).
- `scripts/refine_best_codes.py`: refine best_codes distances for length n; then publish.
- `scripts/refine_best_codes_length.sh` / `scripts/refine_group_m4ri_pipeline.sh`: refinement pipelines.
- `scripts/rebuild_best_codes_artifacts_from_meta.py`: rebuild data.json/index.tsv for a track.

## Results layout
- Search outputs: `results/progressive_*` with `best_codes*/`, `milestones.jsonl`, histograms.
- New best artifacts: `codes/pending` (633), `codes/pending_844` (844).
- Published best codes: `best_codes*/meta/*.json` + `best_codes*/matrices/*__Hx.mtx/__Hz.mtx`.

## Publishing workflow
- After a search: run `scripts/scrape_and_publish_best_codes.py --track 633` (or 844).
- This syncs best_codes folders, rebuilds website datasets, and commits/pushes.

## Conventions to remember
- Keep slow trials >= 50k, driven by best_codes.
- Use early-abort during refine to save time.
- Store only promising codes (d >= sqrt(n), k*d >= n).
- All saved matrices must be MatrixMarket `.mtx`.
