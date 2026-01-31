# Local [8,4,4] track plan (progressive search)

## Scope
Add a second search track that uses local [8,4,4] extended Hamming codes (weight-? regime), alongside the existing [6,3,3] track, and publish results in a separate website table.

This doc summarizes what exists today, what changes are needed, and minimal smoke tests. It is based on current repo inspection (Jan 31, 2026).

## Current state (where things live)
- Local code definitions and permutation reps:
  - `src/qtanner/local_codes.py`
    - `hamming_6_3_3_shortened()`
    - `variants_6_3_3()` (distinct column permutations; expected 30)
- HX/HZ construction:
  - `src/qtanner/lift_matrices.py` (builds HX/HZ from group + local codes)
- Progressive search flow:
  - `src/qtanner/progressive_search.py` (uses local codes, builds HX/HZ, runs dist-m4ri)
  - `scripts/search_progressive.py` (CLI wrapper; runs best_codes updater)
- Best codes selection + publishing:
  - `src/qtanner/best_codes_updater.py`
    - writes `best_codes/data.json`, `best_codes/index.tsv`, `best_codes/best_by_group_k.tsv`
  - `scripts/scrape_and_publish_best_codes.py`
  - `scripts/rebuild_best_codes_artifacts_from_meta.py` (already supports `--best-dir`)
- Website:
  - `best_codes/index.html`, `best_codes/simple.html`, `best_codes/app.js`

## Length formulas and feasible group orders
Let nA,nB be local code lengths, and |G| be group order. Then total length is:
- n = nA * nB * |G|

### 8×8 track (both sides [8,4,4])
- n = 64 * |G|
- For n in 200–500, this suggests:
  - |G| = 4  -> n = 256
  - |G| = 5  -> n = 320
  - |G| = 6  -> n = 384
  - |G| = 7  -> n = 448
  - |G| = 3 gives 192 (below), |G| = 8 gives 512 (above)

### Mixed 8×6 track (if supported later)
- n = 48 * |G|
- For n in 200–500:
  - |G| = 5 -> 240
  - |G| = 6 -> 288
  - |G| = 7 -> 336
  - |G| = 8 -> 384
  - |G| = 9 -> 432
  - |G| = 10 -> 480

## Local code choice: A/B sides
- Initial scaffolding assumes **same-length local codes on both sides**.
- Mixed (8×6) is desirable but not implemented yet. It would require separate multiset sizes for A and B and minor refactors to the progressive enumerators.

## Compute implications
- nA = nB = 8 increases slice size and overall HX/HZ sizes by ~ (8/6)^2 ≈ 1.78.
- Classical precompute and dist-m4ri steps will be more expensive; fewer groups/orders are feasible on laptop-scale compute.
- Base-k filtering remains important; use `--min-base-k` to prune weak permutations.

## Changes required (code + data)
### Local code definitions
- `src/qtanner/local_codes.py`
  - Add `hamming_8_4_4_extended()` with parity-check + generator rows (self-dual).
  - Add `variants_8_4_4()` via column-permutation representatives.
  - Add `local_code_from_name()` and `variants_for_local_code()` helper.

### Progressive search wiring
- `src/qtanner/progressive_search.py`
  - New CLI flags: `--local-a {6_3_3,8_4_4}`, `--local-b {6_3_3,8_4_4}`.
  - Use dynamic side length from local codes (instead of fixed `SIDE_LEN=6`).
  - Use local-code selection for A/B in HX/HZ construction.
  - Track `best_codes_dir` and optional `codes/pending_844` outputs.
  - Record local code names in `run_meta.json`.

### Best codes publish (separate track)
- `src/qtanner/best_codes_updater.py`
  - Parameterize `best_dir_name` (default `best_codes`).
  - Scan run-level folders matching best_dir_name (e.g., `best_codes_844`).
  - Use a track-specific pending dir (default: `codes/pending` or `codes/pending_844`).
  - Emit `best_dir_name/data.json`, `index.tsv`, `best_by_group_k.tsv`.
- `scripts/scrape_and_publish_best_codes.py`
  - Add `--track {633,844}` or `--best-dir` to target the desired folder.

### Website page
- Add `best_codes_844/index.html` and `best_codes_844/simple.html` to host a
  separate table that reads `best_codes_844/data.json`.
- Reuse `best_codes/app.js` and `best_codes/style.css`.

## What is already implemented in this branch
- Local-code scaffolding for [8,4,4], with helpers in `src/qtanner/local_codes.py`.
- Progressive search selection via `--local-a/--local-b`, plus dynamic side length handling.
- Track-aware best_codes updates via `--best-codes-dir` (and `--track` in the scraper).
- Minimal website pages: `best_codes_844/index.html` + `best_codes_844/simple.html`.

## Remaining work / checklist
- Decide whether mixed 8×6 (A/B different lengths) should be supported; if yes:
  - Update multiset enumeration to use separate sizes for A and B.
  - Update progress reports to show separate settings per side.
- Update non-progressive search (`src/qtanner/search.py`) to accept local code selection.
- Add documentation updates (README + search_progressive_cli.md) for new flags.
- Ensure best_codes_844 is linked from the main website landing page (if desired).

## Smoke tests (no expensive search)
- Quick CLI parsing + early exit (no publish):
  - `python3 scripts/search_progressive.py --group C4 --target-distance 8 --local-a 8_4_4 --local-b 8_4_4 --classical-steps 5 --quantum-steps-fast 10 --max-quantum-evals 1 --no-best-codes-update --no-publish --no-git --best-codes-dir best_codes_844 --save-new-bests-dir codes/pending_844`
- Best-codes publish dry-run for 844 track:
  - `python3 scripts/scrape_and_publish_best_codes.py --track 844 --dry-run`
