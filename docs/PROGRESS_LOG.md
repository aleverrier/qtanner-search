# Progress log (2026-01-31)

## Overview and goals
We are searching for explicit quantum Tanner codes from left-right Cayley complexes
(Leverrier--Rozendaal--Zemor, arXiv:2512.20532). The current focus is on lengths
roughly 200--500 under laptop-scale compute constraints, using filter-first
workflows and only persisting promising codes (heuristic: d >= sqrt(n) and k*d >= n).

## Current workflow (canonical path)
1) Search: `scripts/search_progressive.py` (wrapper for `src/qtanner/progressive_search.py`).
   Produces run artifacts (best-by-k tables, histograms, `best_codes/` snapshots,
   `codes/pending/` new_best JSONs).
2) Scrape + publish: `scripts/scrape_and_publish_best_codes.py`.
   Scans run artifacts, `best_codes/meta`, `best_codes/collected`, and git history,
   selects best-by-(n,k), syncs matrices + meta, rebuilds `best_codes/data.json` and
   `best_codes/index.tsv`, and optionally commits/pushes.
3) Refine: `scripts/refine_best_codes.py` or length/group pipelines.
   Runs dist-m4ri refinements, updates metadata, rebuilds website artifacts, and
   republish via the best_codes updater.

## Canonical scripts and responsibilities
- `scripts/search_progressive.py`: end-to-end progressive search runner; triggers
  best_codes update on clean exit or Ctrl-C (unless `--no-best-codes-update`).
- `scripts/scrape_and_publish_best_codes.py`: standard scrape/publish entrypoint for
  best_codes (sync + website data + git).
- `scripts/refine_best_codes.py`: refine all best_codes entries at a given length
  (optional group filter), then resync/publish.
- `scripts/refine_best_codes_length.py` and `scripts/refine_best_codes_length.sh`:
  length-based refinement pipeline, including archiving below-trials entries.
- `scripts/refine_best_codes_m4ri.py` and `scripts/refine_best_codes_m4ri_by_length.py`:
  m4ri-only refinement (group or length scoped).
- `scripts/refine_length_m4ri_pipeline.sh` and `scripts/refine_group_m4ri_pipeline.sh`:
  bundled pipelines that refine, rebuild artifacts, sync matrices, and publish.
- Repair/consistency tools used after refinements:
  - `scripts/rebuild_best_codes_artifacts_from_meta.py`
  - `scripts/ensure_best_codes_data_json_from_meta.py`
  - `scripts/sync_best_codes_names_from_meta.py`
  - `scripts/sync_best_codes_matrices.py`
- Website artifacts live in `best_codes/`:
  `data.json` and `index.tsv` are the canonical published datasets.

## Key design choices and rationale
- Best-codes-driven slow distance evaluation:
  - Progressive search loads best_codes (data.json or index.tsv) to set slow-trial
    budgets and decide whether the slow pass is worthwhile for each (n,k).
  - Slow trials = max(50,000, best_codes trials) unless overridden.
  - Slow pass runs only if the fast estimate can plausibly beat the current best_d.
- Early-abort during refinement:
  - During slow refinement chunks, stop early if d_x or d_z cannot exceed the
    best_codes distance threshold. This prevents wasting trials on non-competitive
    candidates.
- Min slow trials = 50k:
  - Enforces a floor for reliability and keeps best_codes entries comparable.
- Base-k filter to reduce search space:
  - `--min-base-k` prunes local-code permutations whose unlifted/base code
    dimension is too small, reducing the search without changing correctness.

## Multi-machine best practice
- Keep a dedicated maintenance clone for publishing; do not `git pull` into an
  actively running search clone.
- Archive untracked artifacts from the search clone into `local_results/` before
  syncing. This avoids conflicts and preserves local run data.
- Only the maintenance clone should run best_codes publish/website rebuild steps.

## Troubleshooting: untracked files blocking git pull/rebase
- Move or archive run artifacts:
  - `mkdir -p local_results/` and move untracked run folders into it.
  - Alternatively: `git stash -u -m "archive run artifacts"`.
- After verifying backups, you can clean leftover untracked files with
  `git clean -fd` (only if you are certain nothing important remains).

## What's next (open tasks)
- Schedule refinement passes for lengths 200--500 to standardize trials across best_codes.
- Add a lightweight summary of best_codes trial coverage to the website dataset.
- Expand base-k filtering experiments to identify safe thresholds per group family.
- Tighten documentation around multi-machine workflows and the refine pipelines.
