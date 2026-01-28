# search_progressive.py CLI reference

This document explains the command-line options of `scripts/search_progressive.py`.

## What the script does

It enumerates candidate quantum Tanner codes (from a group `G` and generator multisets `A,B` plus local-code column permutations),
filters them using a classical slice-distance heuristic, then estimates quantum CSS distance using two distance estimates:
- fast estimate (cheap, used to reject early)
- slow/refined estimate (more expensive, driven by the current best_codes table for each (n,k))

Distances reported are *upper bounds* found by a search routine (i.e., “we found a logical operator of weight w, so distance <= w”).

## Key concepts printed in logs

- `k`: estimated dimension of the quantum code
- `best`: best (largest) distance upper bound found so far for this `k` among *kept* candidates
- `must_exceed`: threshold a candidate must exceed to be kept (depends on `best` and `--target-distance`)
- `dX_best`, `dZ_best`: current best upper bounds for X and Z logical distances
- `decision`: accept/reject

## Options

### Required / main
- `--group GROUP`
  Group specification (examples: "C2xC2", "SmallGroup(16,3)" depending on repo support).

- `--target-distance D`
  Minimum quantum distance target.
  - If `D > 0`: reject candidates that cannot beat `max(best_by_k, D-1)`.
  - If `D = 0`: disable the fixed target and run “best-by-k” mode (keep first code per k, then only try to beat best).

### Classical prefilter (slice codes)
- `--classical-target T`
  Target distance threshold for the classical slice filter.
  Lower values = less filtering (more exhaustive, more compute).
- `--classical-distance-backend {dist-m4ri,fast}`
  Choose how slice distances are estimated.
- `--classical-steps STEPS`
  Steps for dist_m4ri classical distance estimation.
- `--classical-exhaustive-k-max K`
  (fast backend) exact enumeration cutoff.
- `--classical-sample-count N`
  (fast backend) number of random samples.

### Quantum distance estimation
- `--quantum-steps-fast STEPS`
  Steps for the fast quantum distance estimate.
- `--quantum-steps-slow STEPS`
  Deprecated (use `--slow-quantum-trials-override`).
- `--slow-quantum-trials-override STEPS`
  Override slow/refined trials (debugging only; default derives from best_codes).
- `--quantum-refine-chunk STEPS`
  Refinement chunk size.
- `--best-codes-source {auto,origin,working-tree,website}`
  Where to load the best_codes index from (default: auto).
- `--best-codes-refresh-seconds N`
  Minimum seconds between best_codes refreshes (default: 600).

### Runtime / reproducibility
- `--seed SEED`
  RNG seed.
- `--max-quantum-evals N`
  Stop after N quantum evaluations (0 = no limit).
- `--report-every N`
  Print summary every N quantum evals.
- `--results-dir PATH`
  Output directory.
- `--save-new-bests-dir PATH`
  Directory for `decision=new_best` JSON artifacts (default: `codes/pending`).
- `--no-save-new-bests`
  Disable writing new_best artifacts.
- `--timings`
  Print timing breakdown.

### Best codes update (post-run)
After a successful run, `scripts/search_progressive.py` now runs the best-codes
updater to sync `best_codes/`, rebuild website data, and push to GitHub.
If you interrupt the run with Ctrl-C, it will still attempt this update.

Workflow:
- Run `scripts/search_progressive.py` (or `python -m qtanner.search progressive`).
- New best candidates are written to `codes/pending/*.json` when logged as `decision=new_best`.
- Run `scripts/scrape_and_publish_best_codes.py` to scan pending artifacts and publish `best_codes/data.json`.

- `--no-best-codes-update`
  Skip all post-run best-codes steps.
- `--no-git`
  Run the updater but skip git pull/commit/push.
- `--no-publish`
  Run the updater but skip website data updates.
- `--best-codes-no-history`
  Skip git history scanning during the updater (faster, may miss legacy codes).
- `--best-codes-max-attempts N`
  Max push attempts if the remote moved (default: 3).

### Cayley multiset enumeration
- `--dedup-cayley`
  Deduplicate multisets up to Aut(G) (requires GAP).
- `--min-distinct N`
  Require at least N distinct elements in each multiset.
- `--gap-cmd GAP`
  GAP executable (needed for SmallGroup or dedup).

## Suggested “presets”

### Very exhaustive (small groups only)
- classical filter essentially disabled
- quantum target disabled

Example:

python3 -u scripts/search_progressive.py \
  --group "SmallGroup(16,1)" \
  --target-distance 16 \
  --classical-target 16 \
  --classical-distance-backend fast \
  --quantum-steps-fast 5000 \
  --seed 1
