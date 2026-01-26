# search_progressive.py CLI reference

This document explains the command-line options of `scripts/search_progressive.py`.

## What the script does

It enumerates candidate quantum Tanner codes (from a group `G` and generator multisets `A,B` plus local-code column permutations),
filters them using a classical slice-distance heuristic, then estimates quantum CSS distance using two distance estimates:
- fast estimate (cheap, used to reject early)
- slow/refined estimate (more expensive, used on promising candidates)

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
- `--classical-enum-kmax K`
  (fast backend) exact enumeration cutoff.
  Alias: `--classical-exhaustive-k-max`.
- `--classical-sample-count N`
  (fast backend) number of random samples.

### Quantum distance estimation
- `--quantum-steps-fast STEPS`
  Steps for the fast quantum distance estimate.
- `--quantum-steps-slow STEPS`
  Steps for slow/refined estimate.
- `--quantum-refine-chunk STEPS`
  Refinement chunk size.

### Runtime / reproducibility
- `--seed SEED`
  RNG seed.
- `--kmax N`
  Stop after N quantum evaluations (0 = no limit).
  Alias: `--max-quantum-evals`.
- `--report-every N`
  Print summary every N quantum evals.
- `--results-dir PATH`
  Output directory.
- `--timings`
  Print timing breakdown.

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

python3 -u scripts/search_progressive.py \\
  --group "C2xC2" \\
  --target-distance 0 \\
  --classical-target 1 \\
  --classical-distance-backend fast \\
  --quantum-steps-fast 1000 \\
  --quantum-steps-slow 100000 \\
  --seed 1
