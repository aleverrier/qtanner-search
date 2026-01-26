# qtanner-search cleanup spec

## Goals (must implement)

### 1) Search (group or group order)
Provide a CLI that can search for quantum Tanner codes given:
- a specific group name (e.g. C2xC2) OR
- a group order (e.g. 16) and iterate over groups of that order.

The search must:
- enumerate permutations and multisets (exhaustive where feasible; progressive heuristics allowed)
- compute classical distance EXACTLY by enumerating all classical codewords when k <= 8 (<= 2^8)
- allow two-phase quantum distance estimation ("fast" then "slow") using the existing m4ri machinery
- include a more-exhaustive option that tries more candidate local-code settings when classical parameters are best

### 2) Refine distances for published codes (by length)
Provide a CLI command:
- input: code length n, number of trials T
- action: refine distance estimates for all codes of length n that are currently published on the website
- output: updated metadata in best_codes/ (and any artifacts the repo expects)

### 3) Live update best_codes + website
Any search/refine action must, by default:
- update best_codes/ in-repo
- regenerate the website content in-repo
Provide an escape hatch flag like --no-publish for development.

## CLI interface (required)
- qtanner search progressive ...
- qtanner search exhaustive ...
- qtanner refine length --n N --trials T ...
- qtanner publish

## Constraints
- Must be safe for laptop by default:
  - no long-running operations in tests
  - keep default trials small; expose knobs for bigger runs
- Prefer standard library (argparse) over adding new dependencies.
- Keep existing scripts working (don't break current pipelines).
- Logs go under runs/ but runs/ should be gitignored.
