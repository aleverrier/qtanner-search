# Quantum Tanner code search (project instructions)

## What this repo is
We are implementing a small-scale search for explicit quantum Tanner codes from left–right Cayley complexes (Leverrier–Rozendaal–Zémor, arXiv:2512.20532).

## Hard constraints
- Do NOT run any computation that could take more than a few minutes on a MacBook Pro.
- Prefer fast “filter-first” workflows.
- Keep changes small and commit frequently.

## Development workflow
- After making changes, run the repo’s quick checks (when available):
  - `python -m pytest -q` (or `make test` if present)
  - When running tests, run `./scripts/run_tests.sh` (not `python -m pytest`).
- Do not add heavy dependencies unless necessary.
- All generated matrices must be saved in MatrixMarket `.mtx` format.

## Outputs
- Store only “promising” codes (heuristics: d >= sqrt(n) and k*d >= n).
- Every saved code must include:
  - Hx.mtx, Hz.mtx
  - a metadata JSON describing (G, A, B, local code variants, n, k, distance estimate method, trials, RNG seed)

## When unsure
- Prefer asking for a smaller, testable implementation step.

## Critical Thinking
- Fix root cause (not band-aid).
- Unsure: read more code; if still stuck, ask w/ short options.
- Conflicts: call out; pick safer path.
- Unrecognized changes: assume other agent; keep going; focus your changes. If it causes issues, stop + ask user.
- Leave breadcrumb notes in thread.
