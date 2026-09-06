# Quantum Tanner code search (project instructions)

## What this repo is
We are implementing a small-scale search for explicit quantum Tanner codes from left–right Cayley complexes (Leverrier–Rozendaal–Zémor, arXiv:2512.20532).

## Hard constraints
- Before a non-trivial simulation, give a best-effort runtime estimate and obtain a time/CPU budget if not already supplied. Run requested work within that budget; longer runs are allowed when authorized. Use fast smoke checks for correctness validation.
- Prefer fast “filter-first” workflows.
- Keep changes small and commit frequently.

## Development workflow
- After behavioral changes, run focused checks using the repository entrypoints below. For documentation-only edits, validate the affected documents and links:
  - Preferred: `make test`
  - Equivalent: `./scripts/run_tests.sh`
- Do not run `python -m pytest` directly (the test entrypoint is `./scripts/run_tests.sh`).
- Do not add heavy dependencies unless necessary.
- All generated matrices must be saved in MatrixMarket `.mtx` format.

## Outputs
- Store only “promising” codes (heuristics: d >= sqrt(n) and k*d >= n).
- Every saved code must include:
  - Hx.mtx, Hz.mtx
  - a metadata JSON describing (G, A, B, local code variants, n, k, distance estimate method, trials, RNG seed)

## Working agreements
- Preserve unrelated edits and resolve the cause of failures introduced by this task. Ask only when a material decision or missing information blocks progress.

- Use bounded work units and incremental checkpoints so the agreed time/CPU budget and explicit cancellation can be honored. Prefer clean stops between units; make units interruptible when they could exceed the remaining budget.
