# best_codes/

This folder stores *artifacts* for the best quantum Tanner codes found so far.

## What to store
For each saved code we want:
- `meta/<CODE_TAG>.json` : all metadata needed to reconstruct the code
  (group, A/B, local code IDs, permutations, quantum (n,k), distance estimation settings,
   and the 4 classical slice code parameters).
- `matrices/<CODE_TAG>_HX.mtx` and `matrices/<CODE_TAG>_HZ.mtx` : parity-check matrices
  for the quantum code (Matrix Market format recommended).
- `matrices/<CODE_TAG>_classical_*.mtx` : the 4 classical slice Tanner-code parity-check matrices.

## Naming convention
Suggested `<CODE_TAG>`:
`G=<group>_n=<n>_k=<k>_dub=<dub>_<timestamp>_A=<A_id>_B=<B_id>`

Keep it filesystem-friendly: no spaces.
