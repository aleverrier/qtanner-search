# Plan: search for explicit quantum Tanner codes

## Goal
Search over:
- finite groups G (in increasing order),
- generator multisets A,B,
- local code choices and local coordinate permutations,
to find quantum Tanner codes with length 100..1000 maximizing k and d.

We keep only candidates we hope satisfy:
- d >= sqrt(n)
- k*d >= n

## Implementation milestones
1) Local codes library: [6,3,3] only.
2) GAP interface for group enumeration + regular actions.
3) Matrix builder for Hx, Hz (lift formula).
4) Exact k from GF(2) rank.
5) Distance estimator integration:
   - GAP QDistRnd baseline
   - optional dist-m4ri via Docker for speed
6) Search pipeline with filters + persistence.

## Reminders
- [ ] Later: integrate dist-m4ri (m4ri backend) via Docker for faster distance estimation.

## Filters to try
Baseline:
- classical slice Tanner-code distance filter

New filters:
- stabilizer-triviality test for low-weight slice codewords
- short-cycle counts / girth proxy on Cayley graphs
