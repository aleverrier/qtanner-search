Quantum Tanner code search (Leverrier-Rozendaal-Zemor, arXiv:2512.20532).

Local code support is limited to the shortened Hamming [6,3,3].

Quick start:
- `python -m qtanner.smoke`
- `python -m qtanner.search --m-list 6,8,9 --topA 30 --topB 30 --steps 5000`
- `python -m qtanner.search --groups C4,C2xC2 --A-enum multiset --B-enum multiset --max-n 200 --topA 30 --topB 30 --steps 5000`
- Non-abelian via GAP (not runnable under max-n 200, but supported): `python -m qtanner.search --groups "SmallGroup(8,3)" --A-enum multiset --B-enum multiset --max-n 200 --topA 30 --topB 30 --steps 5000`
- Progressive mode (exhaustive classical-first): `python -m qtanner.search progressive --group C2xC2xC2 --target-distance 16 --classical-steps 100 --quantum-steps-fast 2000 --quantum-steps-slow 50000 --report-every 50`

Distance estimation (dist-m4ri):
- `qtanner.search` uses dist-m4ri RW (method=1) for CSS distance estimates.
- When `--target-distance` is set, we pass `wmin=target-1`; a negative `d` from dist-m4ri indicates early stop.
- Install dist-m4ri and ensure `dist_m4ri` is on `PATH` (e.g., build from your local dist-m4ri checkout and add its `src/` to `PATH`).
- GAP is only used for group automorphisms and SmallGroup data; distance estimates do not use GAP.
- Run the C2xC2xC2 target-16 search with: `scripts/run_c2xc2xc2_d16.sh`
- Run the progressive C2xC2xC2 target-16 search with: `scripts/run_progressive_c2xc2xc2_d16.sh`

Progressive mode:
- Enumerates all multisets of size 6 containing the identity (optionally deduplicated).
- Scores classical slice codes with dist-m4ri, then explores quantum pairs in a promising-first order.
- Ctrl-C stops after the current evaluation and prints a final best-by-k summary.
- Outputs go under the run directory (classical JSONL + histograms, `best_codes/`, `milestones.jsonl`).
- You can also use `python -m qtanner.search --mode progressive ...` instead of the subcommand.

Classical selection (A/B candidates):
- Uses a (k,d)-tradeoff frontier based on k_min=min(k_slice) and d_min=min(d_slice).
- Multisets are deduplicated up to Aut(G) (Cayley-unique) before scoring.
- Frontier artifacts are written to `runs/<run>/classical_A_frontier.json` and `runs/<run>/classical_B_frontier.json`.
- Optional caps: `--frontier-max-per-point` (default 50) and `--frontier-max-total` (default 500).
- Legacy flags `--topA/--topB/--topA-d/--topA-k` remain accepted but no longer drive selection.
- Enumeration modes: `--A-enum/--B-enum subset|multiset|ordered` (default is multiset for |G|<=20 and nA=nB=6, otherwise subset).

Best-by-k tracking:
- Tail the live table with: `tail -f runs/<run>/best_by_k.txt`

Distance re-check:
- `python -m qtanner.check_distance --run <RUN_DIR> --id <CODE_ID> --steps 5000 --dist-m4ri-cmd dist_m4ri`

Report generation:
- `python -m qtanner.report --run <RUN_DIR> --out report.tex`
- Optional PDF: `python -m qtanner.report --run <RUN_DIR> --out report.tex --pdf`

Docs:
- `docs/COMMANDS.md` (command cheat sheet)
- `docs/groups_under_20.txt` (SmallGroup identifiers)

Global logs:
- `runs/_global/best_overall.json`
- `runs/_global/best_overall.txt`
- `runs/_global/best_overall.tex`
- `runs/_global/history.jsonl`
