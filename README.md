Quantum Tanner code search (Leverrier-Rozendaal-Zemor, arXiv:2512.20532).

Local code support is limited to the shortened Hamming [6,3,3].

Quick start:
- `python -m qtanner.smoke`
- `python -m qtanner.search --m-list 6,8,9 --topA 30 --topB 30 --trials 20 --mindist 10`
- `python -m qtanner.search --groups C4,C2xC2 --allow-repeats --max-n 200 --topA 30 --topB 30 --trials 20 --mindist 10`
- Non-abelian via GAP (not runnable under max-n 200, but supported): `python -m qtanner.search --groups "SmallGroup(8,3)" --allow-repeats --max-n 200 --topA 30 --topB 30 --trials 20 --mindist 10`

Classical selection (A/B candidates):
- Uses a (k,d)-tradeoff frontier based on k_min=min(k_slice) and d_min=min(d_slice).
- Multisets are deduplicated up to Aut(G) (Cayley-unique) before scoring.
- Frontier artifacts are written to `runs/<run>/classical_A_frontier.json` and `runs/<run>/classical_B_frontier.json`.
- Optional caps: `--frontier-max-per-point` (default 50) and `--frontier-max-total` (default 500).
- Legacy flags `--topA/--topB/--topA-d/--topA-k` remain accepted but no longer drive selection.

Best-by-k tracking:
- Tail the live table with: `tail -f runs/<run>/best_by_k.txt`

Distance re-check:
- `python -m qtanner.check_distance --run <RUN_DIR> --id <CODE_ID> --trials 50000 --uniq-target 5 --gap-cmd gap`

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
