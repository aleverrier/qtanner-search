"""Generate a LaTeX report from a qtanner run directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def _escape(text: str) -> str:
    return text.replace("_", "\\_")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_best_codes(run_dir: Path) -> List[dict]:
    best_root = run_dir / "best_codes"
    results = []
    if not best_root.exists():
        return results
    for group_dir in sorted(best_root.iterdir()):
        if not group_dir.is_dir():
            continue
        for n_dir in sorted(group_dir.glob("n*")):
            if not n_dir.is_dir():
                continue
            for k_dir in sorted(n_dir.glob("k*")):
                if not k_dir.is_dir():
                    continue
                for code_dir in sorted(k_dir.iterdir()):
                    if code_dir.is_dir():
                        results.append(
                            {
                                "group": group_dir.name,
                                "n": n_dir.name.lstrip("n"),
                                "k": k_dir.name.lstrip("k"),
                                "id": code_dir.name,
                                "path": str(code_dir),
                            }
                        )
    return results


def _load_global_best() -> dict:
    global_path = Path("runs") / "_global" / "best_overall.json"
    return _load_json(global_path)


def _best_by_k_lookup(best_by_k: dict) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for group_spec, n_map in best_by_k.items():
        for n_val, rows in n_map.items():
            for row in rows:
                code_id = row.get("id")
                if code_id:
                    lookup[str(code_id)] = row
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX report for a run.")
    parser.add_argument("--run", required=True, help="Run directory.")
    parser.add_argument("--out", required=True, help="Output .tex path.")
    parser.add_argument("--pdf", action="store_true", help="Compile to PDF if pdflatex is available.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_path = Path(args.out)
    run_meta = _load_json(run_dir / "run_meta.json")
    coverage = _load_json(run_dir / "coverage.json")
    best_by_k = _load_json(run_dir / "best_by_k.json")
    best_codes = _collect_best_codes(run_dir)
    best_lookup = _best_by_k_lookup(best_by_k)
    global_best = _load_global_best()

    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{Run Summary}",
        r"\begin{itemize}",
    ]
    for key in sorted(run_meta):
        lines.append(f"  \\item {_escape(str(key))}: {_escape(str(run_meta[key]))}")
    lines.extend([r"\end{itemize}", r"\section*{Methodology}"])
    selection_mode = run_meta.get("classical_keep", "unknown")
    if selection_mode == "frontier":
        selection_desc = "frontier selection on the $(k,d)$ tradeoff curve"
    elif selection_mode == "above_sqrt":
        selection_desc = "keep all candidates with $d_{min} \ge d_0$"
    else:
        selection_desc = "selection policy from run metadata"
    lines.extend(
        [
            r"\begin{itemize}",
            r"\item Parameterization: choose a finite group $G$, multisets $A,B$ of size 6, "
            r"and local $[6,3,3]$ column permutations (30 variants) for each side.",
            r"\item Classical exploration: for each side and permutation, compute two slice Tanner codes. "
            r"Define $k_{min}=\min(k_{slice1},k_{slice2})$ and $d_{min}=\min(d_{slice1},d_{slice2})$.",
            rf"\item Classical selection: use {selection_desc}. The threshold $d_0=\lceil\sqrt{{36|G|}}\rceil$ "
            r"compares classical slice distance to the quantum block length $n=36|G|$.",
            r"\item Quantum stage: build $H_X,H_Z$, compute $k=n-\mathrm{rank}(H_X)-\mathrm{rank}(H_Z)$, "
            r"and skip candidates with $k=0$.",
            r"\item Distance estimation: dist-m4ri RW (method=1) on $(H_X,H_Z)$ and $(H_Z,H_X)$ to estimate "
            r"$d_Z,d_X$, with $d=\min(d_X,d_Z)$. When a target distance is set, we pass $wmin=target-1$ so "
            r"negative $d$ indicates early stop from a low-weight codeword.",
            r"\end{itemize}",
            r"\section*{Coverage}",
        ]
    )

    if coverage:
        lines.append(r"\begin{tabular}{lllllll}")
        lines.append(r"\toprule")
        lines.append(r"Group & $|G|$ & $n$ & $A_{sel}$ & $B_{sel}$ & $Q_{meas}$ & ratio$_{ge}$ \\")
        lines.append(r"\midrule")
        for group_spec in sorted(coverage):
            entry = coverage[group_spec]
            ratio = entry.get("ratios", {}).get("Q_meas_over_possible_ge_sqrt")
            ratio_txt = "n/a" if ratio is None else f"{ratio:.3f}"
            lines.append(
                f"{_escape(group_spec)} & {entry.get('order')} & {entry.get('n_quantum')} & "
                f"{entry.get('A_candidates_selected')} & {entry.get('B_candidates_selected')} & "
                f"{entry.get('Q_meas_run')} & {ratio_txt} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"No coverage.json found.")

    lines.append(r"\section*{Best Quantum Codes Found in This Run}")
    if best_by_k:
        lines.append(r"\begin{tabular}{llrrrrl}")
        lines.append(r"\toprule")
        lines.append(r"Group & $|G|$ & $n$ & $k$ & $d_{est}$ & steps & id \\")
        lines.append(r"\midrule")
        for group_spec in sorted(best_by_k):
            order = coverage.get(group_spec, {}).get("order", "n/a")
            for n_val in sorted(best_by_k[group_spec], key=lambda x: int(x)):
                for row in best_by_k[group_spec][n_val]:
                    lines.append(
                        f"{_escape(group_spec)} & {order} & {n_val} & {row.get('k')} & "
                        f"{row.get('d_ub')} & {row.get('steps')} & "
                        f"{_escape(str(row.get('id')))} \\\\"
                    )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"No best\_by\_k.json found.")

    lines.append(r"\section*{Best-by-k Tables}")
    if best_by_k:
        for group_spec in sorted(best_by_k):
            lines.append(f"\\subsection*{{Group {_escape(group_spec)}}}")
            for n_val in sorted(best_by_k[group_spec], key=lambda x: int(x)):
                lines.append(f"\\paragraph*{{$n={n_val}$}}")
                lines.append(r"\begin{tabular}{rrrrrrl}")
                lines.append(r"\toprule")
                lines.append(r"$k$ & $d_{est}$ & steps & $d_X$ & $d_Z$ & id \\")
                lines.append(r"\midrule")
                for row in best_by_k[group_spec][n_val]:
                    lines.append(
                        f"{row.get('k')} & {row.get('d_ub')} & {row.get('steps')} & "
                        f"{row.get('dx_ub')} & {row.get('dz_ub')} & "
                        f"{_escape(str(row.get('id')))} \\\\"
                    )
                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
    else:
        lines.append(r"No best\_by\_k.json found.")

    lines.append(r"\section*{Saved Best Codes}")
    if best_codes:
        lines.append(r"\begin{tabular}{llll}")
        lines.append(r"\toprule")
        lines.append(r"Group & $n$ & $k$ & id \\")
        lines.append(r"\midrule")
        for rec in best_codes:
            lines.append(
                f"{_escape(rec['group'])} & {rec['n']} & {rec['k']} & {_escape(rec['id'])} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"No best codes saved.")

    lines.append(r"\section*{Global Best So Far (All Runs)}")
    if global_best:
        lines.append(r"\begin{tabular}{llrrrrll}")
        lines.append(r"\toprule")
        lines.append(r"Group & $n$ & $k$ & $d_{est}$ & steps & $d_X$ & $d_Z$ & id \\")
        lines.append(r"\midrule")
        for group_spec in sorted(global_best):
            for n_val in sorted(global_best[group_spec], key=lambda x: int(x)):
                for k_val in sorted(global_best[group_spec][n_val], key=lambda x: int(x)):
                    row = global_best[group_spec][n_val][k_val]
                    lines.append(
                        f"{_escape(group_spec)} & {row.get('n')} & {row.get('k')} & "
                        f"{row.get('d_ub')} & {row.get('steps')} & "
                        f"{row.get('dx_ub')} & {row.get('dz_ub')} & "
                        f"{_escape(str(row.get('id')))} \\\\"
                    )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"No global best file found at runs/\_global/best\_overall.json.")

    lines.append(r"\section*{How to Reproduce/Check}")
    if best_codes:
        steps = run_meta.get("steps", 5000)
        target_distance = run_meta.get("target_distance")
        dist_cmd = run_meta.get("dist_m4ri_cmd", "dist_m4ri")
        lines.append(r"\begin{itemize}")
        for rec in best_codes:
            cmd = (
                f"python -m qtanner.check_distance --run {run_dir} "
                f"--id {rec['id']} --steps {steps} --dist-m4ri-cmd {dist_cmd}"
            )
            if target_distance is not None:
                cmd += f" --target-distance {target_distance}"
            lines.append(f"  \\item \\texttt{{{_escape(cmd)}}}")
        lines.append(r"\end{itemize}")
    else:
        lines.append(r"No best codes available for re-check commands.")

    lines.append(r"\end{document}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.pdf:
        if shutil.which("pdflatex") is None:
            print("[report] pdflatex not found; wrote .tex only.")
            return 0
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(out_path)],
                cwd=str(out_path.parent),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print("[report] pdflatex failed.")
            print(exc.stdout)
            print(exc.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
