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
    lines.extend([r"\end{itemize}", r"\section*{Coverage}"])

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

    lines.append(r"\section*{Best-by-k Tables}")
    if best_by_k:
        for group_spec in sorted(best_by_k):
            lines.append(f"\\subsection*{{Group {_escape(group_spec)}}}")
            for n_val in sorted(best_by_k[group_spec], key=lambda x: int(x)):
                lines.append(f"\\paragraph*{{$n={n_val}$}}")
                lines.append(r"\begin{tabular}{rrrrrrrl}")
                lines.append(r"\toprule")
                lines.append(r"$k$ & $d_{ub}$ & trials & $d_X$ & $d_Z$ & uniq & avg & id \\")
                lines.append(r"\midrule")
                for row in best_by_k[group_spec][n_val]:
                    avg = row.get("avg_lim")
                    avg_txt = "n/a" if avg is None else f"{avg:.3f}"
                    lines.append(
                        f"{row.get('k')} & {row.get('d_ub')} & {row.get('trials_used')} & "
                        f"{row.get('dx_ub')} & {row.get('dz_ub')} & {row.get('uniq_lim')} & "
                        f"{avg_txt} & {_escape(str(row.get('id')))} \\\\"
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

    lines.append(r"\section*{How to Reproduce/Check}")
    if best_codes:
        lines.append(r"\begin{itemize}")
        for rec in best_codes:
            row = best_lookup.get(rec["id"], {})
            trials = row.get("trials_used", 20000)
            cmd = (
                f"python -m qtanner.check_distance --run {run_dir} "
                f"--id {rec['id']} --trials {trials} --uniq-target 5 --gap-cmd gap"
            )
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
