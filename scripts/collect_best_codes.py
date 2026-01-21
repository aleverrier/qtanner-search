#!/usr/bin/env python3
"""
Collect all best codes stored under results/**/best_codes/* into a single canonical folder.

Typical source layout (example):
  results/<run_name>/best_codes/<CODE_ID>/
    (mtx files, json, etc.)

Where <CODE_ID> looks like:
  C2xC2xC2_AAp6_0_1_2_3_4_6_BBp6_0_1_2_4_5_7_k16_d16

What this script does:
- Scans for directories matching: results/**/best_codes/<CODE_ID>
- Parses CODE_ID to extract group, A_id, B_id, k, d (from folder name)
- Copies artifacts into:
    best_codes/collected/<CODE_ID>/
- Writes/updates metadata:
    best_codes/meta/<CODE_ID>.json
- Copies matrices into a flat folder for convenience:
    best_codes/matrices/<CODE_ID>__<original_filename>
- Generates:
    best_codes/index.tsv
    best_codes/best_by_group_k.tsv
- Updates notes/search_log.tex between markers:
    % BEGIN AUTO BEST TABLES
    ...
    % END AUTO BEST TABLES

Notes:
- "d" in CODE_ID is whatever your pipeline recorded (often an upper bound from QDistRnd).
- We only copy a whitelist of file types by default. Use --copy-all to copy everything.

Usage:
  python scripts/collect_best_codes.py --dry-run
  python scripts/collect_best_codes.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

AUTO_BEGIN = "% BEGIN AUTO BEST TABLES"
AUTO_END = "% END AUTO BEST TABLES"

# Example:
#   C2xC2xC2_AAp11_0_0_1_2_2_7_BBp11_0_0_1_2_4_5_k4_d20
CODE_ID_RE = re.compile(
    r"^(?P<group>.+?)_AA(?P<A>.+?)_BB(?P<B>.+?)_k(?P<k>\d+)_d(?P<d>\d+)$"
)

WHITELIST_EXTS = {
    ".mtx", ".json", ".txt", ".md", ".tsv", ".csv", ".npz", ".pkl", ".pickle", ".png", ".pdf"
}


@dataclass
class CodeRecord:
    code_id: str
    group: str
    A_id: str
    B_id: str
    A_elems: List[int]
    B_elems: List[int]
    k: int
    d_recorded: int
    d_recorded_kind: str  # "d_ub" by convention here
    n: Optional[int]      # inferred from matrix shapes if possible

    run_dir: str          # e.g. results/progressive_... (relative to repo root)
    src_dir: str          # e.g. results/.../best_codes/<code_id>
    collected_dir: str    # best_codes/collected/<code_id>

    matrices_flat: List[str]   # best_codes/matrices/* copied for convenience
    collected_files: List[str] # files copied under collected_dir
    collected_at_utc: str

    also_seen_in: List[str]    # other src dirs with same (group,A,B,k,d)


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(out)


def rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_multiset_elems(ms_id: str) -> List[int]:
    """
    Parse something like:
      p6_0_1_2_3_4_6  -> [0,1,2,3,4,6]
      p11_0_0_1_2_2_7 -> [0,0,1,2,2,7]
    Best-effort: take all integer tokens after the first underscore-separated token.
    """
    toks = ms_id.split("_")
    out: List[int] = []
    for t in toks[1:]:
        if re.fullmatch(r"-?\d+", t):
            out.append(int(t))
    return out


def parse_code_id(code_id: str) -> Optional[Tuple[str, str, str, List[int], List[int], int, int]]:
    m = CODE_ID_RE.match(code_id)
    if not m:
        return None
    group = m.group("group")
    A_id = m.group("A")
    B_id = m.group("B")
    k = int(m.group("k"))
    d = int(m.group("d"))
    return group, A_id, B_id, parse_multiset_elems(A_id), parse_multiset_elems(B_id), k, d


def iter_best_code_dirs(root: Path, sources: List[str]) -> Iterable[Tuple[Path, Path]]:
    """
    Yield (run_dir, code_dir) for each <run_dir>/best_codes/<code_id> directory found.
    """
    for src in sources:
        base = root / src
        if not base.exists():
            continue
        for best_codes_dir in base.rglob("best_codes"):
            if not best_codes_dir.is_dir():
                continue
            run_dir = best_codes_dir.parent
            for code_dir in best_codes_dir.iterdir():
                if code_dir.is_dir():
                    yield (run_dir, code_dir)


def read_mtx_shape(fp: Path) -> Optional[Tuple[int, int, int]]:
    """
    Parse Matrix Market header to get (nrows, ncols, nnz) if possible (coordinate format).
    """
    try:
        with fp.open("r", errors="replace") as f:
            header = f.readline()
            if not header.lower().startswith("%%matrixmarket"):
                return None
            # skip comments
            line = f.readline()
            while line and line.strip().startswith("%"):
                line = f.readline()
            if not line:
                return None
            parts = line.strip().split()
            if len(parts) < 3:
                return None
            nrows, ncols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
            return nrows, ncols, nnz
    except Exception:
        return None


def infer_n_from_code_dir(code_dir: Path) -> Optional[int]:
    """
    Try to infer n from any .mtx file (prefer ones containing hx/hz in filename).
    """
    mtx_files = sorted(code_dir.rglob("*.mtx"))
    if not mtx_files:
        return None

    def score(p: Path) -> int:
        s = p.name.lower()
        if "hx" in s or "h_x" in s:
            return 0
        if "hz" in s or "h_z" in s:
            return 1
        return 5

    mtx_files.sort(key=lambda p: (score(p), len(str(p))))
    for fp in mtx_files[:20]:
        shp = read_mtx_shape(fp)
        if shp:
            _, ncols, _ = shp
            return ncols
    return None


def copy_artifacts(src_code_dir: Path, dst_code_dir: Path, copy_all: bool) -> List[Path]:
    """
    Copy whitelisted files (or all files if copy_all) preserving relative structure inside code dir.
    Returns list of destination file paths.
    """
    ensure_dir(dst_code_dir)
    copied: List[Path] = []

    for fp in src_code_dir.rglob("*"):
        if fp.is_dir():
            continue
        if not copy_all and fp.suffix.lower() not in WHITELIST_EXTS:
            continue
        rel_path = fp.relative_to(src_code_dir)
        dst_fp = dst_code_dir / rel_path
        ensure_dir(dst_fp.parent)
        try:
            shutil.copy2(fp, dst_fp)
            copied.append(dst_fp)
        except Exception:
            continue

    return copied


def copy_matrices_flat(dst_flat_dir: Path, code_id: str, dst_code_dir: Path) -> List[Path]:
    """
    Copy all .mtx under collected code dir into best_codes/matrices/ as flat files for convenience.
    """
    ensure_dir(dst_flat_dir)
    out: List[Path] = []
    for fp in dst_code_dir.rglob("*.mtx"):
        flat_name = f"{code_id}__{fp.name}"
        dst = dst_flat_dir / flat_name
        try:
            shutil.copy2(fp, dst)
            out.append(dst)
        except Exception:
            continue
    return out


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def update_or_create_meta(root: Path, meta_dir: Path, record: CodeRecord) -> None:
    ensure_dir(meta_dir)
    meta_path = meta_dir / f"{record.code_id}.json"

    if meta_path.exists():
        old = load_json(meta_path)
        if isinstance(old, dict):
            also = set(old.get("also_seen_in", []))
            also.update(record.also_seen_in)
            also.add(record.src_dir)
            old["also_seen_in"] = sorted(also)

            # refresh/overwrite fields that should stay current
            for k in [
                "group", "A_id", "B_id", "A_elems", "B_elems",
                "k", "d_recorded", "d_recorded_kind", "n",
                "matrices_flat", "collected_dir", "run_dir", "src_dir",
                "collected_files",
            ]:
                old[k] = getattr(record, k)

            old["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            save_json(meta_path, old)
            return

    save_json(meta_path, asdict(record))


def group_to_latex(group: str) -> str:
    parts = group.split("x")
    latex_parts = []
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            latex_parts.append(rf"C_{{{int(p[1:])}}}")
        else:
            latex_parts.append(rf"\mathrm{{{p}}}")
    return r" \times ".join(latex_parts)


def build_latex_block(records: List[CodeRecord]) -> str:
    # best by (group,n,k): max d_recorded
    best: Dict[Tuple[str, Optional[int], int], CodeRecord] = {}
    for r in records:
        key = (r.group, r.n, r.k)
        if key not in best or r.d_recorded > best[key].d_recorded:
            best[key] = r

    grouped: Dict[Tuple[str, Optional[int]], List[CodeRecord]] = {}
    for r in best.values():
        grouped.setdefault((r.group, r.n), []).append(r)

    lines: List[str] = []
    lines.append(r"\section{Auto-collected best results}")
    lines.append(r"This section is generated by \texttt{scripts/collect\_best\_codes.py}.")
    lines.append(r"Here, $d$ is the recorded value stored by the search pipeline (often a QDistRnd upper bound).")
    lines.append("")

    for (g, n), lst in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else 10**18)):
        lst.sort(key=lambda r: r.k)
        g_ltx = group_to_latex(g)
        n_str = str(n) if n is not None else "unknown"
        lines.append(rf"\subsection{{Group $G = {g_ltx}$, length $n={n_str}$}}")
        lines.append("")
        lines.append(r"\begin{longtable}{@{}r r l l@{}}")
        lines.append(r"\toprule")
        lines.append(r"$k$ & recorded $d$ & code id & source run \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        for r in lst:
            run = r.run_dir.replace("_", r"\_")
            cid = r.code_id.replace("_", r"\_")
            lines.append(rf"{r.k} & {r.d_recorded} & \texttt{{{cid}}} & \texttt{{{run}}} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")
    return "\n".join(lines)


def update_tex(tex_path: Path, latex_block: str) -> None:
    """
    Update the AUTO block safely.
    IMPORTANT: we must NOT pass latex_block directly as a regex replacement string
    because backslashes like \\section would be interpreted as regex escapes.
    """
    text = tex_path.read_text(errors="replace") if tex_path.exists() else ""

    if not text.strip():
        text = (
            "\\documentclass[11pt]{article}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage{microtype}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{margin=1in}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\usepackage{hyperref}\n"
            "\\title{Explicit-qT: Search Log for Quantum Tanner Codes}\n"
            "\\author{}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            "\\section{Purpose}\n"
            "Auto-generated log of best codes.\n\n"
            f"{AUTO_BEGIN}\n{AUTO_END}\n\n"
            "\\end{document}\n"
        )

    if AUTO_BEGIN not in text or AUTO_END not in text:
        if "\\end{document}" in text:
            text = text.replace("\\end{document}", f"\n{AUTO_BEGIN}\n{AUTO_END}\n\n\\end{{document}}\n")
        else:
            text += f"\n{AUTO_BEGIN}\n{AUTO_END}\n"

    pattern = re.compile(re.escape(AUTO_BEGIN) + r".*?" + re.escape(AUTO_END), re.DOTALL)
    replacement = f"{AUTO_BEGIN}\n{latex_block}\n{AUTO_END}"

    # Use a function replacement to avoid backslash interpretation
    new_text, nsubs = pattern.subn(lambda _m: replacement, text, count=1)
    if nsubs != 1:
        raise RuntimeError("Could not update TeX AUTO block (unexpected marker structure).")

    tex_path.write_text(new_text)


def write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    out = ["\t".join(header)]
    out.extend(["\t".join(r) for r in rows])
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["results"], help="Relative dirs to scan (default: results)")
    ap.add_argument("--dest", default="best_codes/collected", help="Destination folder under repo root")
    ap.add_argument("--copy-all", action="store_true", help="Copy all files (not only whitelisted extensions)")
    ap.add_argument("--update-tex", default="notes/search_log.tex", help="TeX log file to update")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report, do not copy/write")
    args = ap.parse_args()

    root = repo_root()
    dest_root = root / args.dest
    meta_dir = root / "best_codes" / "meta"
    flat_mtx_dir = root / "best_codes" / "matrices"

    found = list(iter_best_code_dirs(root, args.sources))
    if not found:
        raise SystemExit(f"No best_codes directories found under {args.sources}.")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Dedup by (group, A, B, k, d)
    by_key: Dict[Tuple[str, str, str, int, int], CodeRecord] = {}
    records: List[CodeRecord] = []

    for run_dir, code_dir in found:
        code_id = code_dir.name
        parsed = parse_code_id(code_id)
        if not parsed:
            continue
        group, A_id, B_id, A_elems, B_elems, k, drec = parsed
        n = infer_n_from_code_dir(code_dir)

        run_rel = rel(root, run_dir)
        src_rel = rel(root, code_dir)

        key = (group, A_id, B_id, k, drec)
        if key in by_key:
            by_key[key].also_seen_in.append(src_rel)
            continue

        collected_dir = dest_root / code_id
        rec = CodeRecord(
            code_id=code_id,
            group=group,
            A_id=A_id,
            B_id=B_id,
            A_elems=A_elems,
            B_elems=B_elems,
            k=k,
            d_recorded=drec,
            d_recorded_kind="d_ub",
            n=n,
            run_dir=run_rel,
            src_dir=src_rel,
            collected_dir=rel(root, collected_dir),
            matrices_flat=[],
            collected_files=[],
            collected_at_utc=now,
            also_seen_in=[],
        )
        by_key[key] = rec
        records.append(rec)

    if not records:
        raise SystemExit("Found best_codes folders, but none matched expected CODE_ID naming convention.")

    records.sort(key=lambda r: (r.group, r.n if r.n is not None else 10**18, r.k, -r.d_recorded, r.code_id))

    print(f"Found {len(found)} candidate code dirs; collected {len(records)} unique codes (deduped).")
    for r in records[:30]:
        n_str = str(r.n) if r.n is not None else "?"
        print(f"  {r.code_id}  (G={r.group}, n={n_str}, k={r.k}, d={r.d_recorded})  from {r.src_dir}")
    if len(records) > 30:
        print(f"  ... ({len(records)-30} more)")

    if args.dry_run:
        return

    ensure_dir(dest_root)
    ensure_dir(meta_dir)
    ensure_dir(flat_mtx_dir)

    # Copy artifacts + write meta
    for r in records:
        src = root / r.src_dir
        dst = root / r.collected_dir
        copied = copy_artifacts(src, dst, copy_all=args.copy_all)
        r.collected_files = [rel(root, p) for p in copied]

        flat = copy_matrices_flat(flat_mtx_dir, r.code_id, dst)
        r.matrices_flat = [rel(root, p) for p in flat]

        update_or_create_meta(root, meta_dir, r)

    # Write index TSV
    index_rows: List[List[str]] = []
    for r in records:
        index_rows.append([
            r.code_id,
            r.group,
            str(r.n) if r.n is not None else "",
            str(r.k),
            str(r.d_recorded),
            r.A_id,
            r.B_id,
            r.run_dir,
            r.src_dir,
            r.collected_dir,
        ])

    write_tsv(
        root / "best_codes" / "index.tsv",
        ["code_id", "group", "n", "k", "d_recorded", "A_id", "B_id", "run_dir", "src_dir", "collected_dir"],
        index_rows,
    )

    # best_by_group_k.tsv
    best_gk: Dict[Tuple[str, int], CodeRecord] = {}
    for r in records:
        key = (r.group, r.k)
        if key not in best_gk or r.d_recorded > best_gk[key].d_recorded:
            best_gk[key] = r

    best_rows = []
    for (g, k), r in sorted(best_gk.items(), key=lambda x: (x[0][0], x[0][1], -x[1].d_recorded)):
        best_rows.append([g, str(k), str(r.d_recorded), r.code_id, str(r.n) if r.n is not None else ""])

    write_tsv(
        root / "best_codes" / "best_by_group_k.tsv",
        ["group", "k", "best_d_recorded", "code_id", "n"],
        best_rows,
    )

    # Update TeX
    tex_path = root / args.update_tex
    latex_block = build_latex_block(records)
    try:
        update_tex(tex_path, latex_block)
    except Exception as e:
        print(f"[warn] Failed to update {rel(root, tex_path)}: {e}")

    print("Wrote:")
    print("  best_codes/index.tsv")
    print("  best_codes/best_by_group_k.tsv")
    print("  best_codes/meta/*.json")
    print("  best_codes/collected/*")
    print("  best_codes/matrices/*")
    print(f"  {rel(root, tex_path)}")


if __name__ == "__main__":
    main()
