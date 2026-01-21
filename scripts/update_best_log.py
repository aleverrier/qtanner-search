#!/usr/bin/env python3
"""
Update notes/search_log.tex + best_codes artifacts from existing run logs.

What it does (best-effort, robust to missing info):
- Scans results/ and runs/ for text files containing the "best table" header:
    k | best d_ub | when_found(eval) | A_id | B_id
- Extracts rows and tries to infer (group, n) from nearby context in the same file.
- Aggregates best entries per (group, n, k) using maximum d_ub (tie-break by earlier timestamp).
- Writes a LaTeX section between markers in notes/search_log.tex:
    % BEGIN AUTO BEST TABLES
    ...
    % END AUTO BEST TABLES
- For each best code, writes metadata JSON to best_codes/meta/<CODE_TAG>.json
- Tries to find and copy matching .mtx matrices into best_codes/matrices/

Usage:
  python scripts/update_best_log.py --dry-run
  python scripts/update_best_log.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


AUTO_BEGIN = "% BEGIN AUTO BEST TABLES"
AUTO_END = "% END AUTO BEST TABLES"


@dataclass(frozen=True)
class BestRow:
    group: str                 # e.g. "C2xC2xC2"
    n: Optional[int]           # code length if inferred
    k: int
    d_ub: int
    when_found_raw: str        # e.g. "20260121T101724Z(2745)"
    timestamp_utc: Optional[str]  # ISO-ish string if parsed
    eval_when_found: Optional[int]
    A_id: str
    B_id: str
    source_file: str           # where we parsed this from
    source_group_context: str  # raw context line(s) used to infer group/n if any
    trials_in_run: Optional[int]  # best-effort; usually unknown


def repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        return Path(out)
    except Exception as e:
        raise SystemExit(f"ERROR: not inside a git repo (or git missing): {e}")


def iter_candidate_text_files(root: Path, rel_dirs: List[str]) -> Iterable[Path]:
    exts = {".txt", ".log", ".out", ".tsv", ".csv", ".md", ".tex"}
    for d in rel_dirs:
        p = root / d
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                yield fp


def safe_read_text(fp: Path, max_bytes: int = 30_000_000) -> Optional[str]:
    try:
        if fp.stat().st_size > max_bytes:
            return None
        return fp.read_text(errors="replace")
    except Exception:
        return None


TABLE_HEADER_RE = re.compile(
    r"^\s*k\s*\|\s*best\s*d_ub\s*\|\s*when_found\(eval\)\s*\|\s*A_id\s*\|\s*B_id\s*$",
    re.IGNORECASE | re.MULTILINE,
)

ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9TZ\-\:\.]+(?:\(\d+\))?)\s*\|\s*([A-Za-z0-9_\-\.]+)\s*\|\s*([A-Za-z0-9_\-\.]+)\s*$",
    re.MULTILINE,
)

# Context hints like:
#   [pilot] C2xC2xC2 n=288 d0=17 ...
# or:
#   C2xC2xC2 n=288
GROUP_N_RE = re.compile(r"\b([A-Za-z0-9x]+)\b.*\bn\s*=\s*(\d+)\b")
TRIALS_RE = re.compile(r"\b(total\s+)?(evals|trials)\s*[:=]\s*(\d+)\b", re.IGNORECASE)


def parse_when_found(s: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse strings like:
      20260121T101724Z(2745)
      20260121T101724Z
      2026-01-21T10:17:24Z(2745)
    """
    eval_m = re.search(r"\((\d+)\)\s*$", s)
    eval_i = int(eval_m.group(1)) if eval_m else None
    ts_part = re.sub(r"\(\d+\)\s*$", "", s).strip()

    # normalize a couple of formats
    ts_part = ts_part.replace(" ", "")
    try:
        if re.fullmatch(r"\d{8}T\d{6}Z", ts_part):
            dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), eval_i
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts_part):
            # already ISO-ish
            return ts_part, eval_i
    except Exception:
        pass
    return None, eval_i


def infer_group_n_from_context(text: str, header_pos: int) -> Tuple[str, Optional[int], str, Optional[int]]:
    """
    Look backwards a bit from the header to find group + n hints.
    Returns (group, n, context_line, trials_in_run)
    """
    window = text[max(0, header_pos - 3000):header_pos]
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
    lines_rev = list(reversed(lines))

    group = "UNKNOWN"
    n = None
    ctx_line = ""
    for ln in lines_rev[:80]:
        m = GROUP_N_RE.search(ln)
        if m:
            group = m.group(1)
            n = int(m.group(2))
            ctx_line = ln
            break

    trials = None
    for ln in lines_rev[:120]:
        tm = TRIALS_RE.search(ln)
        if tm:
            trials = int(tm.group(3))
            break

    return group, n, ctx_line, trials


def parse_best_tables_from_text(text: str, source_file: str) -> List[BestRow]:
    rows: List[BestRow] = []
    for hm in TABLE_HEADER_RE.finditer(text):
        group, n, ctx_line, trials = infer_group_n_from_context(text, hm.start())

        # Parse rows after this header until we hit an empty line or another header
        tail = text[hm.end():]
        # Look only in a limited tail window to avoid accidental later tables
        tail_window = tail[:6000]

        for rm in ROW_RE.finditer(tail_window):
            k = int(rm.group(1))
            d_ub = int(rm.group(2))
            when_raw = rm.group(3)
            A_id = rm.group(4)
            B_id = rm.group(5)
            ts_iso, eval_i = parse_when_found(when_raw)
            rows.append(
                BestRow(
                    group=group,
                    n=n,
                    k=k,
                    d_ub=d_ub,
                    when_found_raw=when_raw,
                    timestamp_utc=ts_iso,
                    eval_when_found=eval_i,
                    A_id=A_id,
                    B_id=B_id,
                    source_file=source_file,
                    source_group_context=ctx_line,
                    trials_in_run=trials,
                )
            )
        # continue scanning; there might be multiple headers in one file
    return rows


def group_order(group: str) -> Optional[int]:
    """
    Very simple parser for strings like:
      C2xC2xC2
      C2x4
      C2xC4xC4
    Returns product, or None if unknown.
    """
    if group == "UNKNOWN":
        return None
    parts = group.split("x")
    prod = 1
    ok = False
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            prod *= int(p[1:])
            ok = True
        elif p.isdigit():
            prod *= int(p)
            ok = True
        else:
            return None
    return prod if ok else None


def group_to_latex(group: str) -> str:
    if group == "UNKNOWN":
        return r"\mathrm{UNKNOWN}"
    parts = group.split("x")
    latex_parts = []
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            latex_parts.append(rf"C_{{{int(p[1:])}}}")
        elif p.isdigit():
            latex_parts.append(rf"C_{{{int(p)}}}")
        else:
            latex_parts.append(rf"\mathrm{{{p}}}")
    return r" \times ".join(latex_parts)


def sanitize_tag(s: str, max_len: int = 180) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_\-\.=]+", "_", s)
    s2 = re.sub(r"_+", "_", s2).strip("_")
    if len(s2) <= max_len:
        return s2
    # If too long, keep the front and last part
    return s2[:120] + "__" + s2[-50:]


def code_tag(row: BestRow) -> str:
    ts = row.timestamp_utc.replace(":", "").replace("-", "") if row.timestamp_utc else "unknown_ts"
    n_str = str(row.n) if row.n is not None else "unknownN"
    base = f"G={row.group}_n={n_str}_k={row.k}_dub={row.d_ub}_{ts}_A={row.A_id}_B={row.B_id}"
    return sanitize_tag(base)


def find_mtx_candidates(root: Path, row: BestRow) -> List[Path]:
    """
    Best-effort: find .mtx files that mention A_id or B_id or both in name/path.
    Also catch HX/HZ naming.
    """
    candidates: List[Path] = []
    search_dirs = [root / "results", root / "runs"]
    for d in search_dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*.mtx"):
            s = str(fp)
            if row.A_id in s or row.B_id in s:
                candidates.append(fp)
    return candidates


def classify_mtx(fp: Path) -> str:
    s = fp.name.lower()
    if "hx" in s or "h_x" in s:
        return "HX"
    if "hz" in s or "h_z" in s:
        return "HZ"
    if "classical" in s:
        return "CLASSICAL"
    if "ha" in s and "hb" in s:
        return "MIXED"
    return "MTX"


def choose_best_rows(rows: List[BestRow]) -> List[BestRow]:
    """
    Aggregate by (group, n, k): pick max d_ub, tiebreak by earlier timestamp (if parseable),
    then by smaller eval_when_found.
    """
    def key(row: BestRow) -> Tuple[str, Optional[int], int]:
        return (row.group, row.n, row.k)

    best: Dict[Tuple[str, Optional[int], int], BestRow] = {}

    def ts_sort_val(r: BestRow) -> Tuple[int, str]:
        # earlier timestamp = better tie-break (more reproducible to reference first find)
        if r.timestamp_utc is None:
            return (1, "9999")
        return (0, r.timestamp_utc)

    for r in rows:
        k = key(r)
        if k not in best:
            best[k] = r
            continue
        cur = best[k]
        if r.d_ub > cur.d_ub:
            best[k] = r
        elif r.d_ub == cur.d_ub:
            if ts_sort_val(r) < ts_sort_val(cur):
                best[k] = r
            else:
                # if timestamps tie/unknown, prefer smaller eval_when_found if available
                if (r.eval_when_found is not None) and (cur.eval_when_found is not None):
                    if r.eval_when_found < cur.eval_when_found:
                        best[k] = r
                elif r.eval_when_found is not None and cur.eval_when_found is None:
                    best[k] = r

    out = list(best.values())
    # Sort by group, n, k
    out.sort(key=lambda r: (r.group, r.n if r.n is not None else 10**18, r.k))
    return out


def build_latex_block(best_rows: List[BestRow], root: Path) -> str:
    # group by (group,n)
    grouped: Dict[Tuple[str, Optional[int]], List[BestRow]] = {}
    for r in best_rows:
        grouped.setdefault((r.group, r.n), []).append(r)

    lines: List[str] = []
    lines.append(r"\section{Auto-updated best results}")
    lines.append(r"This section is generated by \texttt{scripts/update\_best\_log.py}.")
    lines.append("")

    for (g, n), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else 10**18)):
        g_ltx = group_to_latex(g)
        ord_g = group_order(g)
        n_str = str(n) if n is not None else "unknown"
        ord_str = f", order {ord_g}" if ord_g is not None else ""
        lines.append(rf"\subsection{{Group $G = {g_ltx}$ ({ord_str[2:] if ord_str.startswith(', ') else ord_str}), length $n={n_str}$}}")
        lines.append("")
        lines.append(r"\begin{longtable}{@{}r r l r l l@{}}")
        lines.append(r"\toprule")
        lines.append(r"$k$ & best $d_{\mathrm{ub}}$ & when\_found & eval & $A$ id & $B$ id \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")

        for r in rows:
            when = r.timestamp_utc if r.timestamp_utc is not None else r.when_found_raw
            ev = str(r.eval_when_found) if r.eval_when_found is not None else ""
            lines.append(rf"{r.k} & {r.d_ub} & {when} & {ev} & \texttt{{{r.A_id}}} & \texttt{{{r.B_id}}} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")

        lines.append(r"\paragraph{Artifacts in \texttt{best\_codes/}}")
        lines.append(r"\begin{itemize}")
        for r in rows:
            tag = code_tag(r)
            meta = f"best_codes/meta/{tag}.json"
            lines.append(rf"\item \texttt{{{tag}}}: \texttt{{{meta}}}")
        lines.append(r"\end{itemize}")
        lines.append("")

    return "\n".join(lines) + "\n"


def update_tex_file(tex_path: Path, new_block: str) -> None:
    if not tex_path.exists():
        raise SystemExit(f"ERROR: TeX file does not exist: {tex_path}")

    text = tex_path.read_text(errors="replace")
    if AUTO_BEGIN not in text or AUTO_END not in text:
        # Append markers at end before \end{document} if possible
        if r"\end{document}" in text:
            text = text.replace(
                r"\end{document}",
                f"\n{AUTO_BEGIN}\n{AUTO_END}\n\n\\end{{document}}\n"
            )
        else:
            text = text + f"\n{AUTO_BEGIN}\n{AUTO_END}\n"

    pattern = re.compile(re.escape(AUTO_BEGIN) + r".*?" + re.escape(AUTO_END), re.DOTALL)
    replacement = f"{AUTO_BEGIN}\n{new_block}{AUTO_END}"
    new_text, nsubs = pattern.subn(replacement, text, count=1)
    if nsubs != 1:
        raise SystemExit("ERROR: failed to update AUTO block (unexpected marker structure).")

    tex_path.write_text(new_text)


def write_meta_and_copy_mtx(root: Path, row: BestRow, do_copy: bool = True) -> Dict[str, List[str]]:
    """
    Write best_codes/meta/<tag>.json and copy mtx files (best-effort).
    Returns a dict of what we copied.
    """
    tag = code_tag(row)
    meta_dir = root / "best_codes" / "meta"
    mtx_dir = root / "best_codes" / "matrices"
    meta_dir.mkdir(parents=True, exist_ok=True)
    mtx_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / f"{tag}.json"
    meta = asdict(row)
    meta["code_tag"] = tag
    meta["generated_by"] = "scripts/update_best_log.py"
    meta["generated_at_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    copied: Dict[str, List[str]] = {"meta": [str(meta_path.relative_to(root))], "matrices": []}

    if not do_copy:
        return copied

    # Try to find matrices in results/runs that mention A_id/B_id
    cands = find_mtx_candidates(root, row)
    # prefer HX/HZ if present
    # sort by classification priority
    prio = {"HX": 0, "HZ": 1, "CLASSICAL": 2, "MTX": 3, "MIXED": 4}
    cands.sort(key=lambda fp: (prio.get(classify_mtx(fp), 9), len(str(fp))))

    # Copy up to a reasonable number to avoid dumping huge sets
    max_copy = 20
    for fp in cands[:max_copy]:
        kind = classify_mtx(fp)
        dest_name = f"{tag}_{kind}_{fp.name}"
        dest = mtx_dir / dest_name
        try:
            shutil.copy2(fp, dest)
            copied["matrices"].append(str(dest.relative_to(root)))
        except Exception:
            continue

    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dirs", nargs="+", default=["results", "runs"],
                    help="Relative dirs to scan under repo root (default: results runs)")
    ap.add_argument("--tex", default="notes/search_log.tex", help="TeX logbook path")
    ap.add_argument("--dry-run", action="store_true", help="Parse and print summary only")
    ap.add_argument("--write", action="store_true", help="Update TeX + write meta + copy matrices")
    args = ap.parse_args()

    root = repo_root()
    tex_path = root / args.tex

    all_rows: List[BestRow] = []
    for fp in iter_candidate_text_files(root, args.source_dirs):
        text = safe_read_text(fp)
        if text is None:
            continue
        rows = parse_best_tables_from_text(text, str(fp.relative_to(root)))
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No best tables found in results/ or runs/. Nothing to do.")

    best = choose_best_rows(all_rows)

    # Print a concise summary
    print(f"Found {len(all_rows)} rows across logs; keeping {len(best)} best rows.")
    for r in best:
        n_str = r.n if r.n is not None else "?"
        print(f"  {r.group} n={n_str} k={r.k}: d_ub={r.d_ub}  A={r.A_id}  B={r.B_id}  ({r.when_found_raw})")

    if args.dry_run and not args.write:
        return

    if not args.write:
        raise SystemExit("Refusing to modify files: pass --write to update TeX and best_codes artifacts.")

    # Build latex and update file
    new_block = build_latex_block(best, root)
    update_tex_file(tex_path, new_block)
    print(f"Updated {tex_path.relative_to(root)}")

    # Write meta and copy matrices
    copied_all: List[Dict[str, List[str]]] = []
    for r in best:
        copied = write_meta_and_copy_mtx(root, r, do_copy=True)
        copied_all.append(copied)

    print("Wrote metadata and attempted to copy matrices into best_codes/.")
    print("Done.")


if __name__ == "__main__":
    main()
