#!/usr/bin/env python3
"""
Collect all best codes stored under results/**/best_codes/* into a single canonical folder,
and curate metadata for the website.

Source layout example:
  results/<run_name>/best_codes/<CODE_ID>/
    Hx.mtx
    Hz.mtx
    meta.json

CODE_ID example:
  C2xC2xC2_AAp6_0_1_2_3_4_6_BBp6_0_1_2_4_5_7_k16_d16

What this script does:
- Finds all results/**/best_codes/<CODE_ID> directories
- Copies artifacts into best_codes/collected/<CODE_ID>/
- Copies all .mtx into best_codes/matrices/ (flat convenience copies)
- Writes canonical metadata into best_codes/meta/<CODE_ID>.json
  * includes extracted distance-trial information when present in the source meta.json
  * embeds the full source meta.json as "source_meta" so we never lose construction data
- Generates best_codes/index.tsv and best_codes/best_by_group_k.tsv
- Updates notes/search_log.tex between:
    % BEGIN AUTO BEST TABLES
    ...
    % END AUTO BEST TABLES

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

AUTO_BEGIN = "% BEGIN AUTO BEST TABLES"
AUTO_END = "% END AUTO BEST TABLES"

# CODE_ID example:
#   C2xC2xC2_AAp11_0_0_1_2_2_7_BBp11_0_0_1_2_4_5_k4_d20
CODE_ID_RE = re.compile(
    r"^(?P<group>.+?)_AA(?P<A>.+?)_BB(?P<B>.+?)_k(?P<k>\d+)_d(?P<d>\d+)$"
)

WHITELIST_EXTS = {
    ".mtx", ".json", ".txt", ".md", ".tsv", ".csv", ".npz", ".pkl", ".pickle", ".png", ".pdf"
}


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(out)


def rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_multiset_elems(ms_id: str) -> List[int]:
    """
    Parse:
      p6_0_1_2_3_4_6  -> [0,1,2,3,4,6]
      p11_0_0_1_2_2_7 -> [0,0,1,2,2,7]
    """
    toks = ms_id.split("_")
    out: List[int] = []
    for t in toks[1:]:
        if re.fullmatch(r"-?\d+", t):
            out.append(int(t))
    return out


def parse_code_id(code_id: str) -> Optional[Dict[str, Any]]:
    m = CODE_ID_RE.match(code_id)
    if not m:
        return None
    group = m.group("group")
    A_id = m.group("A")
    B_id = m.group("B")
    k = int(m.group("k"))
    d_in_id = int(m.group("d"))
    return {
        "group": group,
        "A_id": A_id,
        "B_id": B_id,
        "A_elems": parse_multiset_elems(A_id),
        "B_elems": parse_multiset_elems(B_id),
        "k": k,
        "d_in_id": d_in_id,
    }


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


def safe_load_json(fp: Path) -> Optional[Dict[str, Any]]:
    try:
        x = json.loads(fp.read_text(errors="replace"))
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def find_source_meta(code_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Prefer code_dir/meta.json. Otherwise try first *meta*.json in code_dir.
    """
    p = code_dir / "meta.json"
    if p.exists():
        return p, safe_load_json(p)
    cands = sorted(code_dir.glob("*meta*.json"))
    for c in cands:
        d = safe_load_json(c)
        if d is not None:
            return c, d
    return None, None


def walk_json(obj: Any, path: List[str]) -> Iterable[Tuple[List[str], Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_path = path + [str(k)]
            yield from walk_json(v, walk_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_path = path + [str(i)]
            yield from walk_json(v, walk_path)


def extract_distance_info(source_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best-effort extraction. We DO NOT assume a fixed schema of the source meta.json.

    Returns a dict with keys:
      method, trials, trials_path, trials_key, seed, candidates
    """
    out: Dict[str, Any] = {
        "method": None,
        "trials": None,
        "trials_path": None,
        "trials_key": None,
        "seed": None,
        "candidates": [],
    }
    if not source_meta:
        return out

    # Method heuristics
    meta_text = json.dumps(source_meta).lower()
    if "qdistrnd" in meta_text or "qdist" in meta_text:
        out["method"] = "QDistRnd"
    elif "randomwalk" in meta_text or "random walk" in meta_text:
        out["method"] = "RandomWalk"

    # Gather integer candidates that look like trial budgets
    def looks_like_trials_key(k: str) -> bool:
        kk = k.lower()
        return any(t in kk for t in ["trial", "trials", "eval", "evals", "step", "steps", "iter", "iters", "attempt"])

    def path_relevance_score(path_str: str) -> int:
        s = path_str.lower()
        score = 0
        # more relevant if it mentions distance estimation machinery
        if any(t in s for t in ["distance", "dist", "qdistrnd", "qdist", "estimator", "estimate"]):
            score -= 6
        if any(t in s for t in ["trial", "eval", "step", "iter", "attempt"]):
            score -= 3
        return score

    candidates: List[Dict[str, Any]] = []
    for path, val in walk_json(source_meta, []):
        if not path:
            continue
        key = path[-1]
        if isinstance(val, int) and looks_like_trials_key(key):
            pstr = "/".join(path)
            candidates.append({
                "path": pstr,
                "key": key,
                "value": val,
                "score": path_relevance_score(pstr),
            })

    # Also try to find a seed (common key name)
    seed_candidates: List[Dict[str, Any]] = []
    for path, val in walk_json(source_meta, []):
        if not path:
            continue
        key = path[-1].lower()
        if isinstance(val, int) and "seed" in key:
            pstr = "/".join(path)
            seed_candidates.append({
                "path": pstr,
                "key": path[-1],
                "value": val,
                "score": path_relevance_score(pstr),
            })

    # Choose best trials candidate
    if candidates:
        candidates.sort(key=lambda c: (c["score"], -c["value"]))
        best = candidates[0]
        out["trials"] = int(best["value"])
        out["trials_path"] = best["path"]
        out["trials_key"] = best["key"]
        out["candidates"] = candidates[:20]  # keep some for debugging

    # Choose best seed candidate
    if seed_candidates:
        seed_candidates.sort(key=lambda c: (c["score"], c["path"]))
        out["seed"] = int(seed_candidates[0]["value"])

    return out


def extract_permutation_candidates(source_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find arrays that look like permutations: list[int] under keys containing 'perm'.
    We do not assume any fixed key names.
    """
    if not source_meta:
        return []
    perms: List[Dict[str, Any]] = []

    for path, val in walk_json(source_meta, []):
        if not path:
            continue
        key = path[-1].lower()
        if "perm" not in key:
            continue
        if isinstance(val, list) and val and all(isinstance(x, int) for x in val):
            perms.append({
                "path": "/".join(path),
                "length": len(val),
                "value_preview": val[:20],
            })

    return perms[:50]


def read_mtx_shape(fp: Path) -> Optional[Tuple[int, int, int]]:
    """
    Parse Matrix Market header for (nrows,ncols,nnz).
    """
    try:
        with fp.open("r", errors="replace") as f:
            header = f.readline()
            if not header.lower().startswith("%%matrixmarket"):
                return None
            line = f.readline()
            while line and line.strip().startswith("%"):
                line = f.readline()
            if not line:
                return None
            parts = line.strip().split()
            if len(parts) < 3:
                return None
            return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None


def infer_n_from_code_dir(code_dir: Path) -> Optional[int]:
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


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def group_to_latex(group: str) -> str:
    parts = group.split("x")
    latex_parts = []
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            latex_parts.append(rf"C_{{{int(p[1:])}}}")
        else:
            latex_parts.append(rf"\mathrm{{{p}}}")
    return r" \times ".join(latex_parts)


def build_latex_block(records: List["CodeRecord"]) -> str:
    # best by (group,n,k): max d_recorded
    best: Dict[Tuple[str, Optional[int], int], "CodeRecord"] = {}
    for r in records:
        key = (r.group, r.n, r.k)
        if key not in best or r.d_recorded > best[key].d_recorded:
            best[key] = r

    grouped: Dict[Tuple[str, Optional[int]], List["CodeRecord"]] = {}
    for r in best.values():
        grouped.setdefault((r.group, r.n), []).append(r)

    lines: List[str] = []
    lines.append(r"\section{Auto-collected best results}")
    lines.append(r"This section is generated by \texttt{scripts/collect\_best\_codes.py}.")
    lines.append(r"$d$ is the recorded value saved by the search pipeline (often a QDistRnd upper bound).")
    lines.append("")

    for (g, n), lst in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else 10**18)):
        lst.sort(key=lambda r: r.k)
        g_ltx = group_to_latex(g)
        n_str = str(n) if n is not None else "unknown"
        lines.append(rf"\subsection{{Group $G = {g_ltx}$, length $n={n_str}$}}")
        lines.append("")
        lines.append(r"\begin{longtable}{@{}r r r l l@{}}")
        lines.append(r"\toprule")
        lines.append(r"$k$ & recorded $d$ & trials & code id & source run \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        for r in lst:
            run = r.run_dir.replace("_", r"\_")
            cid = r.code_id.replace("_", r"\_")
            trials = str(r.distance_trials) if r.distance_trials is not None else ""
            lines.append(rf"{r.k} & {r.d_recorded} & {trials} & \texttt{{{cid}}} & \texttt{{{run}}} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")
    return "\n".join(lines)


def update_tex(tex_path: Path, latex_block: str) -> None:
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
    new_text, nsubs = pattern.subn(lambda _m: replacement, text, count=1)
    if nsubs != 1:
        raise RuntimeError("Could not update TeX AUTO block (unexpected marker structure).")
    tex_path.write_text(new_text)


def write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    out = ["\t".join(header)]
    out.extend(["\t".join(r) for r in rows])
    path.write_text("\n".join(out) + "\n")


@dataclass
class CodeRecord:
    schema: str
    code_id: str

    group: str
    A_id: str
    B_id: str
    A_elems: List[int]
    B_elems: List[int]

    n: Optional[int]
    k: int

    d_recorded: int
    d_recorded_kind: str
    d_in_id: int

    distance_method: Optional[str]
    distance_trials: Optional[int]
    distance_trials_path: Optional[str]
    distance_seed: Optional[int]

    permutation_candidates: List[Dict[str, Any]]

    source_meta_path_in_src: Optional[str]
    source_meta_path_in_collected: Optional[str]
    source_meta: Optional[Dict[str, Any]]

    run_dir: str
    src_dir: str
    collected_dir: str
    settings_path: str

    matrices_flat: List[str]
    collected_files: List[str]

    collected_at_utc: str
    also_seen_in: List[str]


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

    # Dedup key
    by_key: Dict[Tuple[str, str, str, int, int], CodeRecord] = {}
    records: List[CodeRecord] = []

    for run_dir, code_dir in found:
        code_id = code_dir.name
        parsed = parse_code_id(code_id)
        if not parsed:
            continue

        group = parsed["group"]
        A_id = parsed["A_id"]
        B_id = parsed["B_id"]
        A_elems = parsed["A_elems"]
        B_elems = parsed["B_elems"]
        k = parsed["k"]
        d_in_id = parsed["d_in_id"]

        # Load source meta.json if present
        src_meta_path, src_meta = find_source_meta(code_dir)
        dist = extract_distance_info(src_meta)
        perms = extract_permutation_candidates(src_meta)

        # Infer n from matrices; prefer meta if it explicitly has "n" (common)
        n = None
        if isinstance(src_meta, dict):
            if isinstance(src_meta.get("n"), int):
                n = int(src_meta["n"])
            elif isinstance(src_meta.get("N"), int):
                n = int(src_meta["N"])
            elif isinstance(src_meta.get("quantum"), dict) and isinstance(src_meta["quantum"].get("n"), int):
                n = int(src_meta["quantum"]["n"])
        if n is None:
            n = infer_n_from_code_dir(code_dir)

        # Prefer d from meta if it is present; otherwise folder name
        d_recorded = d_in_id
        d_kind = "from_code_id"
        if isinstance(src_meta, dict):
            if isinstance(src_meta.get("d_ub"), int):
                d_recorded = int(src_meta["d_ub"])
                d_kind = "d_ub"
            elif isinstance(src_meta.get("d"), int):
                d_recorded = int(src_meta["d"])
                d_kind = "d"

        run_rel = rel(root, run_dir)
        src_rel = rel(root, code_dir)

        key = (group, A_id, B_id, k, d_in_id)
        if key in by_key:
            # Merge provenance + fill missing fields
            rec = by_key[key]
            rec.also_seen_in.append(src_rel)
            if rec.distance_trials is None and dist.get("trials") is not None:
                rec.distance_trials = dist["trials"]
                rec.distance_trials_path = dist.get("trials_path")
            if rec.distance_method is None and dist.get("method") is not None:
                rec.distance_method = dist["method"]
            continue

        collected_dir = dest_root / code_id
        settings_path = collected_dir / "settings.json"

        rec = CodeRecord(
            schema="explicitqt_best_code_meta_v1",
            code_id=code_id,
            group=group,
            A_id=A_id,
            B_id=B_id,
            A_elems=A_elems,
            B_elems=B_elems,
            n=n,
            k=k,
            d_recorded=d_recorded,
            d_recorded_kind=d_kind,
            d_in_id=d_in_id,
            distance_method=dist.get("method"),
            distance_trials=dist.get("trials"),
            distance_trials_path=dist.get("trials_path"),
            distance_seed=dist.get("seed"),
            permutation_candidates=perms,
            source_meta_path_in_src=rel(root, src_meta_path) if src_meta_path else None,
            source_meta_path_in_collected=str(Path(args.dest) / code_id / (src_meta_path.name if src_meta_path else "meta.json")),
            source_meta=src_meta,
            run_dir=run_rel,
            src_dir=src_rel,
            collected_dir=rel(root, collected_dir),
            settings_path=rel(root, settings_path),
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
        t_str = str(r.distance_trials) if r.distance_trials is not None else "?"
        print(f"  {r.code_id}  (G={r.group}, n={n_str}, k={r.k}, d={r.d_recorded}, trials={t_str})  from {r.src_dir}")
    if len(records) > 30:
        print(f"  ... ({len(records)-30} more)")

    if args.dry_run:
        return

    ensure_dir(dest_root)
    ensure_dir(meta_dir)
    ensure_dir(flat_mtx_dir)

    # Copy + write curated meta + settings.json
    for r in records:
        src = root / r.src_dir
        dst = root / r.collected_dir

        copied = copy_artifacts(src, dst, copy_all=args.copy_all)
        r.collected_files = [rel(root, p) for p in copied]

        flat = copy_matrices_flat(flat_mtx_dir, r.code_id, dst)
        r.matrices_flat = [rel(root, p) for p in flat]

        # Write settings.json (small stable summary for humans + scripts)
        settings_abs = root / r.settings_path
        ensure_dir(settings_abs.parent)
        settings = {
            "schema": "explicitqt_best_code_settings_v1",
            "code_id": r.code_id,
            "group": r.group,
            "n": r.n,
            "k": r.k,
            "d_recorded": r.d_recorded,
            "d_recorded_kind": r.d_recorded_kind,
            "distance_estimation": {
                "method": r.distance_method,
                "trials": r.distance_trials,
                "seed": r.distance_seed,
                "trials_extracted_from": r.distance_trials_path,
            },
            "A_id": r.A_id,
            "B_id": r.B_id,
            "A_elems": r.A_elems,
            "B_elems": r.B_elems,
            "permutation_candidates": r.permutation_candidates,
            "matrices_flat": r.matrices_flat,
            "source_meta_path_in_collected": r.source_meta_path_in_collected,
            "provenance": {
                "run_dir": r.run_dir,
                "src_dir": r.src_dir,
                "also_seen_in": r.also_seen_in,
                "collected_at_utc": r.collected_at_utc,
            },
        }
        save_json(settings_abs, settings)
        if rel(root, settings_abs) not in r.collected_files:
            r.collected_files.append(rel(root, settings_abs))

        # Write canonical meta in best_codes/meta/
        save_json(meta_dir / f"{r.code_id}.json", asdict(r))

    # index.tsv
    index_rows: List[List[str]] = []
    for r in records:
        index_rows.append([
            r.code_id,
            r.group,
            str(r.n) if r.n is not None else "",
            str(r.k),
            str(r.d_recorded),
            str(r.distance_trials) if r.distance_trials is not None else "",
            r.distance_method or "",
            r.A_id,
            r.B_id,
            r.run_dir,
            r.src_dir,
            r.collected_dir,
        ])
    write_tsv(
        root / "best_codes" / "index.tsv",
        ["code_id", "group", "n", "k", "d_recorded", "distance_trials", "distance_method", "A_id", "B_id", "run_dir", "src_dir", "collected_dir"],
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
        best_rows.append([
            g,
            str(k),
            str(r.d_recorded),
            str(r.distance_trials) if r.distance_trials is not None else "",
            r.code_id,
            str(r.n) if r.n is not None else "",
        ])
    write_tsv(
        root / "best_codes" / "best_by_group_k.tsv",
        ["group", "k", "best_d_recorded", "distance_trials", "code_id", "n"],
        best_rows,
    )

    # TeX
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
