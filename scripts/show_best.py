#!/usr/bin/env python3
"""Show top saved and evaluated candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.leaderboard import load_best_by_nk


def _safe_float(value: float) -> str:
    return f"{value:.3f}"


def _extract_d_ub(qd_stats: Dict[str, Any]) -> Optional[int]:
    if not qd_stats:
        return None
    if "dX_ub" in qd_stats and "dZ_ub" in qd_stats:
        return min(int(qd_stats["dX_ub"]), int(qd_stats["dZ_ub"]))
    if "d_ub" in qd_stats:
        return int(qd_stats["d_ub"])
    return None


def _passes_early_filter(qd_stats: Dict[str, Any]) -> bool:
    return not (
        qd_stats.get("terminated_early_X") and qd_stats.get("terminated_early_Z")
    )


def _not_rejected(qd_stats: Dict[str, Any]) -> bool:
    if not isinstance(qd_stats, dict):
        return False
    return (
        qd_stats.get("terminated_early_X") is False
        and qd_stats.get("terminated_early_Z") is False
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skipping invalid JSONL at line {line_num}", file=sys.stderr)
    return records


def _load_promising_meta(promising_dir: Path) -> List[Dict[str, Any]]:
    if not promising_dir.exists():
        return []
    records = []
    for meta_path in promising_dir.rglob("meta.json"):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["saved_path"] = str(meta_path.parent)
            records.append(meta)
        except (OSError, json.JSONDecodeError):
            print(f"[warn] skipping unreadable {meta_path}", file=sys.stderr)
    return records


def _candidate_from_result(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qd_stats = record.get("qdistrnd")
    if not isinstance(qd_stats, dict):
        return None
    n = record.get("n")
    k = record.get("k")
    if n is None or k is None:
        return None
    d_ub = _extract_d_ub(qd_stats)
    if d_ub is None:
        return None
    return {
        "n": int(n),
        "k": int(k),
        "d_ub": int(d_ub),
        "kd_over_n": (int(k) * int(d_ub)) / int(n),
        "d_over_sqrt_n": int(d_ub) / math.sqrt(int(n)),
        "group": record.get("group", {}),
        "a1v": record.get("a1v"),
        "b1v": record.get("b1v"),
        "saved_path": record.get("saved_path"),
        "qdistrnd": qd_stats,
    }


def _candidate_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qd_stats = meta.get("qdistrnd")
    if not isinstance(qd_stats, dict):
        return None
    n = meta.get("n")
    k = meta.get("k")
    if n is None or k is None:
        return None
    d_ub = _extract_d_ub(qd_stats)
    if d_ub is None:
        return None
    local_codes = meta.get("local_codes", {})
    return {
        "n": int(n),
        "k": int(k),
        "d_ub": int(d_ub),
        "kd_over_n": (int(k) * int(d_ub)) / int(n),
        "d_over_sqrt_n": int(d_ub) / math.sqrt(int(n)),
        "group": meta.get("group", {}),
        "a1v": local_codes.get("a1v"),
        "b1v": local_codes.get("b1v"),
        "saved_path": meta.get("saved_path"),
        "qdistrnd": qd_stats,
    }


def _format_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row.get(h, "")))
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-" * len(header_line)
    lines = [header_line, sep_line]
    if not rows:
        lines.append("(none)")
        return "\n".join(lines)
    for row in rows:
        lines.append("  ".join(row.get(h, "").ljust(widths[h]) for h in headers))
    return "\n".join(lines)


def _render_rows(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for entry in entries:
        group = entry.get("group", {})
        order = group.get("order", "?")
        gid = group.get("gid", "?")
        rows.append(
            {
                "n": str(entry.get("n", "?")),
                "k": str(entry.get("k", "?")),
                "d_ub": str(entry.get("d_ub", "?")),
                "kd/n": _safe_float(entry.get("kd_over_n", 0.0)),
                "d/sqrt(n)": _safe_float(entry.get("d_over_sqrt_n", 0.0)),
                "group": f"{order},{gid}",
                "a1v": str(entry.get("a1v", "?")),
                "b1v": str(entry.get("b1v", "?")),
                "saved_path": str(entry.get("saved_path") or ""),
            }
        )
    return rows


def _top_entries(entries: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    return sorted(
        entries, key=lambda e: (e["kd_over_n"], e["d_over_sqrt_n"]), reverse=True
    )[:top_n]


def _record_key(record: Dict[str, Any]) -> tuple:
    group = record.get("group", {})
    order = group.get("order")
    gid = group.get("gid")
    A = record.get("A")
    B = record.get("B")
    return (
        order,
        gid,
        tuple(A) if isinstance(A, list) else A,
        tuple(B) if isinstance(B, list) else B,
        record.get("a1v"),
        record.get("b1v"),
    )


def _render_best_by_nk_rows(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for entry in entries:
        group = entry.get("group", {})
        order = group.get("order", "?")
        gid = group.get("gid", "?")
        n = int(entry.get("n", 0) or 0)
        k = int(entry.get("k", 0) or 0)
        d_obs = int(entry.get("d_obs", 0) or 0)
        kd_over_n = (k * d_obs) / n if n else 0.0
        d_over_sqrt_n = d_obs / math.sqrt(n) if n else 0.0
        rows.append(
            {
                "n": str(n or "?"),
                "k": str(k or "?"),
                "d_obs": str(d_obs or "?"),
                "kd/n": _safe_float(kd_over_n),
                "d/sqrt(n)": _safe_float(d_over_sqrt_n),
                "trials": str(entry.get("trials_used", "?")),
                "group": f"{order},{gid}",
                "a1v": str(entry.get("a1v", "?")),
                "b1v": str(entry.get("b1v", "?")),
                "timestamp": str(entry.get("timestamp", "")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Show best saved and evaluated codes.")
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "search_w9_smallgroups.jsonl",
    )
    parser.add_argument(
        "--promising-dir",
        type=Path,
        default=REPO_ROOT / "data" / "promising",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--only-saved", action="store_true")
    parser.add_argument("--only-passed", action="store_true")
    args = parser.parse_args()

    best_by_nk_path = REPO_ROOT / "results" / "best_by_nk.json"
    best_by_nk = load_best_by_nk(str(best_by_nk_path))
    best_entries = list(best_by_nk.values())
    best_entries.sort(
        key=lambda e: (
            (int(e.get("d_obs", 0) or 0) / math.sqrt(int(e.get("n", 1) or 1))),
            ((int(e.get("k", 0) or 0) * int(e.get("d_obs", 0) or 0)) / int(e.get("n", 1) or 1)),
        ),
        reverse=True,
    )
    best_entries = best_entries[: args.top]
    best_headers = [
        "n",
        "k",
        "d_obs",
        "kd/n",
        "d/sqrt(n)",
        "trials",
        "group",
        "a1v",
        "b1v",
        "timestamp",
    ]
    print(f"Best by (n,k) so far (top {args.top}):")
    print(_format_table(_render_best_by_nk_rows(best_entries), best_headers))
    print("")

    promising_meta = _load_promising_meta(args.promising_dir)
    saved_entries = []
    for meta in promising_meta:
        entry = _candidate_from_meta(meta)
        if entry is None:
            continue
        if args.only_passed and not _passes_early_filter(entry["qdistrnd"]):
            continue
        saved_entries.append(entry)

    saved_top = _top_entries(saved_entries, args.top)
    headers = ["n", "k", "d_ub", "kd/n", "d/sqrt(n)", "group", "a1v", "b1v", "saved_path"]
    print(f"Saved promising codes (top {args.top}):")
    print(_format_table(_render_rows(saved_top), headers))

    if args.only_saved:
        return

    records = _load_jsonl(args.results)
    candidate_map: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        entry = _candidate_from_result(record)
        if entry is None:
            continue
        if not _not_rejected(entry["qdistrnd"]):
            continue
        if args.only_passed and not _passes_early_filter(entry["qdistrnd"]):
            continue
        key = _record_key(record)
        prev = candidate_map.get(key)
        if prev is None:
            candidate_map[key] = entry
        else:
            prev_score = (prev["kd_over_n"], prev["d_over_sqrt_n"])
            next_score = (entry["kd_over_n"], entry["d_over_sqrt_n"])
            if next_score > prev_score:
                candidate_map[key] = entry
    candidates_top = _top_entries(list(candidate_map.values()), args.top)

    print("")
    print(f"Best evaluated candidates so far (top {args.top}):")
    print(_format_table(_render_rows(candidates_top), headers))


if __name__ == "__main__":
    main()
