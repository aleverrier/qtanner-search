#!/usr/bin/env python3
"""
Sync the GitHub Pages dataset so ONLY the most reliable distance estimate is shown per [[n,k]].

Your rule (enforced):
  For a fixed [[n,k]], keep ONLY the entry with the largest number of "distance trials".
  (Tie-break: larger distance, then keep first.)

Important: many of your JSON datasets do NOT store trials in a top-level "trials" column.
They often store it inside nested structures (e.g. vote_stats / vote_histograms),
or under keys like quantum_refine_chunk/refine_chunk.
This script can extract the trial count from nested JSON automatically.

By default, rows whose text contains "archived" or "below_min_trials" are ignored,
so archived results won't reappear on the website.

Run (from repo root):
  python3 scripts/update_website.py

If auto-detection ever picks the wrong dataset file, force it:
  python3 scripts/update_website.py --data-file docs/some_file.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# We avoid crawling huge directories when trying to auto-detect the website dataset file.
EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "best_codes",  # IMPORTANT: huge; the website should NOT directly use these meta JSONs
}

DEFAULT_HIDE_SUBSTRINGS = ("archived", "below_min_trials")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())


def _parse_intish(x: Any) -> Optional[int]:
    """
    Parse integers written as:
      50000, "50_000", "50 000", "50k", "3M", "≤22", ">= 24", etc.
    Returns None if cannot parse.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None

    s = s.replace(",", "").replace("_", "").replace(" ", "")
    m = re.match(r"^[<>≤≥=]*?(\d+(?:\.\d+)?)([kKmM])?.*$", s)
    if not m:
        return None
    num = float(m.group(1))
    suf = m.group(2)
    mult = 1
    if suf in ("k", "K"):
        mult = 1000
    elif suf in ("m", "M"):
        mult = 1_000_000
    return int(round(num * mult))


@dataclass(frozen=True)
class Dataset:
    path: Path
    kind: str  # "tsv" | "csv" | "json"
    headers: List[str]
    rows: List[Dict[str, Any]]
    delimiter: Optional[str] = None  # for csv/tsv
    # JSON container preservation: if original JSON was {"rows":[...], ...}, keep outer dict
    json_outer: Optional[Dict[str, Any]] = None
    json_rows_key: Optional[str] = None


def _iter_files(root: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            for p in d.iterdir():
                if p.is_dir():
                    if p.name in EXCLUDE_DIR_NAMES:
                        continue
                    if p.name.startswith(".") and p.name not in {".github"}:
                        continue
                    stack.append(p)
                else:
                    if p.suffix.lower() in suffixes:
                        yield p
        except PermissionError:
            continue


def _detect_pages_root(repo: Path) -> Path:
    docs = repo / "docs"
    if (docs / "index.html").exists() or (docs / "index.md").exists():
        return docs
    if (repo / "index.html").exists() or (repo / "index.md").exists():
        return repo
    for p in _iter_files(repo, suffixes=[".html"]):
        if p.name == "index.html":
            rel = p.relative_to(repo)
            if len(rel.parts) <= 2:
                return p.parent
    return docs if docs.exists() else repo


def _extract_data_refs_from_html(index_html: Path) -> List[str]:
    """
    Find local references to *.tsv/*.csv/*.json in index.html.
    """
    text = index_html.read_text(encoding="utf-8", errors="ignore")
    refs = re.findall(r"""['"]([^'"]+\.(?:tsv|csv|json)(?:\?[^'"]*)?)['"]""", text, flags=re.I)
    out: List[str] = []
    for r in refs:
        if re.match(r"^[a-zA-Z]+://", r):
            continue
        r = r.split("?", 1)[0].split("#", 1)[0]
        r = r.lstrip("/")
        if r:
            out.append(r)
    seen = set()
    uniq = []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _read_delimited(path: Path, delimiter: str) -> Dataset:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError(f"{path}: empty file")

    start_idx = 0
    while start_idx < len(lines) and lines[start_idx].lstrip().startswith("#"):
        start_idx += 1
    if start_idx >= len(lines):
        raise ValueError(f"{path}: only comments, no header")

    data_str = "\n".join(lines[start_idx:]) + "\n"
    f = io.StringIO(data_str)
    reader = csv.DictReader(f, delimiter=delimiter)
    headers = reader.fieldnames or []
    rows = list(reader)

    kind = "tsv" if delimiter == "\t" else "csv"
    return Dataset(path=path, kind=kind, headers=headers, rows=rows, delimiter=delimiter)


def _read_json(path: Path) -> Dataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    json_outer: Optional[Dict[str, Any]] = None
    rows_key: Optional[str] = None

    if isinstance(data, dict):
        # common patterns: {"rows":[...]} or {"data":[...]} (we support rows/data)
        for key in ("rows", "data"):
            if key in data and isinstance(data[key], list):
                rows_key = key
                json_outer = dict(data)
                rows = data[key]
                break
        else:
            raise ValueError(f"{path}: JSON dict must contain a list under 'rows' or 'data'")
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"{path}: JSON must be a list of objects, or a dict with a 'rows'/'data' list")

    if not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: JSON rows must be objects (dicts)")

    headers: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)

    return Dataset(
        path=path,
        kind="json",
        headers=headers,
        rows=rows,
        delimiter=None,
        json_outer=json_outer,
        json_rows_key=rows_key,
    )


def _read_dataset_any(path: Path) -> Dataset:
    suf = path.suffix.lower()
    if suf == ".tsv":
        return _read_delimited(path, "\t")
    if suf == ".csv":
        return _read_delimited(path, ",")
    if suf == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported dataset type: {path}")


def _score_headers(headers: List[str]) -> int:
    hn = [_norm(h) for h in headers]

    def has_any(cands: Sequence[str]) -> bool:
        return any(c in hn for c in cands)

    score = 0
    if has_any(["n", "length", "nqubits", "numqubits", "blocklength"]):
        score += 3
    if has_any(["k", "dimension", "klogical", "nlogical"]):
        score += 3
    if has_any(["d", "distance", "dist", "dmin"]):
        score += 2
    if has_any(["trials", "trial", "distancetrials", "ntrials", "numtrials", "nbtrials"]):
        score += 4
    # Some repos call this refine_chunk or quantum_refine_chunk
    if has_any(["refinechunk", "quantumrefinechunk", "refine"]):
        score += 1
    return score


def _guess_cols(headers: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Return (n_col, k_col, d_col, trials_col, params_col).
    trials_col may be None; for JSON we can extract trials from nested data.
    """
    hn = {h: _norm(h) for h in headers}

    def pick_exact(cands: Sequence[str]) -> Optional[str]:
        for h, nh in hn.items():
            if nh in cands:
                return h
        return None

    def pick_contains(substrs: Sequence[str]) -> Optional[str]:
        for h, nh in hn.items():
            for sub in substrs:
                if sub in nh:
                    return h
        return None

    n_col = pick_exact(["n", "length", "nqubits", "numqubits", "blocklength"]) or pick_contains(["nqubit", "numqubit"])
    k_col = pick_exact(["k", "dimension", "klogical", "nlogical"]) or pick_contains(["klogic", "dimension"])
    d_col = pick_exact(["d", "distance", "dist", "dmin", "mindistance"]) or pick_contains(["distance", "dmin"])

    trials_col = (
        pick_exact(["trials", "trial", "distancetrials", "ntrials", "numtrials", "nbtrials", "distance_trials"])
        or pick_contains(["trial", "trials"])
        or pick_contains(["refinechunk", "quantumrefinechunk", "refine_chunk", "refinechunk"])
    )

    params_col = pick_contains(["params", "codeparams", "parameters", "code"])
    return n_col, k_col, d_col, trials_col, params_col


PARAMS_RE = re.compile(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*([<>≤≥=]*\s*\d+)")


def _row_get_nkd(
    row: Dict[str, Any],
    n_col: Optional[str],
    k_col: Optional[str],
    d_col: Optional[str],
    params_col: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    n = _parse_intish(row.get(n_col)) if n_col else None
    k = _parse_intish(row.get(k_col)) if k_col else None
    d = _parse_intish(row.get(d_col)) if d_col else None

    if (n is None or k is None or d is None) and params_col and row.get(params_col):
        m = PARAMS_RE.search(str(row.get(params_col)))
        if m:
            n = n if n is not None else _parse_intish(m.group(1))
            k = k if k is not None else _parse_intish(m.group(2))
            d = d if d is not None else _parse_intish(m.group(3))

    return n, k, d


def _row_is_hidden(row: Dict[str, Any], hide_substrings: Sequence[str]) -> bool:
    blob = " ".join(str(v) for v in row.values()).lower()
    return any(s in blob for s in hide_substrings)


def _trials_from_vote_histograms(row: Dict[str, Any]) -> Optional[int]:
    """
    If vote_histograms is a dict of counts, sum(counts) ~= number of trials.
    """
    def tot(obj: Any) -> Optional[int]:
        if isinstance(obj, dict):
            vals = []
            for v in obj.values():
                pv = _parse_intish(v)
                if pv is None:
                    return None
                vals.append(pv)
            return sum(vals) if vals else None
        if isinstance(obj, list):
            subs = []
            for it in obj:
                t = tot(it)
                if t is not None:
                    subs.append(t)
            return sum(subs) if subs else None
        return None

    if "vote_histograms" in row:
        t = tot(row.get("vote_histograms"))
        if t is not None and 0 < t <= 100_000_000:
            return t
    return None


def _extract_trials_from_row(row: Dict[str, Any]) -> Optional[int]:
    """
    Extract trials count from nested JSON. We look for keys containing 'trial'/'trials'
    (or refine_chunk-like keys), and also fall back to vote_histograms totals.
    """
    candidates: List[Tuple[int, int]] = []  # (score, value)

    good_exact = {
        "trials",
        "trial",
        "ntrials",
        "numtrials",
        "nbtrials",
        "distance_trials",
        "distancetrials",
        "qdist_trials",
        "qdistrnd_trials",
        "qdistrndtrials",
        "random_trials",
        "total_trials",
        "totaltrials",
    }

    def add_candidate(key: str, val: Any) -> None:
        v = _parse_intish(val)
        if v is None:
            return
        # sanity: avoid grabbing random seeds/IDs
        if v <= 0 or v > 100_000_000:
            return

        kn = _norm(key)
        score = 0
        if kn in good_exact:
            score += 30
        if "distance" in kn:
            score += 15
        if "qdist" in kn or "quantum" in kn:
            score += 6
        if "trial" in kn:
            score += 3
        if "refine" in kn and ("chunk" in kn or "trial" in kn):
            score += 1
        candidates.append((score, v))

    def rec(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                kn = ks.lower()
                if "trial" in kn or "trials" in kn:
                    add_candidate(ks, v)
                elif "refine" in kn and ("chunk" in kn or "trial" in kn or "trials" in kn):
                    add_candidate(ks, v)
                rec(v)
        elif isinstance(obj, list):
            for v in obj:
                rec(v)

    rec(row)

    # vote_histograms fallback (sum of counts)
    vh = _trials_from_vote_histograms(row)
    if vh is not None:
        candidates.append((5, vh))

    if not candidates:
        return None
    return max(candidates, key=lambda x: (x[0], x[1]))[1]


def _trial_signal(ds: Dataset, trials_col: Optional[str]) -> bool:
    """
    Used only for auto-detection scoring: do we see trials information?
    """
    if trials_col:
        for r in ds.rows[:50]:
            if _parse_intish(r.get(trials_col)) is not None:
                return True
    if ds.kind == "json":
        for r in ds.rows[:50]:
            if _extract_trials_from_row(r) is not None:
                return True
    return False


def _detect_dataset(repo: Path, pages_root: Path, forced_path: Optional[str] = None) -> Dataset:
    if forced_path:
        p = (repo / forced_path).resolve() if not Path(forced_path).is_absolute() else Path(forced_path)
        if not p.exists():
            raise FileNotFoundError(f"--data-file points to missing file: {p}")
        return _read_dataset_any(p)

    # If index.html exists, try any referenced dataset files, in order.
    index_html = pages_root / "index.html"
    if index_html.exists():
        refs = _extract_data_refs_from_html(index_html)
        existing = []
        for r in refs:
            p = pages_root / r
            if p.exists():
                existing.append(p)
            else:
                p2 = pages_root / Path(r).name
                if p2.exists():
                    existing.append(p2)

        for p in existing:
            try:
                ds = _read_dataset_any(p)
                n_col, k_col, d_col, trials_col, params_col = _guess_cols(ds.headers)
                if n_col and k_col and _trial_signal(ds, trials_col):
                    return ds
            except Exception:
                continue

        for p in existing:
            try:
                ds = _read_dataset_any(p)
                n_col, k_col, d_col, trials_col, params_col = _guess_cols(ds.headers)
                if n_col and k_col and d_col:
                    return ds
            except Exception:
                continue

    candidates = list(_iter_files(pages_root, suffixes=[".tsv", ".csv", ".json"]))

    scored: List[Tuple[int, int, Path]] = []
    for p in candidates:
        try:
            ds = _read_dataset_any(p)
            score = _score_headers(ds.headers)
            n_col, k_col, d_col, trials_col, params_col = _guess_cols(ds.headers)
            if n_col and k_col:
                if _trial_signal(ds, trials_col):
                    score += 10
            else:
                score -= 5
            scored.append((score, len(ds.rows), p))
        except Exception:
            continue

    if scored:
        scored.sort(reverse=True)
        return _read_dataset_any(scored[0][2])

    raise FileNotFoundError(
        "Could not auto-detect a website dataset file (tsv/csv/json). "
        "Run again with --data-file PATH (relative to repo root)."
    )


def _write_dataset(ds: Dataset, rows: List[Dict[str, Any]], dry_run: bool) -> bool:
    """
    Return True if content changed.
    """
    if ds.kind in ("tsv", "csv"):
        delimiter = ds.delimiter or ("\t" if ds.kind == "tsv" else ",")
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=ds.headers, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in ds.headers})
        new_text = out.getvalue()
    else:
        if ds.json_outer is not None and ds.json_rows_key is not None:
            outer = dict(ds.json_outer)
            outer[ds.json_rows_key] = rows
            new_text = json.dumps(outer, indent=2, ensure_ascii=False) + "\n"
        else:
            new_text = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"

    old_text = ds.path.read_text(encoding="utf-8", errors="ignore")
    changed = (old_text != new_text)
    if changed and not dry_run:
        ds.path.write_text(new_text, encoding="utf-8")
    return changed


def _round_up(x: int, step: int) -> int:
    if step <= 0:
        return x
    return ((x + step - 1) // step) * step


def _make_suggestions_report(
    grouped: Dict[Tuple[int, int], List[Tuple[int, int, Dict[str, Any]]]],
    best: Dict[Tuple[int, int], Tuple[int, int, Dict[str, Any]]],
    suggest_step: int,
) -> str:
    lines: List[str] = []
    lines.append("# Suggestions: under-tested higher-distance candidates\n\n")
    lines.append(
        "Rule used for the website: for each **[[n,k]]**, we display the entry with the **largest number of distance trials**.\n\n"
        "This report lists cases where another entry has a **larger distance** but **fewer trials** than the displayed one.\n"
        "Those are good candidates to re-run with more trials.\n\n"
    )

    any_sug = False
    for nk in sorted(best.keys()):
        best_trials, best_d, best_row = best[nk]
        candidates = []
        for t, d, row in grouped.get(nk, []):
            if t < best_trials and d > best_d:
                candidates.append((t, d, row))
        if not candidates:
            continue

        any_sug = True
        n, k = nk
        lines.append(f"## [[{n},{k}]] displayed: d={best_d}, trials={best_trials}\n\n")

        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        for t, d, row in candidates:
            suggested = _round_up(best_trials + 1, suggest_step)
            ident = None
            for key in ("code_id", "id", "name", "filename", "file", "path", "folder"):
                if key in row and str(row[key]).strip():
                    ident = str(row[key]).strip()
                    break
            if ident:
                lines.append(f"- Candidate d={d}, trials={t} (id: `{ident}`). Suggested: rerun to ≥ {suggested} trials.\n")
            else:
                lines.append(f"- Candidate d={d}, trials={t}. Suggested: rerun to ≥ {suggested} trials.\n")
        lines.append("\n")

    if not any_sug:
        lines.append("No under-tested higher-distance candidates found.\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default=None, help="Path to the dataset file used by Pages (relative to repo root)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files, just report what would change")
    ap.add_argument(
        "--hide-substrings",
        default=",".join(DEFAULT_HIDE_SUBSTRINGS),
        help="Comma-separated substrings; rows containing any are ignored (default: archived,below_min_trials)",
    )
    ap.add_argument(
        "--min-trials",
        type=int,
        default=0,
        help="Ignore rows with trials < MIN (default: 0, i.e. keep everything).",
    )
    ap.add_argument(
        "--suggest-step",
        type=int,
        default=50_000,
        help="Rounding step for suggested rerun trials in the report (default: 50000).",
    )
    ap.add_argument(
        "--debug-trials",
        action="store_true",
        help="Print how trials were extracted for a few sample rows.",
    )
    args = ap.parse_args()

    repo = repo_root()
    pages_root = _detect_pages_root(repo)

    ds = _detect_dataset(repo, pages_root, forced_path=args.data_file)

    hide_substrings = [s.strip().lower() for s in args.hide_substrings.split(",") if s.strip()]

    n_col, k_col, d_col, trials_col, params_col = _guess_cols(ds.headers)

    if n_col is None or k_col is None:
        raise SystemExit(
            f"Could not identify n/k columns in {ds.path}.\n"
            f"Headers: {ds.headers}\n"
            f"Re-run with --data-file PATH if this isn't the right dataset."
        )

    if ds.kind in ("tsv", "csv") and trials_col is None:
        raise SystemExit(
            f"Could not identify a trials-like column in {ds.path} (tsv/csv).\n"
            f"Headers: {ds.headers}\n"
            "I need a trials column to enforce 'keep max trials per [[n,k]]'."
        )

    parsed_rows: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
    dropped_hidden = 0
    dropped_min_trials = 0
    dropped_unparseable = 0
    missing_trials = 0
    debug_shown = 0

    for row in ds.rows:
        if hide_substrings and _row_is_hidden(row, hide_substrings):
            dropped_hidden += 1
            continue

        n, k, d = _row_get_nkd(row, n_col, k_col, d_col, params_col)
        if n is None or k is None:
            dropped_unparseable += 1
            continue
        if d is None:
            d = -1

        trials: Optional[int] = None
        if trials_col is not None:
            trials = _parse_intish(row.get(trials_col))

        if trials is None:
            trials = _extract_trials_from_row(row) if ds.kind == "json" else None

        if trials is None:
            missing_trials += 1
            trials = 0

        if trials < args.min_trials:
            dropped_min_trials += 1
            continue

        if args.debug_trials and debug_shown < 5:
            debug_shown += 1
            print(f"[debug] n={n} k={k} d={d} trials={trials} code_id={row.get('code_id','')}", file=sys.stderr)

        parsed_rows.append((n, k, trials, d, row))

    if not parsed_rows:
        raise SystemExit(
            f"After filtering, no rows remained from {ds.path}. "
            "Try --min-trials 0 and/or --hide-substrings ''"
        )

    grouped: Dict[Tuple[int, int], List[Tuple[int, int, Dict[str, Any]]]] = {}
    best: Dict[Tuple[int, int], Tuple[int, int, Dict[str, Any]]] = {}

    for n, k, trials, d, row in parsed_rows:
        key = (n, k)
        grouped.setdefault(key, []).append((trials, d, row))

        if key not in best:
            best[key] = (trials, d, row)
        else:
            bt, bd, _ = best[key]
            if (trials, d) > (bt, bd):
                best[key] = (trials, d, row)

    out_rows = []
    for (n, k) in sorted(best.keys()):
        out_rows.append(best[(n, k)][2])

    reports_dir = repo / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sug_path = reports_dir / "trials_to_increase.md"
    sug_text = _make_suggestions_report(grouped, best, suggest_step=args.suggest_step)
    if not args.dry_run:
        sug_path.write_text(sug_text, encoding="utf-8")

    changed = _write_dataset(ds, out_rows, dry_run=args.dry_run)

    before = len(ds.rows)
    after = len(out_rows)
    print("Pages root:", pages_root)
    print("Website dataset:", ds.path)
    print("Rows: before =", before, "after =", after)
    if dropped_hidden:
        print("Ignored (archived/below_min_trials filter):", dropped_hidden)
    if dropped_min_trials:
        print("Ignored (min trials filter):", dropped_min_trials)
    if dropped_unparseable:
        print("Ignored (unparseable rows):", dropped_unparseable)
    if missing_trials:
        print("Rows missing trials info (treated as trials=0):", missing_trials)
    print("Suggestions report:", sug_path)

    if args.dry_run:
        print("[dry-run] No files were written.")
    else:
        if changed:
            print("✅ Dataset updated (website will only show max-trials entry per [[n,k]]).")
        else:
            print("✅ No change needed (already clean).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
