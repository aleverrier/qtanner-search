#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Entry:
    code_id: str
    group: str
    n: int
    k: int
    d_recorded: int
    distance_trials: int
    distance_method: str


Key = Tuple[str, int, int]  # (group, n, k)


def _git_show(path: str, rev: str) -> str:
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{path}"], text=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[ERROR] Cannot read {path} at rev {rev}. Did you `git fetch`? ({e})")


def _parse_index_tsv(tsv_text: str) -> List[Entry]:
    r = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    if not r.fieldnames:
        raise SystemExit("[ERROR] index.tsv has no header")

    need = {"code_id", "group", "n", "k", "d_recorded", "distance_trials", "distance_method"}
    missing = need.difference(set(r.fieldnames))
    if missing:
        raise SystemExit(f"[ERROR] index.tsv missing columns: {sorted(missing)}. Have: {r.fieldnames}")

    out: List[Entry] = []
    for row in r:
        try:
            code_id = row["code_id"].strip()
            group = row["group"].strip()
            n = int(row["n"])
            k = int(row["k"])
            d = int(row["d_recorded"])
            trials = int(row["distance_trials"]) if row["distance_trials"].strip() else 0
            method = row["distance_method"].strip()
        except Exception:
            # Skip malformed rows
            continue
        out.append(Entry(code_id, group, n, k, d, trials, method))
    return out


def _best_by_key(entries: Iterable[Entry], prefer: str) -> Dict[Key, Entry]:
    best: Dict[Key, Entry] = {}
    for e in entries:
        key: Key = (e.group, e.n, e.k)
        cur = best.get(key)
        if cur is None:
            best[key] = e
            continue

        if prefer == "higher":
            # Best = larger d_recorded, tie-break larger trials
            if (e.d_recorded, e.distance_trials) > (cur.d_recorded, cur.distance_trials):
                best[key] = e
        else:
            # Best = smaller d_recorded, tie-break larger trials
            if (e.d_recorded, -e.distance_trials) < (cur.d_recorded, -cur.distance_trials):
                best[key] = e
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare best_codes/index.tsv between two git revisions.")
    ap.add_argument("--base", default="origin/main", help="Base git revision (default: origin/main)")
    ap.add_argument("--new", default="HEAD", help="New git revision (default: HEAD)")
    ap.add_argument("--prefer", choices=["higher", "lower"], default="higher",
                    help="Whether higher or lower d_recorded is considered better when selecting best-per-(group,n,k).")
    ap.add_argument("--show", type=int, default=12, help="How many examples to show per category.")
    args = ap.parse_args()

    base_txt = _git_show("best_codes/index.tsv", args.base)
    new_txt = _git_show("best_codes/index.tsv", args.new)

    base_entries = _parse_index_tsv(base_txt)
    new_entries = _parse_index_tsv(new_txt)

    base_best = _best_by_key(base_entries, args.prefer)
    new_best = _best_by_key(new_entries, args.prefer)

    ok_improved: List[Tuple[Key, Entry, Entry]] = []
    ok_more_trials: List[Tuple[Key, Entry, Entry]] = []
    needs_trials: List[Tuple[Key, Entry, Entry]] = []
    bad_lower_d: List[Tuple[Key, Entry, Entry]] = []
    bad_lower_trials: List[Tuple[Key, Entry, Entry]] = []
    missing_keys: List[Tuple[Key, Entry]] = []
    new_keys: List[Tuple[Key, Entry]] = []

    for key, old in base_best.items():
        new = new_best.get(key)
        if new is None:
            missing_keys.append((key, old))
            continue

        # Comparison depends on prefer direction.
        if args.prefer == "higher":
            if new.d_recorded > old.d_recorded and new.distance_trials >= old.distance_trials:
                ok_improved.append((key, old, new))
            elif new.d_recorded == old.d_recorded and new.distance_trials > old.distance_trials:
                ok_more_trials.append((key, old, new))
            elif new.d_recorded > old.d_recorded and new.distance_trials < old.distance_trials:
                needs_trials.append((key, old, new))
            elif new.d_recorded < old.d_recorded:
                bad_lower_d.append((key, old, new))
            elif new.distance_trials < old.distance_trials:
                bad_lower_trials.append((key, old, new))
        else:
            # prefer lower: "improved" means d decreased (tighter)
            if new.d_recorded < old.d_recorded and new.distance_trials >= old.distance_trials:
                ok_improved.append((key, old, new))
            elif new.d_recorded == old.d_recorded and new.distance_trials > old.distance_trials:
                ok_more_trials.append((key, old, new))
            elif new.d_recorded < old.d_recorded and new.distance_trials < old.distance_trials:
                needs_trials.append((key, old, new))
            elif new.d_recorded > old.d_recorded:
                bad_lower_d.append((key, old, new))
            elif new.distance_trials < old.distance_trials:
                bad_lower_trials.append((key, old, new))

    for key, e in new_best.items():
        if key not in base_best:
            new_keys.append((key, e))

    def show_triplets(title: str, arr: List[Tuple[Key, Entry, Entry]]) -> None:
        if not arr:
            return
        print(f"\n{title} (showing up to {args.show})")
        for (g, n, k), o, nn in arr[: args.show]:
            print(
                f"  ({g!r}, {n}, {k})  "
                f"{o.d_recorded}@{o.distance_trials} ({o.distance_method}) -> "
                f"{nn.d_recorded}@{nn.distance_trials} ({nn.distance_method})  "
                f"new={nn.code_id}"
            )

    def show_missing(title: str, arr: List[Tuple[Key, Entry]]) -> None:
        if not arr:
            return
        print(f"\n{title} (showing up to {args.show})")
        for (g, n, k), o in arr[: args.show]:
            print(f"  ({g!r}, {n}, {k})  was {o.d_recorded}@{o.distance_trials}  ({o.code_id})  -> MISSING")

    def show_new(title: str, arr: List[Tuple[Key, Entry]]) -> None:
        if not arr:
            return
        print(f"\n{title} (showing up to {args.show})")
        for (g, n, k), e in arr[: args.show]:
            print(f"  ({g!r}, {n}, {k})  new {e.d_recorded}@{e.distance_trials} ({e.distance_method})  ({e.code_id})")

    print(f"[summary] prefer={args.prefer}  base={args.base}  new={args.new}")
    print(f"  base keys: {len(base_best)}")
    print(f"  new  keys: {len(new_best)}")
    print(f"  OK improved: {len(ok_improved)}")
    print(f"  OK more trials (same d): {len(ok_more_trials)}")
    print(f"  NEEDS trials: {len(needs_trials)}")
    print(f"  BAD lower d: {len(bad_lower_d)}")
    print(f"  BAD lower trials: {len(bad_lower_trials)}")
    print(f"  MISSING keys: {len(missing_keys)}")
    print(f"  NEW keys: {len(new_keys)}")

    show_triplets("NEEDS trials", needs_trials)
    show_triplets("BAD lower d", bad_lower_d)
    show_triplets("BAD lower trials", bad_lower_trials)
    show_missing("MISSING keys", missing_keys)
    show_new("NEW keys", new_keys)

    # Exit non-zero if there are dangerous regressions.
    if missing_keys or bad_lower_d or bad_lower_trials:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
