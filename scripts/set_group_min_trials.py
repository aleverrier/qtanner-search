#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import argparse
import datetime as dt

def nowz():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

ap = argparse.ArgumentParser()
ap.add_argument("--best-dir", default="best_codes")
ap.add_argument("--group", required=True)
ap.add_argument("--min-trials", type=int, required=True)
args = ap.parse_args()

best = Path(args.best_dir)
p = best / "min_trials_by_group.json"
data = {}
if p.exists():
    data = json.loads(p.read_text())

cur = int(data.get(args.group, 0) or 0)
newv = max(cur, int(args.min_trials))
data[args.group] = newv
data["_updated_at_utc"] = nowz()

p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(f"[ok] {p}: {args.group} min_trials {cur} -> {newv}")
