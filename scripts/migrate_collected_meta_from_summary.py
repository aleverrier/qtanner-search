#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Optional

def read(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text()) if p.exists() else {}

def write(p: Path, d: Dict[str, Any]) -> None:
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")

def ensure_int(x: Any) -> Optional[int]:
    try: return int(x)
    except: return None

def main() -> int:
    ap = argparse.ArgumentParser(description="Copy m4ri-refined distance/trials from best_codes/meta/*.json into best_codes/collected/*/meta.json.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", required=True)
    ap.add_argument("--min-trials", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    meta_dir = best / "meta"
    collected = best / "collected"

    pref = args.group + "_"
    codes = [p.name for p in collected.iterdir() if p.is_dir() and p.name.startswith(pref)]
    codes.sort()

    changed = 0
    skipped = 0

    for cid in codes:
        sm = read(meta_dir / f"{cid}.json")
        if sm.get("distance_backend") != "dist-m4ri":
            skipped += 1
            continue
        steps = ensure_int(sm.get("m4ri_steps")) or ensure_int(sm.get("trials")) or 0
        if steps < args.min_trials:
            skipped += 1
            continue

        cm_path = collected / cid / "meta.json"
        cm = read(cm_path)

        # Preserve old distance block if present
        if "distance" in cm and "distance_prev" not in cm:
            cm["distance_prev"] = cm["distance"]

        # Copy over the canonical refined fields
        for k in ["distance_backend","distance_method","m4ri_steps","m4ri_seed","dX_ub","dZ_ub","d_ub","d","trials","updated_at","distance_m4ri"]:
            if k in sm:
                cm[k] = sm[k]

        # Overwrite the displayed 'distance' block using the refined summary if available
        if "distance" in sm:
            cm["distance"] = sm["distance"]

        cm["code_id"] = cid

        if args.dry_run:
            changed += 1
            continue

        write(cm_path, cm)
        changed += 1

    print(f"[done] group={args.group} min_trials={args.min_trials} updated_collected_meta={changed} skipped={skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
