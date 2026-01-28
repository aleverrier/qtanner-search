#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()

import json
from pathlib import Path
from typing import Any, Dict, Tuple

def extract_distance(meta: Dict[str, Any]) -> Tuple[int|None, int|None, int|None, int|None]:
    # returns (d, trials, dX, dZ)
    # tolerate several schemas
    d = meta.get("d_ub", meta.get("d"))
    trials = meta.get("m4ri_steps", meta.get("trials", meta.get("steps")))
    dX = meta.get("dX_ub", meta.get("dX"))
    dZ = meta.get("dZ_ub", meta.get("dZ"))

    # nested?
    dist = meta.get("distance")
    if isinstance(dist, dict):
        d = d if d is not None else dist.get("d_ub", dist.get("d"))
        trials = trials if trials is not None else dist.get("steps", dist.get("trials"))
        dX = dX if dX is not None else dist.get("dX_ub", dist.get("dX"))
        dZ = dZ if dZ is not None else dist.get("dZ_ub", dist.get("dZ"))

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    return to_int(d), to_int(trials), to_int(dX), to_int(dZ)

def main() -> int:
    best = Path("best_codes")
    meta_dir = best / "meta"
    data_path = best / "data.json"

    if not meta_dir.is_dir():
        raise SystemExit(f"ERROR: {meta_dir} not found")
    if not data_path.exists():
        raise SystemExit(f"ERROR: {data_path} not found")

    # Load meta map
    meta_map: Dict[str, Dict[str, Any]] = {}
    for p in meta_dir.glob("*.json"):
        try:
            meta_map[p.stem] = json.loads(p.read_text())
        except Exception:
            pass

    data = json.loads(data_path.read_text())

    updated = 0

    def walk(obj: Any) -> None:
        nonlocal updated
        if isinstance(obj, dict):
            # If this dict looks like a code record, patch it
            cid = obj.get("code_id") or obj.get("id") or obj.get("name")
            if isinstance(cid, str) and cid in meta_map:
                d, trials, dX, dZ = extract_distance(meta_map[cid])
                if d is not None:
                    obj["d"] = d
                    obj["d_ub"] = d
                if trials is not None:
                    obj["trials"] = trials
                if dX is not None:
                    obj["dX_ub"] = dX
                if dZ is not None:
                    obj["dZ_ub"] = dZ
                # keep a marker for debugging
                if meta_map[cid].get("distance_backend") == "dist-m4ri":
                    obj["distance_backend"] = "dist-m4ri"
                updated += 1

            # recurse
            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)

    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"[ok] patched {updated} dict(s) inside best_codes/data.json using best_codes/meta/*.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
