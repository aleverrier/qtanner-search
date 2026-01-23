#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

def now_utc_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_json(p: Path) -> Any:
    return json.loads(p.read_text())

def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def ensure_int(x: Any) -> Optional[int]:
    try: return int(x)
    except: return None

def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild best_codes/data.json from best_codes/collected/*/meta.json.")
    ap.add_argument("--best-dir", default="best_codes")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    collected = best / "collected"
    data_path = best / "data.json"
    if not collected.is_dir():
        raise SystemExit("ERROR: best_codes/collected not found.")

    codes: List[Dict[str, Any]] = []
    for d in sorted(collected.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = read_json(mp)

        # Ensure top-level display fields exist
        cid = m.get("code_id") or d.name
        m["code_id"] = cid

        # Canonical trials: prefer m4ri_steps else trials else distance.steps_used_total
        trials = ensure_int(m.get("m4ri_steps")) or ensure_int(m.get("trials"))
        dist = m.get("distance")
        if trials is None and isinstance(dist, dict):
            trials = ensure_int(dist.get("steps")) or ensure_int(dist.get("trials")) or ensure_int(dist.get("steps_used_total"))
        if trials is not None:
            m["trials"] = trials
            m["m4ri_steps"] = m.get("m4ri_steps", trials)

        # Canonical d_ub/d
        d_ub = ensure_int(m.get("d_ub")) or ensure_int(m.get("d"))
        if d_ub is None and isinstance(dist, dict):
            d_ub = ensure_int(dist.get("d_ub")) or ensure_int(dist.get("d"))
        if d_ub is not None:
            m["d_ub"] = d_ub
            m["d"] = m.get("d", d_ub)

        codes.append(m)

    data = {
        "repo": str(best),
        "generated_at_utc": now_utc_z(),
        "dropped_codes_missing_n_or_k": 0,
        "codes": codes,
    }
    write_json(data_path, data)
    print(f"[ok] wrote {data_path} with {len(codes)} codes")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
