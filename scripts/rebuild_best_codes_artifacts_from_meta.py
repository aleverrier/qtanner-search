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

def group_of(cid: str) -> str:
    i = cid.find("_")
    return cid[:i] if i >= 0 else cid

def trials_from_meta(m: Dict[str, Any]) -> int:
    # Prefer refined fields; else progressive steps_used_total
    for k in ("m4ri_steps","trials","steps","steps_used","distance_trials","distance_steps"):
        if k in m:
            v = ensure_int(m.get(k))
            if v is not None and v > 0:
                return v
    dist = m.get("distance")
    if isinstance(dist, dict):
        for k in ("steps","trials","steps_used_total","steps_used_x","steps_used_z"):
            if k in dist:
                v = ensure_int(dist.get(k))
                if v is not None and v > 0:
                    return v
    dm = m.get("distance_m4ri")
    if isinstance(dm, dict):
        v = ensure_int(dm.get("steps_per_side"))
        if v is not None and v > 0:
            return v
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default="best_codes")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    collected = best / "collected"
    out_path = best / "data.json"
    min_trials_path = best / "min_trials_by_group.json"

    if not collected.is_dir():
        raise SystemExit("ERROR: best_codes/collected not found")

    min_trials_by_group: Dict[str,int] = {}
    if min_trials_path.exists():
        raw = json.loads(min_trials_path.read_text())
        for k,v in raw.items():
            if k.startswith("_"): 
                continue
            try:
                min_trials_by_group[k] = int(v)
            except:
                pass

    codes: List[Dict[str, Any]] = []
    dropped_low = 0

    for d in sorted(collected.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = read_json(mp)
        cid = m.get("code_id") or d.name
        m["code_id"] = cid

        g = m.get("group", {}).get("spec") if isinstance(m.get("group"), dict) else None
        if not isinstance(g, str):
            g = group_of(cid)

        trials = trials_from_meta(m)
        threshold = int(min_trials_by_group.get(g, 0))

        # **Key rule**: if group has a threshold, drop entries below it.
        if threshold > 0 and trials < threshold:
            dropped_low += 1
            continue

        # Normalize fields used by table
        if trials > 0:
            m["trials"] = trials
            m.setdefault("m4ri_steps", trials)

        dist = m.get("distance")
        if isinstance(dist, dict):
            dub = ensure_int(m.get("d_ub")) or ensure_int(m.get("d")) or ensure_int(dist.get("d_ub")) or ensure_int(dist.get("d"))
            if dub is not None:
                m["d_ub"] = dub
                m.setdefault("d", dub)

        codes.append(m)

    data = {
        "generated_at_utc": now_utc_z(),
        "repo": str(best),
        "dropped_codes_missing_n_or_k": 0,
        "dropped_codes_below_group_min_trials": dropped_low,
        "codes": codes,
    }
    write_json(out_path, data)
    print(f"[ok] wrote {out_path} with {len(codes)} codes; dropped_below_min_trials={dropped_low}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
