#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None

def pick_first_int(*vals: Any) -> Optional[int]:
    for v in vals:
        w = to_int(v)
        if w is not None:
            return w
    return None

def extract_code_id_from_record(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("code_id", "id", "name"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    meta = rec.get("meta")
    if isinstance(meta, dict):
        for k in ("code_id", "id", "name"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                return v
    return None

def extract_group_spec(meta: Dict[str, Any]) -> Optional[str]:
    g = meta.get("group")
    if isinstance(g, dict) and isinstance(g.get("spec"), str) and g.get("spec"):
        return g["spec"]

    for side in ("A", "B"):
        s = meta.get(side)
        if isinstance(s, dict):
            gg = s.get("group")
            if isinstance(gg, dict) and isinstance(gg.get("spec"), str) and gg.get("spec"):
                return gg["spec"]

    cid = meta.get("code_id")
    if isinstance(cid, str) and "_" in cid:
        return cid.split("_", 1)[0]
    return None

def extract_n_k_from_meta(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    n = pick_first_int(meta.get("n"), meta.get("N"))
    k = pick_first_int(meta.get("k"), meta.get("K"))
    params = meta.get("params")
    if isinstance(params, dict):
        n = n if n is not None else pick_first_int(params.get("n"), params.get("N"))
        k = k if k is not None else pick_first_int(params.get("k"), params.get("K"))
    return n, k

def extract_distance_fields(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str], int]:
    dist = meta.get("distance")
    method = None
    d_ub = None
    dX_ub = None
    dZ_ub = None

    trials_total: Optional[int] = None
    steps_x: Optional[int] = None
    steps_z: Optional[int] = None

    trials_total = pick_first_int(meta.get("steps_used_total"), meta.get("m4ri_trials_total"), meta.get("m4ri_trials"))
    if trials_total is None:
        per_side = pick_first_int(meta.get("m4ri_steps"), meta.get("trials"))
        if per_side is not None:
            trials_total = 2 * per_side

    if isinstance(dist, dict):
        method = dist.get("method") if isinstance(dist.get("method"), str) else None

        d_ub = pick_first_int(dist.get("d_ub"))
        dX_ub = pick_first_int(dist.get("dX_ub"), dist.get("dX_best"))
        dZ_ub = pick_first_int(dist.get("dZ_ub"), dist.get("dZ_best"))

        trials_total = trials_total if trials_total is not None else pick_first_int(
            dist.get("steps_used_total"),
            dist.get("steps_total"),
            dist.get("steps"),
            dist.get("trials"),
            dist.get("steps_fast"),
            dist.get("steps_slow"),
        )

        steps_x = pick_first_int(dist.get("steps_used_x"), dist.get("steps_x"))
        steps_z = pick_first_int(dist.get("steps_used_z"), dist.get("steps_z"))

        fast = dist.get("fast")
        if isinstance(fast, dict):
            dx = fast.get("dx")
            dz = fast.get("dz")
            if isinstance(dx, dict):
                dX_ub = dX_ub if dX_ub is not None else pick_first_int(dx.get("d_ub"), dx.get("signed"))
                sx = pick_first_int(dx.get("steps"), dx.get("trials"))
                steps_x = steps_x if steps_x is not None else sx
            if isinstance(dz, dict):
                dZ_ub = dZ_ub if dZ_ub is not None else pick_first_int(dz.get("d_ub"), dz.get("signed"))
                sz = pick_first_int(dz.get("steps"), dz.get("trials"))
                steps_z = steps_z if steps_z is not None else sz

        if trials_total is None and steps_x is not None and steps_z is not None:
            trials_total = steps_x + steps_z

    if d_ub is None:
        cid = meta.get("code_id")
        if isinstance(cid, str):
            m = re.search(r"_d(\d+)$", cid)
            if m:
                d_ub = int(m.group(1))

    trials_total_int = int(trials_total) if trials_total is not None else 0
    return dX_ub, dZ_ub, d_ub, method, trials_total_int

def load_meta(best_dir: Path, code_id: str) -> Optional[Dict[str, Any]]:
    p1 = best_dir / "meta" / f"{code_id}.json"
    if p1.exists():
        try:
            return json.loads(p1.read_text())
        except Exception:
            return None
    p2 = best_dir / "collected" / code_id / "meta.json"
    if p2.exists():
        try:
            return json.loads(p2.read_text())
        except Exception:
            return None
    return None

def main() -> int:
    ap = argparse.ArgumentParser(description="Fill summary fields in best_codes/data.json from best_codes/meta/*.json.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--drop-missing-meta", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir)
    data_path = best_dir / "data.json"
    if not data_path.exists():
        raise SystemExit(f"ERROR: missing {data_path}")

    data = json.loads(data_path.read_text())
    codes = data.get("codes")

    if isinstance(codes, dict):
        items = [(cid, rec) for cid, rec in codes.items() if isinstance(rec, dict)]
        out_list = None
        out_dict = dict(codes)
    elif isinstance(codes, list):
        items = []
        for rec in codes:
            if not isinstance(rec, dict):
                continue
            cid = extract_code_id_from_record(rec)
            if isinstance(cid, str) and cid:
                items.append((cid, rec))
        out_dict = None
        out_list = list(codes)
    else:
        raise SystemExit("ERROR: data.json['codes'] is neither a list nor a dict.")

    changed = 0
    dropped = 0
    min_trials_by_group: Dict[str, int] = {}

    def update_min(group: Optional[str], trials: int):
        if not group:
            return
        prev = min_trials_by_group.get(group)
        if prev is None or trials < prev:
            min_trials_by_group[group] = trials

    for cid, rec in items:
        meta = load_meta(best_dir, cid)
        if meta is None:
            if args.drop_missing_meta:
                if out_dict is not None and cid in out_dict:
                    del out_dict[cid]
                else:
                    rec["_DROP_"] = True
                dropped += 1
            if args.verbose:
                print(f"[warn] missing meta for {cid}")
            continue

        group = extract_group_spec(meta)
        n, k = extract_n_k_from_meta(meta)
        dX_ub, dZ_ub, d_ub, method, trials_total = extract_distance_fields(meta)

        before = (
            rec.get("group"), rec.get("n"), rec.get("k"),
            rec.get("d_ub"), rec.get("dX_ub"), rec.get("dZ_ub"),
            rec.get("m4ri_trials"), rec.get("trials"),
            rec.get("distance_method"),
        )

        if group is not None:
            rec["group"] = group
        if n is not None:
            rec["n"] = n
        if k is not None:
            rec["k"] = k
        if d_ub is not None:
            rec["d_ub"] = d_ub
        if dX_ub is not None:
            rec["dX_ub"] = dX_ub
        if dZ_ub is not None:
            rec["dZ_ub"] = dZ_ub

        rec["m4ri_trials"] = int(trials_total)
        rec["trials"] = int(trials_total)
        if method is not None:
            rec["distance_method"] = method

        if isinstance(rec.get("meta"), dict):
            m = rec["meta"]
            if group is not None:
                m["group"] = group
            if n is not None:
                m["n"] = n
            if k is not None:
                m["k"] = k

        after = (
            rec.get("group"), rec.get("n"), rec.get("k"),
            rec.get("d_ub"), rec.get("dX_ub"), rec.get("dZ_ub"),
            rec.get("m4ri_trials"), rec.get("trials"),
            rec.get("distance_method"),
        )

        if before != after:
            changed += 1
            if args.verbose:
                print(f"[fix] {cid}")

        update_min(group, int(trials_total))

    if out_list is not None:
        new_list = [rec for rec in out_list if not (isinstance(rec, dict) and rec.get("_DROP_") is True)]
        data["codes"] = new_list
    elif out_dict is not None:
        data["codes"] = out_dict

    data["generated_at_utc"] = iso_utc_now()

    mt_path = best_dir / "min_trials_by_group.json"
    if min_trials_by_group and not args.check_only:
        mt_path.write_text(json.dumps(min_trials_by_group, indent=2, sort_keys=True) + "\n")

    if args.check_only:
        print(f"[check] would_update={changed} would_drop_missing={dropped}")
        return 0

    data_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    print(f"[ok] ensured summary fields in {data_path} (updated={changed}, dropped={dropped})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
