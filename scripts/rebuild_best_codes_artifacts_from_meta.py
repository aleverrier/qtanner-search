#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Tuple, Set, Optional, List

def read_json(p: Path) -> Any:
    return json.loads(p.read_text())

def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def ensure_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def looks_like_code_id(s: str) -> bool:
    return isinstance(s, str) and ("_k" in s and "_d" in s)

def parse_group(code_id: str) -> str:
    return code_id.split("_", 1)[0] if "_" in code_id else code_id

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def parse_d(code_id: str) -> Optional[int]:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def extract_distance(meta: Dict[str, Any], code_id: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    d = meta.get("d_ub", meta.get("d", parse_d(code_id)))
    trials = meta.get("m4ri_steps", meta.get("trials", meta.get("steps")))
    dX = meta.get("dX_ub", meta.get("dX"))
    dZ = meta.get("dZ_ub", meta.get("dZ"))
    dist = meta.get("distance")
    if isinstance(dist, dict):
        d = d if d is not None else dist.get("d_ub", dist.get("d"))
        trials = trials if trials is not None else dist.get("steps", dist.get("trials"))
        dX = dX if dX is not None else dist.get("dX_ub", dist.get("dX"))
        dZ = dZ if dZ is not None else dist.get("dZ_ub", dist.get("dZ"))
    return ensure_int(d), ensure_int(trials), ensure_int(dX), ensure_int(dZ)

def patch_record(rec: Any, cid: str, meta_map: Dict[str, Dict[str, Any]]) -> Any:
    """Patch a code record dict from meta, writing trials into all common fields."""
    if cid not in meta_map:
        return rec
    meta = meta_map[cid]
    d, trials, dX, dZ = extract_distance(meta, cid)

    if not isinstance(rec, dict):
        return rec

    # always have a code_id field available
    rec.setdefault("code_id", cid)

    if d is not None:
        rec["d"] = d
        rec["d_ub"] = d

    if trials is not None:
        # write into many keys (frontends differ)
        rec["m4ri_steps"] = trials
        rec["trials"] = trials
        rec["steps"] = trials
        rec["steps_used"] = trials
        rec["distance_trials"] = trials
        rec["distance_steps"] = trials

    if dX is not None:
        rec["dX_ub"] = dX
    if dZ is not None:
        rec["dZ_ub"] = dZ

    # nested distance object for compatibility
    dist = rec.get("distance")
    if not isinstance(dist, dict):
        dist = {}
        rec["distance"] = dist
    if d is not None:
        dist["d"] = d
        dist["d_ub"] = d
    if trials is not None:
        dist["steps"] = trials
        dist["trials"] = trials
    if dX is not None:
        dist["dX_ub"] = dX
    if dZ is not None:
        dist["dZ_ub"] = dZ

    if meta.get("distance_backend") == "dist-m4ri":
        rec["distance_backend"] = "dist-m4ri"
        dist["backend"] = "dist-m4ri"

    return rec

def get_code_id_from_record(rec: Any) -> Optional[str]:
    if not isinstance(rec, dict):
        return None
    cid = rec.get("code_id") or rec.get("id") or rec.get("name")
    return cid if isinstance(cid, str) else None

def now_utc_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild/patch website artifacts using best_codes/meta and best_codes/collected.")
    ap.add_argument("--best-dir", default="best_codes")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    col = best / "collected"
    meta_dir = best / "meta"
    data_path = best / "data.json"

    if not col.is_dir() or not meta_dir.is_dir() or not data_path.exists():
        raise SystemExit("ERROR: need best_codes/collected, best_codes/meta, best_codes/data.json")

    collected_codes: Set[str] = {p.name for p in col.iterdir() if p.is_dir()}

    meta_map: Dict[str, Dict[str, Any]] = {}
    rename_old_to_new: Dict[str, str] = {}
    for mp in meta_dir.glob("*.json"):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        cid = mp.stem
        meta_map[cid] = m
        old = m.get("original_code_id")
        if isinstance(old, str) and old != cid:
            rename_old_to_new[old] = cid

    data = read_json(data_path)
    if not isinstance(data, dict) or "codes" not in data:
        raise SystemExit("ERROR: data.json must be a dict with a top-level 'codes' key.")

    codes_obj = data["codes"]
    updated = 0
    dropped = 0
    renamed = 0

    if isinstance(codes_obj, dict):
        new_codes: Dict[str, Any] = {}
        for key, rec in codes_obj.items():
            cid = rename_old_to_new.get(key, key)
            if cid != key:
                renamed += 1
            if looks_like_code_id(cid) and cid not in collected_codes:
                dropped += 1
                continue
            if looks_like_code_id(cid):
                rec = patch_record(rec, cid, meta_map)
                updated += 1
            new_codes[cid] = rec
        data["codes"] = new_codes

    elif isinstance(codes_obj, list):
        new_list: List[Any] = []
        for rec in codes_obj:
            cid = get_code_id_from_record(rec)
            if isinstance(cid, str):
                cid2 = rename_old_to_new.get(cid, cid)
                if cid2 != cid:
                    renamed += 1
                    if isinstance(rec, dict):
                        if "code_id" in rec: rec["code_id"] = cid2
                        elif "id" in rec: rec["id"] = cid2
                        elif "name" in rec: rec["name"] = cid2
                    cid = cid2

                if looks_like_code_id(cid) and cid not in collected_codes:
                    dropped += 1
                    continue

                if looks_like_code_id(cid):
                    rec = patch_record(rec, cid, meta_map)
                    updated += 1

            new_list.append(rec)
        data["codes"] = new_list

    else:
        raise SystemExit(f"ERROR: unsupported type for data['codes']: {type(codes_obj)}")

    data["generated_at_utc"] = now_utc_z()
    write_json(data_path, data)
    print(f"[ok] patched {data_path} from meta; updated={updated} dropped={dropped} renamed={renamed} collected={len(collected_codes)}")

    # Rebuild index.tsv and best_by_group_k.tsv from collected+meta
    index_tsv = best / "index.tsv"
    rows = []
    for cid in sorted(collected_codes):
        m = meta_map.get(cid, {})
        d, trials, dX, dZ = extract_distance(m, cid)
        rows.append({
            "code_id": cid,
            "group": parse_group(cid),
            "n": m.get("n"),
            "k": m.get("k", parse_k(cid)),
            "d": d,
            "trials": trials,
            "dX_ub": dX,
            "dZ_ub": dZ,
            "distance_backend": m.get("distance_backend"),
            "updated_at": m.get("updated_at"),
        })

    header = ["code_id","group","n","k","d","trials","dX_ub","dZ_ub","distance_backend","updated_at"]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join("" if r[h] is None else str(r[h]) for h in header))
    index_tsv.write_text("\n".join(lines) + "\n")

    best_by: Dict[Tuple[str,int], int] = {}
    for r in rows:
        if r["k"] is None or r["d"] is None:
            continue
        key = (r["group"], int(r["k"]))
        best_by[key] = max(best_by.get(key, 0), int(r["d"]))
    out = ["group\tk\tbest_d"]
    for (g,k), d in sorted(best_by.items()):
        out.append(f"{g}\t{k}\t{d}")
    (best / "best_by_group_k.tsv").write_text("\n".join(out) + "\n")

    print(f"[ok] wrote {index_tsv} and {best/'best_by_group_k.tsv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
