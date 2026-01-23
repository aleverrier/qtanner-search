#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Tuple, Set, Optional

def read_json(p: Path) -> Any:
    return json.loads(p.read_text())

def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def ensure_int(x: Any) -> Optional[int]:
    try: return int(x)
    except: return None

def parse_group(code_id: str) -> str:
    return code_id.split("_", 1)[0] if "_" in code_id else code_id

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def extract_distance(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    d = meta.get("d_ub", meta.get("d"))
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

def looks_like_code_id(s: str) -> bool:
    return ("_k" in s and "_d" in s)

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

    codes: Set[str] = {p.name for p in col.iterdir() if p.is_dir()}
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

    def patch_obj(obj: Any) -> Any:
        # returns patched obj; may return None to indicate "drop this element"
        if isinstance(obj, dict):
            # If dict has code_id-ish key/value, patch it
            cid = obj.get("code_id") or obj.get("id") or obj.get("name")
            if isinstance(cid, str):
                cid2 = rename_old_to_new.get(cid, cid)
                if cid2 != cid:
                    # update id field (keep key name stable)
                    if "code_id" in obj: obj["code_id"] = cid2
                    elif "id" in obj: obj["id"] = cid2
                    elif "name" in obj: obj["name"] = cid2
                    cid = cid2

                # If this is a code entry, drop if code no longer exists
                if cid in meta_map and looks_like_code_id(cid) and cid not in codes:
                    return None

                # Patch distances if we have meta
                if cid in meta_map:
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
                    if meta_map[cid].get("distance_backend") == "dist-m4ri":
                        obj["distance_backend"] = "dist-m4ri"

            # Also handle dicts keyed by code_id
            newd: Dict[str, Any] = {}
            for k, v in obj.items():
                k2 = rename_old_to_new.get(k, k) if isinstance(k, str) else k
                pv = patch_obj(v)
                if pv is None:
                    continue
                # if key looks like a code id, filter it
                if isinstance(k2, str) and looks_like_code_id(k2) and k2 not in codes:
                    continue
                newd[k2] = pv
            return newd

        if isinstance(obj, list):
            out = []
            for it in obj:
                pit = patch_obj(it)
                if pit is None:
                    continue
                out.append(pit)
            return out

        if isinstance(obj, str):
            return rename_old_to_new.get(obj, obj)

        return obj

    data2 = patch_obj(data)
    write_json(data_path, data2)
    print(f"[ok] patched {data_path} from meta; codes={len(codes)} renames={len(rename_old_to_new)}")

    # Rebuild index.tsv (simple stable format)
    index_tsv = best / "index.tsv"
    rows = []
    for cid in sorted(codes):
        m = meta_map.get(cid, {})
        d, trials, dX, dZ = extract_distance(m)
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
    print(f"[ok] wrote {index_tsv}")

    # Rebuild best_by_group_k.tsv
    best_by = {}
    for r in rows:
        if r["k"] is None or r["d"] is None:
            continue
        key = (r["group"], int(r["k"]))
        best_by[key] = max(best_by.get(key, 0), int(r["d"]))
    out = ["group\tk\tbest_d"]
    for (g,k), d in sorted(best_by.items()):
        out.append(f"{g}\t{k}\t{d}")
    (best / "best_by_group_k.tsv").write_text("\n".join(out) + "\n")
    print(f"[ok] wrote {best/'best_by_group_k.tsv'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
