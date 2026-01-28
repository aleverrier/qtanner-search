#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from _best_codes_common import (
    atomic_write_json,
    extract_code_id,
    extract_group_spec,
    extract_distance_bounds,
    extract_trials,
    extract_n,
    extract_k,
    extract_A_elems,
    extract_B_elems,
    utc_now_iso,
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_group_tag(tag: str) -> str:
    t = tag.strip()
    t = t.replace("(", "_").replace(")", "").replace(",", "_")
    t = t.replace("__", "_").replace(" ", "")
    return t.strip("_").lower()


def group_prefix_from_code_id(code_id: str) -> Optional[str]:
    for sep in ("__AA", "_AA"):
        if sep in code_id:
            return code_id.split(sep, 1)[0]
    return None


def group_order_from_spec(group: str) -> Optional[int]:
    if not group:
        return None
    m = re.search(r"SmallGroup[_\(](\d+)", group)
    if m:
        return int(m.group(1))
    parts = group.split("x")
    prod = 1
    ok = False
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            prod *= int(p[1:])
            ok = True
        elif p.isdigit():
            prod *= int(p)
            ok = True
        else:
            return None
    return prod if ok else None


def extract_group_order(meta: Dict[str, Any]) -> Optional[int]:
    g = meta.get("group")
    if isinstance(g, dict) and isinstance(g.get("order"), int):
        return int(g["order"])
    if isinstance(g, str):
        val = group_order_from_spec(g)
        if val is not None:
            return val
    for key in ("A", "B"):
        side = meta.get(key)
        if isinstance(side, dict):
            sg = side.get("group")
            if isinstance(sg, dict) and isinstance(sg.get("order"), int):
                return int(sg["order"])
            if isinstance(sg, str):
                val = group_order_from_spec(sg)
                if val is not None:
                    return val
    run_meta = meta.get("run_meta")
    if isinstance(run_meta, dict):
        grp = run_meta.get("group")
        if isinstance(grp, dict) and isinstance(grp.get("order"), int):
            return int(grp["order"])
    return None


def matches_group(meta: Dict[str, Any], group: Optional[str], order: Optional[int]) -> bool:
    if group is None and order is None:
        return True
    code_id = extract_code_id(meta, "") or ""
    cand_tags: List[str] = []
    g = extract_group_spec(meta)
    if isinstance(g, str) and g:
        cand_tags.append(g)
    g2 = group_prefix_from_code_id(code_id)
    if g2:
        cand_tags.append(g2)
    if isinstance(meta.get("group"), str):
        cand_tags.append(meta["group"])
    if group is not None:
        tgt = normalize_group_tag(group)
        for c in cand_tags:
            if normalize_group_tag(c) == tgt:
                return True
        return False
    if order is not None:
        ord_meta = extract_group_order(meta)
        if ord_meta is None:
            for c in cand_tags:
                val = group_order_from_spec(c)
                if val is not None:
                    ord_meta = val
                    break
        return ord_meta == order
    return False


def find_best_code_dirs(results_dir: Path) -> Iterable[Path]:
    for best_dir in results_dir.rglob("best_codes"):
        if not best_dir.is_dir():
            continue
        for code_dir in best_dir.iterdir():
            if code_dir.is_dir():
                yield code_dir


def find_candidate_meta(code_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = code_dir / "meta.json"
    if meta_path.exists():
        return read_json(meta_path)
    return None


def get_steps(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    total, sx, sz = extract_trials(meta)
    per_side = None
    for k in ("m4ri_steps", "trials", "steps"):
        v = meta.get(k)
        if isinstance(v, int):
            per_side = v
            break
    if per_side is None and (isinstance(sx, int) or isinstance(sz, int)):
        per_side = max([v for v in (sx, sz) if isinstance(v, int)], default=None)
    return per_side, total, sx if isinstance(sx, int) else None, sz if isinstance(sz, int) else None


def merge_distance(meta: Dict[str, Any], incoming: Dict[str, Any], source: str) -> None:
    dX0, dZ0, d0 = extract_distance_bounds(meta)
    dX1, dZ1, d1 = extract_distance_bounds(incoming)
    t0, sx0, sz0 = extract_trials(meta)
    t1, sx1, sz1 = extract_trials(incoming)
    side0, _, _, _ = get_steps(meta)
    side1, _, _, _ = get_steps(incoming)

    best_dX = min([x for x in (dX0, dX1) if isinstance(x, int)], default=None)
    best_dZ = min([x for x in (dZ0, dZ1) if isinstance(x, int)], default=None)
    best_d = min([x for x in (d0, d1, best_dX, best_dZ) if isinstance(x, int)], default=None)

    best_sx = max([x for x in (sx0, sx1) if isinstance(x, int)], default=None)
    best_sz = max([x for x in (sz0, sz1) if isinstance(x, int)], default=None)
    best_total = max([x for x in (t0, t1) if isinstance(x, int)], default=None)
    if isinstance(best_sx, int) and isinstance(best_sz, int):
        best_total = max(best_total or 0, best_sx + best_sz)

    dist = meta.get("distance")
    if not isinstance(dist, dict):
        dist = {}
    if isinstance(best_dX, int):
        meta["dX_ub"] = best_dX
        dist["dX_best"] = best_dX
    if isinstance(best_dZ, int):
        meta["dZ_ub"] = best_dZ
        dist["dZ_best"] = best_dZ
    if isinstance(best_d, int):
        meta["d_ub"] = best_d
        meta["d"] = best_d
        dist["d_ub"] = best_d
    if isinstance(best_sx, int):
        dist["steps_used_x"] = best_sx
    if isinstance(best_sz, int):
        dist["steps_used_z"] = best_sz
    if isinstance(best_total, int):
        dist["steps_used_total"] = best_total
    meta["distance"] = dist

    per_side = max([v for v in (side0, side1, best_sx, best_sz) if isinstance(v, int)], default=None)
    if isinstance(per_side, int):
        meta["m4ri_steps"] = max(int(per_side), int(meta.get("m4ri_steps", 0) or 0))
        meta["trials"] = max(int(per_side), int(meta.get("trials", 0) or 0))
        meta["steps"] = max(int(per_side), int(meta.get("steps", 0) or 0))
    if isinstance(best_total, int):
        meta["m4ri_trials"] = max(int(best_total), int(meta.get("m4ri_trials", 0) or 0))

    sources = meta.get("distance_ub_sources")
    if not isinstance(sources, dict):
        sources = {}
    if isinstance(best_dX, int) and isinstance(dX1, int) and best_dX == dX1 and dX1 != dX0:
        sources["dX_ub"] = source
    if isinstance(best_dZ, int) and isinstance(dZ1, int) and best_dZ == dZ1 and dZ1 != dZ0:
        sources["dZ_ub"] = source
    if isinstance(best_d, int) and isinstance(d1, int) and best_d == d1 and d1 != d0:
        sources["d_ub"] = source
    if sources:
        meta["distance_ub_sources"] = sources


def update_metadata(meta: Dict[str, Any], code_id: str, best_dir: Path, source: str) -> None:
    meta["code_id"] = code_id
    if "also_seen_in" not in meta or not isinstance(meta.get("also_seen_in"), list):
        meta["also_seen_in"] = []
    if source and source not in meta["also_seen_in"]:
        meta["also_seen_in"].append(source)
    meta["updated_at"] = utc_now_iso()
    if extract_group_spec(meta) is None:
        prefix = group_prefix_from_code_id(code_id)
        if prefix:
            meta["group"] = prefix
    if extract_n(meta) is None or extract_k(meta) is None:
        meta.setdefault("n", meta.get("N"))
        meta.setdefault("k", meta.get("K"))

    if "A_elems" not in meta:
        elems = extract_A_elems(meta)
        if elems is not None:
            meta["A_elems"] = elems
    if "B_elems" not in meta:
        elems = extract_B_elems(meta)
        if elems is not None:
            meta["B_elems"] = elems

    collected_dir = best_dir / "collected" / code_id
    if "collected_dir" in meta:
        meta["collected_dir"] = str(collected_dir)
    if "collected_files" in meta:
        files = []
        if collected_dir.exists():
            for p in sorted(collected_dir.iterdir()):
                if p.is_file():
                    files.append(str(p))
        meta["collected_files"] = files
    if "matrices_flat" in meta:
        mats = []
        mats_dir = best_dir / "matrices"
        if mats_dir.exists():
            for p in sorted(mats_dir.glob(f"{code_id}*.mtx")):
                mats.append(str(p))
        meta["matrices_flat"] = mats


def fill_missing_fields(meta: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    for key in ("group", "n", "k", "A", "B", "A_elems", "B_elems", "A_id", "B_id", "local_codes"):
        if key not in meta and key in incoming:
            meta[key] = incoming[key]


def copy_matrix_if_missing(code_dir: Path, best_dir: Path, code_id: str, kind: str) -> None:
    mats_dir = best_dir / "matrices"
    mats_dir.mkdir(parents=True, exist_ok=True)
    dest = mats_dir / f"{code_id}__H{kind}.mtx"
    if dest.exists():
        return
    for name in (f"H{kind}.mtx", f"H{kind.upper()}.mtx"):
        src = code_dir / name
        if src.exists():
            shutil.copy2(src, dest)
            return


def copy_collected_dir(code_dir: Path, best_dir: Path, code_id: str) -> None:
    dest = best_dir / "collected" / code_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copytree(code_dir, dest)
        return
    for fname in ("Hx.mtx", "Hz.mtx", "meta.json", "settings.json"):
        src = code_dir / fname
        if src.exists() and not (dest / fname).exists():
            shutil.copy2(src, dest / fname)


def verify_group(best_dir: Path, group: Optional[str], order: Optional[int]) -> int:
    data_path = best_dir / "data.json"
    min_trials_path = best_dir / "min_trials_by_group.json"
    if not data_path.exists():
        raise SystemExit(f"ERROR: missing {data_path}")
    if not min_trials_path.exists():
        raise SystemExit(f"ERROR: missing {min_trials_path}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    min_trials = json.loads(min_trials_path.read_text(encoding="utf-8"))

    failures = 0
    for rec in data.get("codes", []):
        if not isinstance(rec, dict):
            continue
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        if not matches_group(meta, group, order):
            continue
        gspec = extract_group_spec(meta) or rec.get("group")
        if not isinstance(gspec, str):
            continue
        required = int(min_trials.get(gspec, min_trials.get("SmallGroup", 0) or 0) or 0)
        trials = rec.get("trials") or rec.get("m4ri_trials") or 0
        if int(trials) < required:
            failures += 1
            print(f"[fail] {rec.get('code_id')} group={gspec} trials={trials} < min={required}")
    if failures:
        print(f"[verify] failures={failures}")
        return 2
    print("[verify] ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish best_codes for a group/order by merging results/**/best_codes.")
    ap.add_argument("--group", default=None)
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--groups-out", default=None, help="Optional path to write matched group specs (one per line).")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    if args.verify:
        return verify_group(best_dir, args.group, args.order)

    if not results_dir.exists():
        raise SystemExit(f"ERROR: results dir not found: {results_dir}")

    touched_groups: List[str] = []
    new_codes = 0
    updated = 0

    for code_dir in find_best_code_dirs(results_dir):
        meta = find_candidate_meta(code_dir)
        if not meta:
            continue
        if not matches_group(meta, args.group, args.order):
            continue
        code_id = extract_code_id(meta, code_dir.name) or code_dir.name
        source = str(code_dir)

        dest_meta_path = best_dir / "meta" / f"{code_id}.json"
        dest_meta_path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_json(dest_meta_path)

        if existing:
            merge_distance(existing, meta, source)
            fill_missing_fields(existing, meta)
            update_metadata(existing, code_id, best_dir, source)
            atomic_write_json(dest_meta_path, existing)
            updated += 1
        else:
            merge_distance(meta, meta, source)
            update_metadata(meta, code_id, best_dir, source)
            atomic_write_json(dest_meta_path, meta)
            new_codes += 1

        copy_collected_dir(code_dir, best_dir, code_id)
        collected_meta_path = best_dir / "collected" / code_id / "meta.json"
        if collected_meta_path.exists():
            collected_meta = read_json(collected_meta_path)
            merge_distance(collected_meta, meta, source)
            fill_missing_fields(collected_meta, meta)
            update_metadata(collected_meta, code_id, best_dir, source)
            atomic_write_json(collected_meta_path, collected_meta)
        copy_matrix_if_missing(code_dir, best_dir, code_id, "x")
        copy_matrix_if_missing(code_dir, best_dir, code_id, "z")

        grp = group_prefix_from_code_id(code_id) or extract_group_spec(meta)
        if isinstance(grp, str) and grp and grp not in touched_groups:
            touched_groups.append(grp)

    if args.groups_out:
        out_path = Path(args.groups_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(sorted(touched_groups)) + "\n", encoding="utf-8")

    print(f"[done] new={new_codes} updated={updated} groups={len(touched_groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
