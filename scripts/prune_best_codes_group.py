#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text()) if p.exists() else {}

def ensure_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except: return default

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def parse_d(code_id: str) -> Optional[int]:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def get_trials_anywhere(meta: Dict[str, Any]) -> int:
    # Prefer refined m4ri fields
    for k in ("m4ri_steps","trials","steps","steps_used","distance_trials","distance_steps"):
        if k in meta:
            v = ensure_int(meta.get(k), 0)
            if v: return v

    # Old progressive block
    dist = meta.get("distance")
    if isinstance(dist, dict):
        for k in ("steps","trials","steps_used_total","steps_used_x","steps_used_z"):
            if k in dist:
                v = ensure_int(dist.get(k), 0)
                if v: return v

    dm = meta.get("distance_m4ri")
    if isinstance(dm, dict):
        v = ensure_int(dm.get("steps_per_side", 0), 0)
        if v: return v

    return 0

def get_d_ub(meta: Dict[str, Any], code_id: str) -> int:
    d = meta.get("d_ub", meta.get("d", parse_d(code_id) or 0))
    return ensure_int(d, 0)

def archive_code(best_dir: Path, code_id: str, archive_root: Path) -> None:
    (archive_root / "collected").mkdir(parents=True, exist_ok=True)
    (archive_root / "meta").mkdir(parents=True, exist_ok=True)
    (archive_root / "matrices").mkdir(parents=True, exist_ok=True)

    src_col = best_dir / "collected" / code_id
    src_meta = best_dir / "meta" / f"{code_id}.json"
    mats = best_dir / "matrices"

    if src_col.exists():
        shutil.move(str(src_col), str(archive_root / "collected" / code_id))
    if src_meta.exists():
        shutil.move(str(src_meta), str(archive_root / "meta" / f"{code_id}.json"))
    if mats.exists():
        for f in mats.glob(f"{code_id}*"):
            if f.is_file():
                shutil.move(str(f), str(archive_root / "matrices" / f.name))

def main() -> int:
    ap = argparse.ArgumentParser(description="Prune best_codes: enforce min trials (from embedded collected meta.json) + keep best per k.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", required=True)
    ap.add_argument("--min-trials", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    col = best / "collected"
    meta_dir = best / "meta"

    if not col.is_dir():
        raise SystemExit("ERROR: best_codes/collected not found")

    pref = args.group + "_"
    codes = [p.name for p in col.iterdir() if p.is_dir() and p.name.startswith(pref)]
    codes.sort()

    low: List[Tuple[str,int]] = []
    ok: List[str] = []

    for cid in codes:
        embedded = read_json(col / cid / "meta.json")
        steps = get_trials_anywhere(embedded)
        if steps < args.min_trials:
            low.append((cid, steps))
        else:
            ok.append(cid)

    print(f"[info] group={args.group} codes={len(codes)} min_trials={args.min_trials} below={len(low)}")
    for cid, s in low[:25]:
        print(f"  below-threshold {cid} steps={s}")

    # Best-per-k among ok codes
    best_by_k: Dict[int, int] = {}
    d_by_code: Dict[str, Tuple[int,int]] = {}

    for cid in ok:
        k = parse_k(cid)
        if k is None:
            continue
        embedded = read_json(col / cid / "meta.json")
        d = get_d_ub(embedded, cid)
        if d == 0:
            m = read_json(meta_dir / f"{cid}.json")
            d = get_d_ub(m, cid)
        d_by_code[cid] = (k, d)
        best_by_k[k] = max(best_by_k.get(k, 0), d)

    pruned: List[Tuple[str,int,int,int]] = []
    for cid, (k, d) in d_by_code.items():
        if d < best_by_k.get(k, d):
            pruned.append((cid, k, d, best_by_k[k]))

    print(f"[info] prune-by-best-per-k: {len(pruned)}")
    for cid,k,d,b in pruned[:25]:
        print(f"  prune {cid} (k={k} d={d} < best={b})")

    if args.dry_run:
        return 0

    arch = best / "archived" / f"pruned_{args.group}"
    archived = 0
    for cid,_ in low:
        if (col / cid).exists():
            archive_code(best, cid, arch / "below_min_trials")
            archived += 1
    for cid,_,_,_ in pruned:
        if (col / cid).exists():
            archive_code(best, cid, arch / "not_best")
            archived += 1

    print(f"[done] archived {archived} codes under {arch}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
