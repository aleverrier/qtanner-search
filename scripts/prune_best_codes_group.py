#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def parse_d_from_name(code_id: str) -> Optional[int]:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def ensure_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except: return default

def get_steps(meta: Dict[str, Any]) -> int:
    # prefer m4ri fields, fallback to trials/steps
    for k in ("m4ri_steps", "distance_steps_m4ri", "trials_m4ri", "trials", "steps"):
        if k in meta:
            return ensure_int(meta.get(k), 0)
    dist = meta.get("distance")
    if isinstance(dist, dict):
        for k in ("steps", "trials"):
            if k in dist:
                return ensure_int(dist.get(k), 0)
    return 0

def get_d(meta: Dict[str, Any], code_id: str) -> int:
    return ensure_int(meta.get("d_ub", meta.get("d", parse_d_from_name(code_id) or 0)), 0)

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
    ap = argparse.ArgumentParser(description="Prune best_codes entries for a group: keep only best per k, and enforce min trials.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", required=True)
    ap.add_argument("--min-trials", type=int, required=True, help="Require m4ri_steps >= this; otherwise archive the code.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    col = best_dir / "collected"
    meta_dir = best_dir / "meta"
    if not col.is_dir() or not meta_dir.is_dir():
        raise SystemExit("ERROR: expected best_codes/collected and best_codes/meta to exist.")

    group_prefix = args.group + "_"
    codes = [p.name for p in col.iterdir() if p.is_dir() and p.name.startswith(group_prefix)]
    codes.sort()
    if not codes:
        print(f"[info] no codes for group {args.group}")
        return 0

    # First: enforce min trials (archive anything below threshold or missing)
    low: List[Tuple[str,int]] = []
    ok_codes: List[str] = []
    for cid in codes:
        mp = meta_dir / f"{cid}.json"
        meta = read_json(mp) if mp.exists() else {}
        steps = get_steps(meta)
        if steps < args.min_trials:
            low.append((cid, steps))
        else:
            ok_codes.append(cid)

    print(f"[info] group={args.group} codes={len(codes)} min_trials={args.min_trials} below={len(low)}")
    for cid, steps in low[:20]:
        print(f"  below-threshold {cid} steps={steps}")

    # Second: among remaining codes, keep only best per k (by d_ub/d)
    best_by_k: Dict[int, int] = {}
    d_by_code: Dict[str, Tuple[int,int]] = {}  # cid -> (k,d)
    for cid in ok_codes:
        k = parse_k(cid)
        if k is None:
            continue
        mp = meta_dir / f"{cid}.json"
        meta = read_json(mp) if mp.exists() else {}
        d = get_d(meta, cid)
        d_by_code[cid] = (k, d)
        best_by_k[k] = max(best_by_k.get(k, 0), d)

    pruned: List[Tuple[str,int,int,int]] = []
    for cid, (k, d) in d_by_code.items():
        if d < best_by_k.get(k, d):
            pruned.append((cid, k, d, best_by_k[k]))

    print(f"[info] prune-by-best-per-k: {len(pruned)}")
    for cid,k,d,b in pruned[:20]:
        print(f"  prune {cid} (k={k} d={d} < best={b})")

    if args.dry_run:
        return 0

    archive_root = best_dir / "archived" / f"pruned_{args.group}"
    archived = 0

    for cid, _steps in low:
        archive_code(best_dir, cid, archive_root / "below_min_trials")
        archived += 1

    for cid, _k, _d, _b in pruned:
        # Might already have been archived above, but safe: check existence
        if (col / cid).exists():
            archive_code(best_dir, cid, archive_root / "not_best")
            archived += 1

    print(f"[done] archived {archived} codes under {archive_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
