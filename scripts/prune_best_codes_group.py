#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from typing import Dict, Any

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def parse_k(code_id: str) -> int | None:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def parse_group(code_id: str) -> str:
    return code_id.split("_", 1)[0] if "_" in code_id else code_id

def parse_d_from_name(code_id: str) -> int | None:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def ensure_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except: return default

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
    ap = argparse.ArgumentParser(description="Prune best_codes entries for a group: keep only best per k.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    col = best_dir / "collected"
    meta = best_dir / "meta"
    if not col.is_dir() or not meta.is_dir():
        raise SystemExit("ERROR: expected best_codes/collected and best_codes/meta to exist.")

    codes = [p.name for p in col.iterdir() if p.is_dir() and p.name.startswith(args.group + "_")]
    codes.sort()
    if not codes:
        print(f"[info] no codes for group {args.group}")
        return 0

    # best d_ub per k
    best_by_k: Dict[int, int] = {}
    d_by_code: Dict[str, int] = {}

    for cid in codes:
        k = parse_k(cid)
        if k is None:
            continue
        mp = meta / f"{cid}.json"
        if not mp.exists():
            d = parse_d_from_name(cid) or 0
        else:
            m = read_json(mp)
            d = ensure_int(m.get("d_ub", m.get("d", parse_d_from_name(cid) or 0)), 0)
        d_by_code[cid] = d
        best_by_k[k] = max(best_by_k.get(k, 0), d)

    pruned = []
    for cid, d in d_by_code.items():
        k = parse_k(cid)
        if k is None:
            continue
        if d < best_by_k.get(k, d):
            pruned.append((cid, k, d, best_by_k[k]))

    print(f"[info] group={args.group} codes={len(codes)} prune={len(pruned)}")
    for cid,k,d,b in pruned[:20]:
        print(f"  prune {cid}  (k={k} d={d} < best={b})")

    if args.dry_run or not pruned:
        return 0

    archive_root = best_dir / "archived" / f"pruned_{args.group}"
    for cid,_,_,_ in pruned:
        archive_code(best_dir, cid, archive_root)

    print(f"[done] archived {len(pruned)} codes into {archive_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
