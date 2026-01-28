#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def write_json(p: Path, d: Dict[str, Any]) -> None:
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")

def parse_d_from_name(code_id: str) -> int | None:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def replace_d_suffix(code_id: str, new_d: int) -> str:
    if re.search(r"_d\d+$", code_id):
        return re.sub(r"_d\d+$", f"_d{new_d}", code_id)
    return code_id + f"_d{new_d}"

def main() -> int:
    ap = argparse.ArgumentParser(description="Rename best_codes artifacts so _dNN matches meta.json (d_ub/d).")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", default=None, help="Only process codes whose folder starts with GROUP_")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--archive-collisions", action="store_true",
                    help="If target name already exists, archive the old one instead of erroring.")
    args = ap.parse_args()

    best = Path(args.best_dir).resolve()
    col = best / "collected"
    meta_dir = best / "meta"
    mats = best / "matrices"
    archive_root = best / "archived" / "name_sync_collisions"

    if not col.is_dir() or not meta_dir.is_dir() or not mats.is_dir():
        raise SystemExit(f"ERROR: expected {col}, {meta_dir}, {mats} to exist.")

    codes = [p.name for p in col.iterdir() if p.is_dir()]
    if args.group:
        pref = args.group + "_"
        codes = [c for c in codes if c.startswith(pref)]
    codes.sort()

    renames: List[Tuple[str, str, int, int]] = []

    for code in codes:
        mp = meta_dir / f"{code}.json"
        if not mp.exists():
            continue
        meta = read_json(mp)
        d_meta = meta.get("d_ub", meta.get("d"))
        if d_meta is None:
            continue
        try:
            d_meta_i = int(d_meta)
        except Exception:
            continue

        d_name = parse_d_from_name(code)
        if d_name is None:
            continue

        if d_meta_i == d_name:
            continue

        new_code = replace_d_suffix(code, d_meta_i)
        renames.append((code, new_code, d_name, d_meta_i))

    print(f"[info] planned renames: {len(renames)}")
    for old,new,dn,dm in renames[:20]:
        print(f"  {old}  ->  {new}   (name_d={dn} meta_d={dm})")
    if args.dry_run:
        return 0

    for old, new, dn, dm in renames:
        old_col = col / old
        old_meta = meta_dir / f"{old}.json"
        new_col = col / new
        new_meta = meta_dir / f"{new}.json"

        if new_col.exists() or new_meta.exists():
            if not args.archive_collisions:
                raise SystemExit(f"ERROR: target already exists for {old} -> {new}. Use --archive-collisions.")
            archive_root.mkdir(parents=True, exist_ok=True)
            target = archive_root / old
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            # Move the *old* artifacts aside
            if old_col.exists():
                shutil.move(str(old_col), str(target / "collected"))
            if old_meta.exists():
                shutil.move(str(old_meta), str(target / "meta.json"))
            for f in mats.glob(f"{old}*"):
                if f.is_file():
                    shutil.move(str(f), str(target / f.name))
            print(f"[archive] collision: archived old {old} into {target}")
            continue

        # 1) rename collected dir
        shutil.move(str(old_col), str(new_col))

        # 2) rename matrices
        for f in list(mats.glob(f"{old}*")):
            if not f.is_file():
                continue
            new_name = f.name.replace(old, new, 1)
            shutil.move(str(f), str(mats / new_name))

        # 3) rename meta json and update internal fields
        meta = read_json(old_meta)
        meta["d"] = dm
        meta["d_ub"] = dm
        meta.setdefault("original_code_id", old)
        meta["code_id"] = new
        write_json(new_meta, meta)
        old_meta.unlink(missing_ok=True)

        print(f"[rename] {old} -> {new}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
