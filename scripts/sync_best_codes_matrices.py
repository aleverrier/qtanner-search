#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Set, Tuple


def _glob_first(search_dirs: List[Path], patterns: List[str]) -> Optional[Path]:
    cands: List[Path] = []
    seen: Set[str] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                if p.is_file() and p.name not in seen:
                    seen.add(p.name)
                    cands.append(p)
    if not cands:
        return None
    # Prefer canonical-looking names first.
    def key(p: Path) -> Tuple[int, int, str]:
        name = p.name
        return (0 if "__H" in name else 1, len(name), name)
    cands.sort(key=key)
    return cands[0]


def find_matrix(best_dir: Path, code_id: str, kind: str) -> Optional[Path]:
    # kind in {"x","z"}
    K = kind.upper()
    collected = best_dir / "collected" / code_id
    search_dirs = [
        best_dir / "matrices",
        collected,
        collected / "matrices",
        collected / "data",
    ]
    patterns = [
        f"{code_id}__H{kind}.mtx",
        f"{code_id}__H{K}.mtx",
        f"{code_id}*__H{kind}.mtx",
        f"{code_id}*__H{K}.mtx",
        f"*{code_id}*__H{kind}.mtx",
        f"*{code_id}*__H{K}.mtx",
    ]
    return _glob_first(search_dirs, patterns)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure best_codes/matrices contains canonical lifted Hx/Hz for all codes.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--fail-if-missing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only-group", default=None, help="Optional: restrict to code_ids starting with '<group>_'")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    mats_dir = best_dir / "matrices"
    mats_dir.mkdir(parents=True, exist_ok=True)

    collected_dir = best_dir / "collected"
    meta_dir = best_dir / "meta"

    code_ids: Set[str] = set()
    if collected_dir.is_dir():
        for p in collected_dir.iterdir():
            if p.is_dir():
                code_ids.add(p.name)
    if meta_dir.is_dir():
        for p in meta_dir.glob("*.json"):
            code_ids.add(p.stem)

    if args.only_group:
        prefix = args.only_group + "_"
        code_ids = {cid for cid in code_ids if cid.startswith(prefix)}

    missing: List[str] = []
    copied = 0

    for cid in sorted(code_ids):
        for kind in ("x", "z"):
            dest = mats_dir / f"{cid}__H{kind}.mtx"  # canonical: __Hx / __Hz
            if dest.exists():
                continue
            src = find_matrix(best_dir, cid, kind)
            if src is None:
                missing.append(f"{cid} (H{kind})")
                continue
            shutil.copy2(src, dest)
            copied += 1
            if args.verbose:
                print(f"[copy] {src} -> {dest}")

    if args.verbose or copied:
        print(f"[ok] matrices sync: copied={copied} missing={len(missing)}")

    if missing:
        print("[warn] missing matrices for:")
        for x in missing[:50]:
            print(" ", x)
        if len(missing) > 50:
            print(f"  ... and {len(missing)-50} more")
        if args.fail_if_missing:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
