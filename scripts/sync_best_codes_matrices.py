#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DIST_RE = re.compile(r"^(.*)_d(\d+)$")

def drop_distance(code_id: str) -> str:
    m = DIST_RE.match(code_id)
    return m.group(1) if m else code_id

def load_active_code_ids(best_dir: Path) -> List[str]:
    data_path = best_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}.")
    data = json.loads(data_path.read_text())
    codes = data.get("codes")

    out: List[str] = []
    if isinstance(codes, dict):
        for cid, rec in codes.items():
            if isinstance(cid, str) and cid:
                out.append(cid)
            elif isinstance(rec, dict):
                cid2 = rec.get("code_id") or rec.get("id") or rec.get("name")
                if isinstance(cid2, str) and cid2:
                    out.append(cid2)
    elif isinstance(codes, list):
        for rec in codes:
            if not isinstance(rec, dict):
                continue
            cid = rec.get("code_id") or rec.get("id") or rec.get("name")
            if isinstance(cid, str) and cid:
                out.append(cid)
    else:
        raise TypeError(f"Unexpected data.json['codes'] type: {type(codes)}")

    return sorted(set(out))

@dataclass(frozen=True)
class Hit:
    path: Path
    code_id: str
    key: str
    kind: str  # "x" or "z"

def parse_matrix_name(name: str) -> Optional[Tuple[str, str]]:
    # Return (code_id, kind) if filename matches known patterns.
    for suf, kind in [
        ("__Hx.mtx", "x"), ("__HX.mtx", "x"),
        ("__Hz.mtx", "z"), ("__HZ.mtx", "z"),
        ("__X.mtx", "x"),  ("__Z.mtx", "z"),
        ("_Hx.mtx", "x"),  ("_HX.mtx", "x"),
        ("_Hz.mtx", "z"),  ("_HZ.mtx", "z"),
    ]:
        if name.endswith(suf):
            return name[: -len(suf)], kind
    return None

def build_index(search_roots: List[Path]) -> Dict[Tuple[str, str], List[Hit]]:
    idx: Dict[Tuple[str, str], List[Hit]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.mtx"):
            if not p.is_file():
                continue
            parsed = parse_matrix_name(p.name)
            if not parsed:
                continue
            cid, kind = parsed
            if not cid:
                continue
            key = drop_distance(cid)
            hit = Hit(path=p, code_id=cid, key=key, kind=kind)
            # bucket by exact code id
            idx.setdefault((cid, kind), []).append(hit)
            # bucket by "key" (code id with _d... removed), for renamed distance cases
            idx.setdefault((key, kind), []).append(hit)
    return idx

def choose_best(hits: List[Hit]) -> Hit:
    # Prefer matrices located under results/**/best_codes/**, then newest mtime.
    def score(h: Hit) -> Tuple[int, float, int, str]:
        s = str(h.path)
        looks_like_best = 1 if "/best_codes/" in s else 0
        try:
            mtime = h.path.stat().st_mtime
        except Exception:
            mtime = 0.0
        # shorter path slightly preferred
        plen = -len(h.path.parts)
        return (looks_like_best, mtime, plen, s)

    hits = list({h.path.resolve(): h for h in hits}.values())
    hits.sort(key=score, reverse=True)
    return hits[0]

def canonical_dest(matrices_dir: Path, code_id: str, kind: str) -> Path:
    # IMPORTANT: GitHub Pages is case-sensitive. Use __Hx and __Hz exactly.
    assert kind in ("x", "z")
    return matrices_dir / f"{code_id}__H{'x' if kind=='x' else 'z'}.mtx"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail-if-missing", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir)
    matrices_dir = best_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    active = load_active_code_ids(best_dir)
    print(f"[info] active codes from {best_dir/'data.json'}: {len(active)}")

    roots = [
        Path("results"),
        best_dir / "archived",
        best_dir / "collected",
        best_dir / "matrices",
    ]
    idx = build_index(roots)

    copied = 0
    already = 0
    missing: List[Tuple[str, str]] = []

    for cid in active:
        key = drop_distance(cid)
        for kind in ("x", "z"):
            dest = canonical_dest(matrices_dir, cid, kind)
            if dest.exists():
                already += 1
                continue

            hits = idx.get((cid, kind)) or idx.get((key, kind)) or []
            if not hits:
                missing.append((cid, kind))
                if args.verbose:
                    print(f"[miss] {cid} H{kind}")
                continue

            src = choose_best(hits).path
            if args.verbose:
                print(f"[copy] {cid} H{kind}: {src} -> {dest}")
            if not args.dry_run:
                shutil.copy2(src, dest)
            copied += 1

    print(f"[done] already present: {already}")
    print(f"[done] copied: {copied}")
    print(f"[done] missing: {len(missing)}")

    if missing:
        for cid, kind in missing[:30]:
            print(f"  MISSING {cid}__H{kind}.mtx")
        if args.fail_if_missing:
            return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
