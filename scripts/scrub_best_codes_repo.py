from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _best_codes_common import (
    CodeScore,
    archive_code,
    code_id_with_d,
    code_id_without_d,
    extract_code_id,
    extract_distance_bounds,
    extract_trials,
    load_json,
    atomic_write_json,
    rename_code,
    utc_now_iso,
)

def score_from_meta(meta: Dict[str, Any]) -> CodeScore:
    _, _, d = extract_distance_bounds(meta)
    t, _, _ = extract_trials(meta)
    return CodeScore(d_ub=d, trials_total=t)

def merge_trials(meta_keep: Dict[str, Any], trials_total_max: int) -> None:
    dist = meta_keep.setdefault("distance", {})
    if not isinstance(dist, dict):
        meta_keep["distance"] = dist = {}
    dist["steps_used_total"] = int(trials_total_max)
    dist.setdefault("scrub", {})
    if isinstance(dist["scrub"], dict):
        dist["scrub"]["trials_total_max"] = int(trials_total_max)
        dist["scrub"]["scrubbed_at_utc"] = utc_now_iso()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--group", default=None)
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir)
    meta_dir = best_dir / "meta"
    ts = utc_now_iso().replace(":","").replace("-","")
    archive_root = Path(args.archive_root) if args.archive_root else (best_dir / "archived" / f"scrub_{ts}")
    dupes_root = archive_root / "dupes"
    archive_root.mkdir(parents=True, exist_ok=True)

    metas = sorted([p for p in meta_dir.glob("*.json") if p.is_file()])
    if args.group:
        metas = [p for p in metas if p.stem.startswith(args.group + "_")]

    buckets: Dict[str, List[Tuple[str, Path, Dict[str, Any], CodeScore]]] = {}
    for p in metas:
        cid_file = p.stem
        meta = load_json(p)
        cid_meta = extract_code_id(meta, cid_file) or cid_file
        canon = code_id_without_d(cid_meta)
        buckets.setdefault(canon, []).append((cid_file, p, meta, score_from_meta(meta)))

    renames: List[Tuple[str,str]] = []
    archives: List[str] = []

    for canon, items in buckets.items():
        if len(items) == 1:
            cid_file, p, meta, sc = items[0]
            if isinstance(sc.d_ub, int):
                desired = code_id_with_d(canon, sc.d_ub)
                if desired != cid_file:
                    renames.append((cid_file, desired))
            continue

        items_sorted = sorted(items, key=lambda x: x[3].sort_key_conservative())
        keep_cid, keep_path, keep_meta, keep_sc = items_sorted[0]

        trials_max = 0
        for _cid,_p,_m,sc in items:
            if sc.trials_total is not None:
                trials_max = max(trials_max, int(sc.trials_total))
        if trials_max > 0:
            merge_trials(keep_meta, trials_max)

        best_d = keep_sc.d_ub
        if isinstance(best_d, int):
            desired = code_id_with_d(canon, best_d)
            keep_meta["code_id"] = desired
            if desired != keep_cid:
                renames.append((keep_cid, desired))
                keep_cid = desired

        if not args.dry_run:
            atomic_write_json(keep_path, keep_meta)

        for cid_file,p,meta,sc in items_sorted[1:]:
            archives.append(cid_file)
            if not args.dry_run:
                archive_code(best_dir, cid_file, dupes_root / canon / cid_file)

    if args.dry_run:
        print("[dry] archives:", len(archives), "renames:", len(renames))
        return 0

    if renames:
        coll_root = archive_root / "name_sync_collisions"
        coll_root.mkdir(parents=True, exist_ok=True)
        for old,new in renames:
            if old != new:
                rename_code(best_dir, old, new, archive_collisions_root=coll_root)

    print(f"[done] scrubbed: archived_dupes={len(archives)} renames={len(renames)} archive_root={archive_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
