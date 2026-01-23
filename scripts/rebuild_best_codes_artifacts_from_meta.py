from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _best_codes_common import (
    atomic_write_json,
    extract_A_elems,
    extract_B_elems,
    extract_code_id,
    extract_distance_bounds,
    extract_group_spec,
    extract_k,
    extract_n,
    extract_trials,
    load_json,
    utc_now_iso,
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default="best_codes")
    args = ap.parse_args()
    best_dir = Path(args.best_dir)
    meta_dir = best_dir / "meta"
    meta_files = sorted([p for p in meta_dir.glob("*.json") if p.is_file()])

    codes: List[Dict[str, Any]] = []
    for p in meta_files:
        cid_file = p.stem
        try:
            meta = load_json(p)
        except Exception:
            continue

        code_id = extract_code_id(meta, cid_file) or cid_file
        group = extract_group_spec(meta)
        n = extract_n(meta)
        k = extract_k(meta)
        if group is None or n is None or k is None:
            continue

        A = extract_A_elems(meta)
        B = extract_B_elems(meta)
        dX, dZ, d = extract_distance_bounds(meta)
        t_total, _, _ = extract_trials(meta)

        codes.append({
            "code_id": code_id,
            "group": group,
            "n": int(n),
            "k": int(k),
            "d_ub": int(d) if isinstance(d, int) else None,
            "dX_ub": int(dX) if isinstance(dX, int) else None,
            "dZ_ub": int(dZ) if isinstance(dZ, int) else None,
            "m4ri_trials": int(t_total) if isinstance(t_total, int) else None,
            "A_elems": A,
            "B_elems": B,
            "meta": meta,
        })

    def sort_key(r: Dict[str, Any]) -> Tuple:
        d = r.get("d_ub"); t=r.get("m4ri_trials")
        return (r["group"], r["k"], -(d if isinstance(d,int) else -1), -(t if isinstance(t,int) else -1), r["code_id"])
    codes.sort(key=sort_key)

    out = {"generated_at_utc": utc_now_iso(), "total_codes": len(codes), "codes": codes}
    atomic_write_json(best_dir / "data.json", out)

    # index.tsv
    idx = best_dir / "index.tsv"
    with idx.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["group","n","k","d_ub","m4ri_trials","code_id"])
        for r in codes:
            w.writerow([r["group"], r["n"], r["k"], r.get("d_ub"), r.get("m4ri_trials"), r["code_id"]])

    # best_by_group_k.tsv
    best = {}
    for r in codes:
        key=(r["group"], r["k"])
        cur=best.get(key)
        if cur is None:
            best[key]=r
            continue
        d1=r.get("d_ub"); d0=cur.get("d_ub")
        t1=r.get("m4ri_trials"); t0=cur.get("m4ri_trials")
        s1=(-(d1 if isinstance(d1,int) else -1), -(t1 if isinstance(t1,int) else -1))
        s0=(-(d0 if isinstance(d0,int) else -1), -(t0 if isinstance(t0,int) else -1))
        if s1 < s0:
            best[key]=r

    bp = best_dir / "best_by_group_k.tsv"
    with bp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["group","k","n","d_ub","m4ri_trials","code_id"])
        for (g,k), r in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
            w.writerow([g,k,r["n"], r.get("d_ub"), r.get("m4ri_trials"), r["code_id"]])

    print(f"[ok] rebuilt best_codes/data.json with {len(codes)} codes")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
