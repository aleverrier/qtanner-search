#!/usr/bin/env python3
"""Build one lifted quantum Tanner code instance and write outputs."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.gf2 import gf2_rank
from qtanner.gap_groups import smallgroup
from qtanner.group import FiniteGroup
from qtanner.lift_matrices import build_hx_hz, css_commutes
from qtanner.local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from qtanner.mtx import write_mtx_from_bitrows
from qtanner.qdistrnd import dist_rand_css_mtx


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def _apply_variant(code: LocalCode, variant_idx: int) -> LocalCode:
    variants = variants_6_3_3()
    if variant_idx < 0 or variant_idx >= len(variants):
        raise ValueError(f"Variant index {variant_idx} out of range for 6_3_3.")
    perm = variants[variant_idx]
    H_rows = apply_col_perm_to_rows(code.H_rows, perm, code.n)
    G_rows = apply_col_perm_to_rows(code.G_rows, perm, code.n)
    return LocalCode(
        name=f"{code.name}_var{variant_idx}",
        n=code.n,
        k=code.k,
        H_rows=H_rows,
        G_rows=G_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one lifted [6,3,3]x[6,3,3] Tanner code."
    )
    parser.add_argument("--out", default="data/tmp/build_one", help="Output directory.")
    parser.add_argument("--group", choices=["cyclic", "smallgroup"], default="cyclic")
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--gid", type=int, help="SmallGroup ID (required for smallgroup).")
    parser.add_argument("--A", required=True, help="Comma-separated A list, e.g. 0,1,2.")
    parser.add_argument("--B", required=True, help="Comma-separated B list, e.g. 0,1.")
    parser.add_argument("--A1-variant", type=int, default=0)
    parser.add_argument("--B1-variant", type=int, default=0)
    parser.add_argument("--qdistrnd", action="store_true", help="Estimate distances via GAP/QDistRnd.")
    parser.add_argument("--qd-num", type=int, default=50000)
    parser.add_argument("--qd-mindist", type=int, default=0)
    parser.add_argument("--qd-debug", type=int, default=2)
    parser.add_argument("--qd-maxav", type=float, default=None)
    parser.add_argument("--qd-seed", type=int, default=None)
    parser.add_argument("--qd-timeout", type=float, default=None)
    args = parser.parse_args()

    A = _parse_int_list(args.A)
    B = _parse_int_list(args.B)

    if args.group == "cyclic":
        group = FiniteGroup.cyclic(args.order)
        group_meta = {"type": "cyclic", "order": args.order}
    else:
        if args.gid is None:
            raise ValueError("--gid is required for smallgroup.")
        group = smallgroup(args.order, args.gid)
        group_meta = {"type": "smallgroup", "order": args.order, "gid": args.gid}

    C0 = hamming_6_3_3_shortened()
    C0p = hamming_6_3_3_shortened()
    C1 = _apply_variant(C0, args.A1_variant)
    C1p = _apply_variant(C0p, args.B1_variant)

    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)
    if not css_commutes(hx_rows, hz_rows):
        raise RuntimeError("HX and HZ do not commute.")

    rank_hx = gf2_rank(hx_rows[:], n_cols)
    rank_hz = gf2_rank(hz_rows[:], n_cols)
    k = n_cols - rank_hx - rank_hz

    os.makedirs(args.out, exist_ok=True)
    hx_path = os.path.join(args.out, "Hx.mtx")
    hz_path = os.path.join(args.out, "Hz.mtx")
    write_mtx_from_bitrows(hx_path, hx_rows, n_cols)
    write_mtx_from_bitrows(hz_path, hz_rows, n_cols)

    meta = {
        "group": group_meta,
        "A": A,
        "B": B,
        "local_codes": {
            "C0": C0.name,
            "C1": C1.name,
            "C0p": C0p.name,
            "C1p": C1p.name,
            "a1v": args.A1_variant,
            "b1v": args.B1_variant,
        },
        "n": n_cols,
        "k": k,
        "hx_rows": len(hx_rows),
        "hz_rows": len(hz_rows),
        "column_order": "col = ((i*nB + j)*|G| + g), g fastest",
        "distance_estimate": {
            "method": "QDistRnd" if args.qdistrnd else "none",
            "trials": args.qd_num if args.qdistrnd else 0,
            "rng_seed": args.qd_seed,
        },
    }

    if args.qdistrnd:
        log_path = os.path.join(args.out, "qdistrnd.log")
        try:
            qd_summary = dist_rand_css_mtx(
                hx_path,
                hz_path,
                num=args.qd_num,
                mindist=args.qd_mindist,
                debug=args.qd_debug,
                maxav=args.qd_maxav,
                seed=args.qd_seed,
                timeout_sec=args.qd_timeout,
                log_path=log_path,
            )
        except RuntimeError:
            print(f"QDistRnd failed; see {log_path}")
            sys.exit(2)
        meta["qdistrnd"] = {**qd_summary, "log_file": "qdistrnd.log"}
        print(
            "QDistRnd upper bounds:",
            f"dX_ub={qd_summary['dX_ub']}",
            f"dZ_ub={qd_summary['dZ_ub']}",
            f"d_ub={qd_summary['d_ub']}",
            f"runtime_sec={qd_summary['runtime_sec']:.2f}",
        )
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(
        "Built code:",
        f"n={n_cols}",
        f"k={k}",
        f"hx_rank={rank_hx}",
        f"hz_rank={rank_hz}",
        f"Hx_rows={len(hx_rows)}",
        f"Hz_rows={len(hz_rows)}",
        f"out={args.out}",
    )


if __name__ == "__main__":
    main()
