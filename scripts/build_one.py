#!/usr/bin/env python3
"""Build one lifted quantum Tanner code instance and write outputs."""

from __future__ import annotations

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
    hamming_8_4_4_extended,
    repetition_2,
    variants_6_3_3,
    variants_8_4_4,
)
from qtanner.mtx import write_mtx_from_bitrows


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _base_code(name: str) -> LocalCode:
    if name == "2_1_2":
        return repetition_2()
    if name == "6_3_3":
        return hamming_6_3_3_shortened()
    if name == "8_4_4":
        return hamming_8_4_4_extended()
    raise ValueError(f"Unknown code name: {name}")


def _variants_for(name: str) -> List[List[int]]:
    if name == "6_3_3":
        return variants_6_3_3()
    if name == "8_4_4":
        return variants_8_4_4()
    return []


def _apply_variant(code: LocalCode, name: str, variant_idx: int) -> LocalCode:
    if name == "2_1_2":
        return code
    variants = _variants_for(name)
    if variant_idx < 0 or variant_idx >= len(variants):
        raise ValueError(f"Variant index {variant_idx} out of range for {name}.")
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
    parser = argparse.ArgumentParser(description="Build one lifted Tanner code.")
    parser.add_argument("--out", default="data/tmp/build_one", help="Output directory.")
    parser.add_argument("--group", choices=["cyclic", "smallgroup"], default="cyclic")
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--gid", type=int, help="SmallGroup ID (required for smallgroup).")
    parser.add_argument("--A", required=True, help="Comma-separated A list, e.g. 0,1,2.")
    parser.add_argument("--B", required=True, help="Comma-separated B list, e.g. 0,1.")
    parser.add_argument("--Acode", choices=["2_1_2", "6_3_3", "8_4_4"], default="6_3_3")
    parser.add_argument("--Bcode", choices=["2_1_2", "6_3_3", "8_4_4"], default="2_1_2")
    parser.add_argument("--A1-variant", type=int, default=0)
    parser.add_argument("--B1-variant", type=int, default=0)
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

    C0 = _base_code(args.Acode)
    C0p = _base_code(args.Bcode)
    C1 = _apply_variant(C0, args.Acode, args.A1_variant)
    C1p = _apply_variant(C0p, args.Bcode, args.B1_variant)

    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)
    if not css_commutes(hx_rows, hz_rows):
        raise RuntimeError("HX and HZ do not commute.")

    rank_hx = gf2_rank(hx_rows, n_cols)
    rank_hz = gf2_rank(hz_rows, n_cols)
    k = n_cols - rank_hx - rank_hz

    os.makedirs(args.out, exist_ok=True)
    write_mtx_from_bitrows(os.path.join(args.out, "Hx.mtx"), hx_rows, n_cols)
    write_mtx_from_bitrows(os.path.join(args.out, "Hz.mtx"), hz_rows, n_cols)

    meta = {
        "group": group_meta,
        "A": A,
        "B": B,
        "Acode": args.Acode,
        "Bcode": args.Bcode,
        "A1_variant": args.A1_variant,
        "B1_variant": args.B1_variant,
        "n": n_cols,
        "k": k,
        "hx_rows": len(hx_rows),
        "hz_rows": len(hz_rows),
        "column_order": "col = ((i*nB + j)*|G| + g), g fastest",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(
        "Built code:",
        f"n={n_cols}",
        f"k={k}",
        f"Hx_rows={len(hx_rows)}",
        f"Hz_rows={len(hz_rows)}",
        f"out={args.out}",
    )


if __name__ == "__main__":
    main()
