"""Tiny end-to-end build/commutation check for [6,3,3]x[6,3,3] codes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from .classical_distance import analyze_parity_check_bitrows
from .gf2 import gf2_rank
from .group import FiniteGroup
from .lift_matrices import build_hx_hz, css_commutes
from .local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from .mtx import write_mtx_from_bitrows


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


def run_smoke(
    *,
    out_dir: Path | None,
    a1v: int,
    b1v: int,
    seed: int | None,
) -> int:
    tmp_dir = None
    created_out_dir = False
    if out_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        out_path = Path(tmp_dir.name)
    else:
        out_path = out_dir
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            created_out_dir = True

    group = FiniteGroup.cyclic(1)
    A = [0, 0, 0, 0, 0, 0]
    B = [0, 0, 0, 0, 0, 0]

    base_code = hamming_6_3_3_shortened()
    C1 = _apply_variant(base_code, a1v)
    C1p = _apply_variant(base_code, b1v)
    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, base_code, C1, base_code, C1p)
    if not css_commutes(hx_rows, hz_rows):
        raise RuntimeError("HX and HZ do not commute.")

    rank_hx = gf2_rank(hx_rows[:], n_cols)
    rank_hz = gf2_rank(hz_rows[:], n_cols)
    k = n_cols - rank_hx - rank_hz

    hx_path = out_path / "Hx.mtx"
    hz_path = out_path / "Hz.mtx"
    write_mtx_from_bitrows(str(hx_path), hx_rows, n_cols)
    write_mtx_from_bitrows(str(hz_path), hz_rows, n_cols)

    dx_stats = analyze_parity_check_bitrows(
        hx_rows, n_cols, max_k_exact=18, samples=2000, seed=seed
    )
    dz_stats = analyze_parity_check_bitrows(
        hz_rows, n_cols, max_k_exact=18, samples=2000, seed=seed
    )
    d_lb = min(dx_stats.d, dz_stats.d)
    target = math.isqrt(n_cols)
    promising = d_lb >= target and (k * d_lb >= n_cols)
    trials_total = dx_stats.codewords_checked + dz_stats.codewords_checked

    meta = {
        "group": {"type": "cyclic", "order": group.order},
        "A": A,
        "B": B,
        "local_codes": {
            "C0": base_code.name,
            "C1": C1.name,
            "C0p": base_code.name,
            "C1p": C1p.name,
            "a1v": a1v,
            "b1v": b1v,
        },
        "n": n_cols,
        "k": k,
        "distance_estimate": {
            "method": "classical_lower_bound",
            "trials": trials_total,
            "rng_seed": seed,
            "details": {"hx": dx_stats.as_dict(), "hz": dz_stats.as_dict()},
        },
        "distance_lower_bound": d_lb,
        "promising": promising,
    }

    meta_path = out_path / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(
        "tiny-instance:",
        f"n={n_cols}",
        f"k={k}",
        f"d_lb={d_lb}",
        f"target={target}",
        f"promising={promising}",
        f"out={out_path}",
    )

    if not promising:
        if out_dir is not None:
            for path in (hx_path, hz_path, meta_path):
                if path.exists():
                    path.unlink()
            if created_out_dir:
                try:
                    out_path.rmdir()
                except OSError:
                    pass
            print("Not promising; outputs removed.", file=sys.stderr)
            return 2
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return 0

    if tmp_dir is not None:
        tmp_dir.cleanup()
        print("Promising code found, but outputs were temporary.", file=sys.stderr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a tiny [6,3,3]x[6,3,3] instance and assert HX*HZ^T=0."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory to keep results (only if promising).",
    )
    parser.add_argument("--a1v", type=int, default=0, help="Variant index for C1.")
    parser.add_argument("--b1v", type=int, default=0, help="Variant index for C1p.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for sampling.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    return run_smoke(out_dir=out_dir, a1v=args.a1v, b1v=args.b1v, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
