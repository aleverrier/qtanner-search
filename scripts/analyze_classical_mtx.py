#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import json
from pathlib import Path
from typing import List

from qtanner.classical_distance import analyze_mtx_parity_check


def _fmt_bool(b: bool) -> str:
    return "yes" if b else "no"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast (often exact) distance computation for small binary classical codes given by a parity-check matrix in MatrixMarket (.mtx)."
    )
    ap.add_argument("mtx", nargs="+", help="Path(s) to MatrixMarket coordinate .mtx file(s) (binary/GF(2)).")
    ap.add_argument("--stop-at", type=int, default=None, help="Early stop if a codeword of weight <= STOP_AT is found.")
    ap.add_argument("--max-k-exact", type=int, default=16, help="Exact enumeration when k <= MAX_K_EXACT (default: 16).")
    ap.add_argument("--samples", type=int, default=20000, help="Number of random samples when k > max-k-exact (default: 20000).")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for sampling mode.")
    ap.add_argument("--json", action="store_true", help="Output JSON (one object per input file).")

    args = ap.parse_args(argv)

    analyses = []
    for p_str in args.mtx:
        p = Path(p_str)
        a = analyze_mtx_parity_check(
            p,
            stop_at=args.stop_at,
            max_k_exact=args.max_k_exact,
            samples=args.samples,
            seed=args.seed,
        )
        rec = {"path": str(p), **a.as_dict()}
        analyses.append(rec)

    if args.json:
        for rec in analyses:
            print(json.dumps(rec, sort_keys=True))
        return 0

    # Human table
    print("path\tm\tn\trank\tk\td\texact\tchecked\tmethod")
    for rec in analyses:
        print(
            f"{rec['path']}\t{rec['m']}\t{rec['n']}\t{rec['rank']}\t{rec['k']}\t{rec['d']}\t{_fmt_bool(rec['exact'])}\t{rec['codewords_checked']}\t{rec['method']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
