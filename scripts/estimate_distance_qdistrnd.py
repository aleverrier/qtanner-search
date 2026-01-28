#!/usr/bin/env python3
"""Estimate CSS distance upper bounds from Hx/Hz .mtx files via QDistRnd."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qtanner.qdistrnd import dist_rand_css_mtx


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate CSS distance upper bounds.")
    parser.add_argument("--hx", required=True, help="Path to Hx.mtx.")
    parser.add_argument("--hz", required=True, help="Path to Hz.mtx.")
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--mindist", type=int, default=0)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--maxav", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--gap-cmd", default="gap")
    args = parser.parse_args()

    summary = dist_rand_css_mtx(
        args.hx,
        args.hz,
        num=args.num,
        mindist=args.mindist,
        debug=args.debug,
        maxav=args.maxav,
        seed=args.seed,
        gap_cmd=args.gap_cmd,
        timeout_sec=args.timeout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
