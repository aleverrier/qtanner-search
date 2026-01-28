#!/usr/bin/env python3
"""Run a fast w=9 search for a single SmallGroup."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_search_w9_smallgroups.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fast w=9 search for a single SmallGroup.",
        epilog=(
            "Example:\n"
            "  scripts/run_one_group_w9.py --only-group 12,5\n"
            "  scripts/run_one_group_w9.py --only-group 12,5 --qd-num 500"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only-group",
        "--group",
        dest="only_group",
        required=True,
        help="Group identifier as 'order,gid' (required).",
    )
    parser.add_argument("--slice-d-min", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args, extra_args = parser.parse_known_args()

    fast_defaults = [
        "--use-slice-filter",
        "--max-A",
        "8",
        "--max-B",
        "8",
        "--max-pairs",
        "10",
        "--qd-num",
        "200",
        "--qd-batches",
        "50,200",
    ]
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--only-group",
        args.only_group,
        *fast_defaults,
        "--verbose",
        str(args.verbose),
    ]
    if args.slice_d_min is not None:
        cmd.extend(["--slice-d-min", str(args.slice_d_min)])
    cmd.extend(extra_args)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
