#!/usr/bin/env python3
"""Run a deeper (more exhaustive) w=9 search for a single SmallGroup.

This is a convenience wrapper around scripts/run_search_w9_smallgroups.py.

Typical usage:

  ./scripts/py scripts/run_one_group_w9_deep.py --only-group 4,2
  ./scripts/py scripts/run_one_group_w9_deep.py --only-group 4,2 --slice-d-min 9

Any extra arguments are forwarded to run_search_w9_smallgroups.py.
"""
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
        description="Run a deeper (more exhaustive) w=9 search for a single SmallGroup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only-group",
        required=True,
        help="Group identifier as 'order,gid' (required).",
    )
    parser.add_argument(
        "--slice-d-min",
        type=int,
        default=None,
        help="Optional slice distance floor used by the slice filter.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level forwarded to the main search script.",
    )
    args, extra_args = parser.parse_known_args()

    # Heavier defaults than scripts/run_one_group_w9.py.
    #
    # Notes:
    # - Keep GAP warm by default.
    # - Use ~2k QDistRnd trials by default (enough for n=144 in practice).
    # - Explore more A/B generator multisets and more (a1v,b1v) permutations.
    deep_defaults = [
        "--use-gap-session",
        "--use-slice-filter",
        "--slice-fast-num",
        "30",
        "--max-A",
        "200",
        "--max-B",
        "200",
        "--max-pairs",
        "5000",
        "--a1v-max",
        "30",
        "--b1v-max",
        "30",
        "--qd-num",
        "2000",
        "--qd-batches",
        "20,200,2000",
        "--qd-timeout",
        "120",
    ]

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--only-group",
        args.only_group,
        *deep_defaults,
        "--verbose",
        str(args.verbose),
    ]
    if args.slice_d_min is not None:
        cmd.extend(["--slice-d-min", str(args.slice_d_min)])
    cmd.extend(extra_args)

    # Replace the current process (so CTRL-C behaves like the main script).
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
