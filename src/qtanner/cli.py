from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "scripts").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def _run(cmd: List[str]) -> int:
    return subprocess.call(cmd, cwd=str(_repo_root()))


def _run_publish() -> int:
    root = _repo_root()
    collect = root / "scripts" / "collect_best_codes.py"
    generate = root / "scripts" / "generate_best_codes_site.py"
    ret = _run([sys.executable, str(collect)])
    if ret != 0:
        return ret
    return _run([sys.executable, str(generate)])


def _run_search_progressive(extra: List[str]) -> int:
    cmd = [sys.executable, "-m", "qtanner.search", "progressive", *extra]
    return _run(cmd)


def _run_search_exhaustive(extra: List[str]) -> int:
    cmd = [sys.executable, "-m", "qtanner.search", *extra]
    return _run(cmd)


def _run_refine_length(n: int, trials: int, extra: List[str]) -> int:
    root = _repo_root()
    script = root / "scripts" / "refine_best_codes_length.py"
    cmd = [
        sys.executable,
        str(script),
        "--n",
        str(n),
        "--trials-per-side",
        str(trials),
        *extra,
    ]
    return _run(cmd)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="qtanner")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search for quantum Tanner codes.")
    search_sub = search.add_subparsers(dest="mode", required=True)
    prog = search_sub.add_parser(
        "progressive",
        help="Progressive search (wraps qtanner.search progressive).",
    )
    prog.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publish step after search.",
    )
    exhaustive = search_sub.add_parser(
        "exhaustive",
        help="Exhaustive/pilot search (wraps qtanner.search).",
    )
    exhaustive.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publish step after search.",
    )

    refine = sub.add_parser("refine", help="Refine distances for published codes.")
    refine_sub = refine.add_subparsers(dest="mode", required=True)
    length = refine_sub.add_parser(
        "length",
        help="Refine distances for all published codes of a given length n.",
    )
    length.add_argument("--n", type=int, required=True)
    length.add_argument(
        "--trials",
        "--trials-per-side",
        dest="trials",
        type=int,
        required=True,
    )
    length.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publish step after refine.",
    )

    publish = sub.add_parser("publish", help="Update best_codes and website artifacts.")

    args, extra = parser.parse_known_args(argv)

    if args.command == "search":
        if args.mode == "progressive":
            ret = _run_search_progressive(extra)
        else:
            ret = _run_search_exhaustive(extra)
        if ret != 0 or args.no_publish:
            return ret
        return _run_publish()

    if args.command == "refine":
        if args.mode != "length":
            raise SystemExit("Unsupported refine mode.")
        ret = _run_refine_length(args.n, args.trials, extra)
        if ret != 0 or args.no_publish:
            return ret
        return _run_publish()

    if args.command == "publish":
        return _run_publish()

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
