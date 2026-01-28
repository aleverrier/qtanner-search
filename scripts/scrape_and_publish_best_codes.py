#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.best_codes_updater import (
    GitNonFastForwardError,
    run_best_codes_update,
)


def _repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        if out:
            return Path(out)
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def _print_summary(selected, records) -> None:
    eligible = [r for r in records if r.n is not None and r.k is not None and r.d is not None]
    print(f"[summary] scanned={len(records)} eligible={len(eligible)} selected={len(selected)}")
    print("n\tk\td\ttrials\tcode_id\tsource")
    for (n, k), rec in sorted(selected.items(), key=lambda x: (x[0][0], x[0][1])):
        d = rec.d if rec.d is not None else ""
        t = rec.trials if rec.trials is not None else ""
        print(f"{n}\t{k}\t{d}\t{t}\t{rec.code_id}\t{rec.source_kind}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scrape repo for codes (including codes/pending) and publish best_codes updates."
    )
    ap.add_argument("--dry-run", action="store_true", help="Scan + select, but do not write or git.")
    ap.add_argument("--no-git", action="store_true", help="Skip git pull/commit/push.")
    ap.add_argument("--no-publish", action="store_true", help="Skip website data.json/index updates.")
    ap.add_argument("--verbose", action="store_true", help="Verbose scan/sync logging.")
    ap.add_argument("--max-attempts", type=int, default=3, help="Max push attempts if non-fast-forward.")
    args = ap.parse_args(argv)

    root = _repo_root()

    try:
        result = run_best_codes_update(
            root,
            dry_run=args.dry_run,
            no_git=args.no_git,
            no_publish=args.no_publish,
            verbose=args.verbose,
            max_attempts=args.max_attempts,
        )
        _print_summary(result.selected, result.records)
        return 0
    except GitNonFastForwardError as exc:
        print(f"[error] push failed after {args.max_attempts} attempts: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
