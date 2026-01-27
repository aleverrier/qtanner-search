#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.best_codes_updater import (
    GitNonFastForwardError,
    scan_all_codes,
    select_best_by_nk,
    sync_best_codes_folder,
    update_best_codes_webpage_data,
    git_commit_and_push,
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


def _git_pull_rebase(root: Path, verbose: bool) -> None:
    cmd = ["git", "pull", "--rebase", "--autostash"]
    if verbose:
        print("[git] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))


def _commit_message() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"best_codes: refresh best-by-nk ({ts})"


def _run_once(root: Path, args) -> Tuple[dict, List]:
    records = scan_all_codes(root, verbose=args.verbose)
    selected = select_best_by_nk(records)

    _print_summary(selected, records)

    if args.dry_run:
        return selected, records

    sync_best_codes_folder(selected, root, dry_run=False, verbose=args.verbose)
    if not args.no_publish:
        update_best_codes_webpage_data(selected, root, dry_run=False, verbose=args.verbose)

    return selected, records


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape repo for codes and publish best_codes updates.")
    ap.add_argument("--dry-run", action="store_true", help="Scan + select, but do not write or git.")
    ap.add_argument("--no-git", action="store_true", help="Skip git pull/commit/push.")
    ap.add_argument("--no-publish", action="store_true", help="Skip website data.json/index updates.")
    ap.add_argument("--verbose", action="store_true", help="Verbose scan/sync logging.")
    ap.add_argument("--max-attempts", type=int, default=3, help="Max push attempts if non-fast-forward.")
    args = ap.parse_args(argv)

    root = _repo_root()

    attempts = 0
    while True:
        attempts += 1
        _run_once(root, args)

        if args.dry_run or args.no_git:
            return 0

        _git_pull_rebase(root, args.verbose)

        try:
            git_commit_and_push(root, _commit_message(), retry_on_nonfastforward=True)
            return 0
        except GitNonFastForwardError as exc:
            if attempts >= args.max_attempts:
                print(f"[error] push failed after {attempts} attempts: {exc}", file=sys.stderr)
                return 2
            if args.verbose:
                print(f"[git] non-fast-forward, retrying (attempt {attempts}/{args.max_attempts})")
            _git_pull_rebase(root, args.verbose)
            # Recompute selection after rebasing onto remote changes.
            continue


if __name__ == "__main__":
    raise SystemExit(main())
