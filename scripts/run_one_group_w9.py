#!/usr/bin/env python3
"""
scripts/run_one_group_w9.py

Thin wrapper around scripts/run_search_w9_smallgroups.py to support:
- --group / --only-group
- --pairs like 1-200 (maps to --max-pairs)
- --tmpdir / --outdir by *temporarily symlinking* the fixed paths that
  run_search_w9_smallgroups.py uses internally.

It also accepts --top-per-nk for compatibility but does not forward it to
run_search_w9_smallgroups.py (which does not know that flag).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple


DEFAULT_BASE_TMP = Path("data/tmp/search_w9_smallgroups")
DEFAULT_BASE_OUT = Path("data/results")


def _parse_pairs_arg(pairs: str) -> Tuple[int, int]:
    s = pairs.strip()
    if "-" in s:
        a_s, b_s = s.split("-", 1)
        a, b = int(a_s), int(b_s)
        if a <= 0 or b <= 0:
            raise ValueError("pairs endpoints must be positive")
        if a > b:
            raise ValueError("pairs must be in increasing order, e.g. 1-200")
        return a, b
    # single integer means 1-N
    n = int(s)
    if n <= 0:
        raise ValueError("pairs must be positive")
    return 1, n


def _ensure_symlink(path: Path, target: Path) -> Optional[Path]:
    """
    Ensure `path` is a symlink to `target`. If `path` exists and is not the right
    symlink, move it aside and return the backup path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)

    if path.exists() or path.is_symlink():
        # If it's already the desired symlink, do nothing.
        if path.is_symlink():
            try:
                cur = path.resolve()
            except OSError:
                cur = None
            if cur is not None and cur == target.resolve():
                return None

        # Move aside (directory/file/symlink) to backup
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = path.with_name(path.name + f".bak_{ts}")
        path.rename(backup)
    else:
        backup = None

    # Create symlink
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing path: {path}")
    path.symlink_to(target, target_is_directory=True)
    return backup


@contextmanager
def _temp_symlink(path: Path, target: Path) -> Iterator[None]:
    # If the caller points tmpdir/outdir to the default base path, don't create a self-symlink.
    if path.resolve() == target.resolve():
        path.mkdir(parents=True, exist_ok=True)
        yield
        return

    backup = _ensure_symlink(path, target)
    try:
        yield
    finally:
        # Remove the symlink we created (if still a symlink).
        try:
            if path.is_symlink():
                path.unlink()
        finally:
            # Restore backup if any.
            if backup is not None and backup.exists():
                backup.rename(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only-group", "--group", dest="only_group", required=True, help="SmallGroup id as 'order,gid' (e.g. 4,2)")
    p.add_argument("--pairs", default="1-200", help="Pair range like 1-200 (maps to --max-pairs)")
    p.add_argument("--tmpdir", default=str(DEFAULT_BASE_TMP), help="Where to store temp candidate dirs")
    p.add_argument("--outdir", default=str(DEFAULT_BASE_OUT), help="Where to store result files (if any)")
    p.add_argument("--trial-cap", type=int, default=0, help="Optional cap on number of pairs tried (0 disables)")
    p.add_argument("--top-per-nk", type=int, default=0, help="(compat) accepted but not forwarded to run_search_w9_smallgroups.py")
    args, extra = p.parse_known_args()

    # Derive max_pairs from --pairs and optional trial-cap
    _, p_hi = _parse_pairs_arg(args.pairs)
    max_pairs = p_hi
    if args.trial_cap and args.trial_cap > 0:
        max_pairs = min(max_pairs, args.trial_cap)

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_search_w9_smallgroups.py"
    if not script.exists():
        print(f"[run_one_group_w9] ERROR: cannot find {script}", file=sys.stderr)
        return 2

    # We rely on run_search using DEFAULT_BASE_TMP/OUT. We temporarily symlink them.
    tmp_target = Path(args.tmpdir).resolve()
    out_target = Path(args.outdir).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--only-group",
        args.only_group,
        "--max-pairs",
        str(max_pairs),
        *extra,
    ]
    print("[run_one_group_w9] cmd:", " ".join(cmd))

    with _temp_symlink(repo_root / DEFAULT_BASE_TMP, tmp_target), _temp_symlink(repo_root / DEFAULT_BASE_OUT, out_target):
        proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
