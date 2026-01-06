#!/usr/bin/env python3
"""
Run a single-group w=9 search by delegating to scripts/run_search_w9_smallgroups.py, while
providing a stable CLI and optional tmp/results directory redirection.

This wrapper exists because run_search_w9_smallgroups.py currently does *not* accept --tmpdir/--outdir,
and some higher-level drivers (e.g. run_one_group_w9_twopass.py) want to separate runs.

Key features:
  - Accepts --group as an alias for --only-group.
  - Accepts --pairs START-END (e.g. 1-200) and maps it to --max-pairs END for run_search_w9_smallgroups.py.
    (If START != 1, we warn because upstream currently cannot skip pairs efficiently.)
  - Accepts --tmpdir and --outdir; implements them by temporarily redirecting the default
    data/tmp/search_w9_smallgroups and data/results paths via symlinks (restored on exit).
  - Ensures PYTHONPATH includes <repo>/src for the subprocess, so qtanner imports work without pip install.

Anthony: copy this file to scripts/run_one_group_w9.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_TMP_REL = Path("data/tmp/search_w9_smallgroups")
DEFAULT_OUT_REL = Path("data/results")


def _repo_root_from_this_file() -> Path:
    # scripts/run_one_group_w9.py -> repo root is parent of "scripts"
    return Path(__file__).resolve().parents[1]


def _parse_pairs_range(pairs: str) -> Tuple[int, int]:
    """
    Accepts "A-B" or "A:B" or "B" (interpreted as 1-B).
    Returns (start, end), inclusive.
    """
    s = pairs.strip()
    if not s:
        raise ValueError("empty --pairs")
    if "-" in s:
        a, b = s.split("-", 1)
    elif ":" in s:
        a, b = s.split(":", 1)
    else:
        a, b = "1", s
    start = int(a)
    end = int(b)
    if start < 1 or end < 1:
        raise ValueError("--pairs values must be >= 1")
    if end < start:
        raise ValueError(f"--pairs end ({end}) must be >= start ({start})")
    return start, end


@dataclass
class _RedirectState:
    existed: bool
    was_symlink: bool
    old_link_target: Optional[str]  # only if was_symlink
    backup_path: Optional[Path]     # only if existed and not symlink


class _TempSymlinkRedirect:
    """
    Context manager that temporarily makes link_path a symlink to target_dir.

    If link_path already exists:
      - If it's a symlink, we swap it (and restore original target).
      - If it's a real dir/file, we move it aside to a timestamped backup name,
        create the symlink, then restore on exit.
    """
    def __init__(self, link_path: Path, target_dir: Optional[Path]):
        self.link_path = link_path
        self.target_dir = target_dir
        self.state: Optional[_RedirectState] = None

    def __enter__(self):
        if self.target_dir is None:
            self.state = _RedirectState(existed=False, was_symlink=False, old_link_target=None, backup_path=None)
            return self

        target = self.target_dir
        link = self.link_path

        link_parent = link.parent
        link_parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)

        existed = link.exists() or link.is_symlink()
        if existed and link.is_symlink():
            old_target = os.readlink(link)
            self.state = _RedirectState(existed=True, was_symlink=True, old_link_target=old_target, backup_path=None)
            link.unlink()
        elif existed:
            ts = time.strftime("%Y%m%dT%H%M%S")
            backup = link.with_name(link.name + f"__bak_{ts}")
            # Avoid collisions
            i = 0
            while backup.exists() or backup.is_symlink():
                i += 1
                backup = link.with_name(link.name + f"__bak_{ts}_{i}")
            link.rename(backup)
            self.state = _RedirectState(existed=True, was_symlink=False, old_link_target=None, backup_path=backup)
        else:
            self.state = _RedirectState(existed=False, was_symlink=False, old_link_target=None, backup_path=None)

        # Create the new symlink
        os.symlink(str(target), str(link))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.target_dir is None or self.state is None:
            return False

        link = self.link_path
        # Remove the temporary symlink (if it exists)
        try:
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                # Should not happen, but be defensive
                if link.is_dir():
                    for p in link.iterdir():
                        # We do not delete user data; just leave it.
                        pass
                link.unlink()
        except FileNotFoundError:
            pass

        # Restore prior state
        st = self.state
        if st.existed and st.was_symlink and st.old_link_target is not None:
            os.symlink(st.old_link_target, str(link))
        elif st.existed and (not st.was_symlink) and st.backup_path is not None:
            st.backup_path.rename(link)

        return False


def _build_env(repo_root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    src = repo_root / "src"
    # Prepend repo/src to PYTHONPATH if not already present.
    pp = env.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p]
    src_s = str(src)
    if src_s not in parts:
        parts = [src_s] + parts
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = _repo_root_from_this_file()

    p = argparse.ArgumentParser(
        description="Wrapper around run_search_w9_smallgroups.py with tmp/results redirection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary selector
    p.add_argument("--only-group", dest="only_group", help="Group identifier, e.g. '4,2'.")
    p.add_argument("--group", dest="only_group", help="Alias for --only-group (preferred).")

    # Optional convenience args (handled here, not passed to run_search)
    p.add_argument("--pairs", default=None, help="Pair range like '1-200' (mapped to --max-pairs END).")
    p.add_argument("--tmpdir", default=None, help="Redirect default tmp dir to this path for this run.")
    p.add_argument("--outdir", default=None, help="Redirect default results dir to this path for this run.")
    p.add_argument("--trial-cap", type=int, default=None, help="Accepted for compatibility; currently ignored.")
    p.add_argument("--top-per-nk", type=int, default=None, help="Accepted for compatibility; currently ignored.")

    # Everything else should be passed through to run_search_w9_smallgroups.py
    args, extra = p.parse_known_args(argv)

    if not args.only_group:
        p.error("missing required --group/--only-group")

    # Resolve requested redirections
    tmp_target: Optional[Path] = None
    out_target: Optional[Path] = None

    if args.tmpdir:
        tmp_target = (repo_root / args.tmpdir).resolve() if not Path(args.tmpdir).is_absolute() else Path(args.tmpdir).resolve()
        default_tmp = (repo_root / DEFAULT_TMP_REL).resolve()
        if tmp_target == default_tmp:
            tmp_target = None  # no-op

    if args.outdir:
        out_target = (repo_root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
        default_out = (repo_root / DEFAULT_OUT_REL).resolve()
        if out_target == default_out:
            out_target = None  # no-op

    # Map --pairs to --max-pairs (END)
    if args.pairs:
        try:
            start, end = _parse_pairs_range(args.pairs)
        except ValueError as e:
            p.error(str(e))
        # Upstream can only cap, not offset.
        if start != 1:
            print(
                f"[run_one_group_w9] WARNING: --pairs {start}-{end} requested, but upstream run_search_w9_smallgroups.py "
                f"cannot skip to pair {start}. Running up to --max-pairs {end} instead.",
                file=sys.stderr,
            )
        # Add/override --max-pairs in extra args (ensure it appears last)
        # Remove any existing --max-pairs occurrences in extra to avoid confusion.
        cleaned: List[str] = []
        it = iter(extra)
        for tok in it:
            if tok == "--max-pairs":
                # skip its value
                try:
                    next(it)
                except StopIteration:
                    break
                continue
            cleaned.append(tok)
        cleaned += ["--max-pairs", str(end)]
        extra = cleaned

    # Always ensure slice filter is enabled unless user explicitly disables it (no flag exists),
    # so we just add it if not present.
    if "--use-slice-filter" not in extra:
        extra = ["--use-slice-filter"] + extra

    run_search = repo_root / "scripts" / "run_search_w9_smallgroups.py"
    if not run_search.exists():
        print(f"[run_one_group_w9] ERROR: expected {run_search} to exist.", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(run_search), "--only-group", args.only_group] + extra

    print("[run_one_group_w9] exec:", " ".join(cmd))
    env = _build_env(repo_root)

    # Apply temporary redirections (if requested)
    default_tmp_link = (repo_root / DEFAULT_TMP_REL).resolve()
    default_out_link = (repo_root / DEFAULT_OUT_REL).resolve()

    with _TempSymlinkRedirect(default_tmp_link, tmp_target), _TempSymlinkRedirect(default_out_link, out_target):
        import subprocess

        proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
