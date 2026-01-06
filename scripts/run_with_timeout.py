#!/usr/bin/env python3
"""
Run a command with a hard wall-clock timeout.

Usage:
  python scripts/run_with_timeout.py --seconds 180 --log results/logs/run.log -- <command> [args...]

Exit codes:
  - returns the command's exit code if it finishes in time
  - returns 124 on timeout
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0, help="Timeout in seconds.")
    ap.add_argument("--kill-delay", type=float, default=5.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    ap.add_argument("--log", type=str, default=None, help="Optional log file path (stdout+stderr).")
    ap.add_argument("--cwd", type=str, default=None, help="Optional working directory.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (put after --).")
    args = ap.parse_args()

    cmd = list(args.cmd)

    # argparse includes the literal "--" in REMAINDER; strip it if present.
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        ap.error("No command provided. Example: -- python scripts/run_search_w9_smallgroups.py")

    log_fh = None
    stdout = None
    stderr = None

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        stdout = log_fh
        stderr = log_fh

    def write_msg(msg: str) -> None:
        if log_fh is not None:
            log_fh.write(msg)
            log_fh.flush()
        else:
            sys.stderr.write(msg)
            sys.stderr.flush()

    try:
        # start_new_session=True puts the child in its own process group (POSIX),
        # so we can terminate the whole group on timeout.
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

        try:
            return proc.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            write_msg(f"\n[TIMEOUT] exceeded {args.seconds:.1f}s, terminating: {' '.join(cmd)}\n")

            # Terminate process group
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return 124

            try:
                proc.wait(timeout=args.kill_delay)
            except subprocess.TimeoutExpired:
                write_msg("[TIMEOUT] SIGTERM did not stop it; sending SIGKILL.\n")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            return 124
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
