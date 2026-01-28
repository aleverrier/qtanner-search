#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qtanner.progressive_search import progressive_main  # noqa: E402


def _repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT),
            text=True,
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass

    for parent in [ROOT, *ROOT.parents]:
        if (parent / ".git").exists():
            return parent
    return ROOT


def _parse_best_codes_flags(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-best-codes-update", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--best-codes-no-history", action="store_true")
    parser.add_argument("--best-codes-max-attempts", type=int, default=3)
    flags, _ = parser.parse_known_args(argv)
    if flags.best_codes_max_attempts <= 0:
        raise SystemExit("--best-codes-max-attempts must be positive.")
    return flags


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    flags = _parse_best_codes_flags(args)

    exit_code = 1
    run_exc: BaseException | None = None
    try:
        exit_code = progressive_main(args)
    except BaseException as exc:  # pragma: no cover - passthrough for CLI
        run_exc = exc
        raise
    finally:
        if flags.no_best_codes_update:
            pass
        elif isinstance(run_exc, SystemExit) and (run_exc.code in (0, None)):
            # --help exits cleanly; avoid noisy post-run logs.
            pass
        elif isinstance(run_exc, KeyboardInterrupt):
            print(
                "[best_codes] interrupted; running best_codes update anyway.",
                file=sys.stderr,
            )
            try:
                from qtanner.best_codes_updater import run_best_codes_update

                repo_root = _repo_root()
                include_history = not flags.best_codes_no_history
                result = run_best_codes_update(
                    repo_root,
                    dry_run=False,
                    no_git=flags.no_git,
                    no_publish=flags.no_publish,
                    include_git_history=include_history,
                    max_attempts=flags.best_codes_max_attempts,
                )
                print(
                    "[best_codes] done "
                    f"scanned={len(result.records)} selected={len(result.selected)} "
                    f"attempts={result.attempts} committed={'yes' if result.committed else 'no'}"
                )
            except Exception as exc:  # pragma: no cover - best-effort post step
                print(f"[best_codes] update failed: {exc}", file=sys.stderr)
        elif run_exc is not None:
            print(
                "[best_codes] skipping update because the search failed.",
                file=sys.stderr,
            )
        elif exit_code != 0:
            print(
                "[best_codes] skipping update due to nonzero exit code.",
                file=sys.stderr,
            )
        else:
            try:
                # Lazy import so the search is unaffected when updates are disabled.
                from qtanner.best_codes_updater import run_best_codes_update

                repo_root = _repo_root()
                include_history = not flags.best_codes_no_history
                print(
                    "[best_codes] updating best_codes "
                    f"(history={'on' if include_history else 'off'} "
                    f"publish={'off' if flags.no_publish else 'on'} "
                    f"git={'off' if flags.no_git else 'on'})"
                )
                result = run_best_codes_update(
                    repo_root,
                    dry_run=False,
                    no_git=flags.no_git,
                    no_publish=flags.no_publish,
                    include_git_history=include_history,
                    max_attempts=flags.best_codes_max_attempts,
                )
                print(
                    "[best_codes] done "
                    f"scanned={len(result.records)} selected={len(result.selected)} "
                    f"attempts={result.attempts} committed={'yes' if result.committed else 'no'}"
                )
            except Exception as exc:  # pragma: no cover - best-effort post step
                print(f"[best_codes] update failed: {exc}", file=sys.stderr)
                if exit_code == 0:
                    exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
