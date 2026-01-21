#!/usr/bin/env python3
"""
Run the progressive search from the repo, even without installing the package.

This wrapper is intentionally non-silent:
- It tries to find and call a CLI entrypoint in qtanner.progressive_search.
- It prefers progressive_main() (which exists in this repo).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _try_call(fn) -> None:
    """
    Try a few common calling conventions:
    - fn(sys.argv[1:]) if it accepts >=1 parameter
    - fn() otherwise
    Also fallback if the signature inspection lies (C-accelerated callables etc.).
    """
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) >= 1:
            fn(sys.argv[1:])
        else:
            fn()
        return
    except Exception:
        # Fallback attempts
        try:
            fn(sys.argv[1:])
            return
        except Exception:
            fn()
            return


def main() -> int:
    mod = importlib.import_module("qtanner.progressive_search")

    # Prefer the known entrypoint name in this repo.
    preferred = ["progressive_main"]

    # Common alternative names (keep for future refactors).
    others = ["main", "cli", "run_cli", "entrypoint", "run", "prog_main", "progressive_search_main"]

    for name in preferred + others:
        fn = getattr(mod, name, None)
        if callable(fn):
            _try_call(fn)
            return 0

    # No entrypoint: print diagnostics
    print("ERROR: qtanner.progressive_search has no callable CLI entrypoint.", file=sys.stderr)
    print("Looked for:", ", ".join(preferred + others), file=sys.stderr)

    callables = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj):
            callables.append(name)
    callables.sort()

    print("Available callables in qtanner.progressive_search:", file=sys.stderr)
    for name in callables:
        print("  -", name, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
