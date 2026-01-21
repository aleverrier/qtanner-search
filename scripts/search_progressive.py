#!/usr/bin/env python3
"""
Run the progressive search from the repo, even without installing the package.

This wrapper is intentionally non-silent:
- It tries to find and call a CLI entrypoint in qtanner.progressive_search.
- If it can't, it prints available callables so we can wire it correctly.
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

def _call_entrypoint(mod) -> int:
    # Common entrypoint names
    candidates = [
        "main",
        "cli",
        "run_cli",
        "entrypoint",
        "run",
        "prog_main",
        "progressive_search_main",
    ]

    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            # Try passing argv if it looks supported; otherwise call with no args.
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) >= 1:
                    fn(sys.argv[1:])
                else:
                    fn()
            except TypeError:
                fn()
            return 0

    # No obvious entrypoint: print diagnostics (so it's never "nothing happens")
    print("ERROR: qtanner.progressive_search has no obvious CLI entrypoint.", file=sys.stderr)
    print("I looked for:", ", ".join(candidates), file=sys.stderr)

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

    print("\nNext step:", file=sys.stderr)
    print("  Run: PYTHONPATH=src python - <<'PY'\n"
          "import qtanner.progressive_search as m\n"
          "import inspect\n"
          "for name in [n for n in dir(m) if not n.startswith('_')]:\n"
          "    obj = getattr(m, name)\n"
          "    if callable(obj):\n"
          "        try:\n"
          "            print(name, inspect.signature(obj))\n"
          "        except Exception:\n"
          "            print(name)\n"
          "PY", file=sys.stderr)
    return 2

def main() -> int:
    mod = importlib.import_module("qtanner.progressive_search")
    return _call_entrypoint(mod)

if __name__ == "__main__":
    raise SystemExit(main())
