#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# If the repo uses a src/ layout (common), make it importable.
if [[ -d "$ROOT/src" ]]; then
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
fi

echo "[$(date -Iseconds)] ROOT=$ROOT"
echo "[$(date -Iseconds)] RUN_TAG=${RUN_TAG:-}"
echo "[$(date -Iseconds)] SEED=${SEED:-}"

# Two ways to tell this wrapper what to run:
#  - QTANNER_CMD: full command string, e.g. QTANNER_CMD="./scripts/py src/qtanner/search.py"
#  - QTANNER_MODULE: ./scripts/py module for -m, e.g. QTANNER_MODULE="qtanner.search"
if [[ -n "${QTANNER_CMD:-}" ]]; then
  # shellcheck disable=SC2206
  CMD=($QTANNER_CMD)
  echo "[$(date -Iseconds)] CMD: ${CMD[*]} $*"
  exec "${CMD[@]}" "$@"
else
  # Default module for THIS repo is qtanner.search (not qtanner_search.*)
  MOD="${QTANNER_MODULE:-qtanner.search}"

  # Check module existence, but DO NOT crash if parent package is missing.
  ./scripts/py - <<'PY' "$MOD"
import importlib.util, os, pkgutil, sys

mod = sys.argv[1]
try:
    spec = importlib.util.find_spec(mod)
except ModuleNotFoundError:
    spec = None

if spec is None:
    print(f"\nERROR: Cannot import ./scripts/py module: {mod}\n")
    print(f"Python executable: {sys.executable}\n")
    print("Hint: in this repo the module is usually 'qtanner.search' (qtanner, not qtanner_search).")
    for base in ["src", "."]:
        if os.path.isdir(base):
            pkgs = sorted([m.name for m in pkgutil.iter_modules([base]) if m.ispkg])
            if pkgs:
                print(f"\nPackages visible under {base}/: {', '.join(pkgs)}")
    print("\nFix options:")
    print("  - Ensure src/ exists (ls src)")
    print("  - Or install editable: ./scripts/py -m pip install -e .")
    sys.exit(2)
PY

  echo "[$(date -Iseconds)] CMD: ./scripts/py -m ${MOD} $*"
  exec ./scripts/py -m "$MOD" "$@"
fi
