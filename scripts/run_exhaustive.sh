#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# If the repo uses a src/ layout (common in Python projects), make it importable.
if [[ -d "$ROOT/src" ]]; then
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
fi

echo "[$(date -Iseconds)] ROOT=$ROOT"
echo "[$(date -Iseconds)] RUN_TAG=${RUN_TAG:-}"
echo "[$(date -Iseconds)] SEED=${SEED:-}"

# Two ways to tell this wrapper what to run:
#  - QTANNER_CMD: full command string, e.g. QTANNER_CMD="python scripts/search_progressive.py"
#  - QTANNER_MODULE: python module for -m, e.g. QTANNER_MODULE="qtanner_search.exhaustive"
if [[ -n "${QTANNER_CMD:-}" ]]; then
  # shellcheck disable=SC2206
  CMD=($QTANNER_CMD)
  echo "[$(date -Iseconds)] CMD: ${CMD[*]} $*"
  exec "${CMD[@]}" "$@"
else
  MOD="${QTANNER_MODULE:-qtanner_search.exhaustive}"

  # Check the module exists before exec, and print helpful info otherwise.
  python - <<'PY' "$MOD"
import importlib.util, os, pkgutil, sys
mod = sys.argv[1]
spec = importlib.util.find_spec(mod)
if spec is None:
    print(f"\nERROR: Cannot import python module: {mod}\n")
    print("This usually means either:")
    print("  (1) the repo package is not on PYTHONPATH / not installed, or")
    print("  (2) the module name is different.\n")

    for base in ["src", "."]:
        if os.path.isdir(base):
            pkgs = sorted([m.name for m in pkgutil.iter_modules([base]) if m.ispkg])
            if pkgs:
                print(f"Packages visible under {base}/: {', '.join(pkgs)}")
    print("\nFix options:")
    print("  - Install the repo (recommended):  python -m pip install -e .")
    print("  - Or set QTANNER_CMD to a script:  QTANNER_CMD='python scripts/XYZ.py' bash scripts/run_exhaustive.sh ...")
    print("  - Or set QTANNER_MODULE to the right module: QTANNER_MODULE='your.module' bash scripts/run_exhaustive.sh ...\n")
    sys.exit(2)
PY

  echo "[$(date -Iseconds)] CMD: python -m ${MOD} $*"
  exec python -m "$MOD" "$@"
fi
