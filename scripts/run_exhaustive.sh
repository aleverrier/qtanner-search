#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the exhaustive search runner.
# You can run it like:
#   bash scripts/run_exhaustive.sh --group C2xC2 --seed 1 ...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default python entrypoint (adjust if your repo uses a different module/CLI).
# You can override like:
#   QTANNER_CMD="python -m your_module.your_entry" bash scripts/run_exhaustive.sh ...
QTANNER_CMD_STR="${QTANNER_CMD:-python -m qtanner_search.exhaustive}"
# shellcheck disable=SC2206
QTANNER_CMD=($QTANNER_CMD_STR)

echo "[$(date -Iseconds)] RUN_TAG=${RUN_TAG:-}"
echo "[$(date -Iseconds)] SEED=${SEED:-}"
echo "[$(date -Iseconds)] CMD: ${QTANNER_CMD[*]} $*"

exec "${QTANNER_CMD[@]}" "$@"
