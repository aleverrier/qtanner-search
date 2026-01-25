#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/research/bin/activate}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: venv activate not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"
pytest -q
