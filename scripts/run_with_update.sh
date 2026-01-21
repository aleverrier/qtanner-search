#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

cleanup() {
  echo "[run_with_update] Post-processing + pushing best codes..."
  bash scripts/update_best_codes_repo.sh || true
}
trap cleanup EXIT

"$@"
