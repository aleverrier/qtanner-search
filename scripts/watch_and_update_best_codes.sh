#!/usr/bin/env bash
set -euo pipefail

INTERVAL_MIN="${1:-10}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[watch] updating best codes every ${INTERVAL_MIN} minutes"
echo "[watch] stop with Ctrl+C"

while true; do
  echo ""
  echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) running update_best_codes_repo.sh"
  bash scripts/update_best_codes_repo.sh || echo "[watch] update script failed (will retry next round)"
  echo "[watch] sleeping..."
  sleep "$((INTERVAL_MIN*60))"
done
