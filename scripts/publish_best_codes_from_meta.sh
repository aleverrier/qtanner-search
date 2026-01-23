#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 scripts/rebuild_best_codes_artifacts_from_meta.py

# Commit/push if something changed
if [[ -n "$(git status --porcelain)" ]]; then
  git add best_codes scripts/rebuild_best_codes_artifacts_from_meta.py scripts/prune_best_codes_group.py scripts/publish_best_codes_from_meta.sh 2>/dev/null || true
  git add -A
  git commit -m "best_codes: rebuild website artifacts from meta ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git push origin main
else
  echo "[publish] no changes to commit."
fi
