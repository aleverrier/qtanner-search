#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 scripts/rebuild_best_codes_artifacts_from_meta.py

# Stage everything, but DO NOT push here
git add -A

# Commit if needed
if [[ -n "$(git status --porcelain)" ]]; then
  git commit -m "best_codes: rebuild website artifacts from meta ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
else
  echo "[publish] no changes to commit."
fi
