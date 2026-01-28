#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "[update_best_codes_repo] Collecting best codes..."
./scripts/py scripts/collect_best_codes.py

echo "[update_best_codes_repo] Regenerating website data.json..."
./scripts/py scripts/generate_best_codes_site.py

# Stage ONLY curated outputs (never add results/ or runs/)
git add \
  best_codes \
  best_codes/data.json \
  notes/search_log.tex

if git diff --cached --quiet; then
  echo "[update_best_codes_repo] No changes to commit."
  exit 0
fi

TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git commit -m "Update best codes + website ($TS)"

echo "[update_best_codes_repo] Pushing to origin..."
git push

echo "[update_best_codes_repo] Done."
