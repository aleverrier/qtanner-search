#!/usr/bin/env bash
set -euo pipefail
BEST_DIR="best_codes"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[auto] working tree not clean -> stashing (including untracked)"
  git stash push -u -m "auto-stash before scrub_best_codes_repo_and_publish ${TS}"
fi

echo "[1/3] scrub duplicates conservatively (keep smallest d_ub, max trials)"
python3 scripts/scrub_best_codes_repo.py --best-dir "${BEST_DIR}" --archive-root "${BEST_DIR}/archived/scrub_all_${TS}"

echo "[2/3] rebuild website artifacts from active meta only (non-recursive)"
python3 scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir "${BEST_DIR}"

echo "[3/3] commit + push"
if [[ -n "$(git status --porcelain)" ]]; then
  git add "${BEST_DIR}" scripts
  git commit -m "best_codes: scrub duplicates + rebuild artifacts (${TS})"
  git push origin main
else
  echo "[done] no changes to commit."
fi
