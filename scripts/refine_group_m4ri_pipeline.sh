#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/refine_group_m4ri_pipeline.sh GROUP TRIALS"
  exit 2
fi

GROUP="$1"
TRIALS="$2"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "ERROR: run this on main:"
  echo "  git switch main"
  exit 1
fi

AUTO_STASH=0
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[auto] working tree not clean -> stashing (including untracked)"
  git stash push -u -m "auto-stash before refine_group_m4ri_pipeline $STAMP"
  AUTO_STASH=1
fi

echo "[1/4] m4ri refine for group=$GROUP trials=$TRIALS"
python3 scripts/refine_best_codes_m4ri.py --group "$GROUP" --trials "$TRIALS"

echo "[2/4] rename code IDs if refined distance changed"
python3 scripts/sync_best_codes_names_from_meta.py --group "$GROUP" --archive-collisions

echo "[3/4] prune non-best codes within this group"
python3 scripts/prune_best_codes_group.py --group "$GROUP"

echo "[4/4] rebuild website artifacts from meta"
bash scripts/publish_best_codes_from_meta.sh

echo "[push] pushing one final commit (if any)"
git push origin main || true

echo "[done] pipeline completed for $GROUP trials=$TRIALS"

if [[ "$AUTO_STASH" -eq 1 ]]; then
  echo ""
  echo "[note] Your previous local changes were stashed:"
  echo "  git stash list"
  echo "Restore later with:"
  echo "  git stash pop"
fi
