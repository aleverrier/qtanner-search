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

# Enforce main branch for GitHub Pages
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "ERROR: You are on branch '$BRANCH'. Switch to main first:"
  echo "  git switch main"
  exit 1
fi

AUTO_STASH=0
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Auto-stash if dirty (tracked or untracked)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[auto] working tree not clean -> stashing (including untracked)"
  git stash push -u -m "auto-stash before refine_group_m4ri_pipeline $STAMP"
  AUTO_STASH=1
fi

echo "[1/4] m4ri refine for group=$GROUP trials=$TRIALS (only if trials increases)"
python3 scripts/refine_best_codes_m4ri.py --group "$GROUP" --trials "$TRIALS"

echo "[2/4] rename code IDs if refined distance changed"
python3 scripts/sync_best_codes_names_from_meta.py --group "$GROUP" --archive-collisions

echo "[3/4] prune non-best codes within this group (best per k)"
python3 scripts/prune_best_codes_group.py --group "$GROUP"

echo "[4/4] rebuild website artifacts from meta and publish to GitHub"
bash scripts/publish_best_codes_from_meta.sh

echo "[done] pipeline completed for $GROUP trials=$TRIALS"

if [[ "$AUTO_STASH" -eq 1 ]]; then
  echo ""
  echo "[note] I stashed your previous local changes to keep commits clean."
  echo "       They are still available via:"
  echo "         git stash list"
  echo "         git stash show -p stash@{0}"
  echo "       If you really want them back later:"
  echo "         git stash pop"
fi
