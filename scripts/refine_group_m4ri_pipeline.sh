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

# Enforce main branch so the website updates where GitHub Pages serves from.
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "ERROR: You are on branch '$BRANCH'. Switch to main first:"
  echo "  git switch main"
  exit 1
fi

# Keep commits clean/reproducible
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Working tree is not clean. Please stash or commit your changes first:"
  echo "  git stash push -u -m 'wip before refine'"
  exit 1
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
