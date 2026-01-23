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
  echo "ERROR: You are on branch '$BRANCH'. Switch to main first:"
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

echo "[1/5] m4ri refine for group=$GROUP trials=$TRIALS (only if trials increases)"
python3 scripts/refine_best_codes_m4ri.py --group "$GROUP" --trials "$TRIALS"

echo "[2/5] rename code IDs if refined distance changed"
python3 scripts/sync_best_codes_names_from_meta.py --group "$GROUP" --archive-collisions

echo "[3/5] prune: enforce min trials AND keep only best per k"
python3 scripts/prune_best_codes_group.py --group "$GROUP" --min-trials "$TRIALS"

echo "[4/5] rebuild website artifacts from meta"
python3 scripts/rebuild_best_codes_artifacts_from_meta.py

echo "[5/5] commit + push (if changes)"
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "pipeline: refine $GROUP with $TRIALS m4ri steps + publish ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git push origin main
else
  echo "[publish] no changes to commit."
fi

if [[ "$AUTO_STASH" -eq 1 ]]; then
  echo ""
  echo "[note] previous local changes were stashed to keep commits clean."
  echo "       See: git stash list"
fi

echo "[done] $GROUP refined and website published"
