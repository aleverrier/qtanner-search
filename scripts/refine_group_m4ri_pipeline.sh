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

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "ERROR: switch to main first: git switch main"
  exit 1
fi

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[auto] working tree not clean -> stashing (including untracked)"
  git stash push -u -m "auto-stash before refine_group_m4ri_pipeline $STAMP"
fi

echo "[1/6] m4ri refine for group=$GROUP trials=$TRIALS (only if trials increases)"
python3 scripts/refine_best_codes_m4ri.py --group "$GROUP" --trials "$TRIALS"

echo "[2/6] rename code IDs if refined distance changed"
python3 scripts/sync_best_codes_names_from_meta.py --group "$GROUP" --archive-collisions

echo "[3/6] prune: enforce min trials AND keep only best per k"
python3 scripts/prune_best_codes_group.py --group "$GROUP" --min-trials "$TRIALS"

echo "[4/6] rebuild website artifacts from meta"
python3 scripts/rebuild_best_codes_artifacts_from_meta.py

echo "[5/6] ASSERT: no remaining $GROUP code has trials < $TRIALS in data.json"
python3 - <<PY
import json
from pathlib import Path

GROUP="$GROUP"
T=int("$TRIALS")
d=json.loads(Path("best_codes/data.json").read_text())
codes=d["codes"]

def iter_codes(codes):
    if isinstance(codes, dict):
        for cid, rec in codes.items():
            yield cid, rec
    elif isinstance(codes, list):
        for rec in codes:
            if not isinstance(rec, dict): 
                continue
            cid = rec.get("code_id") or rec.get("id") or rec.get("name")
            if isinstance(cid, str):
                yield cid, rec

def get_trials(rec):
    if not isinstance(rec, dict): return None
    for k in ("m4ri_steps","trials","steps","steps_used","distance_trials","distance_steps"):
        if k in rec:
            try: return int(rec[k])
            except: pass
    dist = rec.get("distance")
    if isinstance(dist, dict):
        for k in ("steps","trials"):
            if k in dist:
                try: return int(dist[k])
                except: pass
    return None

bad=[]
for cid, rec in iter_codes(codes):
    if not cid.startswith(GROUP+"_"):
        continue
    t=get_trials(rec)
    if t is None or t < T:
        bad.append((cid,t))

if bad:
    print("ERROR: data.json contains group codes with too few trials:")
    for cid,t in bad[:30]:
        print("  ", cid, "trials=", t)
    raise SystemExit(2)
print("[ok] trials invariant holds for group", GROUP)
PY

echo "[6/6] commit + push (if changes)"
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "pipeline: refine $GROUP with $TRIALS m4ri steps + publish ($STAMP)"
  git push origin main
else
  echo "[publish] no changes to commit."
fi

echo "[done]"
