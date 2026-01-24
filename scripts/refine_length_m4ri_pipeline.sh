#!/usr/bin/env bash
set -euo pipefail

N="${1:-}"
TRIALS="${2:-}"
if [[ -z "$N" || -z "$TRIALS" ]]; then
  echo "usage: $0 <n> <trials>"
  echo "  n      = code length (integer)"
  echo "  trials = m4ri RW steps per side (X and Z), integer"
  exit 2
fi

BEST_DIR="${BEST_DIR:-best_codes}"
JOBS="${JOBS:-5}"
SEED="${SEED:-1}"
TIMEOUT="${TIMEOUT:-}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# auto-stash dirty tree (keep commits clean)
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  git stash push -u -m "auto-stash before refine_length_m4ri_pipeline $(date -u +%Y%m%dT%H%M%SZ)"
  STASHED=1
fi

echo "[1/6] ensure matrices exist locally (so refine won't skip)"
python3 scripts/sync_best_codes_matrices.py --best-dir "$BEST_DIR" || true

echo "[2/6] refine ALL codes with n=$N to trials=$TRIALS (jobs=$JOBS)"
ARGS=(--best-dir "$BEST_DIR" --n "$N" --trials "$TRIALS" --jobs "$JOBS" --seed "$SEED")
if [[ -n "$TIMEOUT" ]]; then ARGS+=(--timeout "$TIMEOUT"); fi
python3 scripts/refine_best_codes_m4ri_by_length.py "${ARGS[@]}"

echo "[3/6] rename code IDs if refined distance changed"
python3 scripts/sync_best_codes_names_from_meta.py --best-dir "$BEST_DIR"

echo "[4/6] prune per-group for groups that have length n=$N (enforce min trials + keep best per k)"
python3 - <<'PY'
import json
from pathlib import Path

best=Path("best_codes").resolve()
meta=best/"meta"
N=int(__import__("os").environ.get("N_FOR_PIPE", "0") or 0)
if N==0:
    raise SystemExit(2)

groups=set()
for p in meta.glob("*.json"):
    try:
        d=json.loads(p.read_text())
    except Exception:
        continue
    if int(d.get("n", -1)) != N:
        continue
    cid=d.get("code_id") or p.stem
    if isinstance(cid,str) and "_" in cid:
        groups.add(cid.split("_",1)[0])

print("\n".join(sorted(groups)))
PY
# Feed N into the python snippet via env without relying on bash export inside heredoc
# (portable)
GROUPS="$(N_FOR_PIPE="$N" python3 - <<'PY'
import json
from pathlib import Path
import os

best=Path("best_codes").resolve()
meta=best/"meta"
N=int(os.environ["N_FOR_PIPE"])

groups=set()
for p in meta.glob("*.json"):
    try:
        d=json.loads(p.read_text())
    except Exception:
        continue
    if int(d.get("n", -1)) != N:
        continue
    cid=d.get("code_id") or p.stem
    if isinstance(cid,str) and "_" in cid:
        groups.add(cid.split("_",1)[0])

print(" ".join(sorted(groups)))
PY
)"

if [[ -z "$GROUPS" ]]; then
  echo "[warn] no groups found with n=$N in $BEST_DIR/meta; skipping prune"
else
  for G in $GROUPS; do
    echo "  prune group=$G min_trials=$TRIALS"
    python3 scripts/prune_best_codes_group.py --best-dir "$BEST_DIR" --group "$G" --min-trials "$TRIALS" || true
  done
fi

echo "[5/6] rebuild website artifacts from meta"
python3 scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir "$BEST_DIR"

echo "[6/6] sync matrices (canonical names) + commit/push"
python3 scripts/sync_best_codes_matrices.py --best-dir "$BEST_DIR" --fail-if-missing || true

git add "$BEST_DIR" scripts/best_codes 2>/dev/null || true
git add best_codes scripts 2>/dev/null || true

if git diff --cached --quiet; then
  echo "[info] no changes to commit"
else
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git commit -m "pipeline: refine n=$N with $TRIALS m4ri steps + publish ($TS)"
  git push origin main
fi

if [[ "$STASHED" -eq 1 ]]; then
  echo
  echo "[note] previous local changes were stashed to keep commits clean."
  echo "       See: git stash list"
fi

echo "[done] n=$N refined and website published"
