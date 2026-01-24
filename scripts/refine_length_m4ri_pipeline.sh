#!/usr/bin/env bash
set -euo pipefail

N="${1:-}"
TRIALS="${2:-}"
if [[ -z "$N" || -z "$TRIALS" ]]; then
  echo "usage: $0 <n> <trials_per_side>"
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

DIST_M4RI="${DIST_M4RI:-$HOME/.local/bin/dist_m4ri}"
if [[ ! -x "$DIST_M4RI" ]]; then
  echo "ERROR: dist_m4ri not found/executable at: $DIST_M4RI"
  exit 1
fi

JOBS="${JOBS:-5}"
SEED="${SEED:-1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# stash if dirty
if [[ -n "$(git status --porcelain)" ]]; then
  git stash push -u -m "auto-stash before refine_length_m4ri_pipeline ${TS}" >/dev/null
  echo "[note] stashed local changes (git stash list / git stash pop)"
fi

echo "[1/5] refine ALL codes with n=$N to trials=$TRIALS (per side), jobs=$JOBS"
python3 scripts/refine_best_codes_m4ri_by_length.py \
  --best-dir best_codes \
  --n "$N" \
  --trials "$TRIALS" \
  --jobs "$JOBS" \
  --seed "$SEED" \
  --dist-m4ri "$DIST_M4RI"

echo "[2/5] prune-by-length: archive any remaining n=$N entries with trials < $TRIALS"
python3 - <<PY
import json, re, shutil
from pathlib import Path
from datetime import datetime, timezone

N=int("$N")
T=int("$TRIALS")
best=Path("best_codes")
meta_dir=best/"meta"
col_dir=best/"collected"
mat_dir=best/"matrices"
ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
arch=best/"archived"/f"pruned_n{N}_{ts}"/"below_min_trials"
(arch/"meta").mkdir(parents=True, exist_ok=True)
(arch/"collected").mkdir(parents=True, exist_ok=True)
(arch/"matrices").mkdir(parents=True, exist_ok=True)

def to_int(x):
    try: return int(x)
    except: return None

def trials_from(meta: dict) -> int:
    for k in ("m4ri_steps","m4ri_trials","trials","steps","steps_used","distance_trials","distance_steps"):
        v=to_int(meta.get(k))
        if v is not None and v>0: return v
    dist=meta.get("distance")
    if isinstance(dist, dict):
        for k in ("steps_used_total","steps_used_x","steps_used_z","steps","trials","steps_fast","steps_slow"):
            v=to_int(dist.get(k))
            if v is not None and v>0: return v
    return 0

def uniq(dst: Path) -> Path:
    if not dst.exists(): return dst
    i=1
    while True:
        cand=dst.with_name(dst.name+f"__dup{i}")
        if not cand.exists(): return cand
        i+=1

def move(src: Path, dst: Path):
    if not src.exists(): return
    dst=uniq(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

bad=[]
for mp in sorted(meta_dir.glob("*.json")):
    try: meta=json.loads(mp.read_text())
    except Exception: continue
    if to_int(meta.get("n")) != N: 
        continue
    t=trials_from(meta)
    if t < T:
        bad.append((mp.stem, t))

print(f"[info] n={N} below-threshold in meta: {len(bad)}")
for cid,t in bad[:20]:
    print("  below", cid, "trials", t)

for cid,_t in bad:
    move(meta_dir/f"{cid}.json", arch/"meta"/f"{cid}.json")
    move(col_dir/cid, arch/"collected"/cid)
    if mat_dir.exists():
        for p in list(mat_dir.glob(f"{cid}*")):
            if p.is_file():
                move(p, arch/"matrices"/p.name)

PY

echo "[3/5] rebuild website artifacts"
python3 scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir best_codes

echo "[4/5] sync matrices (canonical __Hx/__Hz)"
python3 scripts/sync_best_codes_matrices.py --best-dir best_codes || true

echo "[5/5] commit + push"
git add best_codes scripts
if git diff --cached --quiet; then
  echo "[info] nothing to commit"
else
  git commit -m "pipeline: refine n=$N with $TRIALS trials/side + prune-by-length + publish (${TS})"
  git push origin main
fi

echo "[done] n=$N updated"
