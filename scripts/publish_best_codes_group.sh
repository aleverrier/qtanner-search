#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/publish_best_codes_group.sh --group GROUP [--best-dir DIR] [--results-dir DIR] [--no-pull] [--no-push]
  bash scripts/publish_best_codes_group.sh --order N [--best-dir DIR] [--results-dir DIR] [--no-pull] [--no-push]
  bash scripts/publish_best_codes_group.sh --group GROUP --verify
  bash scripts/publish_best_codes_group.sh --order N --verify

What it does:
  - Pulls latest main (unless --no-pull)
  - Merges results/**/best_codes for the selected group/order into best_codes/
  - Prunes by min_trials_by_group.json + best-per-k
  - Rebuilds website artifacts and syncs matrices
  - Commits + pushes (unless --no-push)
EOF
}

GROUP=""
ORDER=""
BEST_DIR="best_codes"
RESULTS_DIR="results"
PULL=1
PUSH=1
VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --order) ORDER="$2"; shift 2 ;;
    --best-dir) BEST_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    --no-push) PUSH=0; shift ;;
    --verify) VERIFY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "${GROUP}" && -z "${ORDER}" ]]; then
  usage
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ "${PULL}" -eq 1 ]]; then
  git pull --rebase origin main
fi

if [[ "${VERIFY}" -eq 1 ]]; then
  python3 scripts/publish_best_codes_group.py \
    ${GROUP:+--group "$GROUP"} \
    ${ORDER:+--order "$ORDER"} \
    --best-dir "$BEST_DIR" \
    --verify
  exit $?
fi

TMP_GROUPS="$(mktemp)"
python3 scripts/publish_best_codes_group.py \
  ${GROUP:+--group "$GROUP"} \
  ${ORDER:+--order "$ORDER"} \
  --best-dir "$BEST_DIR" \
  --results-dir "$RESULTS_DIR" \
  --groups-out "$TMP_GROUPS"

if [[ ! -s "$TMP_GROUPS" ]]; then
  echo "[warn] no matching groups found in results."
  rm -f "$TMP_GROUPS"
  exit 0
fi

while IFS= read -r G; do
  [[ -n "$G" ]] || continue
  MIN_TRIALS="$(python3 - <<PY
import json
from pathlib import Path
path = Path("$BEST_DIR") / "min_trials_by_group.json"
data = json.loads(path.read_text()) if path.exists() else {}
print(int(data.get("$G", data.get("SmallGroup", 0) or 0)))
PY
  )"
  python3 scripts/prune_best_codes_group.py --best-dir "$BEST_DIR" --group "$G" --min-trials "$MIN_TRIALS"
done < "$TMP_GROUPS"

rm -f "$TMP_GROUPS"

python3 scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir "$BEST_DIR"
python3 scripts/sync_best_codes_matrices.py --best-dir "$BEST_DIR" || true

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LABEL="${GROUP:-order=${ORDER}}"

git add -A
if git diff --cached --quiet; then
  echo "[info] no changes to commit."
  exit 0
fi

git commit -m "best_codes: publish ${LABEL} (${TS})"
if [[ "${PUSH}" -eq 1 ]]; then
  git push origin main
fi
