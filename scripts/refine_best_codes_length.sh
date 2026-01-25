#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/refine_best_codes_length.sh --n N --trials-per-side T [--jobs J] [--best-dir DIR] [--dist-m4ri PATH]
  bash scripts/refine_best_codes_length.sh --n N --trials-per-side T --verify

What it does:
  - Pulls latest main (unless --no-pull)
  - Refines published best_codes/meta/*.json for length n using dist_m4ri
  - Archives any remaining n entries with trials < requested
  - Rebuilds website artifacts and syncs matrices
  - Commits + pushes (unless --no-push)
EOF
}

N=""
TRIALS=""
JOBS=5
BEST_DIR="best_codes"
DIST_M4RI=""
PULL=1
PUSH=1
VERIFY=0
PROFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --trials-per-side|--trials) TRIALS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --best-dir) BEST_DIR="$2"; shift 2 ;;
    --dist-m4ri) DIST_M4RI="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    --no-push) PUSH=0; shift ;;
    --verify) VERIFY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "${N}" || -z "${TRIALS}" ]]; then
  usage
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ "${PULL}" -eq 1 ]]; then
  git pull --rebase origin main
fi

if [[ -z "${DIST_M4RI}" && -n "${PROFILE}" ]]; then
  case "${PROFILE}" in
    macbook)
      for p in "$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri" "$HOME/.local/bin/dist_m4ri"; do
        if [[ -x "$p" ]]; then
          DIST_M4RI="$p"
          break
        fi
      done
      ;;
    macmini)
      for p in "/opt/homebrew/bin/dist_m4ri" "$HOME/.local/bin/dist_m4ri"; do
        if [[ -x "$p" ]]; then
          DIST_M4RI="$p"
          break
        fi
      done
      ;;
    *) echo "ERROR: unknown profile '${PROFILE}'"; exit 2 ;;
  esac
fi

if [[ -n "${DIST_M4RI}" ]]; then
  export DIST_M4RI
fi

if [[ "${VERIFY}" -eq 1 ]]; then
  python3 scripts/refine_best_codes_length.py \
    --best-dir "$BEST_DIR" \
    --n "$N" \
    --trials-per-side "$TRIALS" \
    --verify
  exit $?
fi

python3 scripts/refine_best_codes_length.py \
  --best-dir "$BEST_DIR" \
  --n "$N" \
  --trials-per-side "$TRIALS" \
  --jobs "$JOBS" \
  ${DIST_M4RI:+--dist-m4ri "$DIST_M4RI"}

python3 scripts/rebuild_best_codes_artifacts_from_meta.py --best-dir "$BEST_DIR"
python3 scripts/sync_best_codes_matrices.py --best-dir "$BEST_DIR" || true

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git add -A
if git diff --cached --quiet; then
  echo "[info] no changes to commit."
  exit 0
fi

git commit -m "best_codes: refine n=${N} trials=${TRIALS} (${TS})"
if [[ "${PUSH}" -eq 1 ]]; then
  git push origin main
fi
