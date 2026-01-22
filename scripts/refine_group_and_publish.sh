#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash scripts/refine_group_and_publish.sh --group GROUP --trials N [--best-dir DIR] [--dry-run] [--no-push]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/refine_group_and_publish.sh --group C11 --trials 500000"
  echo "  bash scripts/refine_group_and_publish.sh --group 'C6 x C2' --trials 200000 --dry-run"
}

GROUP=""
TRIALS=""
BEST_DIR=""
DRYRUN=0
PUSH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --best-dir) BEST_DIR="$2"; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    --no-push) PUSH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "${GROUP}" || -z "${TRIALS}" ]]; then
  usage
  exit 2
fi

# Go to repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Safety: don’t accidentally commit unrelated changes
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Your working tree is not clean."
  echo "Please commit or stash your current changes first, then rerun."
  echo ""
  git status --porcelain
  exit 1
fi

# Locate the best-codes directory automatically (common names)
if [[ -z "${BEST_DIR}" ]]; then
  if [[ -d "best_codes" ]]; then
    BEST_DIR="best_codes"
  elif [[ -d "best codes" ]]; then
    BEST_DIR="best codes"
  else
    echo "ERROR: Could not find a 'best_codes/' or 'best codes/' directory."
    echo "Use: --best-dir PATH"
    exit 1
  fi
fi

if [[ ! -d "${BEST_DIR}" ]]; then
  echo "ERROR: best-dir does not exist: ${BEST_DIR}"
  exit 1
fi

GROUP_DIR="${BEST_DIR}/${GROUP}"
if [[ ! -d "${GROUP_DIR}" ]]; then
  echo "ERROR: group directory not found: ${GROUP_DIR}"
  echo "Available groups under ${BEST_DIR}:"
  ls -1 "${BEST_DIR}" || true
  exit 1
fi

# Clean macOS junk that breaks filename parsing
find "${BEST_DIR}" -name '.DS_Store' -print -delete || true
find "${BEST_DIR}" -name '._*' -print -delete || true
find "${BEST_DIR}" -type d -name '__MACOSX' -print -prune -exec rm -rf {} + || true

# Detect CLI flags supported by refine_best_codes.py
HELP="$(python3 scripts/refine_best_codes.py --help 2>&1 || true)"

BEST_FLAG=""
if echo "${HELP}" | grep -q -- "--best-dir"; then BEST_FLAG="--best-dir"; fi
if [[ -z "${BEST_FLAG}" ]] && echo "${HELP}" | grep -q -- "--best_dir"; then BEST_FLAG="--best_dir"; fi

TRIALS_FLAG=""
for f in --trials --distance-trials --qdistrnd-trials --num-trials --n-trials; do
  if echo "${HELP}" | grep -q -- "${f}"; then
    TRIALS_FLAG="${f}"
    break
  fi
done

DRY_FLAG=""
if echo "${HELP}" | grep -q -- "--dry-run"; then DRY_FLAG="--dry-run"; fi

# If refine script does not let us pass a best-dir, we temporarily expose ONLY that group
# under the expected best directory name, run refine, then restore everything.
TEMP_MODE=0
BACKUP_DIR=""

restore_best_dir() {
  if [[ "${TEMP_MODE}" -eq 1 ]]; then
    echo "Restoring ${BEST_DIR} from backup..."
    rm -rf "${BEST_DIR}"
    mv "${BACKUP_DIR}" "${BEST_DIR}"
  fi
}

trap restore_best_dir EXIT

CMD=(python3 scripts/refine_best_codes.py)

if [[ -n "${BEST_FLAG}" ]]; then
  CMD+=("${BEST_FLAG}" "${GROUP_DIR}")
else
  echo "NOTE: refine_best_codes.py does not expose a --best-dir flag."
  echo "      Using a safe temporary mode so it only sees group '${GROUP}'."
  TEMP_MODE=1
  BACKUP_DIR="${BEST_DIR}.bak_refine_$(date +%s)"
  mv "${BEST_DIR}" "${BACKUP_DIR}"
  mkdir -p "${BEST_DIR}"
  ln -s "${REPO_ROOT}/${BACKUP_DIR}/${GROUP}" "${BEST_DIR}/${GROUP}"
fi

if [[ -n "${TRIALS_FLAG}" ]]; then
  CMD+=("${TRIALS_FLAG}" "${TRIALS}")
else
  echo "ERROR: Could not detect a trials flag in refine_best_codes.py --help"
  echo "Please run:"
  echo "  python3 scripts/refine_best_codes.py --help"
  echo "and paste the output here so I can wire it correctly."
  exit 1
fi

if [[ "${DRYRUN}" -eq 1 && -n "${DRY_FLAG}" ]]; then
  CMD+=("${DRY_FLAG}")
fi

echo "Running refine:"
echo "  ${CMD[*]}"
"${CMD[@]}"

# If we used temp mode, restore full best_dir BEFORE updating the webpage
restore_best_dir
trap - EXIT

echo "Updating best-codes webpage:"
bash scripts/update_best_codes_repo.sh

# Commit and push changes
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "Refine distances for group ${GROUP} (${TRIALS} trials) + update best-codes page"
  if [[ "${PUSH}" -eq 1 ]]; then
    git push
  else
    echo "Skipping git push (--no-push)."
  fi
else
  echo "No changes detected; nothing to commit."
fi

echo "Done."
