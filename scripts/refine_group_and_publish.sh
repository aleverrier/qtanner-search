#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash scripts/refine_group_and_publish.sh --group GROUP --trials N [--best-dir DIR] [--dry-run] [--no-push]"
  echo "  bash scripts/refine_group_and_publish.sh --list-groups [--best-dir DIR]"
  echo ""
  echo "Notes:"
  echo "  - Codes are expected under: BEST_DIR/collected/"
  echo "  - GROUP matches directories starting with: GROUP_"
  echo ""
  echo "Examples:"
  echo "  bash scripts/refine_group_and_publish.sh --list-groups"
  echo "  bash scripts/refine_group_and_publish.sh --group C2xC2xC2xC2 --trials 500000 --dry-run"
  echo "  bash scripts/refine_group_and_publish.sh --group C2xC2xC2xC2 --trials 500000"
}

GROUP=""
TRIALS=""
BEST_DIR=""
DRYRUN=0
PUSH=1
LIST_GROUPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --best-dir) BEST_DIR="$2"; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    --no-push) PUSH=0; shift ;;
    --list-groups) LIST_GROUPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Locate best_codes directory automatically
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

COLLECTED_DIR="${BEST_DIR}/collected"
if [[ ! -d "${COLLECTED_DIR}" ]]; then
  echo "ERROR: expected codes under: ${COLLECTED_DIR}"
  echo "But that directory does not exist."
  exit 1
fi

# If requested: list available groups (prefix before first underscore)
if [[ "${LIST_GROUPS}" -eq 1 ]]; then
  echo "Groups found under ${COLLECTED_DIR}:"
  ls -1 "${COLLECTED_DIR}" \
    | sed 's/_.*$//' \
    | sort -u
  exit 0
fi

if [[ -z "${GROUP}" || -z "${TRIALS}" ]]; then
  usage
  exit 2
fi

# Safety: avoid committing unrelated changes
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Your working tree is not clean."
  echo "Please commit or stash current changes first, then rerun."
  echo ""
  git status --porcelain
  exit 1
fi

# Clean macOS junk that can break filename parsing
find "${BEST_DIR}" -name '.DS_Store' -print -delete || true
find "${BEST_DIR}" -name '._*' -print -delete || true
find "${BEST_DIR}" -type d -name '__MACOSX' -print -prune -exec rm -rf {} + || true

# Find codes for this group (directories starting with GROUP_)
shopt -s nullglob
CANDIDATES=( "${COLLECTED_DIR}/${GROUP}_"* )
shopt -u nullglob

MATCHED=()
for p in "${CANDIDATES[@]}"; do
  if [[ -d "${p}" ]]; then
    MATCHED+=( "${p}" )
  fi
done

if (( ${#MATCHED[@]} == 0 )); then
  echo "ERROR: no code directories found for group '${GROUP}' under:"
  echo "  ${COLLECTED_DIR}"
  echo ""
  echo "Tip: run:"
  echo "  bash scripts/refine_group_and_publish.sh --list-groups"
  exit 1
fi

echo "Found ${#MATCHED[@]} codes for group '${GROUP}' in ${COLLECTED_DIR}"
echo "Temporarily hiding other groups so refine_best_codes.py only sees this group..."

STAMP="$(date +%s)"
STASH_DIR="${BEST_DIR}/.tmp_stash_other_groups_${GROUP}_${STAMP}"
RESTORED=0

restore_other_groups() {
  if [[ "${RESTORED}" -eq 1 ]]; then
    return
  fi
  RESTORED=1

  if [[ -d "${STASH_DIR}/collected" ]]; then
    shopt -s nullglob
    for d in "${STASH_DIR}/collected/"*; do
      mv "${d}" "${COLLECTED_DIR}/"
    done
    shopt -u nullglob
  fi

  rm -rf "${STASH_DIR}" || true
}

trap restore_other_groups EXIT

mkdir -p "${STASH_DIR}/collected"

# Move non-matching directories out of collected/
shopt -s nullglob
for d in "${COLLECTED_DIR}/"*; do
  name="$(basename "${d}")"
  if [[ -d "${d}" && "${name}" != "${GROUP}_"* ]]; then
    mv "${d}" "${STASH_DIR}/collected/"
  fi
done
shopt -u nullglob

# Detect the trials flag supported by refine_best_codes.py
HELP="$(python3 scripts/refine_best_codes.py --help 2>&1 || true)"

TRIALS_FLAG=""
for f in --trials --qdistrnd-trials --distance-trials --num-trials --n-trials --ntrials; do
  if echo "${HELP}" | grep -q -- "${f}"; then
    TRIALS_FLAG="${f}"
    break
  fi
done

if [[ -z "${TRIALS_FLAG}" ]]; then
  echo "ERROR: Could not detect a trials flag in refine_best_codes.py --help"
  echo "Please run and paste the output:"
  echo "  python3 scripts/refine_best_codes.py --help"
  exit 1
fi

CMD=(python3 scripts/refine_best_codes.py "${TRIALS_FLAG}" "${TRIALS}")
if echo "${HELP}" | grep -q -- "--dry-run" && [[ "${DRYRUN}" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

echo "Running refine:"
echo "  ${CMD[*]}"
"${CMD[@]}"

# Restore other groups before updating website
restore_other_groups
trap - EXIT

if [[ "${DRYRUN}" -eq 1 ]]; then
  echo "Dry-run complete (no update/commit/push)."
  exit 0
fi

echo "Updating best-codes webpage:"
bash scripts/update_best_codes_repo.sh

# Commit and push (supports best_codes being a separate git repo or part of this repo)
commit_push_if_needed() {
  local repo_path="$1"
  local msg="$2"
  local push="$3"

  if [[ ! -d "${repo_path}" ]]; then
    return
  fi

  if git -C "${repo_path}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git -C "${repo_path}" status --porcelain)" ]]; then
      git -C "${repo_path}" add -A
      git -C "${repo_path}" commit -m "${msg}"
      if [[ "${push}" -eq 1 ]]; then
        git -C "${repo_path}" push
      fi
    fi
  fi
}

# If best_codes is its own git repo (e.g., a submodule), commit there first
if [[ -e "${BEST_DIR}/.git" ]]; then
  commit_push_if_needed "${BEST_DIR}" "Update best_codes after refining ${GROUP} (${TRIALS} trials)" "${PUSH}"
fi

# Then commit at repo root if needed (e.g., submodule pointer changed or other files updated)
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "Refine distances for ${GROUP} (${TRIALS} trials) and update best-codes page"
  if [[ "${PUSH}" -eq 1 ]]; then
    git push
  fi
else
  echo "No changes detected at repo root; nothing to commit."
fi

echo "Done."
