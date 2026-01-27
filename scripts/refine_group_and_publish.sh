#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/refine_group_and_publish.sh --group GROUP --trials N [--best-dir DIR] [--dry-run] [--no-push]
  bash scripts/refine_group_and_publish.sh --list-groups [--best-dir DIR]

What it does:
  - Finds all codes in BEST_DIR/collected/ whose folder name starts with GROUP_
  - Runs:
      python3 scripts/refine_best_codes_m4ri.py --best-dir BEST_DIR --group GROUP --trials N
  - Then runs:
      bash scripts/update_best_codes_repo.sh
  - Then commits + pushes the changes (including inside BEST_DIR if it is its own git repo)

Examples:
  bash scripts/refine_group_and_publish.sh --list-groups
  bash scripts/refine_group_and_publish.sh --group C2xC2xC2xC2 --trials 500000 --dry-run
  bash scripts/refine_group_and_publish.sh --group C2xC2xC2xC2 --trials 500000
EOF
}

GROUP=""
STEPS=""
BEST_DIR=""
DRYRUN=0
PUSH=1
LIST_GROUPS=0

# Optional knobs (some legacy flags are accepted but ignored).
LIMIT=""
MINDIST=""
TIMEOUT=""
FORCE=0
GAP=""
PATTERN=""
PATTERN_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --trials|--steps) STEPS="$2"; shift 2 ;;
    --best-dir) BEST_DIR="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; PATTERN_SET=1; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --mindist) MINDIST="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --gap) GAP="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --no-push) PUSH=0; shift ;;
    --list-groups) LIST_GROUPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Auto-detect best_codes directory name if not provided
if [[ -z "${BEST_DIR}" ]]; then
  if [[ -d "best_codes" ]]; then
    BEST_DIR="best_codes"
  elif [[ -d "best-codes" ]]; then
    BEST_DIR="best-codes"
  else
    echo "ERROR: Could not find best_codes/ or best-codes/."
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
  exit 1
fi

# List groups mode
if [[ "${LIST_GROUPS}" -eq 1 ]]; then
  echo "Groups found under ${COLLECTED_DIR}:"
  shopt -s nullglob
  for d in "${COLLECTED_DIR}"/*; do
    [[ -d "${d}" ]] || continue
    name="$(basename "${d}")"
    echo "${name%%_*}"
  done | sort -u
  shopt -u nullglob
  exit 0
fi

if [[ -z "${GROUP}" || -z "${STEPS}" ]]; then
  usage
  exit 2
fi

# Clean macOS junk that sometimes breaks parsing (e.g. .DS_Store)
find "${BEST_DIR}" -name '.DS_Store' -print -delete || true
find "${BEST_DIR}" -name '._*' -print -delete || true
find "${BEST_DIR}" -type d -name '__MACOSX' -print -prune -exec rm -rf {} + || true

# Refuse to run if there are unrelated pending changes (keeps commits clean)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Your working tree is not clean."
  echo "Please commit or stash current changes first, then rerun."
  echo ""
  git status --porcelain
  exit 1
fi

# Confirm the group actually exists, and show how many codes will be processed
shopt -s nullglob
CANDIDATES=( "${COLLECTED_DIR}/${GROUP}_"* )
shopt -u nullglob

COUNT=0
for p in "${CANDIDATES[@]}"; do
  if [[ -d "${p}" ]]; then
    COUNT=$((COUNT + 1))
  fi
done

if [[ "${COUNT}" -eq 0 ]]; then
  echo "ERROR: no code directories found for group '${GROUP}' under:"
  echo "  ${COLLECTED_DIR}"
  echo ""
  echo "Tip: run:"
  echo "  bash scripts/refine_group_and_publish.sh --list-groups"
  exit 1
fi

echo "Found ${COUNT} codes for group '${GROUP}' in ${COLLECTED_DIR}"

# Pattern: use GROUP_ by default to avoid matching longer groups (e.g. C2xC2 inside C2xC2xC2)
if [[ -z "${PATTERN}" ]]; then
  PATTERN="${GROUP}_"
fi

if [[ "${PATTERN_SET}" -eq 1 || -n "${LIMIT}" || -n "${MINDIST}" || -n "${GAP}" ]]; then
  echo "[warn] --pattern/--limit/--mindist/--gap are not supported by refine_best_codes_m4ri.py; ignoring."
fi

CMD=(
  python3 scripts/refine_best_codes_m4ri.py
  --best-dir "${BEST_DIR}"
  --group "${GROUP}"
  --trials "${STEPS}"
)

if [[ -n "${TIMEOUT}" ]]; then CMD+=(--timeout "${TIMEOUT}"); fi
if [[ "${FORCE}" -eq 1 ]]; then CMD+=(--force); fi

echo "Running refine:"
echo "  ${CMD[*]}"
if [[ "${DRYRUN}" -eq 1 ]]; then
  echo "Dry-run: skipping refine/update/commit/push."
  exit 0
fi
"${CMD[@]}"

if [[ "${DRYRUN}" -eq 1 ]]; then
  echo "Dry-run complete (no update/commit/push)."
  exit 0
fi

echo "Updating best-codes webpage:"
bash scripts/update_best_codes_repo.sh

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

# If BEST_DIR is its own git repo (submodule), commit there first
if [[ -e "${BEST_DIR}/.git" ]]; then
  commit_push_if_needed "${BEST_DIR}" "Update best_codes after refining ${GROUP} (${STEPS} steps)" "${PUSH}"
fi

# Then commit at repo root if needed (e.g. submodule pointer changed or other files updated)
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "Refine distances for ${GROUP} (${STEPS} steps) and update best-codes page"
  if [[ "${PUSH}" -eq 1 ]]; then
    git push
  fi
else
  echo "No changes detected at repo root; nothing to commit."
fi

echo "Done."
