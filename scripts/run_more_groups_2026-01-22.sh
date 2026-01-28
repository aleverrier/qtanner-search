#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="runs/2026-01-22_more-groups"
mkdir -p "$RUN_DIR"

echo "[info] Repo: $REPO_ROOT"
echo "[info] Run dir: $RUN_DIR"

# -------------------------
# 1) Try to find a runnable CLI
# -------------------------
RUNNER=()

# (a) already-installed CLI?
if command -v qtanner-search >/dev/null 2>&1; then
  RUNNER=(qtanner-search)
elif command -v qtanner_search >/dev/null 2>&1; then
  RUNNER=(qtanner_search)
elif [ -x ./qtanner-search ]; then
  RUNNER=(./qtanner-search)
fi

# (b) if it looks like a ./scripts/py package, try installing it in editable mode
if [ "${#RUNNER[@]}" -eq 0 ]; then
  if [ -f pyproject.toml ] || [ -f setup.py ]; then
    echo "[info] No CLI found yet. Trying: ./scripts/py -m pip install -e ."
    ./scripts/py -m pip install -e . >/dev/null 2>&1 || true

    if command -v qtanner-search >/dev/null 2>&1; then
      RUNNER=(qtanner-search)
    elif command -v qtanner_search >/dev/null 2>&1; then
      RUNNER=(qtanner_search)
    fi
  fi
fi

# (c) last resort: search for a ./scripts/py/sh entrypoint mentioning '--group'
if [ "${#RUNNER[@]}" -eq 0 ]; then
  echo "[info] Searching repo for an entrypoint mentioning '--group'..."

  # Find up to ~50 candidate files (./scripts/py or shell), skipping common bulky dirs.
  CANDS=""
  CANDS="$(find . \
    -path './.git' -prune -o \
    -path './.venv' -prune -o \
    -path './venv' -prune -o \
    -path './dist' -prune -o \
    -path './build' -prune -o \
    -type f \( -name '*.py' -o -name '*.sh' \) -print | head -n 200)"

  for f in $CANDS; do
    if grep -q -- "--group" "$f" 2>/dev/null; then
      # Try ./scripts/py module form first (handles relative imports better)
      if [[ "$f" == *.py ]]; then
        mod="${f#./}"
        mod="${mod#src/}"
        mod="${mod%.py}"
        mod="${mod//\//.}"

        if PYTHONPATH=".:src:${PYTHONPATH:-}" ./scripts/py -m "$mod" --help >/tmp/qt_help 2>&1; then
          if grep -q -- "--group" /tmp/qt_help; then
            RUNNER=(./scripts/py -m "$mod")
            break
          fi
        fi

        # Fallback: run as a script
        if PYTHONPATH=".:src:${PYTHONPATH:-}" ./scripts/py "$f" --help >/tmp/qt_help 2>&1; then
          if grep -q -- "--group" /tmp/qt_help; then
            RUNNER=(./scripts/py "$f")
            break
          fi
        fi
      else
        # shell script candidate
        if bash "$f" --help >/tmp/qt_help 2>&1; then
          if grep -q -- "--group" /tmp/qt_help; then
            RUNNER=(bash "$f")
            break
          fi
        fi
      fi
    fi
  done
fi

if [ "${#RUNNER[@]}" -eq 0 ]; then
  echo "[error] Could not find how to run the search in this repo."
  echo "[error] Please run these two commands and paste their output here:"
  echo "  ls"
  echo "  find . -maxdepth 3 -type f \\( -name 'README*' -o -name '*.py' -o -name '*.sh' -o -name 'pyproject.toml' -o -name 'setup.py' -o -name 'Makefile' \\) | sort"
  exit 2
fi

echo "[info] Using runner: ${RUNNER[*]}"

# Some CLIs use a 'search' subcommand (typer/click style). If it exists, use it.
if "${RUNNER[@]}" search --help >/tmp/qt_help 2>&1; then
  RUNNER+=(search)
  echo "[info] Runner has a 'search' subcommand → using: ${RUNNER[*]}"
fi

# -------------------------
# 2) Detect flag names from --help (so we don't guess wrong)
# -------------------------
HELP="$("${RUNNER[@]}" --help 2>&1 || true)"

pick_flag () {
  local f
  for f in "$@"; do
    if echo "$HELP" | grep -q -- "$f"; then
      echo "$f"
      return 0
    fi
  done
  return 1
}

GROUP_FLAG="$(pick_flag --group -g || true)"
TRIAL_FLAG="$(pick_flag --qdist-trials --distance-trials --trials --qdist_trials --distance_trials || true)"
LIMIT_FLAG="$(pick_flag --qdist-limit --qdist-max --max-qdist --qdist_limit --qdist_max || true)"
OUT_FLAG="$(pick_flag --out --output --output-file --results --save --save-path || true)"

if [ -z "$GROUP_FLAG" ] || [ -z "$TRIAL_FLAG" ] || [ -z "$LIMIT_FLAG" ]; then
  echo "[error] I found the runner but couldn't auto-detect the flags."
  echo "[error] Here is the first part of --help; paste it back to me and I'll give you exact commands."
  echo "$HELP" | sed -n '1,160p'
  exit 3
fi

echo "[info] Detected flags:"
echo "       GROUP_FLAG=$GROUP_FLAG"
echo "       TRIAL_FLAG=$TRIAL_FLAG"
echo "       LIMIT_FLAG=$LIMIT_FLAG"
echo "       OUT_FLAG=${OUT_FLAG:-<none>}"

run_one () {
  local group="$1"
  local trials="$2"
  local limit="$3"
  local tag="$4"

  echo "[info] Running: group=$group trials=$trials limit=$limit"

  if [ -n "${OUT_FLAG:-}" ]; then
    "${RUNNER[@]}" \
      "$GROUP_FLAG" "$group" \
      "$TRIAL_FLAG" "$trials" \
      "$LIMIT_FLAG" "$limit" \
      "$OUT_FLAG" "$RUN_DIR/${tag}.jsonl" \
      2>&1 | tee "$RUN_DIR/${tag}.log"
  else
    "${RUNNER[@]}" \
      "$GROUP_FLAG" "$group" \
      "$TRIAL_FLAG" "$trials" \
      "$LIMIT_FLAG" "$limit" \
      2>&1 | tee "$RUN_DIR/${tag}.log"
  fi
}

# -------------------------
# 3) NON-EQUIVALENT groups only
# -------------------------
run_one "C6"     50000  16 "C6_t50000_lim16"
run_one "C2xC6"  500000 20 "C2xC6_t500000_lim20"

echo "[info] Done."
echo "[info] Logs/results are in: $RUN_DIR"
