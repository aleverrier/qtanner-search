#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate venv if present
if [ -f research/bin/activate ]; then
  # shellcheck disable=SC1091
  source research/bin/activate
fi

# Basic checks
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found." >&2
  exit 1
fi
if ! command -v dist_m4ri >/dev/null 2>&1; then
  echo "ERROR: dist_m4ri not found on PATH. Install it first." >&2
  exit 1
fi
if [ ! -f scripts/run_c2xc2xc2_d16.sh ]; then
  echo "ERROR: scripts/run_c2xc2xc2_d16.sh not found." >&2
  exit 1
fi

# -------------------------
# Tunable knobs (override by env vars)
# -------------------------
TARGET_DISTANCE="${TARGET_DISTANCE:-16}"

# How many repeats per steps value
BATCHES="${BATCHES:-5}"

# RW steps schedule (higher = more thorough but slower)
# Good default: moderate + heavier pass.
STEPS_LIST="${STEPS_LIST:-5000 20000}"

# If you want to be very thorough, you can do:
# STEPS_LIST="5000 20000 80000"
# BATCHES=10

# Lower CPU priority (helps keep laptop responsive)
NICE_N="${NICE_N:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/campaign_c2xc2xc2_d${TARGET_DISTANCE}_${STAMP}"
mkdir -p "$LOGDIR"

echo "== Campaign starting =="
echo "LOGDIR=$LOGDIR"
echo "TARGET_DISTANCE=$TARGET_DISTANCE"
echo "BATCHES=$BATCHES"
echo "STEPS_LIST=$STEPS_LIST"
echo

# Detect which flags exist on qtanner.search (so we don't pass unknown args)
HELP="$(python -m qtanner.search --help 2>&1 || true)"
HAS_STEPS=0;  echo "$HELP" | grep -q -- '--steps' && HAS_STEPS=1 || true
HAS_SEED=0;   echo "$HELP" | grep -q -- '--seed' && HAS_SEED=1 || true
HAS_TARGET=0; echo "$HELP" | grep -q -- '--target-distance' && HAS_TARGET=1 || true
HAS_ENUM=0;   echo "$HELP" | grep -q -- '--A-enum' && HAS_ENUM=1 || true

run_one () {
  local steps="$1"
  local run_id="$2"
  local seed="$3"
  local log="$LOGDIR/run_steps${steps}_batch${run_id}.log"

  echo "=== $(date) steps=$steps batch=$run_id seed=$seed ===" | tee "$log"

  # Build extra args (only if supported by qtanner.search)
  extra=()
  if [ "$HAS_STEPS" -eq 1 ]; then
    extra+=(--steps "$steps")
  fi
  if [ "$HAS_TARGET" -eq 1 ]; then
    extra+=(--target-distance "$TARGET_DISTANCE")
  fi
  if [ "$HAS_SEED" -eq 1 ]; then
    extra+=(--seed "$seed")
  fi
  if [ "$HAS_ENUM" -eq 1 ]; then
    extra+=(--A-enum multiset --B-enum multiset)
  fi

  # Also vary PYTHONHASHSEED to introduce harmless variation even if --seed isn't supported
  export PYTHONHASHSEED="$seed"

  # Run the existing entrypoint (it should keep best-by-k and apply target filtering)
  nice -n "$NICE_N" bash scripts/run_c2xc2xc2_d16.sh "${extra[@]}" 2>&1 | tee -a "$log"
  echo | tee -a "$log"
}

seed0="$(date +%s)"

for steps in $STEPS_LIST; do
  for i in $(seq 1 "$BATCHES"); do
    seed=$((seed0 + steps + i))
    run_one "$steps" "$i" "$seed"
  done
done

echo "== Campaign done =="
echo "Logs in: $LOGDIR"
