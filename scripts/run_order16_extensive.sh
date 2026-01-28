#!/usr/bin/env bash
set -euo pipefail

# Bash nounset + arrays can be tricky on macOS (bash 3.2).
# Declare optional arrays up-front to avoid 'unbound variable' errors.
declare -a EXTRA_CLASSICAL_ARGS=()

# ===== user parameters =====
MIN_DIST=24
QFAST=4000
QSLOW=500000
PARALLEL=6
SEEDS=(1 2 3)   # "extensive" but still laptop-friendly; increase if you want
MAX_CODEWORDS=256  # 2^8

# ===== environment =====
export CLASSICAL_WORKERS=1          # keep each job single-threaded; we parallelize at job level
export OMP_NUM_THREADS=1            # avoid oversubscription (m4ri/openmp etc.)
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# If the codebase supports env-based caps, these may be used; otherwise they do nothing.
export CLASSICAL_FAST_MAX_CODEWORDS="$MAX_CODEWORDS"
export CLASSICAL_MAX_CODEWORDS="$MAX_CODEWORDS"
export CLASSICAL_SAMPLE_MAX_CODEWORDS="$MAX_CODEWORDS"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="results/order16_min${MIN_DIST}_qf${QFAST}_qs${QSLOW}_${STAMP}"
mkdir -p "$RUN_ROOT"

# ===== detect optional CLI flags =====
HELP="$(./scripts/py scripts/search_progressive.py --help 2>&1 || true)"

EXTRA_CLASSICAL_ARGS=()
if echo "$HELP" | grep -q -- '--classical-fast-max-codewords'; then
  EXTRA_CLASSICAL_ARGS+=(--classical-fast-max-codewords "$MAX_CODEWORDS")
elif echo "$HELP" | grep -q -- '--fast-max-codewords'; then
  EXTRA_CLASSICAL_ARGS+=(--fast-max-codewords "$MAX_CODEWORDS")
elif echo "$HELP" | grep -q -- '--classical-max-codewords'; then
  EXTRA_CLASSICAL_ARGS+=(--classical-max-codewords "$MAX_CODEWORDS")
fi

USE_GROUP_SPEC=0
if echo "$HELP" | grep -q -- '--group-spec'; then
  USE_GROUP_SPEC=1
fi

# ===== build group list =====
GROUP_ITEMS=()

if [[ "$USE_GROUP_SPEC" -eq 1 ]]; then
  # Standard GAP library: 14 groups of order 16
  for i in $(seq 1 14); do
    GROUP_ITEMS+=("spec:SmallGroup(16,${i})")
  done
else
  # Fallback: common aliases (your repo may support these names directly)
  GROUP_ITEMS+=(
    "name:C16"
    "name:C8xC2"
    "name:C4xC4"
    "name:C4xC2xC2"
    "name:C2xC2xC2xC2"
    "name:D16"
    "name:Q16"
    "name:D8xC2"
    "name:Q8xC2"
    "name:SD16"
    "name:M16"
  )
fi

# ===== job generator =====
JOBLIST="$RUN_ROOT/jobs.txt"
: > "$JOBLIST"

for seed in "${SEEDS[@]}"; do
  for item in "${GROUP_ITEMS[@]}"; do
    kind="${item%%:*}"
    val="${item#*:}"

    # Safe tag for filenames
    tag="$(echo "$val" | tr '() ,/' '____' | tr -cd 'A-Za-z0-9_+=-')"
    OUTDIR="$RUN_ROOT/${kind}_${tag}_seed${seed}"
    mkdir -p "$OUTDIR"

    if [[ "$kind" == "spec" ]]; then
      # If your script supports --group-spec, use it.
      printf '%q ' ./scripts/py -u scripts/search_progressive.py \
        --group-spec "$val" \
        --target-distance "$MIN_DIST" \
        --seed "$seed" \
        --classical-distance-backend fast \
        --quantum-steps-fast "$QFAST" \
        --quantum-steps-slow "$QSLOW" \
        ${EXTRA_CLASSICAL_ARGS[@]+"${EXTRA_CLASSICAL_ARGS[@]}"} \
        '2>&1' '|' tee "$OUTDIR/run.log" \
        >> "$JOBLIST"
      printf '\n' >> "$JOBLIST"
    else
      # Otherwise use --group alias mode.
      printf '%q ' ./scripts/py -u scripts/search_progressive.py \
        --group "$val" \
        --target-distance "$MIN_DIST" \
        --seed "$seed" \
        --classical-distance-backend fast \
        --quantum-steps-fast "$QFAST" \
        --quantum-steps-slow "$QSLOW" \
        ${EXTRA_CLASSICAL_ARGS[@]+"${EXTRA_CLASSICAL_ARGS[@]}"} \
        '2>&1' '|' tee "$OUTDIR/run.log" \
        >> "$JOBLIST"
      printf '\n' >> "$JOBLIST"
    fi
  done
done

echo "[run] jobs written to $JOBLIST"
echo "[run] launching with parallelism = $PARALLEL"
echo

DRYRUN="${DRYRUN:-0}"
if [[ "$DRYRUN" == "1" ]]; then
  echo "[dryrun] total jobs: $(wc -l < "$JOBLIST")"
  echo "[dryrun] first 5 jobs:"
  head -n 5 "$JOBLIST"
  exit 0
fi

# macOS has xargs -P
cat "$JOBLIST" | xargs -I{} -P "$PARALLEL" bash -lc "{}"

echo
echo "[done] all order-16 jobs completed in $RUN_ROOT"

# Update best_codes + website data (safe even if no new best codes)
bash scripts/update_best_codes_repo.sh
