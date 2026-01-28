#!/usr/bin/env bash
set -u
set -o pipefail

# ===== Parameters you asked for =====
TARGET_DIST=24
FAST_Q=4000
SLOW_Q=500000
CLASSICAL_MAX_CODEWORDS=256   # 2^8
PARALLEL=6

# "Extensive" = multiple random seeds per group.
# Start with 4 seeds (reasonable on a laptop); increase to 8 or 12 later if you want.
N_SEEDS=4

# ===== Housekeeping =====
SEARCH_SCRIPT="scripts/search_progressive.py"
if [[ ! -f "$SEARCH_SCRIPT" ]]; then
  echo "ERROR: cannot find $SEARCH_SCRIPT"
  echo "Are you in the repo root? (the folder containing scripts/)"
  exit 1
fi

# Avoid oversubscription (important when running 6 processes in parallel)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
BASE_DIR="results/order16_d${TARGET_DIST}_fq${FAST_Q}_sq${SLOW_Q}_c${CLASSICAL_MAX_CODEWORDS}_${RUN_TAG}"
mkdir -p "$BASE_DIR"

GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "=== Order-16 search ==="
echo "Run tag:        $RUN_TAG"
echo "Repo commit:    $GIT_HASH"
echo "Output folder:  $BASE_DIR"
echo "Groups:         SmallGroup(16,1..14)"
echo "Seeds:          1..$N_SEEDS"
echo "Parallel jobs:  $PARALLEL"
echo

# Detect which flag name your code uses for "classical sample size"
HELP_OUT="$(./scripts/py "$SEARCH_SCRIPT" --help 2>&1 || true)"

CLASSICAL_MAX_FLAG=""
if echo "$HELP_OUT" | grep -q -- '--classical-sample-codewords'; then
  CLASSICAL_MAX_FLAG="--classical-sample-codewords"
elif echo "$HELP_OUT" | grep -q -- '--classical-max-codewords'; then
  CLASSICAL_MAX_FLAG="--classical-max-codewords"
elif echo "$HELP_OUT" | grep -q -- '--classical-sample'; then
  CLASSICAL_MAX_FLAG="--classical-sample"
else
  echo "ERROR: I couldn't find a flag in $SEARCH_SCRIPT --help for the classical sampling limit."
  echo "I searched for: --classical-sample-codewords / --classical-max-codewords / --classical-sample"
  echo
  echo "Run this and paste the output to me:"
  echo "  ./scripts/py $SEARCH_SCRIPT --help | sed -n '1,200p'"
  exit 1
fi

# Detect whether you use --target-distance or --min-distance
DIST_FLAG="--target-distance"
if echo "$HELP_OUT" | grep -q -- '--min-distance'; then
  DIST_FLAG="--min-distance"
fi

# If you have a specific backend name for the "fast classical sampler", set it here.
# Keeping "fast" consistent with our earlier runs.
CLASSICAL_BACKEND_FLAG="--classical-distance-backend"
CLASSICAL_BACKEND_VALUE="fast"

# ===== Job control (max PARALLEL concurrent background processes) =====
pids=()
failures=0

cleanup() {
  echo
  echo "Caught interrupt; killing running jobs..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch_job() {
  local gid="$1"
  local seed="$2"
  local group="SmallGroup(16,${gid})"
  local run_dir="${BASE_DIR}/SG16_${gid}_seed${seed}"
  mkdir -p "$run_dir"

  # Re-run safety: if a log exists, skip that job (lets you resume by re-running the script).
  if [[ -f "${run_dir}/run.log" ]]; then
    echo "[skip] ${group} seed=${seed}  (already has ${run_dir}/run.log)"
    return
  fi

  # Save parameters for reproducibility
  cat > "${run_dir}/params.txt" <<EOF
group=${group}
seed=${seed}
distance_flag=${DIST_FLAG}
target_dist=${TARGET_DIST}
fast_quantum=${FAST_Q}
slow_quantum=${SLOW_Q}
classical_backend=${CLASSICAL_BACKEND_VALUE}
classical_max_codewords=${CLASSICAL_MAX_CODEWORDS}
git_commit=${GIT_HASH}
EOF

  echo "[start] ${group} seed=${seed}"

  (
    ./scripts/py -u "$SEARCH_SCRIPT" \
      --group "$group" \
      "$DIST_FLAG" "$TARGET_DIST" \
      --seed "$seed" \
      --quantum-steps-fast "$FAST_Q" \
      --quantum-steps-slow "$SLOW_Q" \
      "$CLASSICAL_BACKEND_FLAG" "$CLASSICAL_BACKEND_VALUE" \
      "$CLASSICAL_MAX_FLAG" "$CLASSICAL_MAX_CODEWORDS" \
      2>&1 | tee "${run_dir}/run.log"
  ) &

  pids+=($!)
}

wait_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures+1))
    fi
  done
  pids=()
}

# There are 14 groups of order 16 in the GAP SmallGroups library: ids 1..14
for gid in $(seq 1 14); do
  for seed in $(seq 1 "$N_SEEDS"); do
    launch_job "$gid" "$seed"
    if [[ "${#pids[@]}" -ge "$PARALLEL" ]]; then
      wait_batch
    fi
  done
done

# Wait remaining
wait_batch

echo
echo "All jobs finished. failures=${failures}"
echo "Results are in: $BASE_DIR"
