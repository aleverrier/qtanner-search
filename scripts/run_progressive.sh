#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

# Usage:
#   ./scripts/run_progressive.sh GROUP TARGET_DISTANCE CLASSICAL_TARGET [MAX_EVALS] [QFAST] [QSLOW]
GROUP="${1:?Missing group, e.g. 'C2xC2xC2xC2'}"
TARGET="${2:?Missing quantum target distance, e.g. 24}"
CLASSICAL_TARGET="${3:?Missing classical target distance, e.g. 24}"

MAX_EVALS="${4:-}"   # number of quantum candidates to evaluate
QFAST="${5:-}"       # quantum distance fast estimator steps
QSLOW="${6:-}"       # quantum distance slow estimator steps

CLASSICAL_BACKEND="${CLASSICAL_BACKEND:-fast}"
CLASSICAL_EXHAUSTIVE_K_MAX="${CLASSICAL_EXHAUSTIVE_K_MAX:-8}"
CLASSICAL_SAMPLE_COUNT="${CLASSICAL_SAMPLE_COUNT:-256}"
CLASSICAL_JOBS="${CLASSICAL_JOBS:-0}"
CLASSICAL_STEPS="${CLASSICAL_STEPS:-100}"
QUANTUM_REFINE_CHUNK="${QUANTUM_REFINE_CHUNK:-20000}"

DIST_M4RI_CMD="${DIST_M4RI:-$HOME/research/qtanner-tools/dist-m4ri/src/dist_m4ri}"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/prog_${GROUP//[^A-Za-z0-9]/_}_t${TARGET}_ct${CLASSICAL_TARGET}_${STAMP}.log"

echo "[run] group=$GROUP target_distance=$TARGET classical_target=$CLASSICAL_TARGET"
echo "[run] dist_m4ri_cmd=$DIST_M4RI_CMD"
echo "[run] classical_backend=$CLASSICAL_BACKEND classical_exhaustive_k_max=$CLASSICAL_EXHAUSTIVE_K_MAX classical_sample_count=$CLASSICAL_SAMPLE_COUNT classical_jobs=$CLASSICAL_JOBS"
echo "[run] quantum_refine_chunk=$QUANTUM_REFINE_CHUNK"
[[ -n "$MAX_EVALS" ]] && echo "[run] max_quantum_evals=$MAX_EVALS"
[[ -n "$QFAST"    ]] && echo "[run] quantum_steps_fast=$QFAST"
[[ -n "$QSLOW"    ]] && echo "[run] quantum_steps_slow=$QSLOW"
echo "[run] log=$LOG"

ARGS=(
  --group "$GROUP"
  --target-distance "$TARGET"
  --classical-target "$CLASSICAL_TARGET"
  --classical-steps "$CLASSICAL_STEPS"
  --classical-distance-backend "$CLASSICAL_BACKEND"
  --classical-exhaustive-k-max "$CLASSICAL_EXHAUSTIVE_K_MAX"
  --classical-sample-count "$CLASSICAL_SAMPLE_COUNT"
  --classical-jobs "$CLASSICAL_JOBS"
  --quantum-refine-chunk "$QUANTUM_REFINE_CHUNK"
  --dist-m4ri-cmd "$DIST_M4RI_CMD"
)

[[ -n "$MAX_EVALS" ]] && ARGS+=( --max-quantum-evals "$MAX_EVALS" )
[[ -n "$QFAST"    ]] && ARGS+=( --quantum-steps-fast "$QFAST" )
[[ -n "$QSLOW"    ]] && ARGS+=( --quantum-steps-slow "$QSLOW" )

./scripts/py -u scripts/search_progressive.py "${ARGS[@]}" 2>&1 | tee "$LOG"
