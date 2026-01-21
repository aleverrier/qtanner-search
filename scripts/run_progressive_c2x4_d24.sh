#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure dist_m4ri is visible
export PATH="$HOME/.local/bin:$PATH"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/progressive_c2x4_d24_${STAMP}"
mkdir -p "$OUTDIR"

# Override any of these by prefixing VAR=value before the script.
GROUP="${GROUP:-C2xC2xC2xC2}"
TARGET_DISTANCE="${TARGET_DISTANCE:-24}"

# Quantum block length is 36*|G| = 576 => sqrt(576) = 24 (classical filter default)
CLASSICAL_TARGET="${CLASSICAL_TARGET:-24}"

# Start modest; increase later for more confidence
CLASSICAL_STEPS="${CLASSICAL_STEPS:-100}"
QUANTUM_STEPS_FAST="${QUANTUM_STEPS_FAST:-2000}"
QUANTUM_STEPS_SLOW="${QUANTUM_STEPS_SLOW:-50000}"

# Start small; scale up after you see promising candidates
MAX_QUANTUM_EVALS="${MAX_QUANTUM_EVALS:-3000}"
REPORT_EVERY="${REPORT_EVERY:-50}"
SEED="${SEED:-0}"

# Use whichever progressive CLI interface exists
if python -m qtanner.search progressive --help >/dev/null 2>&1; then
  CMD=(python -m qtanner.search progressive)
else
  CMD=(python -m qtanner.search --mode progressive)
fi

echo "OUTDIR=$OUTDIR" | tee "$OUTDIR/run.log"
echo "GROUP=$GROUP TARGET_DISTANCE=$TARGET_DISTANCE CLASSICAL_TARGET=$CLASSICAL_TARGET" | tee -a "$OUTDIR/run.log"
echo "CLASSICAL_STEPS=$CLASSICAL_STEPS QUANTUM_STEPS_FAST=$QUANTUM_STEPS_FAST QUANTUM_STEPS_SLOW=$QUANTUM_STEPS_SLOW MAX_QUANTUM_EVALS=$MAX_QUANTUM_EVALS REPORT_EVERY=$REPORT_EVERY SEED=$SEED" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"

"${CMD[@]}" \
  --group "$GROUP" \
  --target-distance "$TARGET_DISTANCE" \
  --classical-target "$CLASSICAL_TARGET" \
  --classical-steps "$CLASSICAL_STEPS" \
  --quantum-steps-fast "$QUANTUM_STEPS_FAST" \
  --quantum-steps-slow "$QUANTUM_STEPS_SLOW" \
  --max-quantum-evals "$MAX_QUANTUM_EVALS" \
  --report-every "$REPORT_EVERY" \
  --seed "$SEED" \
  --results-dir "$OUTDIR" \
  2>&1 | tee -a "$OUTDIR/run.log"
