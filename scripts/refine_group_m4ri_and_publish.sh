#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/refine_group_m4ri_and_publish.sh GROUP TRIALS"
  exit 2
fi

GROUP="$1"
TRIALS="$2"

python3 scripts/refine_results_best_codes_m4ri.py --group "${GROUP}" --trials "${TRIALS}"

bash scripts/update_best_codes_repo.sh
