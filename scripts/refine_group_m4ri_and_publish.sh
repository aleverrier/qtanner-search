#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/refine_group_m4ri_and_publish.sh GROUP TRIALS [--force] [--prune] [--dist-m4ri /path/to/dist_m4ri]"
  exit 2
fi

GROUP="$1"
TRIALS="$2"
shift 2

python3 scripts/refine_best_codes_m4ri.py --group "${GROUP}" --trials "${TRIALS}" "$@"

# Regenerate website + commit/push (your existing script already does that)
bash scripts/update_best_codes_repo.sh
