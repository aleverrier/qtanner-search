#!/usr/bin/env bash
set -euo pipefail

python -m qtanner.paper_extract \
  --in data/lrz_paper_mtx \
  --out docs/LRZ_paper_codes_extracted.md \
  --json docs/LRZ_paper_codes_extracted.json
