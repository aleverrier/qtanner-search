#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qtanner.progressive_search import progressive_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(progressive_main(sys.argv[1:]))
