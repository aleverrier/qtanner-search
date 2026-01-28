#!/usr/bin/env python3
"""Run a tiny [6,3,3]x[6,3,3] Tanner instance and assert CSS commutation."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.smoke import main


if __name__ == "__main__":
    sys.exit(main())
