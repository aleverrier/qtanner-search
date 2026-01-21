#!/usr/bin/env python3
"""
Thin wrapper to run qtanner.progressive_search from the repo root.

This makes "python scripts/search_progressive.py ..." work even if the
package is not installed, by prepending ./src to sys.path.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_module("qtanner.progressive_search", run_name="__main__")
