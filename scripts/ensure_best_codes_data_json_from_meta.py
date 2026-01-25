#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compatibility wrapper: rebuild best_codes artifacts from meta."
    )
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--drop-missing-meta", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args, _unknown = ap.parse_known_args()

    script = Path(__file__).resolve().parent / "rebuild_best_codes_artifacts_from_meta.py"
    cmd = ["python3", str(script), "--best-dir", args.best_dir]
    print("[info] ensure_best_codes_data_json_from_meta.py -> rebuild_best_codes_artifacts_from_meta.py")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
