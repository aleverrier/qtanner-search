#!/usr/bin/env python3
"""
Verify parameters of collected best codes.

Default behavior (fast):
- For each best_codes/collected/<CODE_ID>/ folder:
  - locate HX/HZ .mtx files (heuristic by filename)
  - compute rank(HX), rank(HZ) over GF(2)
  - infer n from matrix width
  - compute k_calc = n - rank(HX) - rank(HZ)
  - compare with k from CODE_ID folder name

Optional (off by default):
- Try to call scripts/estimate_distance_qdistrnd.py for a quick distance test.
  This is best-effort: we auto-detect argument names from --help output.
  Use --run-distance to enable; keep steps small.

Usage:
  python scripts/verify_collected_codes.py
  python scripts/verify_collected_codes.py --limit 5
  python scripts/verify_collected_codes.py --run-distance --steps 2000 --timeout 120
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CODE_ID_RE = re.compile(r"^(?P<group>.+?)_AA(?P<A>.+?)_BB(?P<B>.+?)_k(?P<k>\d+)_d(?P<d>\d+)$")


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(out)


def parse_code_id(code_id: str) -> Optional[Tuple[int, int]]:
    m = CODE_ID_RE.match(code_id)
    if not m:
        return None
    return int(m.group("k")), int(m.group("d"))


def read_mtx_as_row_bitmasks(fp: Path) -> Tuple[int, List[int]]:
    """
    Read a MatrixMarket coordinate integer matrix over GF(2) and return:
      ncols, rows_as_int_bitmasks (length = nrows)

    Assumptions:
    - coordinate format
    - entries represent 1s (values ignored; parity matters mod 2)
    - 1-indexed row/col in file
    """
    with fp.open("r", errors="replace") as f:
        header = f.readline()
        if not header.lower().startswith("%%matrixmarket"):
            raise ValueError(f"Not a MatrixMarket file: {fp}")

        # skip comments
        line = f.readline()
        while line and line.strip().startswith("%"):
            line = f.readline()
        if not line:
            raise ValueError(f"Malformed MatrixMarket file: {fp}")

        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Malformed size line in {fp}: {line}")
        nrows, ncols, nnz = int(parts[0]), int(parts[1]), int(parts[2])

        rows = [0] * nrows
        for _ in range(nnz):
            ln = f.readline()
            if not ln:
                break
            if ln.strip().startswith("%") or not ln.strip():
                continue
            p = ln.strip().split()
            if len(p) < 2:
                continue
            i = int(p[0]) - 1
            j = int(p[1]) - 1
            if 0 <= i < nrows and 0 <= j < ncols:
                rows[i] ^= (1 << j)  # XOR for GF(2)
        return ncols, rows


def gf2_rank(bitrows: List[int]) -> int:
    """
    Gaussian elimination over GF(2) using Python ints as bitsets.
    """
    basis: Dict[int, int] = {}
    r = 0
    for x in bitrows:
        v = x
        while v:
            msb = v.bit_length() - 1
            if msb in basis:
                v ^= basis[msb]
            else:
                basis[msb] = v
                r += 1
                break
    return r


def find_hx_hz(code_dir: Path) -> Tuple[Optional[Path], Optional[Path], List[Path]]:
    mtx = sorted(code_dir.rglob("*.mtx"))
    hx = None
    hz = None
    others: List[Path] = []

    for fp in mtx:
        name = fp.name.lower()
        if hx is None and ("hx" in name or "h_x" in name):
            hx = fp
            continue
        if hz is None and ("hz" in name or "h_z" in name):
            hz = fp
            continue
        others.append(fp)

    # fallback: if not found, take first two matrices as (hx,hz) guess
    if hx is None or hz is None:
        if len(mtx) >= 2:
            hx = hx or mtx[0]
            hz = hz or (mtx[1] if mtx[1] != hx else (mtx[2] if len(mtx) > 2 else None))

    return hx, hz, mtx


def autodetect_qdistrnd_args(help_text: str) -> Dict[str, str]:
    """
    Best-effort detection of argument names for hx/hz and steps.
    """
    def find_flag(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if re.search(rf"(^|\s){re.escape(c)}(\s|,|=|$)", help_text):
                return c
        return None

    hx_flag = find_flag(["--hx", "--HX", "--hx_path", "--hx-mtx", "--mtx-hx"])
    hz_flag = find_flag(["--hz", "--HZ", "--hz_path", "--hz-mtx", "--mtx-hz"])
    steps_flag = find_flag(["--steps", "--trials", "--num_steps", "--max_steps", "--iters"])
    return {"hx": hx_flag or "", "hz": hz_flag or "", "steps": steps_flag or ""}


def maybe_run_distance(root: Path, hx: Path, hz: Path, steps: int, timeout: int) -> Optional[str]:
    est = root / "scripts" / "estimate_distance_qdistrnd.py"
    if not est.exists():
        return None

    try:
        help_text = subprocess.check_output([sys.executable, str(est), "--help"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None

    flags = autodetect_qdistrnd_args(help_text)
    if not flags["hx"] or not flags["hz"] or not flags["steps"]:
        return None

    cmd = [sys.executable, str(est), flags["hx"], str(hx), flags["hz"], str(hz), flags["steps"], str(steps)]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
        return out.strip()
    except subprocess.TimeoutExpired:
        return f"[distance] TIMEOUT after {timeout}s: {' '.join(cmd)}"
    except subprocess.CalledProcessError as e:
        return f"[distance] ERROR: {e.output.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default="best_codes/collected", help="Folder with collected codes")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of codes to verify (0 = all)")
    ap.add_argument("--run-distance", action="store_true", help="Also run distance estimator (best-effort)")
    ap.add_argument("--steps", type=int, default=2000, help="Steps/trials for distance estimator (if enabled)")
    ap.add_argument("--timeout", type=int, default=120, help="Timeout seconds per distance run (if enabled)")
    args = ap.parse_args()

    root = repo_root()
    base = root / args.collected
    if not base.exists():
        raise SystemExit(f"Missing folder: {base}. Run collect_best_codes.py first.")

    code_dirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if args.limit and args.limit > 0:
        code_dirs = code_dirs[:args.limit]

    ok = 0
    bad = 0

    for cd in code_dirs:
        parsed = parse_code_id(cd.name)
        if not parsed:
            print(f"[skip] {cd.name} (unexpected code_id format)")
            continue
        k_expected, d_recorded = parsed

        hx, hz, mtx_all = find_hx_hz(cd)
        if hx is None or hz is None:
            print(f"[warn] {cd.name}: could not locate HX/HZ (found {len(mtx_all)} .mtx files)")
            bad += 1
            continue

        try:
            n1, rows_hx = read_mtx_as_row_bitmasks(hx)
            n2, rows_hz = read_mtx_as_row_bitmasks(hz)
            n = max(n1, n2)
            rx = gf2_rank(rows_hx)
            rz = gf2_rank(rows_hz)
            k_calc = n - rx - rz
        except Exception as e:
            print(f"[error] {cd.name}: failed to read/rank matrices: {e}")
            bad += 1
            continue

        status = "OK" if k_calc == k_expected else "MISMATCH"
        print(f"[{status}] {cd.name}: n={n} rank(HX)={rx} rank(HZ)={rz} => k_calc={k_calc}, k_expected={k_expected}, d_recorded={d_recorded}")

        if status == "OK":
            ok += 1
        else:
            bad += 1

        if args.run_distance:
            dist_out = maybe_run_distance(root, hx, hz, args.steps, args.timeout)
            if dist_out is None:
                print("  [distance] skipped (could not auto-detect estimator args or script missing)")
            else:
                # print only last few lines to keep it readable
                lines = dist_out.splitlines()
                tail = "\n".join(lines[-10:]) if len(lines) > 10 else dist_out
                print("  [distance] output tail:")
                print("  " + "\n  ".join(tail.splitlines()))

    print(f"\nSummary: OK={ok}, issues={bad}, total_checked={ok+bad}")


if __name__ == "__main__":
    main()
