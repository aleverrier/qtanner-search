#!/usr/bin/env python3
"""Refine (re-run) QDistRnd on an already-built candidate directory.

This script is intentionally self-contained and does **not** import qtanner.

It expects a candidate directory produced by scripts/run_search_w9_smallgroups.py
that contains:

  - Hx.mtx   (X-check matrix, MTX format)
  - Hz.mtx   (Z-check matrix, MTX format)

Example:

  ./scripts/py scripts/refine_qdistrnd_dir.py data/tmp/search_w9_smallgroups/o4_g2_p12_a7_b19 \
    --qd-num 2000 --qd-mindist 12

Outputs a single machine-readable line starting with "QDISTRESULT".
"""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _gap_script(hx_path: Path, hz_path: Path, qd_num: int, qd_mindist: int, qd_debug: int) -> str:
    # A robust GAP script that prints *only* integers for dX and dZ.
    # DistRandCSS can return either an int or a list depending on the package version.
    hx = str(hx_path).replace("\\", "\\\\")
    hz = str(hz_path).replace("\\", "\\\\")
    return f"""
if LoadPackage(\"QDistRnd\") = fail then
  Print(\"QDISTERROR: failed to LoadPackage(QDistRnd)\\n\");
  QuitGap(2);
fi;

hxinfo := ReadMTXE(\"{hx}\", 0);
hzinfo := ReadMTXE(\"{hz}\", 0);

HX := hxinfo[3];
HZ := hzinfo[3];

dXraw := DistRandCSS(HX, HZ, {qd_num}, {qd_mindist}, {qd_debug});
dZraw := DistRandCSS(HZ, HX, {qd_num}, {qd_mindist}, {qd_debug});

if IsInt(dXraw) then dX := dXraw; else dX := dXraw[1]; fi;
if IsInt(dZraw) then dZ := dZraw; else dZ := dZraw[1]; fi;

if IsInt(dX) = false or IsInt(dZ) = false then
  Print(\"QDISTERROR: unexpected DistRandCSS return types\\n\");
  QuitGap(3);
fi;

if dX < dZ then dObs := dX; else dObs := dZ; fi;

Print(\"QDISTRESULT dX=\", dX, \" dZ=\", dZ, \" d=\", dObs, \" qd_num=\", {qd_num}, \" mindist=\", {qd_mindist}, "");
Print(\"\\n\");
QuitGap(0);
"""


def _run_gap(gap_bin: str, script: str, timeout_s: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [gap_bin, "-q", "-b", "--quitonbreak", "--norepl"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Could not execute GAP binary: {gap_bin!r} ({e})")
    except subprocess.TimeoutExpired:
        return 124, "", f"Timed out after {timeout_s}s"
    return proc.returncode, proc.stdout, proc.stderr


def _parse_result(stdout: str) -> dict[str, int] | None:
    # Expected: QDISTRESULT dX=16 dZ=18 d=16 qd_num=2000 mindist=12
    m = re.search(r"^QDISTRESULT\s+.*$", stdout, flags=re.MULTILINE)
    if not m:
        return None
    line = m.group(0)
    out: dict[str, int] = {}
    for key in ("dX", "dZ", "d", "qd_num", "mindist"):
        km = re.search(rf"\b{key}=(\d+)\b", line)
        if km:
            out[key] = int(km.group(1))
    return out if out else None


def main() -> None:
    p = argparse.ArgumentParser(description="Refine QDistRnd on a candidate directory (Hx.mtx/Hz.mtx).")
    p.add_argument("cand_dir", type=Path, help="Candidate directory containing Hx.mtx and Hz.mtx")
    p.add_argument("--hx", type=Path, default=None, help="Override path to Hx.mtx")
    p.add_argument("--hz", type=Path, default=None, help="Override path to Hz.mtx")
    p.add_argument("--gap", default="gap", help="GAP executable (default: 'gap')")
    p.add_argument("--qd-num", type=int, default=2000, help="Number of QDistRnd trials (default: 2000)")
    p.add_argument("--qd-mindist", type=int, default=0, help="mindist passed to DistRandCSS (default: 0)")
    p.add_argument("--qd-debug", type=int, default=0, help="debug flag passed to DistRandCSS (default: 0)")
    p.add_argument("--timeout", "--qd-timeout", dest="timeout", type=int, default=120, help="Timeout in seconds (default: 120)")
    p.add_argument(
        "--print-gap-output",
        action="store_true",
        help="Also print GAP stdout/stderr (useful for debugging failures).",
    )
    args = p.parse_args()

    cand_dir = args.cand_dir
    hx_path = args.hx or (cand_dir / "Hx.mtx")
    hz_path = args.hz or (cand_dir / "Hz.mtx")

    if not hx_path.is_file():
        raise SystemExit(f"Missing Hx.mtx: {hx_path}")
    if not hz_path.is_file():
        raise SystemExit(f"Missing Hz.mtx: {hz_path}")

    script = _gap_script(hx_path, hz_path, args.qd_num, args.qd_mindist, args.qd_debug)
    rc, stdout, stderr = _run_gap(args.gap, script, args.timeout)

    result = _parse_result(stdout)
    if result is None:
        # Always emit *something* easy to grep.
        print(
            "QDISTERROR: failed to parse QDISTRESULT from GAP output. "
            f"rc={rc} gap={shlex.quote(args.gap)} dir={cand_dir}",
            file=sys.stderr,
        )
        if args.print_gap_output:
            sys.stderr.write("\n--- GAP STDOUT ---\n")
            sys.stderr.write(stdout)
            sys.stderr.write("\n--- GAP STDERR ---\n")
            sys.stderr.write(stderr)
        raise SystemExit(rc if rc != 0 else 1)

    # Always print a single line that can be parsed by log scrapers.
    print(
        "QDISTRESULT "
        f"dX={result.get('dX', -1)} dZ={result.get('dZ', -1)} d={result.get('d', -1)} "
        f"qd_num={result.get('qd_num', args.qd_num)} mindist={result.get('mindist', args.qd_mindist)} "
        f"dir={cand_dir}"
    )

    if args.print_gap_output:
        sys.stdout.write("\n--- GAP STDOUT ---\n")
        sys.stdout.write(stdout)
        sys.stdout.write("\n--- GAP STDERR ---\n")
        sys.stdout.write(stderr)

    # Exit code mirrors GAP (0 on success).
    raise SystemExit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
