#!/usr/bin/env python3
"""
scripts/refine_qdistrnd_dir.py

Run QDistRnd (via GAP) on a candidate directory that contains:
  - Hx.mtx
  - Hz.mtx

Prints a single parseable line:
  QDISTRESULT n=... k=... rX=... rZ=... dX=... dZ=... d=... qd_num=... mindist=...

The 'd' reported is min(dX,dZ) (an upper bound found by QDistRnd's random search).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _gap_script(hx_path: Path, hz_path: Path, qd_num: int, mindist: int) -> str:
    hx = str(hx_path).replace("\\", "\\\\")
    hz = str(hz_path).replace("\\", "\\\\")
    return f"""
LoadPackage("qdistRnd");
hxinfo := ReadMTXE("{hx}");
hzinfo := ReadMTXE("{hz}");
HX := hxinfo[3];
HZ := hzinfo[3];

# n,k computation (CSS): k = n - rank(HX) - rank(HZ)
n := fail;
if IsList(hxinfo) and Length(hxinfo) >= 2 and IsInt(hxinfo[2]) then
  n := hxinfo[2];
fi;
if n = fail then
  if IsList(HX) and Length(HX) > 0 and IsList(HX[1]) then
    n := Length(HX[1]);
  fi;
fi;

rX := RankMat(HX);
rZ := RankMat(HZ);
k := n - rX - rZ;

mindist := {mindist};
res := DistRandCSS(HX, HZ, {qd_num}, mindist);
dX := res[1];
dZ := res[2];
dObs := Minimum(dX, dZ);

Print("QDISTRESULT n=", n, " k=", k, " rX=", rX, " rZ=", rZ,
      " dX=", dX, " dZ=", dZ, " d=", dObs, " qd_num=", {qd_num}, " mindist=", mindist, "\\n");
QuitGap(0);
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Candidate directory containing Hx.mtx and Hz.mtx")
    p.add_argument("--qd-num", type=int, default=2000, help="QDistRnd random trials")
    p.add_argument("--mindist", type=int, default=1, help="Minimum distance parameter passed to DistRandCSS")
    p.add_argument("--timeout", "--qd-timeout", dest="timeout_s", type=int, default=240, help="Timeout in seconds")
    p.add_argument("--gap-cmd", default=os.environ.get("QTANNER_GAP", "gap"), help="GAP executable")
    args = p.parse_args()

    d = Path(args.dir)
    hx_path = d / "Hx.mtx"
    hz_path = d / "Hz.mtx"
    if not hx_path.exists() or not hz_path.exists():
        print(f"[refine] ERROR: missing Hx.mtx/Hz.mtx in {d}", file=sys.stderr)
        return 2

    script = _gap_script(hx_path, hz_path, qd_num=args.qd_num, mindist=args.mindist)
    try:
        proc = subprocess.run(
            [args.gap_cmd, "-q", "-b", "--quitonbreak", "--norepl"],
            input=script,
            text=True,
            capture_output=True,
            timeout=args.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[refine] ERROR: GAP timed out after {args.timeout_s}s", file=sys.stderr)
        return 124

    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
