#!/usr/bin/env python3
"""
scripts/run_one_group_w9_twopass.py

Two-pass driver:
  Pass 1: run_one_group_w9.py with a small qd-num (fast scan)
  Pass 2: refine a subset of candidate dirs with a larger qd-num

This script writes a JSONL file with one record per refinement attempt, including:
  - n, k, d_ub, kd_over_n, d_over_sqrt_n (when available)

It relies on the candidate directories produced in --tmpdir, each containing:
  - Hx.mtx, Hz.mtx
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


_CAND_RE = re.compile(r"^o(?P<order>\d+)_g(?P<gid>\d+)_p(?P<pair>\d+)_a(?P<a1v>\d+)_b(?P<b1v>\d+)$")


def _utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _run(cmd: List[str], *, cwd: Path | None = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _tail(s: str, n: int = 800) -> str:
    s = s or ""
    return s[-n:]


def _parse_qdistresult(stdout: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not stdout:
        return out
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("QDISTRESULT")]
    if not lines:
        return out
    ln = lines[-1]
    for tok in ln.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            continue
    return out


def _scan_tmp(tmp_root: Path) -> List[Path]:
    if not tmp_root.exists():
        return []
    dirs: List[Path] = []
    for p in sorted(tmp_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "Hx.mtx").exists() and (p / "Hz.mtx").exists():
            dirs.append(p)
    return dirs


def _pick_for_refine(tmp_root: Path, refine_max: int) -> List[Path]:
    dirs = _scan_tmp(tmp_root)
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    if refine_max > 0:
        dirs = dirs[:refine_max]
    return dirs


def _cand_meta_from_dirname(d: Path) -> Dict[str, int]:
    m = _CAND_RE.match(d.name)
    if not m:
        return {}
    return {k: int(v) for k, v in m.groupdict().items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, help="SmallGroup id as 'order,gid' (e.g. 4,2)")
    p.add_argument("--pairs", default="1-200", help="Pair range like 1-200")
    p.add_argument("--qd-fast", type=int, default=100, help="Pass-1 qd-num")
    p.add_argument("--qd-refine", type=int, default=2000, help="Refinement qd-num")
    p.add_argument("--qd-timeout", "--qd-fast-timeout", dest="qd_fast_timeout", type=int, default=240, help="Pass-1 timeout (seconds)")
    p.add_argument("--qd-refine-timeout", dest="qd_refine_timeout", type=int, default=240, help="Refinement timeout (seconds)")
    p.add_argument("--refine-max", type=int, default=50, help="Max candidate dirs to refine (0 = no cap)")
    p.add_argument("--tmpdir", default="data/tmp/search_w9_smallgroups_twopass", help="Where pass-1 writes candidate dirs")
    p.add_argument("--outdir", default="data/results", help="Where to write the JSONL output")
    p.add_argument("--no-pass1", action="store_true", help="Skip pass 1 and only refine existing dirs in --tmpdir")
    p.add_argument("--dry-run", action="store_true", help="Print what would run, but do not execute")
    args, passthrough = p.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_one = repo_root / "scripts" / "run_one_group_w9.py"
    refine = repo_root / "scripts" / "refine_qdistrnd_dir.py"

    tmp_root = Path(args.tmpdir).resolve()
    results_root = Path(args.outdir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    if not args.no_pass1:
        pass1_cmd = [
            sys.executable,
            str(run_one),
            "--group",
            args.group,
            "--pairs",
            args.pairs,
            "--tmpdir",
            str(tmp_root),
            "--outdir",
            str(results_root),
            "--trial-cap",
            "0",
            "--qd-num",
            str(args.qd_fast),
            "--qd-batches",
            "1,1",
            "--qd-timeout",
            str(args.qd_fast_timeout),
            *passthrough,
        ]
        print("[twopass] Pass 1 cmd:", " ".join(pass1_cmd))
        if not args.dry_run:
            rc, out, err = _run(pass1_cmd, cwd=repo_root)
            if rc != 0:
                print(f"[twopass] Pass 1 exited with code {rc}. Continuing to refinement anyway.", file=sys.stderr)
            if out:
                sys.stdout.write(out)
            if err:
                sys.stderr.write(err)

    refine_dirs = _pick_for_refine(tmp_root, args.refine_max)
    print(f"[twopass] Selected {len(refine_dirs)} candidate(s) for refinement.")

    out_path = results_root / f"refine_w9_twopass_{args.group.replace(',', '_')}_{_utc_ts()}.jsonl"

    ok = 0
    with open(out_path, "w") as f:
        for d in refine_dirs:
            cmd = [
                sys.executable,
                str(refine),
                "--dir",
                str(d),
                "--qd-num",
                str(args.qd_refine),
                "--qd-timeout",
                str(args.qd_refine_timeout),
            ]
            if args.dry_run:
                rec = {"cand_dir": str(d), "cmd": cmd, "dry_run": True}
                f.write(json.dumps(rec) + "\n")
                continue

            rc, out, err = _run(cmd, cwd=repo_root)
            parsed = _parse_qdistresult(out)
            rec: Dict[str, Any] = {
                "group": args.group,
                "cand_dir": str(d),
                "cmd": cmd,
                "returncode": rc,
                "stdout_tail": _tail(out),
                "stderr_tail": _tail(err),
                "parsed": parsed,
                **_cand_meta_from_dirname(d),
            }

            for key in ("n", "k", "rX", "rZ", "dX", "dZ", "d", "qd_num", "mindist"):
                if key in parsed:
                    rec[key] = parsed[key]
            if "d" in parsed:
                rec["d_ub"] = parsed["d"]

            if isinstance(rec.get("n"), int) and isinstance(rec.get("k"), int) and isinstance(rec.get("d_ub"), int) and rec["n"] > 0:
                n = rec["n"]
                k = rec["k"]
                d_ub = rec["d_ub"]
                rec["kd_over_n"] = (k * d_ub) / n
                rec["d_over_sqrt_n"] = d_ub / math.sqrt(n)

            if rc == 0 and "d_ub" in rec:
                ok += 1

            f.write(json.dumps(rec) + "\n")

    print(f"[twopass] refine ok={ok}/{len(refine_dirs)}")
    print("[twopass] wrote", out_path)

    # Summary
    try:
        rows: List[Tuple[float, Dict[str, Any]]] = []
        with open(out_path, "r") as f:
            for line in f:
                r = json.loads(line)
                kd = r.get("kd_over_n")
                if isinstance(kd, (int, float)):
                    rows.append((float(kd), r))
        rows.sort(key=lambda t: t[0], reverse=True)
        if rows:
            print("[twopass] Top 10 by kd/n:")
            for kd, r in rows[:10]:
                print(f"  kd/n={kd:.3f}  n={r.get('n')} k={r.get('k')} d_ub={r.get('d_ub')}  cand={r.get('cand_dir')}")
        else:
            print("[twopass] No refined records contained kd_over_n (maybe matrices missing or refinement failed).")
    except Exception as e:
        print(f"[twopass] Summary failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
