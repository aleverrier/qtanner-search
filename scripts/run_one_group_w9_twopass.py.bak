#!/usr/bin/env python3
"""
Two-pass driver for the W=9 small-groups search.

Pass 1: run the regular search with a small QDistRnd budget (fast screening).
Pass 2: automatically re-run QDistRnd with a larger budget for the most promising
        candidates (selected either from best_by_nk* files or by scanning tmp dirs).

This script is designed to be forgiving with CLI flags, so you can keep using:

  --group (alias of --only-group)
  --tmpdir (alias of --tmp-root)
  --outdir (alias of --results-root)
  --qd-timeout (alias of --qd-fast-timeout)

All other unknown flags are passed through to the underlying search script.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    # scripts/run_one_group_w9_twopass.py -> repo root
    return Path(__file__).resolve().parents[1]


def _py() -> str:
    return sys.executable


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _subprocess_env() -> Dict[str, str]:
    """
    Ensure 'src' is on PYTHONPATH for child processes, since this repo may not be a pip-installable package.
    """
    env = dict(os.environ)
    src_dir = _repo_root() / "src"
    existing = env.get("PYTHONPATH", "")
    parts = [str(src_dir)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join([p for p in parts if p])
    return env


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str], label: str) -> int:
    print(f"[twopass] {label} cmd: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
        return int(proc.returncode)
    except KeyboardInterrupt:
        print("[twopass] interrupted by user.")
        return 130


def _latest_best_by_nk(results_root: Path) -> Optional[Path]:
    cands = sorted(results_root.glob("best_by_nk*.jsonl"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _read_best_by_nk_dirs(best_path: Path, *, max_dirs: int) -> List[Path]:
    """
    Read candidate dirs from a best_by_nk*.jsonl file.

    We accept several possible key names, because the upstream script may evolve:
      - cand_dir
      - dir
      - tmpdir
      - tmp_dir
    """
    out: List[Path] = []
    seen: set[str] = set()
    with best_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cand = (
                rec.get("cand_dir")
                or rec.get("dir")
                or rec.get("tmpdir")
                or rec.get("tmp_dir")
            )
            if not cand:
                continue
            cand_path = Path(str(cand))
            key = str(cand_path.resolve())
            if key in seen:
                continue
            if cand_path.exists() and cand_path.is_dir():
                out.append(cand_path)
                seen.add(key)
            if len(out) >= max_dirs:
                break
    return out


def _scan_tmp_for_candidate_dirs(tmp_root: Path, *, max_dirs: int) -> List[Path]:
    """
    Heuristic: candidate dirs look like .../o{order}_g{groupid}_p{pair}_a{a}_b{b}
    and contain hx/hz matrix files, or at least slice logs.

    We just take the most recently modified directories up to max_dirs.
    """
    if not tmp_root.exists():
        return []
    dirs = [p for p in tmp_root.iterdir() if p.is_dir()]
    # Prefer leaf dirs (the actual candidates)
    dirs = [p for p in dirs if any((p / name).exists() for name in ("hx.txt", "hz.txt", "slice_a.log", "slice_b.log", "meta.json"))]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[:max_dirs]


def _refine_one(
    refine_script: Path,
    cand_dir: Path,
    *,
    qd_num: int,
    timeout_s: int,
    env: Dict[str, str],
    cwd: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run refine_qdistrnd_dir.py on a single candidate directory and return:
      (ok, record)
    """
    cmd = [
        _py(),
        str(refine_script),
        str(cand_dir),
        "--qd-num",
        str(qd_num),
        "--timeout",
        str(timeout_s),
    ]
    print(f"[twopass] refine: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    ok = (p.returncode == 0)

    rec: Dict[str, Any] = {
        "timestamp": _utc_timestamp(),
        "cand_dir": str(cand_dir),
        "cmd": cmd,
        "returncode": int(p.returncode),
        "stdout_tail": "\n".join(p.stdout.splitlines()[-25:]),
        "stderr_tail": "\n".join(p.stderr.splitlines()[-25:]),
    }

    # If the refine script printed a JSON line, try to parse it.
    # (We keep it best-effort; failures still record stdout/stderr tails.)
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                j = json.loads(line)
                if isinstance(j, dict):
                    rec.update({"refine": j})
            except Exception:
                pass
            break

    return ok, rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass W9 smallgroups search (fast screen + refine).")

    # Accept either spelling; store in the same dest.
    parser.add_argument(
        "--only-group",
        "--group",
        dest="only_group",
        required=True,
        help="Group identifier, e.g. 4,2",
    )

    # Convenience aliases for directory flags
    parser.add_argument(
        "--tmp-root",
        "--tmpdir",
        dest="tmp_root",
        default="data/tmp/search_w9_smallgroups",
        help="Where candidate directories are written (pass1) and scanned (pass2).",
    )
    parser.add_argument(
        "--results-root",
        "--outdir",
        dest="results_root",
        default="data/results",
        help="Where best_by_nk*.jsonl (pass1) and refine results (pass2) are written.",
    )

    # Two-pass parameters
    parser.add_argument("--qd-fast", type=int, default=100, help="Pass 1 QDistRnd budget (number of random trials).")
    parser.add_argument(
        "--qd-fast-timeout",
        "--qd-timeout",
        dest="qd_fast_timeout",
        type=int,
        default=240,
        help="Pass 1 per-call timeout (seconds) for GAP/QDistRnd.",
    )
    parser.add_argument("--qd-refine", type=int, default=2000, help="Pass 2 QDistRnd budget (number of random trials).")
    parser.add_argument(
        "--qd-refine-timeout",
        dest="qd_refine_timeout",
        type=int,
        default=240,
        help="Pass 2 per-call timeout (seconds) for GAP/QDistRnd.",
    )

    parser.add_argument(
        "--refine-max",
        type=int,
        default=25,
        help="Maximum number of candidate dirs to refine in pass 2.",
    )
    parser.add_argument(
        "--refine-from",
        choices=["best_by_nk", "scan_tmp"],
        default="best_by_nk",
        help="How to choose candidates for refinement.",
    )
    parser.add_argument("--no-pass1", action="store_true", help="Skip pass 1; only do refinement.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")

    args, passthrough = parser.parse_known_args()

    repo = _repo_root()
    scripts_dir = repo / "scripts"
    run_one_group = scripts_dir / "run_one_group_w9.py"
    refine_script = scripts_dir / "refine_qdistrnd_dir.py"

    if not run_one_group.exists():
        raise SystemExit(f"Missing script: {run_one_group}")
    if not refine_script.exists():
        raise SystemExit(f"Missing script: {refine_script}")

    tmp_root = (repo / args.tmp_root).resolve() if not Path(args.tmp_root).is_absolute() else Path(args.tmp_root).resolve()
    results_root = (repo / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root).resolve()
    _ensure_dir(tmp_root)
    _ensure_dir(results_root)

    env = _subprocess_env()

    # --- Pass 1 ---
    pass1_rc = 0
    if not args.no_pass1:
        # For a truly "fast" screen, we force qd-batches to 1,1 by default (can be overridden via passthrough).
        pass1_cmd = [
            _py(),
            str(run_one_group),
            "--only-group",
            str(args.only_group),
            "--tmpdir",
            str(tmp_root),
            "--outdir",
            str(results_root),
            "--qd-num",
            str(args.qd_fast),
            "--qd-batches",
            "1,1",
            "--qd-timeout",
            str(args.qd_fast_timeout),
            "--trial-cap",
            "0",
        ] + passthrough

        if args.dry_run:
            print("[twopass] DRY RUN: would run pass 1.")
        else:
            pass1_rc = _run(pass1_cmd, cwd=repo, env=env, label="Pass 1")
            if pass1_rc != 0:
                print(f"[twopass] Pass 1 exited with code {pass1_rc}. Continuing to refinement anyway.")

    # --- Choose candidates for refinement ---
    cand_dirs: List[Path] = []
    if args.refine_from == "best_by_nk":
        best_path = _latest_best_by_nk(results_root)
        if best_path is None:
            print("[twopass] WARNING: could not find any best_by_nk* file; falling back to scan_tmp.")
            cand_dirs = _scan_tmp_for_candidate_dirs(tmp_root, max_dirs=args.refine_max)
        else:
            print(f"[twopass] Using best_by_nk file: {best_path}")
            cand_dirs = _read_best_by_nk_dirs(best_path, max_dirs=args.refine_max)
            if not cand_dirs:
                print("[twopass] WARNING: best_by_nk contained no usable dirs; falling back to scan_tmp.")
                cand_dirs = _scan_tmp_for_candidate_dirs(tmp_root, max_dirs=args.refine_max)
    else:
        cand_dirs = _scan_tmp_for_candidate_dirs(tmp_root, max_dirs=args.refine_max)

    print(f"[twopass] Selected {len(cand_dirs)} dirs for refinement from {args.refine_from}.")

    out_path = results_root / f"refine_w9_twopass_{str(args.only_group).replace(',', '_')}_{_utc_timestamp().replace(':','').replace('-','')}.jsonl"

    if args.dry_run:
        print(f"[twopass] DRY RUN: would write refinement results to {out_path}")
        return

    # --- Pass 2: refine ---
    ok_count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for d in cand_dirs:
            ok, rec = _refine_one(
                refine_script,
                d,
                qd_num=args.qd_refine,
                timeout_s=args.qd_refine_timeout,
                env=env,
                cwd=repo,
            )
            if ok:
                ok_count += 1
            out_f.write(json.dumps(rec, sort_keys=True) + "\n")
            out_f.flush()

    print(f"[twopass] refine ok={ok_count}/{len(cand_dirs)}")
    print(f"[twopass] wrote {out_path}")


if __name__ == "__main__":
    main()
