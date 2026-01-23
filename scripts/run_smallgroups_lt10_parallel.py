#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


GAP_LIST_CMD = r"""
for n in [1..9] do
  for i in [1..NumberSmallGroups(n)] do
    Print("SmallGroup(", n, ",", i, ")\n");
  od;
od;
"""


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.wait()


def get_smallgroups_lt10() -> List[str]:
    # Use GAP as the source of truth
    p = subprocess.run(
        ["gap", "-q", "-c", GAP_LIST_CMD + "quit;"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise SystemExit(
            "ERROR: could not run GAP to list groups.\n"
            f"stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
    groups = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    # Defensive: keep only SmallGroup(...) lines
    groups = [g for g in groups if g.startswith("SmallGroup(") and g.endswith(")")]
    return groups


def parse_order(group_spec: str) -> int:
    # group_spec is "SmallGroup(n,i)"
    m = re.fullmatch(r"SmallGroup\((\d+),(\d+)\)", group_spec.replace(" ", ""))
    if not m:
        raise ValueError(f"Unrecognized group spec: {group_spec}")
    return int(m.group(1))


def safe_log_name(group_spec: str, target: int, seed: int) -> str:
    # SmallGroup(8,3) -> SmallGroup_8_3_target16_seed1.log
    s = group_spec.replace("SmallGroup(", "SmallGroup_").replace(",", "_").replace(")", "")
    return f"{s}_target{target}_seed{seed}.log"


def build_cmd(
    group_spec: str,
    seed: int,
    target_distance: int,
    q_fast: int,
    q_slow: int,
    classical_backend: str,
) -> List[str]:
    # Use current python interpreter, unbuffered
    return [
        sys.executable, "-u", "scripts/search_progressive.py",
        "--group", group_spec,
        "--target-distance", str(target_distance),
        "--seed", str(seed),
        "--classical-distance-backend", classical_backend,
        "--quantum-steps-fast", str(q_fast),
        "--quantum-steps-slow", str(q_slow),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run scripts/search_progressive.py on all SmallGroup(n,i) for n<10, in parallel.")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel jobs (default: 8)")
    ap.add_argument("--seed", type=int, default=1, help="Seed passed to search_progressive.py (default: 1)")
    ap.add_argument("--run-dir", default=None, help="Output directory for logs (default: runs/smallgroups_lt10_<timestamp>)")
    ap.add_argument("--classical-distance-backend", default="fast", help="Classical backend (default: fast)")
    args = ap.parse_args()

    groups = get_smallgroups_lt10()

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"smallgroups_lt10_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] groups={len(groups)} jobs={args.jobs} seed={args.seed}")
    print(f"[info] run_dir={run_dir.resolve()}")

    tasks: List[Tuple[str, List[str], Path]] = []
    for gspec in groups:
        G = parse_order(gspec)
        target = 2 * G
        q_fast = 1000 if G < 5 else 3000
        q_slow = 10000 if G < 5 else 50000

        cmd = build_cmd(
            gspec,
            seed=args.seed,
            target_distance=target,
            q_fast=q_fast,
            q_slow=q_slow,
            classical_backend=args.classical_distance_backend,
        )
        log_path = run_dir / safe_log_name(gspec, target, args.seed)
        tasks.append((gspec, cmd, log_path))

    # Run in parallel; stdout is kept clean (logs go to files)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {}
        for gspec, cmd, log_path in tasks:
            print(f"[queue] {gspec} -> {log_path.name}")
            futs[ex.submit(run, cmd, log_path)] = (gspec, log_path)
        for fut in cf.as_completed(futs):
            gspec, log_path = futs[fut]
            try:
                rc = fut.result()
            except Exception as e:
                print(f"[FAIL] {gspec} exception: {e}")
                results[gspec] = ("EXC", str(log_path))
                continue
            status = "OK" if rc == 0 else f"RC={rc}"
            print(f"[done] {gspec} {status} log={log_path.name}")
            results[gspec] = (status, str(log_path))

    # Summary
    ok = sum(1 for s,_ in results.values() if s == "OK")
    print(f"[summary] OK={ok}/{len(groups)}  run_dir={run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
