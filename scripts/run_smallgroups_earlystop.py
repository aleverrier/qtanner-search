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
from typing import List, Tuple, Optional


GAP_LIST_TEMPLATE = r"""
for n in [{NMIN}..{NMAX}] do
  for i in [1..NumberSmallGroups(n)] do
    Print("SmallGroup(", n, ",", i, ")\n");
  od;
od;
"""


RE_EVAL = re.compile(r"\beval=(\d+)\b")
RE_STEPS = re.compile(r"\bsteps(?:_used)?=(\d+)\b")
RE_NEW_BEST = re.compile(r"\bNEW_BEST\b")


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_smallgroup(spec: str) -> Tuple[int, int]:
    s = spec.replace(" ", "")
    m = re.fullmatch(r"SmallGroup\((\d+),(\d+)\)", s)
    if not m:
        raise ValueError(f"Not a SmallGroup spec: {spec}")
    return int(m.group(1)), int(m.group(2))


def list_smallgroups(nmin: int, nmax: int) -> List[str]:
    gap_cmd = GAP_LIST_TEMPLATE.format(NMIN=nmin, NMAX=nmax) + "quit;"
    p = subprocess.run(["gap", "-q", "-c", gap_cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ERROR running GAP.\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    groups = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    groups = [g for g in groups if g.startswith("SmallGroup(") and g.endswith(")")]
    # Sort by (n,i) increasing
    parsed = [(parse_smallgroup(g)[0], parse_smallgroup(g)[1], g.replace(" ", "")) for g in groups]
    parsed.sort()
    return [g for _,_,g in parsed]


def safe_log_name(spec: str, seed: int, target: int) -> str:
    # SmallGroup(12,3) -> SmallGroup_12_3_target24_seed1.log
    s = spec.replace("SmallGroup(", "SmallGroup_").replace(",", "_").replace(")", "").replace(" ", "")
    return f"{s}_target{target}_seed{seed}.log"


def params_for_order(G: int) -> Tuple[int, int, int]:
    # target-distance = 2G
    target = 2 * G
    # quantum-steps-fast / slow rules (as in your requests so far)
    if G < 5:
        q_fast, q_slow = 1000, 10000
    elif G < 10:
        q_fast, q_slow = 3000, 50000
    else:
        q_fast, q_slow = 4000, 100000
    return target, q_fast, q_slow


def run_one_group(
    group_spec: str,
    seed: int,
    classical_backend: str,
    run_dir: Path,
    no_progress_steps: int,
    eval_halving: bool,
) -> Tuple[str, str]:
    """
    Run search_progressive.py for one group, and stop early if:
      (steps_since_progress >= no_progress_steps) AND (last_progress_eval <= current_eval/2)
    Progress is defined as seeing a NEW_BEST line.
    """
    n, i = parse_smallgroup(group_spec)
    G = n
    target, q_fast, q_slow = params_for_order(G)

    log_path = run_dir / safe_log_name(group_spec, seed, target)
    cmd = [
        sys.executable, "-u", "scripts/search_progressive.py",
        "--group", group_spec,
        "--target-distance", str(target),
        "--seed", str(seed),
        "--classical-distance-backend", classical_backend,
        "--quantum-steps-fast", str(q_fast),
        "--quantum-steps-slow", str(q_slow),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    current_eval: int = 0
    last_progress_eval: int = 0
    steps_since_progress: int = 0

    # We count "steps" primarily from explicit steps= / steps_used=.
    # If absent but eval advances, we count q_fast per eval (pessimistic, but makes the stop criterion work).
    last_seen_eval_for_count: int = 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n")
        f.write(f"[earlystop] no_progress_steps={no_progress_steps} eval_halving={eval_halving}\n\n")
        f.flush()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            # flush so tail -f works
            f.flush()

            # Track eval
            m_eval = RE_EVAL.search(line)
            if m_eval:
                current_eval = int(m_eval.group(1))

            # Track progress
            if RE_NEW_BEST.search(line):
                # progress at current eval (or 0 if unknown)
                last_progress_eval = current_eval or last_progress_eval
                steps_since_progress = 0
                # also reset eval counting anchor
                last_seen_eval_for_count = current_eval or last_seen_eval_for_count

            # Track steps used
            m_steps = RE_STEPS.search(line)
            if m_steps:
                steps_since_progress += int(m_steps.group(1))
            else:
                # If eval advanced and we didn't see steps in this line, pessimistically add q_fast per new eval step
                if current_eval > last_seen_eval_for_count:
                    steps_since_progress += (current_eval - last_seen_eval_for_count) * q_fast
                    last_seen_eval_for_count = current_eval

            # Early stop check (only meaningful once eval is nonzero)
            if current_eval > 0 and steps_since_progress >= no_progress_steps:
                if (not eval_halving) or (last_progress_eval <= current_eval // 2):
                    f.write(
                        f"\n[earlystop-triggered] current_eval={current_eval} "
                        f"last_progress_eval={last_progress_eval} steps_since_progress={steps_since_progress}\n"
                    )
                    f.flush()
                    # terminate and break
                    p.terminate()
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break

        # Ensure process ended
        rc = p.wait()

    status = "OK" if rc == 0 else f"RC={rc}"
    return group_spec, f"{status} log={log_path.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run search_progressive.py over SmallGroup ranges with early-stop per group.")
    ap.add_argument("--nmin", type=int, required=True)
    ap.add_argument("--nmax", type=int, required=True)
    ap.add_argument("--reverse", action="store_true", help="Process groups in decreasing (n,i) order")
    ap.add_argument("--jobs", type=int, default=1, help="How many groups to run concurrently inside this batch (default: 1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--classical-distance-backend", default="fast")
    ap.add_argument("--no-progress-steps", type=int, default=20000)
    ap.add_argument("--eval-halving", action="store_true", default=True)
    ap.add_argument("--run-dir", default=None, help="Run directory for logs")
    args = ap.parse_args()

    groups = list_smallgroups(args.nmin, args.nmax)
    if args.reverse:
        groups = list(reversed(groups))

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"smallgroups_{args.nmin}to{args.nmax}_{'desc' if args.reverse else 'asc'}_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] range={args.nmin}..{args.nmax} groups={len(groups)} jobs={args.jobs} reverse={args.reverse} seed={args.seed}")
    print(f"[info] run_dir={run_dir.resolve()}")
    print(f"[info] earlystop: no_progress_steps={args.no_progress_steps} AND no progress since eval n/2")

    if args.jobs <= 1:
        for g in groups:
            spec, msg = run_one_group(
                g, args.seed, args.classical_distance_backend, run_dir,
                args.no_progress_steps, args.eval_halving
            )
            print(f"[done] {spec} {msg}")
    else:
        # Parallel groups inside this batch
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = []
            for g in groups:
                futs.append(ex.submit(
                    run_one_group, g, args.seed, args.classical_distance_backend, run_dir,
                    args.no_progress_steps, args.eval_halving
                ))
            for fut in cf.as_completed(futs):
                spec, msg = fut.result()
                print(f"[done] {spec} {msg}")

    print(f"[summary] finished range {args.nmin}..{args.nmax}  run_dir={run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
