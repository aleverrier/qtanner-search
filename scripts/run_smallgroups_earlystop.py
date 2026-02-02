#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()

import argparse
import concurrent.futures as cf
import datetime as dt
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
    parsed = [(parse_smallgroup(g)[0], parse_smallgroup(g)[1], g.replace(" ", "")) for g in groups]
    parsed.sort()
    return [g for _, _, g in parsed]


def safe_log_name(spec: str, seed: int, target: int) -> str:
    s = spec.replace("SmallGroup(", "SmallGroup_").replace(",", "_").replace(")", "").replace(" ", "")
    return f"{s}_target{target}_seed{seed}.log"


def run_one_group(
    group_spec: str,
    seed: int,
    classical_backend: str,
    run_dir: Path,
    no_progress_steps: int,
    eval_halving: bool,
    *,
    local_code: str,
    local_n: int,
    target_distance: int,
    classical_target: int,
    quantum_steps_fast: int,
    slow_trials_override: Optional[int],
    best_codes_source: str,
    no_progress_evals: int,
    extra_args: List[str],
) -> Tuple[str, str]:
    n, _ = parse_smallgroup(group_spec)
    log_path = run_dir / safe_log_name(group_spec, seed, target_distance)
    cmd = [
        sys.executable, "-u", "scripts/search_progressive.py",
        "--group", group_spec,
        "--target-distance", str(target_distance),
        "--seed", str(seed),
        "--classical-distance-backend", classical_backend,
        "--quantum-steps-fast", str(quantum_steps_fast),
        "--best-codes-source", best_codes_source,
        "--local-a", local_code,
        "--local-b", local_code,
        "--classical-target", str(classical_target),
    ]
    if slow_trials_override is not None:
        cmd.extend(["--slow-quantum-trials-override", str(slow_trials_override)])
    if extra_args:
        cmd.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    current_eval: int = 0
    last_progress_eval: int = 0
    steps_since_progress: int = 0
    last_seen_eval_for_count: int = 0

    eff_no_progress_steps = no_progress_steps
    if n >= 16:
        eff_no_progress_steps = max(eff_no_progress_steps, 100000)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n")
        f.write(f"[earlystop] no_progress_steps={eff_no_progress_steps} eval_halving={eval_halving} no_progress_evals={no_progress_evals}\n\n")
        f.flush()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            f.flush()

            m_eval = RE_EVAL.search(line)
            if m_eval:
                current_eval = int(m_eval.group(1))

            if RE_NEW_BEST.search(line):
                last_progress_eval = current_eval or last_progress_eval
                steps_since_progress = 0
                last_seen_eval_for_count = current_eval or last_seen_eval_for_count

            m_steps = RE_STEPS.search(line)
            if m_steps:
                steps_since_progress += int(m_steps.group(1))
            else:
                if current_eval > last_seen_eval_for_count:
                    steps_since_progress += (current_eval - last_seen_eval_for_count) * quantum_steps_fast
                    last_seen_eval_for_count = current_eval

            if no_progress_evals > 0 and current_eval > 0:
                if current_eval - last_progress_eval >= no_progress_evals:
                    f.write(
                        f"\n[earlystop-triggered] current_eval={current_eval} "
                        f"last_progress_eval={last_progress_eval} no_progress_evals={no_progress_evals}\n"
                    )
                    f.flush()
                    p.terminate()
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break
            elif current_eval > 0 and steps_since_progress >= eff_no_progress_steps:
                if (not eval_halving) or (last_progress_eval <= current_eval // 2):
                    f.write(
                        f"\n[earlystop-triggered] current_eval={current_eval} "
                        f"last_progress_eval={last_progress_eval} steps_since_progress={steps_since_progress}\n"
                    )
                    f.flush()
                    p.terminate()
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break

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
    ap.add_argument("--no-progress-evals", type=int, default=0, help="Stop if no NEW_BEST for this many evals (0=disabled)")
    ap.add_argument("--eval-halving", action="store_true", default=True)
    ap.add_argument("--run-dir", default=None, help="Run directory for logs")
    ap.add_argument("--local-code", required=True, choices=("6_3_3", "8_4_4", "2_1_2"))
    ap.add_argument("--max-code-length", type=int, default=0, help="Skip groups with n>max (0=disable)")
    ap.add_argument("--target-distance", type=int, default=0, help="Override target distance (0=auto)")
    ap.add_argument("--classical-target", type=int, default=0, help="Override classical target (0=auto)")
    ap.add_argument("--quantum-steps-fast", type=int, default=2000)
    ap.add_argument("--slow-quantum-trials-override", type=int, default=0)
    ap.add_argument("--best-codes-source", default="website")
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra args forwarded to search_progressive.py")
    args = ap.parse_args()

    groups = list_smallgroups(args.nmin, args.nmax)
    if args.reverse:
        groups = list(reversed(groups))

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"smallgroups_{args.nmin}to{args.nmax}_{'desc' if args.reverse else 'asc'}_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] range={args.nmin}..{args.nmax} groups={len(groups)} jobs={args.jobs} reverse={args.reverse} seed={args.seed}")
    print(f"[info] run_dir={run_dir.resolve()}")
    print(f"[info] earlystop: no_progress_steps={args.no_progress_steps} AND no progress since eval n/2")
    if args.no_progress_evals:
        print(f"[info] earlystop: no_progress_evals={args.no_progress_evals}")

    local_n_map = {"6_3_3": 6, "8_4_4": 8, "2_1_2": 2}
    local_n = local_n_map[args.local_code]
    max_len = args.max_code_length

    def maybe_run(gspec: str) -> Optional[Tuple[str, str]]:
        n, _ = parse_smallgroup(gspec)
        code_len = (local_n * local_n) * n
        if max_len and code_len > max_len:
            return None
        if args.target_distance > 0:
            target = args.target_distance
        else:
            target = 4 if code_len < 100 else 8
        if args.classical_target > 0:
            classical_target = args.classical_target
        else:
            classical_target = target
        slow_override = args.slow_quantum_trials_override if args.slow_quantum_trials_override > 0 else None
        return run_one_group(
            gspec,
            args.seed,
            args.classical_distance_backend,
            run_dir,
            args.no_progress_steps,
            args.eval_halving,
            local_code=args.local_code,
            local_n=local_n,
            target_distance=target,
            classical_target=classical_target,
            quantum_steps_fast=args.quantum_steps_fast,
            slow_trials_override=slow_override,
            best_codes_source=args.best_codes_source,
            no_progress_evals=args.no_progress_evals,
            extra_args=args.extra_arg,
        )

    if args.jobs <= 1:
        for g in groups:
            result = maybe_run(g)
            if result is None:
                continue
            spec, msg = result
            print(f"[done] {spec} {msg}")
    else:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = []
            for g in groups:
                futs.append(ex.submit(maybe_run, g))
            for fut in cf.as_completed(futs):
                result = fut.result()
                if result is None:
                    continue
                spec, msg = result
                print(f"[done] {spec} {msg}")

    print(f"[summary] finished range {args.nmin}..{args.nmax}  run_dir={run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
