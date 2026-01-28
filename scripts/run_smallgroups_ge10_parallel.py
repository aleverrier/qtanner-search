#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import concurrent.futures as cf
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

GAP_LIST_TEMPLATE = r"""
for n in [{NMIN}..{NMAX}] do
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
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return p.wait()

def get_smallgroups(nmin: int, nmax: int) -> List[str]:
    gap_cmd = GAP_LIST_TEMPLATE.format(NMIN=nmin, NMAX=nmax) + "quit;"
    p = subprocess.run(["gap", "-q", "-c", gap_cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ERROR running GAP.\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    groups = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    groups = [g for g in groups if g.startswith("SmallGroup(") and g.endswith(")")]
    return groups

def parse_order(group_spec: str) -> int:
    s = group_spec.replace(" ", "")
    n = s[len("SmallGroup("):].split(",", 1)[0]
    return int(n)

def safe_log_name(group_spec: str, target: int, seed: int) -> str:
    s = group_spec.replace("SmallGroup(", "SmallGroup_").replace(",", "_").replace(")", "").replace(" ", "")
    return f"{s}_target{target}_seed{seed}.log"

def build_cmd(group_spec: str, seed: int, target: int, q_fast: int, q_slow: int, classical_backend: str) -> List[str]:
    return [
        sys.executable, "-u", "scripts/search_progressive.py",
        "--group", group_spec,
        "--target-distance", str(target),
        "--seed", str(seed),
        "--classical-distance-backend", classical_backend,
        "--quantum-steps-fast", str(q_fast),
        "--quantum-steps-slow", str(q_slow),
    ]

def main() -> int:
    ap = argparse.ArgumentParser(description="Run search_progressive.py on SmallGroup(n,i) for n in [nmin..nmax], in parallel.")
    ap.add_argument("--nmin", type=int, default=10, help="Minimum order n (default: 10)")
    ap.add_argument("--nmax", type=int, default=19, help="Maximum order n (default: 19)")
    ap.add_argument("--reverse", action="store_true", help="Queue groups in decreasing order (by n then i)")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel jobs (default: 8)")
    ap.add_argument("--seed", type=int, default=1, help="Seed passed to search_progressive.py (default: 1)")
    ap.add_argument("--run-dir", default=None, help="Output directory for logs")
    ap.add_argument("--classical-distance-backend", default="fast", help="Classical backend (default: fast)")
    args = ap.parse_args()

    groups = get_smallgroups(args.nmin, args.nmax)
    if not groups:
        print("[info] no groups found (check nmin/nmax)")
        return 0

    # Sort explicitly: increasing (n,i) or decreasing (n,i)
    parsed = []
    for g in groups:
        s = g.replace(" ", "")
        inside = s[len("SmallGroup("):-1]
        n_str, i_str = inside.split(",")
        parsed.append((int(n_str), int(i_str), s))
    parsed.sort(reverse=args.reverse)
    groups = [g for _,_,g in parsed]

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"smallgroups_{args.nmin}to{args.nmax}_{'desc' if args.reverse else 'asc'}_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] orders={args.nmin}..{args.nmax} groups={len(groups)} jobs={args.jobs} seed={args.seed} reverse={args.reverse}")
    print(f"[info] run_dir={run_dir.resolve()}")

    tasks: List[Tuple[str, List[str], Path]] = []
    for gspec in groups:
        G = parse_order(gspec)
        target = 2 * G
        q_fast = 4000
        q_slow = 100000
        cmd = build_cmd(gspec, args.seed, target, q_fast, q_slow, args.classical_distance_backend)
        log_path = run_dir / safe_log_name(gspec, target, args.seed)
        tasks.append((gspec, cmd, log_path))

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
                continue
            status = "OK" if rc == 0 else f"RC={rc}"
            print(f"[done] {gspec} {status} log={log_path.name}")

    print(f"[summary] finished groups={len(groups)}  run_dir={run_dir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
