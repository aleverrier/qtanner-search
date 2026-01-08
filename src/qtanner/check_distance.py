"""Re-check distance estimates for saved codes."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Optional

from .search import _qd_summary_from_stats, _run_gap_batch


def _find_code_dir(run_dir: Path, code_id: str) -> Path:
    best_root = run_dir / "best_codes"
    if best_root.exists():
        matches = sorted(
            [
                path
                for path in best_root.rglob(code_id)
                if path.is_dir() and path.name == code_id
            ]
        )
        if matches:
            return matches[0]
    promising = run_dir / "promising" / code_id
    if promising.exists():
        return promising
    raise FileNotFoundError(f"Could not find code id={code_id} under {run_dir}.")


def _format_avg(avg: Optional[float]) -> str:
    if avg is None:
        return "n/a"
    return f"{avg:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-check QDistRnd distance estimates.")
    parser.add_argument("--run", required=True, help="Run directory.")
    parser.add_argument("--id", required=True, help="Code ID to check.")
    parser.add_argument("--trials", type=int, default=20000, help="QDistRnd trials.")
    parser.add_argument("--uniq-target", type=int, default=5, help="Uniq target.")
    parser.add_argument("--gap-cmd", type=str, default="gap", help="GAP command.")
    parser.add_argument("--qd-debug", type=int, default=2, help="QDistRnd debug.")
    parser.add_argument("--qd-timeout", type=float, default=None, help="Timeout per batch.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    args = parser.parse_args()

    run_dir = Path(args.run)
    code_dir = _find_code_dir(run_dir, args.id)
    hx_path = code_dir / "Hx.mtx"
    hz_path = code_dir / "Hz.mtx"
    if not hx_path.exists() or not hz_path.exists():
        raise FileNotFoundError(f"Missing Hx.mtx/Hz.mtx under {code_dir}.")

    batch = [(0, str(hx_path), str(hz_path))]
    qd_results, runtime_sec = _run_gap_batch(
        batch=batch,
        num=args.trials,
        mindist=0,
        debug=args.qd_debug,
        seed=args.seed,
        outdir=code_dir,
        batch_id=0,
        gap_cmd=args.gap_cmd,
        timeout_sec=args.qd_timeout,
    )
    qd_stats = qd_results.get(0)
    if not qd_stats or qd_stats.get("qd_failed"):
        raise RuntimeError("QDistRnd failed for requested code.")
    qd, qd_x, qd_z, dx_ub, dz_ub, d_ub = _qd_summary_from_stats(
        qd_stats,
        num=args.trials,
        mindist=0,
        seed=args.seed or 0,
        runtime_sec=runtime_sec,
        gap_cmd=args.gap_cmd,
    )
    meta = {}
    meta_path = code_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    group = (meta.get("group") or {}).get("spec")
    n = meta.get("n")
    k = meta.get("k")

    print(
        "id group n k dx dz d uniqX avgX uniqZ avgZ doneX doneZ",
        flush=True,
    )
    print(
        f"{args.id} {group} {n} {k} {dx_ub} {dz_ub} {d_ub} "
        f"{qd_x.get('uniq')} {_format_avg(qd_x.get('avg'))} "
        f"{qd_z.get('uniq')} {_format_avg(qd_z.get('avg'))} "
        f"{qd_x.get('rounds_done')} {qd_z.get('rounds_done')}",
        flush=True,
    )

    payload = {
        "code_id": args.id,
        "run_dir": str(run_dir),
        "code_dir": str(code_dir),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "trials": args.trials,
        "uniq_target": args.uniq_target,
        "gap_cmd": args.gap_cmd,
        "qd_debug": args.qd_debug,
        "seed": args.seed,
        "qdistrnd": qd,
    }
    (code_dir / "qd_recheck.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
