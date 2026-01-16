"""Re-check distance estimates for saved codes."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .classical_distance import read_mtx_coordinate_binary
from .dist_m4ri import dist_m4ri_is_available, run_dist_m4ri_css_rw


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-check distance estimates using dist-m4ri (RW)."
    )
    parser.add_argument("--run", required=True, help="Run directory.")
    parser.add_argument("--id", required=True, help="Code ID to check.")
    parser.add_argument("--steps", type=int, default=5000, help="dist_m4ri RW steps.")
    parser.add_argument(
        "--target-distance",
        type=int,
        default=None,
        help="Reject distances below this target (sets wmin=target-1).",
    )
    parser.add_argument(
        "--dist-m4ri-cmd", type=str, default="dist_m4ri", help="dist_m4ri command."
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    args = parser.parse_args()

    run_dir = Path(args.run)
    code_dir = _find_code_dir(run_dir, args.id)
    hx_path = code_dir / "Hx.mtx"
    hz_path = code_dir / "Hz.mtx"
    if not hx_path.exists() or not hz_path.exists():
        raise FileNotFoundError(f"Missing Hx.mtx/Hz.mtx under {code_dir}.")

    if not dist_m4ri_is_available(args.dist_m4ri_cmd):
        raise RuntimeError(
            "dist_m4ri not found on PATH; install dist-m4ri and ensure the dist_m4ri "
            "binary is available (see README.md#dist-m4ri)."
        )
    _, n_cols, hx_rows = read_mtx_coordinate_binary(hx_path)
    _, n_cols_z, hz_rows = read_mtx_coordinate_binary(hz_path)
    if n_cols_z != n_cols:
        raise ValueError("Hx/Hz column counts do not match.")
    wmin = 0 if args.target_distance is None else max(0, args.target_distance - 1)
    seed = args.seed or 0
    dz_signed = run_dist_m4ri_css_rw(
        hx_rows,
        hz_rows,
        n_cols,
        steps=args.steps,
        wmin=wmin,
        seed=seed,
        dist_m4ri_cmd=args.dist_m4ri_cmd,
    )
    dz_est = abs(dz_signed)
    early_stop_z = dz_signed < 0
    dx_signed = None
    dx_est = None
    early_stop_x = None
    if args.target_distance is None or (not early_stop_z and dz_est >= args.target_distance):
        dx_signed = run_dist_m4ri_css_rw(
            hz_rows,
            hx_rows,
            n_cols,
            steps=args.steps,
            wmin=wmin,
            seed=seed,
            dist_m4ri_cmd=args.dist_m4ri_cmd,
        )
        dx_est = abs(dx_signed)
        early_stop_x = dx_signed < 0
    d_est = dz_est if dx_est is None else min(dx_est, dz_est)
    meta = {}
    meta_path = code_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    group = (meta.get("group") or {}).get("spec")
    n = meta.get("n")
    k = meta.get("k")

    print("id group n k dx dz d steps wmin earlyX earlyZ", flush=True)
    print(
        f"{args.id} {group} {n} {k} {dx_est} {dz_est} {d_est} "
        f"{args.steps} {wmin} {early_stop_x} {early_stop_z}",
        flush=True,
    )

    payload = {
        "code_id": args.id,
        "run_dir": str(run_dir),
        "code_dir": str(code_dir),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "steps": args.steps,
        "target_distance": args.target_distance,
        "dist_m4ri_cmd": args.dist_m4ri_cmd,
        "seed": args.seed,
        "distance_estimate": {
            "method": "dist_m4ri_rw",
            "steps": args.steps,
            "wmin": wmin,
            "rng_seed": seed,
            "dx_est": dx_est,
            "dz_est": dz_est,
            "d_est": d_est,
            "dx_signed": dx_signed,
            "dz_signed": dz_signed,
            "early_stop_x": early_stop_x,
            "early_stop_z": early_stop_z,
        },
    }
    (code_dir / "qd_recheck.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
