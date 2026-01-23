#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def ensure_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def find_dist_m4ri(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists() and p.is_file():
            return p
        raise SystemExit(f"ERROR: --dist-m4ri '{explicit}' not found.")
    for env in ("DIST_M4RI", "QTANNER_DIST_M4RI"):
        v = os.environ.get(env)
        if v:
            p = Path(v).expanduser()
            if p.exists() and p.is_file():
                return p
    w = shutil.which("dist_m4ri")
    if w:
        return Path(w)
    for p in [
        Path.home() / ".local/bin/dist_m4ri",
        Path.home() / "research/qtanner-tools/dist-m4ri/src/dist_m4ri",
    ]:
        p = p.expanduser().resolve()
        if p.exists() and p.is_file():
            return p
    raise SystemExit("ERROR: dist_m4ri not found on PATH and no known fallback paths exist.")

@dataclass
class MatPair:
    hx: Path
    hz: Path

def find_matrix_pair(best_dir: Path, code_id: str) -> MatPair:
    mats = best_dir / "matrices"
    if not mats.exists():
        raise SystemExit(f"ERROR: matrices folder not found: {mats}")

    def pick(kind: str) -> Path:
        if kind not in ("x", "z"):
            raise ValueError("kind must be x or z")
        K = kind.upper()
        patterns = [
            f"{code_id}__H{kind}.mtx", f"{code_id}*__H{kind}.mtx",
            f"{code_id}__H{K}.mtx",    f"{code_id}*__H{K}.mtx",
            f"{code_id}*{K}.mtx",
        ]
        cands = []
        seen = set()
        for pat in patterns:
            for q in mats.glob(pat):
                if q.is_file() and q.name not in seen:
                    seen.add(q.name)
                    cands.append(q)
        if not cands:
            raise FileNotFoundError(f"no H{kind} matrix for {code_id} under {mats}")
        cands.sort(key=lambda q: (len(q.name), q.name))
        return cands[0]

    return MatPair(hx=pick("x"), hz=pick("z"))

def parse_dist_m4ri_output(text: str) -> int:
    # Try a few patterns, but if none match we will log the raw output.
    for pat in [r"\bdmin=(\d+)\b", r"\bd_ub=(\d+)\b", r"\bd=(\d+)\b"]:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    # Fallback: last integer on last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        m = re.search(r"(\d+)\s*$", lines[-1])
        if m:
            return int(m.group(1))
    raise ValueError("Could not parse distance from dist_m4ri output.")

def _write_fail_log(code_id: str, side: str, steps: int, seed: int, out: str) -> Path:
    logs = Path("logs") / "m4ri_refine_failures"
    logs.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs / f"{code_id}__{side}__steps{steps}__seed{seed}__{ts}.log"
    path.write_text(out)
    return path

def run_dist_m4ri(
    exe: Path,
    finH: Path,
    finG: Path,
    steps: int,
    seed: int,
    timeout_s: Optional[int],
    code_id: str,
    side: str,
) -> Optional[int]:
    cmd = [
        str(exe),
        "debug=0",
        "method=1",
        f"steps={steps}",
        "wmin=1",
        f"seed={seed}",
        f"finH={str(finH)}",
        f"finG={str(finG)}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    try:
        d = parse_dist_m4ri_output(out)
        return d
    except Exception as ex:
        # Log the FULL output for diagnosis and skip this side
        log = _write_fail_log(code_id, side, steps, seed, out)
        print(f"[fail] {code_id} side={side}: parse failed ({ex}); logged: {log}")
        return None

def get_prev_steps(meta: Dict[str, Any]) -> int:
    for k in ("m4ri_steps", "trials", "steps"):
        if k in meta:
            return ensure_int(meta.get(k), 0)
    dm = meta.get("distance_m4ri")
    if isinstance(dm, dict):
        return ensure_int(dm.get("steps_per_side", 0), 0)
    return 0

def inject_m4ri_results(meta: Dict[str, Any], *, steps: int, seed: int, dX: int, dZ: int) -> Dict[str, Any]:
    ts = utc_now_iso()
    d = min(dX, dZ)

    if "distance" in meta and "distance_prev" not in meta:
        meta["distance_prev"] = meta["distance"]

    meta["distance_backend"] = "dist-m4ri"
    meta["distance_method"] = "RW"
    meta["m4ri_steps"] = steps
    meta["m4ri_seed"] = seed
    meta["dX_ub"] = dX
    meta["dZ_ub"] = dZ
    meta["d_ub"] = d
    meta["d"] = d
    meta["trials"] = steps
    meta["updated_at"] = ts

    meta["distance_m4ri"] = {
        "backend": "dist-m4ri",
        "method": "RW",
        "steps_per_side": steps,
        "seed_base": seed,
        "dX_ub": dX,
        "dZ_ub": dZ,
        "d_ub": d,
        "updated_at": ts,
    }

    # This is what your website currently displays
    meta["distance"] = {
        "method": "dist-m4ri",
        "dX_best": dX,
        "dZ_best": dZ,
        "d_ub": d,
        "steps_used_x": steps,
        "steps_used_z": steps,
        "steps_used_total": 2 * steps,
        "fast": {
            "dx": {"d_ub": dX, "steps": steps, "seed": seed + 1, "signed": dX, "early_stop": False},
            "dz": {"d_ub": dZ, "steps": steps, "seed": seed,     "signed": dZ, "early_stop": False},
        },
    }
    return meta

def main() -> int:
    ap = argparse.ArgumentParser(description="Refine best_codes distances using dist_m4ri (no GAP). Updates collected/*/meta.json too.")
    ap.add_argument("--group", required=True)
    ap.add_argument("--trials", type=int, required=True, help="RW steps per side (X and Z).")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--dist-m4ri", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    collected_dir = best_dir / "collected"
    meta_dir = best_dir / "meta"
    if not collected_dir.is_dir():
        raise SystemExit(f"ERROR: {collected_dir} not found.")
    meta_dir.mkdir(parents=True, exist_ok=True)

    exe = find_dist_m4ri(args.dist_m4ri)

    prefix = args.group + "_"
    code_ids = sorted([p.name for p in collected_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    print(f"[info] best_dir={best_dir}")
    print(f"[info] dist_m4ri={exe}")
    print(f"[info] group={args.group} codes={len(code_ids)} trials={args.trials} seed={args.seed}")

    updated = 0
    skipped = 0

    for i, code_id in enumerate(code_ids, start=1):
        meta_summary_path = meta_dir / f"{code_id}.json"
        meta_collected_path = collected_dir / code_id / "meta.json"
        meta_summary = read_json(meta_summary_path)
        meta_collected = read_json(meta_collected_path)

        prev = max(get_prev_steps(meta_summary), get_prev_steps(meta_collected))
        if (not args.force) and args.trials <= prev:
            skipped += 1
            continue

        try:
            mat = find_matrix_pair(best_dir, code_id)
        except FileNotFoundError as ex:
            print(f"[skip] ({i}/{len(code_ids)}) {code_id}: {ex}")
            skipped += 1
            continue

        print(f"[run ] ({i}/{len(code_ids)}) {code_id} prev_steps={prev} -> new_steps={args.trials}")
        print(f"       Hx={mat.hx.name}  Hz={mat.hz.name}")

        dZ = run_dist_m4ri(exe, finH=mat.hx, finG=mat.hz, steps=args.trials, seed=args.seed,
                           timeout_s=args.timeout, code_id=code_id, side="Z")
        dX = run_dist_m4ri(exe, finH=mat.hz, finG=mat.hx, steps=args.trials, seed=args.seed + 1,
                           timeout_s=args.timeout, code_id=code_id, side="X")

        if dX is None or dZ is None:
            skipped += 1
            continue

        meta_summary = inject_m4ri_results(meta_summary, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
        meta_collected = inject_m4ri_results(meta_collected, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
        meta_collected["code_id"] = code_id

        write_json(meta_summary_path, meta_summary)
        write_json(meta_collected_path, meta_collected)

        updated += 1
        print(f"      -> dX_ub={dX} dZ_ub={dZ} d_ub={min(dX,dZ)}")

    print(f"[done] updated={updated} skipped={skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
