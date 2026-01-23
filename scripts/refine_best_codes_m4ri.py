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
from typing import Any, Dict, Optional, Tuple, List


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

    candidates = [
        Path.home() / ".local/bin/dist_m4ri",
        Path.home() / "research/qtanner-tools/dist-m4ri/src/dist_m4ri",
        Path("../qtanner-tools/dist-m4ri/src/dist_m4ri"),
        Path("../dist-m4ri/src/dist_m4ri"),
    ]
    for p in candidates:
        p = p.expanduser().resolve()
        if p.exists() and p.is_file():
            return p

    raise SystemExit(
        "ERROR: dist_m4ri not found.\n"
        "Put it on PATH as dist_m4ri, or pass --dist-m4ri /full/path/to/dist_m4ri, or set DIST_M4RI."
    )

@dataclass
class MatPair:
    hx: Path
    hz: Path

def find_matrix_pair(best_dir: Path, code_id: str) -> MatPair:
    mats = best_dir / "matrices"
    if not mats.exists():
        raise SystemExit(f"ERROR: matrices folder not found: {mats}")

    def pick(kind: str) -> Path:
        # kind is 'x' or 'z'
        if kind not in ("x", "z"):
            raise ValueError("kind must be x or z")
        K = kind.upper()
        patterns = [
            f"{code_id}__H{kind}.mtx", f"{code_id}*__H{kind}.mtx",
            f"{code_id}__H{K}.mtx",    f"{code_id}*__H{K}.mtx",
            f"{code_id}*{K}.mtx",  # legacy
        ]
        seen = {}
        for pat in patterns:
            for q in mats.glob(pat):
                if q.is_file():
                    seen[q.name] = q
        cands = list(seen.values())
        if not cands:
            raise FileNotFoundError(f"no H{kind} matrix for {code_id} under {mats}")
        cands.sort(key=lambda q: (len(q.name), q.name))
        return cands[0]

    hx = pick("x")
    hz = pick("z")
    return MatPair(hx=hx, hz=hz)

def parse_dist_m4ri_output(text: str) -> int:
    # robust parse: prefer dmin=NN else last d=NN
    dmin = None
    for m in re.finditer(r"\bdmin=(\d+)\b", text):
        dmin = int(m.group(1))
    if dmin is not None:
        return dmin
    last = None
    for m in re.finditer(r"\bd=(-?\d+)\b", text):
        last = abs(int(m.group(1)))
    if last is not None:
        return last
    raise ValueError("Could not parse distance from dist_m4ri output.")

def run_dist_m4ri(
    exe: Path,
    finH: Path,
    finG: Path,
    steps: int,
    seed: int,
    timeout_s: Optional[int],
) -> Tuple[int, str]:
    # RW method: method=1 with finH/finG for CSS
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
    if proc.returncode != 0:
        # still try parsing; if not, raise with context
        try:
            return parse_dist_m4ri_output(out), out
        except Exception:
            raise RuntimeError(
                f"dist_m4ri failed rc={proc.returncode}\ncmd={' '.join(cmd)}\n"
                + "\n".join(out.splitlines()[-80:])
            )
    return parse_dist_m4ri_output(out), out

def get_prev_steps(meta: Dict[str, Any]) -> int:
    # prefer our own fields
    for k in ("m4ri_steps", "trials", "steps"):
        if k in meta:
            return ensure_int(meta.get(k), 0)
    # or nested distance_m4ri
    dm = meta.get("distance_m4ri")
    if isinstance(dm, dict):
        return ensure_int(dm.get("steps_per_side", 0), 0)
    return 0

def inject_m4ri_results(meta: Dict[str, Any], *, steps: int, seed: int, dX: int, dZ: int) -> Dict[str, Any]:
    ts = utc_now_iso()
    d = min(dX, dZ)

    # Preserve previous distance block if present and not already preserved
    if "distance" in meta and "distance_prev" not in meta:
        meta["distance_prev"] = meta["distance"]

    # Canonical top-level fields used by the table/pipeline
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

    # A canonical nested block (for future-proofing)
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

    # Overwrite the "distance" block that the website currently displays
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
        "steps_fast": steps,
        "steps_slow": steps,
    }

    return meta

def main() -> int:
    ap = argparse.ArgumentParser(description="Refine best_codes distances using dist_m4ri, and update collected meta.json too.")
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
        # Two meta locations:
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

        dZ, _ = run_dist_m4ri(exe, finH=mat.hx, finG=mat.hz, steps=args.trials, seed=args.seed, timeout_s=args.timeout)
        dX, _ = run_dist_m4ri(exe, finH=mat.hz, finG=mat.hx, steps=args.trials, seed=args.seed + 1, timeout_s=args.timeout)

        meta_summary = inject_m4ri_results(meta_summary, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
        meta_collected = inject_m4ri_results(meta_collected, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)

        # Keep code_id stable inside collected meta
        meta_collected["code_id"] = code_id

        write_json(meta_summary_path, meta_summary)
        write_json(meta_collected_path, meta_collected)

        updated += 1
        print(f"      -> dX_ub={dX} dZ_ub={dZ} d_ub={min(dX,dZ)}")

    print(f"[done] updated={updated} skipped={skipped}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
