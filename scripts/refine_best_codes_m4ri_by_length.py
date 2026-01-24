#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _pick_matrix(search_dirs: List[Path], code_id: str, kind: str) -> Optional[Path]:
    # kind in {"x","z"}
    K = kind.upper()
    patterns = [
        f"{code_id}__H{kind}.mtx",
        f"{code_id}__H{K}.mtx",
        f"{code_id}*__H{kind}.mtx",
        f"{code_id}*__H{K}.mtx",
        f"*{code_id}*__H{kind}.mtx",
        f"*{code_id}*__H{K}.mtx",
    ]
    cands: List[Path] = []
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                if p.is_file() and p.name not in seen:
                    seen.add(p.name)
                    cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: (0 if p.name.startswith(code_id) else 1, len(p.name), p.name))
    return cands[0]


def find_matrix_pair(best_dir: Path, code_id: str) -> MatPair:
    # Prefer canonical centralized matrices, but fall back to collected/<code_id>/...
    mats_dir = best_dir / "matrices"
    collected = best_dir / "collected" / code_id
    search_dirs = [
        mats_dir,
        collected,
        collected / "matrices",
        collected / "data",
    ]
    hx = _pick_matrix(search_dirs, code_id, "x")
    hz = _pick_matrix(search_dirs, code_id, "z")
    if hx is None or hz is None:
        raise FileNotFoundError(f"missing lifted matrices for {code_id}: Hx={hx} Hz={hz}")
    return MatPair(hx=hx, hz=hz)


def parse_dist_m4ri_output(text: str) -> int:
    for pat in [r"\bdmin=(\d+)\b", r"\bd_ub=(\d+)\b", r"\bd=(\d+)\b"]:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
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
    path.write_text(out, encoding="utf-8")
    return path


def run_dist_m4ri(exe: Path, *, finH: Path, finG: Path, steps: int, seed: int, timeout_s: Optional[int], code_id: str, side: str) -> Optional[int]:
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
        return parse_dist_m4ri_output(out)
    except Exception as ex:
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
    meta["trials"] = steps  # steps per side (what the UI/pipeline expects)
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
    # Website compatibility: keep a distance dict with steps_used_*
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


def extract_n(meta: Dict[str, Any]) -> Optional[int]:
    # Most metas have top-level "n"
    if "n" in meta:
        try:
            return int(meta["n"])
        except Exception:
            return None
    # Fallback: sometimes nested
    for k in ("params", "code", "summary"):
        v = meta.get(k)
        if isinstance(v, dict) and "n" in v:
            try:
                return int(v["n"])
            except Exception:
                pass
    return None


def _worker(task: Tuple[str, str, str, str, int, int, Optional[int]]) -> Tuple[str, Optional[int], Optional[int]]:
    code_id, exe_s, hx_s, hz_s, steps, seed, timeout_s = task
    exe = Path(exe_s)
    hx = Path(hx_s)
    hz = Path(hz_s)
    dZ = run_dist_m4ri(exe, finH=hx, finG=hz, steps=steps, seed=seed, timeout_s=timeout_s, code_id=code_id, side="Z")
    dX = run_dist_m4ri(exe, finH=hz, finG=hx, steps=steps, seed=seed + 1, timeout_s=timeout_s, code_id=code_id, side="X")
    return code_id, dX, dZ


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine best_codes distances using dist_m4ri for ALL codes of a given length n.")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--trials", type=int, required=True, help="RW steps per side (X and Z).")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--dist-m4ri", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    collected_dir = best_dir / "collected"
    meta_dir = best_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if not collected_dir.is_dir():
        raise SystemExit(f"ERROR: {collected_dir} not found.")

    exe = find_dist_m4ri(args.dist_m4ri)

    # Build candidate list from collected, using whichever meta has n.
    candidates: List[str] = []
    for p in sorted([q for q in collected_dir.iterdir() if q.is_dir()]):
        code_id = p.name
        meta_col = read_json(p / "meta.json")
        meta_sum = read_json(meta_dir / f"{code_id}.json")
        n = extract_n(meta_col) or extract_n(meta_sum)
        if n == args.n:
            candidates.append(code_id)

    print(f"[info] best_dir={best_dir}")
    print(f"[info] dist_m4ri={exe}")
    print(f"[info] n={args.n} codes={len(candidates)} trials={args.trials} jobs={args.jobs} seed={args.seed}")

    # Decide which actually need refining; also locate matrices up-front.
    tasks: List[Tuple[str, str, str, str, int, int, Optional[int]]] = []
    planned = 0
    skipped = 0

    for code_id in candidates:
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
            print(f"[skip] {code_id}: {ex}")
            skipped += 1
            continue
        planned += 1
        tasks.append((code_id, str(exe), str(mat.hx), str(mat.hz), args.trials, args.seed, args.timeout))

    if not tasks:
        print(f"[done] nothing to do (skipped={skipped})")
        return 0

    updated = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futs = {pool.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            code_id = futs[fut]
            try:
                cid, dX, dZ = fut.result()
            except Exception as ex:
                print(f"[fail] {code_id}: worker crashed: {ex}")
                continue
            if dX is None or dZ is None:
                print(f"[skip] {code_id}: dist_m4ri failed on at least one side")
                continue

            meta_summary_path = meta_dir / f"{code_id}.json"
            meta_collected_path = collected_dir / code_id / "meta.json"
            meta_summary = read_json(meta_summary_path)
            meta_collected = read_json(meta_collected_path)

            meta_summary = inject_m4ri_results(meta_summary, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
            meta_collected = inject_m4ri_results(meta_collected, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
            meta_collected["code_id"] = code_id

            write_json(meta_summary_path, meta_summary)
            write_json(meta_collected_path, meta_collected)
            updated += 1
            print(f"[ok] {code_id}: dX_ub={dX} dZ_ub={dZ} d_ub={min(dX,dZ)} trials={args.trials}")

    print(f"[done] updated={updated} planned={planned} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
