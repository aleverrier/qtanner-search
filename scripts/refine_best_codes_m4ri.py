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


# -------------------------
# Helpers
# -------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def eprint(*args: object) -> None:
    print(*args, file=os.sys.stderr)

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

def strip_d_suffix(code_id: str) -> str:
    # Remove trailing _dNN (if present)
    return re.sub(r"_d\d+$", "", code_id)

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

def parse_group(code_id: str) -> str:
    return code_id.split("_", 1)[0] if "_" in code_id else code_id

def parse_d_from_name(code_id: str) -> Optional[int]:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def find_dist_m4ri(explicit: Optional[str] = None) -> Path:
    # 1) explicit
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists() and p.is_file():
            return p
        raise SystemExit(f"ERROR: --dist-m4ri '{explicit}' not found.")

    # 2) env vars
    for env in ("DIST_M4RI", "QTANNER_DIST_M4RI"):
        v = os.environ.get(env)
        if v:
            p = Path(v).expanduser()
            if p.exists() and p.is_file():
                return p

    # 3) PATH
    w = shutil.which("dist_m4ri")
    if w:
        return Path(w)

    # 4) common project paths
    candidates = [
        Path("dist-m4ri/src/dist_m4ri"),
        Path("../dist-m4ri/src/dist_m4ri"),
        Path("../qtanner-tools/dist-m4ri/src/dist_m4ri"),
        Path.home() / "research/qtanner-tools/dist-m4ri/src/dist_m4ri",
    ]
    for p in candidates:
        p = p.expanduser().resolve()
        if p.exists() and p.is_file():
            return p

    raise SystemExit(
        "ERROR: dist_m4ri not found.\n"
        "  - Put it on PATH (dist_m4ri), or\n"
        "  - pass --dist-m4ri /full/path/to/dist_m4ri, or\n"
        "  - set env DIST_M4RI=/full/path/to/dist_m4ri"
    )

@dataclass
class MatPair:
    hx: Path
    hz: Path

def find_matrix_pair(best_dir: Path, code_id: str) -> MatPair:
    # We expect files like .../{code_id}*X.mtx and .../{code_id}*Z.mtx
    matrices = best_dir / "matrices"
    if not matrices.exists():
        raise SystemExit(f"ERROR: matrices folder not found: {matrices}")

    hx_candidates = sorted(matrices.glob(f"{code_id}*X.mtx"), key=lambda p: len(p.name))
    hz_candidates = sorted(matrices.glob(f"{code_id}*Z.mtx"), key=lambda p: len(p.name))

    if not hx_candidates:
        raise SystemExit(f"ERROR: no Hx matrix found for {code_id} under {matrices} (pattern {code_id}*X.mtx)")
    if not hz_candidates:
        raise SystemExit(f"ERROR: no Hz matrix found for {code_id} under {matrices} (pattern {code_id}*Z.mtx)")

    hx = hx_candidates[0]
    hz = hz_candidates[0]
    return MatPair(hx=hx, hz=hz)

def parse_dist_m4ri_output(text: str) -> int:
    # Try to find dmin=NN or success ... d=NN
    # (Documentation shows "dmin=8" and "success ... d=8" for CC; RW is similar in spirit.) :contentReference[oaicite:3]{index=3}
    dmin = None
    for m in re.finditer(r"\bdmin=(\d+)\b", text):
        dmin = int(m.group(1))
    if dmin is not None:
        return dmin

    d = None
    # Prefer "success ... d=NN" lines if present
    for line in text.splitlines():
        if "success" in line:
            m = re.search(r"\bd=(-?\d+)\b", line)
            if m:
                d = int(m.group(1))
    if d is not None:
        return abs(d)

    # Fallback: last d=NN anywhere
    for m in re.finditer(r"\bd=(-?\d+)\b", text):
        d = int(m.group(1))
    if d is not None:
        return abs(d)

    # Last resort: last integer on the last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        m = re.search(r"(-?\d+)\s*$", tail)
        if m:
            return abs(int(m.group(1)))

    raise ValueError("Could not parse a distance from dist_m4ri output.")

def run_dist_m4ri(
    exe: Path,
    finH: Path,
    finG: Path,
    steps: int,
    seed: int,
    timeout_s: Optional[int] = None,
) -> Tuple[int, str]:
    # Using dist_m4ri RW method (method=1) with steps, finH, finG. :contentReference[oaicite:4]{index=4}
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
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        # Still try to parse; dist_m4ri sometimes returns nonzero in some early-stop cases.
        try:
            d = parse_dist_m4ri_output(out)
            return d, out
        except Exception:
            raise RuntimeError(
                f"dist_m4ri failed (rc={proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output (last 80 lines):\n" + "\n".join(out.splitlines()[-80:])
            )
    d = parse_dist_m4ri_output(out)
    return d, out

def get_prev_m4ri_steps(meta: Dict[str, Any]) -> int:
    # Prefer an explicit m4ri field; fallback to generic trials if it looks like it came from dist_m4ri.
    for key in ("m4ri_steps", "distance_steps_m4ri", "trials_m4ri"):
        if key in meta:
            return ensure_int(meta.get(key), 0)

    # If meta says backend is dist-m4ri, accept generic 'trials' or 'steps'
    backend = str(meta.get("distance_backend", "")).lower()
    if "m4ri" in backend or "dist" in backend:
        if "trials" in meta:
            return ensure_int(meta.get("trials"), 0)
        if "steps" in meta:
            return ensure_int(meta.get("steps"), 0)

    # Nested?
    dist = meta.get("distance", {})
    if isinstance(dist, dict):
        if dist.get("backend") in ("dist-m4ri", "m4ri"):
            return ensure_int(dist.get("steps", 0), 0)

    return 0

def update_meta(
    meta: Dict[str, Any],
    *,
    steps: int,
    seed: int,
    dX: int,
    dZ: int,
) -> Dict[str, Any]:
    ts = utc_now_iso()
    d = min(dX, dZ)

    # Keep any previous 'd' around (if present)
    if "d" in meta and meta.get("d") != d:
        meta.setdefault("d_previous", meta.get("d"))

    meta["distance_backend"] = "dist-m4ri"
    meta["distance_method"] = "RW"
    meta["m4ri_steps"] = steps
    meta["m4ri_seed"] = seed
    meta["dX_ub"] = dX
    meta["dZ_ub"] = dZ
    meta["d_ub"] = d

    # For compatibility with existing website tables, also set these
    meta["d"] = d
    meta["trials"] = steps
    meta["updated_at"] = ts

    return meta

def archive_code(best_dir: Path, code_id: str, archive_root: Path) -> None:
    # Move collected entry, meta JSON, and any matrix files for this code_id.
    (archive_root / "collected").mkdir(parents=True, exist_ok=True)
    (archive_root / "meta").mkdir(parents=True, exist_ok=True)
    (archive_root / "matrices").mkdir(parents=True, exist_ok=True)

    src_col = best_dir / "collected" / code_id
    src_meta = best_dir / "meta" / f"{code_id}.json"
    matrices_dir = best_dir / "matrices"

    if src_col.exists():
        shutil.move(str(src_col), str(archive_root / "collected" / code_id))

    if src_meta.exists():
        shutil.move(str(src_meta), str(archive_root / "meta" / f"{code_id}.json"))

    if matrices_dir.exists():
        for f in matrices_dir.glob(f"{code_id}*"):
            if f.is_file():
                shutil.move(str(f), str(archive_root / "matrices" / f.name))


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine best_codes distances using dist_m4ri (no GAP).")
    ap.add_argument("--group", required=True, help="Group prefix, e.g. C2xC2xC2xC2")
    ap.add_argument("--trials", type=int, required=True, help="Number of RW steps for dist_m4ri (method=1)")
    ap.add_argument("--best-dir", default="best_codes", help="Best codes directory (default: best_codes)")
    ap.add_argument("--dist-m4ri", default=None, help="Path to dist_m4ri executable (optional)")
    ap.add_argument("--seed", type=int, default=1, help="Base RNG seed (default: 1). dZ uses seed, dX uses seed+1.")
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) per dist_m4ri call")
    ap.add_argument("--dry-run", action="store_true", help="List what would run but do not execute dist_m4ri or write files")
    ap.add_argument("--force", action="store_true", help="Recompute even if trials <= previous m4ri_steps")
    ap.add_argument("--prune", action="store_true", help="Archive non-best codes (within this group) after updating distances")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    collected_dir = best_dir / "collected"
    meta_dir = best_dir / "meta"

    if not collected_dir.exists():
        raise SystemExit(f"ERROR: {collected_dir} not found.")
    meta_dir.mkdir(parents=True, exist_ok=True)

    exe = find_dist_m4ri(args.dist_m4ri)

    # Collect codes for this group
    prefix = args.group + "_"
    code_ids = sorted([p.name for p in collected_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])

    if not code_ids:
        eprint(f"[info] no codes found for group '{args.group}' under {collected_dir}")
        return 0

    print(f"[info] best_dir={best_dir}")
    print(f"[info] dist_m4ri={exe}")
    print(f"[info] group={args.group} codes={len(code_ids)} trials={args.trials} seed={args.seed}")

    updated: List[Tuple[str,int,int,int]] = []
    skipped: List[str] = []

    for i, code_id in enumerate(code_ids, start=1):
        meta_path = meta_dir / f"{code_id}.json"
        meta = read_json(meta_path)
        prev_steps = get_prev_m4ri_steps(meta)

        if (not args.force) and args.trials <= prev_steps:
            skipped.append(code_id)
            continue

        mat = find_matrix_pair(best_dir, code_id)

        print(f"[run ] ({i}/{len(code_ids)}) {code_id} prev_steps={prev_steps} -> new_steps={args.trials}")
        print(f"       Hx={mat.hx.name}  Hz={mat.hz.name}")

        if args.dry_run:
            continue

        # dZ: finH=Hx, finG=Hz   (as documented finH=Hx finG=Hz for CSS) :contentReference[oaicite:5]{index=5}
        dZ, _outZ = run_dist_m4ri(exe, finH=mat.hx, finG=mat.hz, steps=args.trials, seed=args.seed, timeout_s=args.timeout)

        # dX: swap roles
        dX, _outX = run_dist_m4ri(exe, finH=mat.hz, finG=mat.hx, steps=args.trials, seed=args.seed + 1, timeout_s=args.timeout)

        # Keep monotone: if we had an older upper bound, take min(old, new)
        old_d = ensure_int(meta.get("d_ub", meta.get("d", parse_d_from_name(code_id) or 10**9)), 10**9)
        new_d = min(dX, dZ)
        d_final = min(old_d, new_d)

        if d_final != new_d:
            # Preserve per-side values but record the best overall UB we have ever seen
            meta["d_ub_previous_best"] = old_d

        meta = update_meta(meta, steps=args.trials, seed=args.seed, dX=dX, dZ=dZ)
        meta["d_ub"] = d_final
        meta["d"] = d_final

        write_json(meta_path, meta)
        updated.append((code_id, args.trials, dX, dZ))

        print(f"      -> dX_ub={dX} dZ_ub={dZ} d_ub={meta['d_ub']}")

    if args.dry_run:
        print(f"[dry] would update {len(code_ids) - len(skipped)} code(s); skipped {len(skipped)}")
        return 0

    print(f"[done] updated={len(updated)} skipped={len(skipped)}")

    if args.prune:
        # Determine best per k within this group, based on current meta d_ub/d
        best_by_k: Dict[int, int] = {}
        d_by_code: Dict[str, int] = {}

        for code_id in code_ids:
            k = parse_k(code_id)
            if k is None:
                continue
            m = read_json(meta_dir / f"{code_id}.json")
            dval = ensure_int(m.get("d_ub", m.get("d", parse_d_from_name(code_id) or 0)), 0)
            d_by_code[code_id] = dval
            best_by_k[k] = max(best_by_k.get(k, 0), dval)

        stamp = utc_now_iso().replace(":", "").replace("+", "").replace("-", "")
        archive_root = best_dir / "archived" / f"pruned_{args.group}_{stamp}"
        pruned = 0

        for code_id, dval in d_by_code.items():
            k = parse_k(code_id)
            if k is None:
                continue
            if dval < best_by_k.get(k, dval):
                print(f"[prune] archiving {code_id} (k={k} d={dval} < best={best_by_k[k]})")
                archive_code(best_dir, code_id, archive_root)
                pruned += 1

        print(f"[prune] archived {pruned} code(s) into {archive_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
