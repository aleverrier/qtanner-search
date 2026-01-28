#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()


import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

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

def parse_d_from_name(code_id: str) -> Optional[int]:
    m = re.search(r"_d(\d+)$", code_id)
    return int(m.group(1)) if m else None

def replace_d_suffix(code_id: str, new_d: int) -> str:
    if re.search(r"_d\d+$", code_id):
        return re.sub(r"_d\d+$", f"_d{new_d}", code_id)
    return f"{code_id}_d{new_d}"

def parse_k(code_id: str) -> Optional[int]:
    m = re.search(r"_k(\d+)_d\d+$", code_id)
    return int(m.group(1)) if m else None

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

def parse_dist_m4ri_output(text: str) -> int:
    # Try dmin=NN
    dmin = None
    for m in re.finditer(r"\bdmin=(\d+)\b", text):
        dmin = int(m.group(1))
    if dmin is not None:
        return dmin

    # Prefer "success ... d=NN"
    d = None
    for line in text.splitlines():
        if "success" in line:
            m = re.search(r"\bd=(-?\d+)\b", line)
            if m:
                d = int(m.group(1))
    if d is not None:
        return abs(d)

    # Fallback: any d=NN
    for m in re.finditer(r"\bd=(-?\d+)\b", text):
        d = int(m.group(1))
    if d is not None:
        return abs(d)

    raise ValueError("Could not parse a distance from dist_m4ri output.")

def run_dist_m4ri(
    exe: Path,
    finH: Path,
    finG: Path,
    steps: int,
    seed: int,
    timeout_s: Optional[int],
) -> Tuple[int, str]:
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
        # still try parse
        try:
            return parse_dist_m4ri_output(out), out
        except Exception:
            tail = "\n".join(out.splitlines()[-120:])
            raise RuntimeError(f"dist_m4ri failed rc={proc.returncode}\ncmd={' '.join(cmd)}\n\n{tail}")
    return parse_dist_m4ri_output(out), out

def find_matrices_for_code(best_codes_dir: Path, code_id: str) -> Tuple[Path, Path]:
    """
    We support two common layouts:
      1) best_codes_dir/matrices/{code_id}*X.mtx and *Z.mtx
      2) inside code dir itself (rare): best_codes_dir/{code_id}/...X.mtx ...Z.mtx
    """
    mats_dir = best_codes_dir / "matrices"
    hx: Optional[Path] = None
    hz: Optional[Path] = None

    if mats_dir.exists():
        hx_cands = sorted(mats_dir.glob(f"{code_id}*X.mtx"), key=lambda p: len(p.name))
        hz_cands = sorted(mats_dir.glob(f"{code_id}*Z.mtx"), key=lambda p: len(p.name))
        if hx_cands and hz_cands:
            return hx_cands[0], hz_cands[0]

    # fallback: search within code dir
    code_dir = best_codes_dir / code_id
    if code_dir.exists():
        hx_cands = sorted(code_dir.rglob("*X.mtx"), key=lambda p: len(p.name))
        hz_cands = sorted(code_dir.rglob("*Z.mtx"), key=lambda p: len(p.name))
        if hx_cands and hz_cands:
            return hx_cands[0], hz_cands[0]

    raise SystemExit(f"ERROR: could not find X/Z .mtx matrices for {code_id} under {best_codes_dir}")

def meta_path_for_code(best_codes_dir: Path, code_id: str) -> Path:
    # Prefer best_codes/meta/{code_id}.json if present, else create it.
    return best_codes_dir / "meta" / f"{code_id}.json"

def prev_steps_from_meta(meta: Dict[str, Any]) -> int:
    for k in ("m4ri_steps", "trials_m4ri", "distance_steps_m4ri", "steps", "trials"):
        if k in meta and meta[k] is not None:
            return ensure_int(meta[k], 0)
    dist = meta.get("distance", {})
    if isinstance(dist, dict):
        if dist.get("backend") in ("dist-m4ri", "m4ri"):
            return ensure_int(dist.get("steps", 0), 0)
    return 0

def update_meta(meta: Dict[str, Any], *, steps: int, seed: int, dX: int, dZ: int, d_final: int) -> Dict[str, Any]:
    meta = dict(meta)
    meta["updated_at"] = utc_now_iso()
    meta["distance_backend"] = "dist-m4ri"
    meta["distance_method"] = "RW"
    meta["m4ri_steps"] = steps
    meta["m4ri_seed"] = seed
    meta["dX_ub"] = dX
    meta["dZ_ub"] = dZ
    meta["d_ub"] = min(dX, dZ)
    meta["d"] = d_final
    meta["trials"] = steps
    return meta

def safe_rename_path(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        # archive the source instead of clobbering
        arch = src.parent / "archived_duplicates"
        arch.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(arch / src.name))
        return
    shutil.move(str(src), str(dst))

def rename_associated_files(best_codes_dir: Path, old_id: str, new_id: str) -> None:
    # Rename matrices that start with old_id
    mats_dir = best_codes_dir / "matrices"
    if mats_dir.exists():
        for f in list(mats_dir.glob(f"{old_id}*")):
            if f.is_file():
                new_name = f.name.replace(old_id, new_id, 1)
                safe_rename_path(f, f.with_name(new_name))

    # Rename meta JSON
    old_meta = best_codes_dir / "meta" / f"{old_id}.json"
    new_meta = best_codes_dir / "meta" / f"{new_id}.json"
    if old_meta.exists():
        safe_rename_path(old_meta, new_meta)

def refine_one_code(
    dist_exe: Path,
    best_codes_dir: Path,
    code_id: str,
    steps: int,
    seed: int,
    timeout: Optional[int],
    force: bool,
    rename: bool,
    dry_run: bool,
) -> Optional[str]:
    meta_p = meta_path_for_code(best_codes_dir, code_id)
    meta = read_json(meta_p)
    prev = prev_steps_from_meta(meta)

    if (not force) and steps <= prev:
        return None

    hx, hz = find_matrices_for_code(best_codes_dir, code_id)
    old_d = parse_d_from_name(code_id) or ensure_int(meta.get("d"), 10**9)

    if dry_run:
        print(f"[dry] {code_id}: prev_steps={prev} -> steps={steps}  (Hx={hx.name} Hz={hz.name})")
        return None

    dZ, _ = run_dist_m4ri(dist_exe, finH=hx, finG=hz, steps=steps, seed=seed, timeout_s=timeout)
    dX, _ = run_dist_m4ri(dist_exe, finH=hz, finG=hx, steps=steps, seed=seed + 1, timeout_s=timeout)

    new_d = min(dX, dZ)
    d_final = min(old_d, new_d)  # monotone upper bound

    meta2 = update_meta(meta, steps=steps, seed=seed, dX=dX, dZ=dZ, d_final=d_final)
    write_json(meta_p, meta2)

    print(f"[ok ] {code_id}: dX_ub={dX} dZ_ub={dZ} d_final={d_final} steps={steps}")

    if rename:
        new_id = replace_d_suffix(code_id, d_final)
        if new_id != code_id:
            src_dir = best_codes_dir / code_id
            dst_dir = best_codes_dir / new_id
            if src_dir.exists():
                safe_rename_path(src_dir, dst_dir)
            rename_associated_files(best_codes_dir, code_id, new_id)
            # also rewrite meta under new name (if it moved)
            new_meta_p = meta_path_for_code(best_codes_dir, new_id)
            meta3 = read_json(new_meta_p)
            meta3["code_id"] = new_id
            write_json(new_meta_p, meta3)
            return new_id

    return code_id

def main() -> int:
    ap = argparse.ArgumentParser(description="Refine distances using dist_m4ri for codes stored under results/**/best_codes (no GAP).")
    ap.add_argument("--group", required=True, help="Group prefix, e.g. C2xC2xC2xC2")
    ap.add_argument("--trials", type=int, required=True, help="Number of RW steps for dist_m4ri (method=1)")
    ap.add_argument("--results-dir", default="results", help="Root results directory (default: results)")
    ap.add_argument("--dist-m4ri", default=None, help="Path to dist_m4ri executable (optional)")
    ap.add_argument("--seed", type=int, default=1, help="Base seed (dZ uses seed, dX uses seed+1)")
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) per dist_m4ri call")
    ap.add_argument("--limit", type=int, default=None, help="Only refine first N codes discovered (for testing)")
    ap.add_argument("--force", action="store_true", help="Recompute even if trials <= previous recorded steps")
    ap.add_argument("--no-rename", action="store_true", help="Do not rename folders/files to update _dXX suffix")
    ap.add_argument("--dry-run", action="store_true", help="List what would run, without executing or writing")
    args = ap.parse_args()

    dist_exe = find_dist_m4ri(args.dist_m4ri)
    results = Path(args.results_dir).resolve()
    if not results.exists():
        raise SystemExit(f"ERROR: results dir not found: {results}")

    # Find all per-run best_codes directories
    best_codes_dirs = sorted({p for p in results.rglob("best_codes") if p.is_dir()})
    if not best_codes_dirs:
        print(f"[info] no best_codes folders found under {results}")
        return 0

    prefix = args.group + "_"
    rename = not args.no_rename

    # Discover code dirs within each run’s best_codes folder
    discovered: List[Tuple[Path, str]] = []
    for bd in best_codes_dirs:
        for code_dir in sorted(bd.iterdir()):
            if code_dir.is_dir() and code_dir.name.startswith(prefix):
                discovered.append((bd, code_dir.name))

    if not discovered:
        print(f"[info] no codes found for group '{args.group}' under {results}/**/best_codes/")
        return 0

    # Deduplicate by exact code_id + folder location; we still refine each occurrence because update_best_codes_repo
    # may pick a specific run as the source. (Keeping it simple and correct.)
    print(f"[info] dist_m4ri={dist_exe}")
    print(f"[info] group={args.group} trials={args.trials} found_instances={len(discovered)} best_codes_folders={len(best_codes_dirs)}")

    n_done = 0
    for bd, code_id in discovered:
        if args.limit is not None and n_done >= args.limit:
            break
        out = refine_one_code(
            dist_exe=dist_exe,
            best_codes_dir=bd,
            code_id=code_id,
            steps=args.trials,
            seed=args.seed,
            timeout=args.timeout,
            force=args.force,
            rename=rename,
            dry_run=args.dry_run,
        )
        n_done += 1

    print(f"[done] processed={n_done} (dry_run={args.dry_run})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
