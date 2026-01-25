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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from _best_codes_common import (
    atomic_write_json,
    code_id_d_from_suffix,
    code_id_with_d,
    extract_distance_bounds,
    extract_trials,
    extract_n,
    utc_now_iso,
    matrices_for_code,
    safe_move,
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def find_dist_m4ri(explicit: Optional[str]) -> Path:
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


def run_dist_m4ri(exe: Path, *, finH: Path, finG: Path, steps: int, seed: int, timeout_s: Optional[int]) -> Optional[int]:
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
    except Exception:
        return None


def _pick_matrix(search_dirs: List[Path], code_id: str, kind: str) -> Optional[Path]:
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


def find_matrix_pair(best_dir: Path, code_id: str) -> Tuple[Path, Path]:
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
    return hx, hz


def trials_per_side(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    total, sx, sz = extract_trials(meta)
    per_side = None
    for k in ("m4ri_steps", "trials", "steps"):
        v = meta.get(k)
        if isinstance(v, int):
            per_side = v
            break
    if per_side is None and (isinstance(sx, int) or isinstance(sz, int)):
        per_side = max([v for v in (sx, sz) if isinstance(v, int)], default=None)
    return per_side, total, sx if isinstance(sx, int) else None, sz if isinstance(sz, int) else None


def merge_refine(meta: Dict[str, Any], *, steps: int, dX: int, dZ: int) -> Dict[str, Any]:
    ts = utc_now_iso()
    old_dX, old_dZ, old_d = extract_distance_bounds(meta)
    old_total, old_sx, old_sz = extract_trials(meta)
    old_side = None
    for k in ("m4ri_steps", "trials", "steps"):
        v = meta.get(k)
        if isinstance(v, int):
            old_side = v
            break

    best_dX = min([x for x in (old_dX, dX) if isinstance(x, int)], default=dX)
    best_dZ = min([x for x in (old_dZ, dZ) if isinstance(x, int)], default=dZ)
    best_d = min([x for x in (old_d, best_dX, best_dZ) if isinstance(x, int)], default=min(dX, dZ))

    best_side = max([x for x in (old_side, steps) if isinstance(x, int)], default=steps)
    best_sx = max([x for x in (old_sx, steps) if isinstance(x, int)], default=steps)
    best_sz = max([x for x in (old_sz, steps) if isinstance(x, int)], default=steps)
    best_total = max([x for x in (old_total, 2 * steps) if isinstance(x, int)], default=2 * steps)

    meta["distance_backend"] = "dist-m4ri"
    meta["distance_method"] = "RW"
    meta["m4ri_steps"] = best_side
    meta["m4ri_seed"] = meta.get("m4ri_seed", meta.get("distance_seed", 1))
    meta["dX_ub"] = best_dX
    meta["dZ_ub"] = best_dZ
    meta["d_ub"] = best_d
    meta["d"] = best_d
    meta["trials"] = best_side
    meta["updated_at"] = ts

    dist = meta.get("distance")
    if not isinstance(dist, dict):
        dist = {}
    dist.update({
        "method": "dist-m4ri",
        "dX_best": best_dX,
        "dZ_best": best_dZ,
        "d_ub": best_d,
        "steps_used_x": best_sx,
        "steps_used_z": best_sz,
        "steps_used_total": best_total,
    })
    meta["distance"] = dist

    dm = meta.get("distance_m4ri")
    if not isinstance(dm, dict):
        dm = {}
    dm.update({
        "backend": "dist-m4ri",
        "method": "RW",
        "steps_per_side": best_side,
        "dX_ub": best_dX,
        "dZ_ub": best_dZ,
        "d_ub": best_d,
        "updated_at": ts,
    })
    meta["distance_m4ri"] = dm
    return meta


def archive_old_name(best_dir: Path, code_id: str, archive_root: Path) -> None:
    (archive_root / "collected").mkdir(parents=True, exist_ok=True)
    (archive_root / "meta").mkdir(parents=True, exist_ok=True)
    (archive_root / "matrices").mkdir(parents=True, exist_ok=True)

    col = best_dir / "collected" / code_id
    meta = best_dir / "meta" / f"{code_id}.json"
    mats = best_dir / "matrices"

    if col.exists():
        dst = archive_root / "collected" / code_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(col, dst)
    if meta.exists():
        dst = archive_root / "meta" / f"{code_id}.json"
        shutil.copy2(meta, dst)
    if mats.exists():
        for p in mats.glob(f"{code_id}*"):
            if p.is_file():
                shutil.copy2(p, archive_root / "matrices" / p.name)


def rename_code(best_dir: Path, old_code_id: str, new_code_id: str) -> None:
    meta_old = best_dir / "meta" / f"{old_code_id}.json"
    meta_new = best_dir / "meta" / f"{new_code_id}.json"
    if meta_old.exists():
        safe_move(meta_old, meta_new)

    col_old = best_dir / "collected" / old_code_id
    col_new = best_dir / "collected" / new_code_id
    if col_old.exists():
        safe_move(col_old, col_new)

    mats = best_dir / "matrices"
    if mats.exists():
        for p in list(mats.glob(f"{old_code_id}*")):
            if p.is_file():
                new_name = p.name.replace(old_code_id, new_code_id, 1)
                safe_move(p, mats / new_name)


def update_code_id_fields(meta: Dict[str, Any], old_code_id: str, new_code_id: str) -> None:
    meta["code_id"] = new_code_id
    d_val = code_id_d_from_suffix(new_code_id)
    if isinstance(d_val, int):
        meta["d_in_id"] = d_val
        meta["d_recorded"] = d_val
        meta["d_recorded_kind"] = "from_code_id"
    if "collected_dir" in meta:
        meta["collected_dir"] = f"best_codes/collected/{new_code_id}"
    if "collected_files" in meta and isinstance(meta["collected_files"], list):
        meta["collected_files"] = [x.replace(f"/{old_code_id}/", f"/{new_code_id}/") for x in meta["collected_files"]]
    if "matrices_flat" in meta and isinstance(meta["matrices_flat"], list):
        meta["matrices_flat"] = [x.replace(old_code_id, new_code_id, 1) for x in meta["matrices_flat"]]


def archive_below_trials(best_dir: Path, n: int, trials: int) -> int:
    meta_dir = best_dir / "meta"
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arch = best_dir / "archived" / f"pruned_n{n}_{ts}" / "below_min_trials"
    archived = 0

    for mp in sorted(meta_dir.glob("*.json")):
        meta = read_json(mp)
        if extract_n(meta) != n:
            continue
        per_side, total, _sx, _sz = trials_per_side(meta)
        ok = False
        if isinstance(per_side, int) and per_side >= trials:
            ok = True
        if isinstance(total, int) and total >= 2 * trials:
            ok = True
        if ok:
            continue
        cid = mp.stem
        safe_move(mp, arch / "meta" / f"{cid}.json")
        safe_move(best_dir / "collected" / cid, arch / "collected" / cid)
        for m in matrices_for_code(best_dir, cid):
            safe_move(m, arch / "matrices" / m.name)
        archived += 1
    if archived:
        print(f"[archive] n={n} archived={archived} -> {arch}")
    return archived


def verify_length(best_dir: Path, n: int, trials: int) -> int:
    data_path = best_dir / "data.json"
    if not data_path.exists():
        raise SystemExit(f"ERROR: missing {data_path}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    failures = 0
    for rec in data.get("codes", []):
        if not isinstance(rec, dict):
            continue
        if rec.get("n") != n:
            continue
        per_side = rec.get("steps") or rec.get("m4ri_steps")
        total = rec.get("trials") or rec.get("m4ri_trials") or 0
        ok = False
        if isinstance(per_side, int) and per_side >= trials:
            ok = True
        if int(total) >= 2 * trials:
            ok = True
        if not ok:
            failures += 1
            print(f"[fail] {rec.get('code_id')} n={n} trials={total} < {trials}")
    if failures:
        print(f"[verify] failures={failures}")
        return 2
    print("[verify] ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine best_codes distances (dist_m4ri) for all codes of length n.")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--trials-per-side", "--trials", dest="trials", type=int, required=True)
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--dist-m4ri", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    best_dir = Path(args.best_dir).resolve()
    meta_dir = best_dir / "meta"
    collected_dir = best_dir / "collected"

    if args.verify:
        return verify_length(best_dir, args.n, args.trials)

    if not meta_dir.is_dir():
        raise SystemExit(f"ERROR: {meta_dir} not found.")

    exe = find_dist_m4ri(args.dist_m4ri)

    candidates: List[str] = []
    for mp in sorted(meta_dir.glob("*.json")):
        meta = read_json(mp)
        if extract_n(meta) == args.n:
            candidates.append(mp.stem)

    tasks: List[Tuple[str, str, str, str, int, int, Optional[int]]] = []
    skipped = 0

    for code_id in candidates:
        meta_summary = read_json(meta_dir / f"{code_id}.json")
        meta_collected = read_json(collected_dir / code_id / "meta.json")
        prev_side, prev_total, _sx, _sz = trials_per_side(meta_summary)
        if isinstance(prev_side, int) and prev_side < args.trials and isinstance(prev_total, int) and prev_total >= 2 * args.trials:
            prev_side = args.trials
        if isinstance(prev_side, int) and prev_side >= args.trials:
            skipped += 1
            continue
        if isinstance(prev_total, int) and prev_total >= 2 * args.trials:
            skipped += 1
            continue
        if meta_collected:
            prev_side2, prev_total2, _sx2, _sz2 = trials_per_side(meta_collected)
            if isinstance(prev_side2, int) and prev_side2 >= args.trials:
                skipped += 1
                continue
            if isinstance(prev_total2, int) and prev_total2 >= 2 * args.trials:
                skipped += 1
                continue
        try:
            hx, hz = find_matrix_pair(best_dir, code_id)
        except FileNotFoundError as ex:
            print(f"[skip] {code_id}: {ex}")
            skipped += 1
            continue
        tasks.append((code_id, str(exe), str(hx), str(hz), args.trials, args.seed, args.timeout))

    if not tasks:
        print(f"[done] nothing to refine (skipped={skipped})")
        archive_below_trials(best_dir, args.n, args.trials)
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

            meta_summary = merge_refine(meta_summary, steps=args.trials, dX=dX, dZ=dZ)
            meta_collected = merge_refine(meta_collected, steps=args.trials, dX=dX, dZ=dZ)

            best_d = meta_summary.get("d_ub") or meta_collected.get("d_ub")
            if isinstance(best_d, int):
                old_suffix = code_id_d_from_suffix(code_id)
                if isinstance(old_suffix, int) and best_d != old_suffix:
                    old_code_id = code_id
                    new_code_id = code_id_with_d(code_id, best_d)
                    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    archive_old_name(best_dir, old_code_id, best_dir / "archived" / f"renamed_{ts}")
                    rename_code(best_dir, old_code_id, new_code_id)
                    code_id = new_code_id
                    update_code_id_fields(meta_summary, old_code_id, new_code_id)
                    update_code_id_fields(meta_collected, old_code_id, new_code_id)
                    meta_summary_path = meta_dir / f"{code_id}.json"
                    meta_collected_path = collected_dir / code_id / "meta.json"

            meta_collected["code_id"] = code_id
            meta_summary["code_id"] = code_id
            atomic_write_json(meta_summary_path, meta_summary)
            if meta_collected_path.parent.exists():
                atomic_write_json(meta_collected_path, meta_collected)
            updated += 1
            print(f"[ok] {code_id}: dX_ub={dX} dZ_ub={dZ} trials={args.trials}")

    print(f"[done] updated={updated} planned={len(tasks)} skipped={skipped}")
    archive_below_trials(best_dir, args.n, args.trials)
    return 0


def _worker(task: Tuple[str, str, str, str, int, int, Optional[int]]) -> Tuple[str, Optional[int], Optional[int]]:
    code_id, exe_s, hx_s, hz_s, steps, seed, timeout_s = task
    exe = Path(exe_s)
    hx = Path(hx_s)
    hz = Path(hz_s)
    dZ = run_dist_m4ri(exe, finH=hx, finG=hz, steps=steps, seed=seed, timeout_s=timeout_s)
    dX = run_dist_m4ri(exe, finH=hz, finG=hx, steps=steps, seed=seed + 1, timeout_s=timeout_s)
    return code_id, dX, dZ


if __name__ == "__main__":
    raise SystemExit(main())
