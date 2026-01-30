#!/usr/bin/env python3
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _best_codes_common import (
    atomic_write_json,
    code_id_d_from_suffix,
    code_id_with_d,
    code_id_without_d,
    extract_code_id,
    extract_distance_bounds_strict,
    extract_group_spec,
    extract_k,
    extract_n,
    extract_trials,
    matrices_for_code,
    utc_now_iso,
)


def _repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path.cwd()


def _is_git_repo(root: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(root), text=True).strip()
        return out.lower() == "true"
    except Exception:
        return False


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _read_json(path: Path, *, verbose: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        _log(verbose, f"[warn] failed to read {path}: {exc}")
        return {}


def _update_code_id_fields(meta: Dict[str, Any], old_code_id: str, new_code_id: str) -> None:
    meta["code_id"] = new_code_id
    d_val = code_id_d_from_suffix(new_code_id)
    if isinstance(d_val, int):
        meta["d_in_id"] = d_val
        meta["d_recorded"] = d_val
        meta["d_recorded_kind"] = "from_code_id"
    if "collected_dir" in meta:
        meta["collected_dir"] = f"best_codes/collected/{new_code_id}"
    if isinstance(meta.get("collected_files"), list):
        meta["collected_files"] = [
            str(x).replace(f"/{old_code_id}/", f"/{new_code_id}/") for x in meta["collected_files"]
        ]
    if isinstance(meta.get("matrices_flat"), list):
        meta["matrices_flat"] = [
            str(x).replace(old_code_id, new_code_id, 1) for x in meta["matrices_flat"]
        ]


def _git_mv(src: Path, dst: Path, *, root: Path, dry_run: bool, verbose: bool) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        _log(verbose, f"[dry-run] git mv {src} -> {dst}")
        return
    res = subprocess.run(["git", "mv", str(src), str(dst)], cwd=str(root), text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"git mv failed: {res.stderr.strip()}")


def _is_git_tracked(path: Path, *, root: Path) -> bool:
    try:
        res = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=str(root),
            text=True,
            capture_output=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def _move_path(src: Path, dst: Path, *, root: Path, use_git: bool, dry_run: bool, verbose: bool) -> None:
    if not src.exists():
        return
    if use_git and _is_git_tracked(src, root=root):
        _git_mv(src, dst, root=root, dry_run=dry_run, verbose=verbose)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        _log(verbose, f"[dry-run] move {src} -> {dst}")
        return
    src.rename(dst)


def _archive_code(best_dir: Path, code_id: str, archive_root: Path, *, root: Path, use_git: bool, dry_run: bool, verbose: bool) -> None:
    meta = best_dir / "meta" / f"{code_id}.json"
    collected = best_dir / "collected" / code_id
    mats = matrices_for_code(best_dir, code_id)
    _move_path(meta, archive_root / "meta" / meta.name, root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)
    _move_path(collected, archive_root / "collected" / code_id, root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)
    for m in mats:
        _move_path(m, archive_root / "matrices" / m.name, root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)


def _rename_code(best_dir: Path, old_id: str, new_id: str, *, root: Path, use_git: bool, dry_run: bool, verbose: bool) -> None:
    if old_id == new_id:
        return
    meta_old = best_dir / "meta" / f"{old_id}.json"
    meta_new = best_dir / "meta" / f"{new_id}.json"
    _move_path(meta_old, meta_new, root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)

    col_old = best_dir / "collected" / old_id
    col_new = best_dir / "collected" / new_id
    _move_path(col_old, col_new, root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)

    for m in matrices_for_code(best_dir, old_id):
        new_name = new_id + m.name[len(old_id):]
        _move_path(m, m.with_name(new_name), root=root, use_git=use_git, dry_run=dry_run, verbose=verbose)


def _trials_total(meta: Dict[str, Any]) -> Optional[int]:
    total, _sx, _sz = extract_trials(meta)
    return total


def _choose_best_record(records: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    def key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
        code_id, meta = item
        trials = _trials_total(meta)
        d_ub = extract_distance_bounds_strict(meta)[2]
        t_val = int(trials) if isinstance(trials, int) else -1
        d_val = int(d_ub) if isinstance(d_ub, int) else -1
        return (t_val, d_val, code_id)

    return sorted(records, key=key)[-1]


def _rebuild_best_codes_from_meta(best_dir: Path, *, dry_run: bool, verbose: bool) -> None:
    meta_dir = best_dir / "meta"
    meta_files = sorted([p for p in meta_dir.glob("*.json") if p.is_file()])
    codes: List[Dict[str, Any]] = []
    for p in meta_files:
        meta = _read_json(p, verbose=verbose)
        if not meta:
            continue
        code_id = extract_code_id(meta, p.stem) or p.stem
        group = extract_group_spec(meta)
        n = extract_n(meta)
        k = extract_k(meta)
        if group is None or n is None or k is None:
            continue
        dX, dZ, d = extract_distance_bounds_strict(meta)
        t_total, sx, sz = extract_trials(meta)
        per_side = None
        if isinstance(sx, int) or isinstance(sz, int):
            per_side = max([v for v in (sx, sz) if isinstance(v, int)], default=None)
        if per_side is None:
            for kkey in ("m4ri_steps", "trials", "steps"):
                v = meta.get(kkey)
                if isinstance(v, int):
                    per_side = v
                    break

        codes.append(
            {
                "code_id": code_id,
                "group": group,
                "n": int(n),
                "k": int(k),
                "d_ub": int(d) if isinstance(d, int) else None,
                "dX_ub": int(dX) if isinstance(dX, int) else None,
                "dZ_ub": int(dZ) if isinstance(dZ, int) else None,
                "trials": int(t_total) if isinstance(t_total, int) else None,
                "steps_used_total": int(t_total) if isinstance(t_total, int) else None,
                "steps_used_x": int(sx) if isinstance(sx, int) else None,
                "steps_used_z": int(sz) if isinstance(sz, int) else None,
                "steps": int(per_side) if isinstance(per_side, int) else None,
                "m4ri_steps": int(per_side) if isinstance(per_side, int) else None,
                "m4ri_trials": int(t_total) if isinstance(t_total, int) else None,
                "meta": meta,
            }
        )

    codes.sort(key=lambda r: (r["group"], r["k"], -(r.get("d_ub") or -1), -(r.get("m4ri_trials") or -1), r["code_id"]))

    out = {"generated_at_utc": utc_now_iso(), "total_codes": len(codes), "codes": codes}
    if dry_run:
        _log(verbose, f"[dry-run] would write best_codes/data.json with {len(codes)} codes")
        return

    atomic_write_json(best_dir / "data.json", out)

    idx_lines = ["group\tn\tk\td_ub\tm4ri_trials\tcode_id"]
    for r in codes:
        idx_lines.append(
            "\t".join(
                [
                    str(r.get("group") or ""),
                    str(r.get("n") or ""),
                    str(r.get("k") or ""),
                    str(r.get("d_ub") if r.get("d_ub") is not None else ""),
                    str(r.get("m4ri_trials") if r.get("m4ri_trials") is not None else ""),
                    r.get("code_id") or "",
                ]
            )
        )
    (best_dir / "index.tsv").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    best_by_group: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in codes:
        g = str(r.get("group") or "")
        k = int(r.get("k")) if isinstance(r.get("k"), int) else None
        if k is None:
            continue
        key = (g, k)
        cur = best_by_group.get(key)
        if cur is None:
            best_by_group[key] = r
            continue
        d_new = r.get("d_ub") if isinstance(r.get("d_ub"), int) else -1
        t_new = r.get("m4ri_trials") if isinstance(r.get("m4ri_trials"), int) else -1
        d_old = cur.get("d_ub") if isinstance(cur.get("d_ub"), int) else -1
        t_old = cur.get("m4ri_trials") if isinstance(cur.get("m4ri_trials"), int) else -1
        if (d_new, t_new) > (d_old, t_old):
            best_by_group[key] = r
        elif (d_new, t_new) == (d_old, t_old):
            if str(r.get("code_id") or "") < str(cur.get("code_id") or ""):
                best_by_group[key] = r

    best_lines = ["group\tk\tn\td_ub\tm4ri_trials\tcode_id"]
    for (g, k), r in sorted(best_by_group.items(), key=lambda x: (x[0][0], x[0][1])):
        best_lines.append(
            "\t".join(
                [
                    g,
                    str(k),
                    str(r.get("n") or ""),
                    str(r.get("d_ub") if r.get("d_ub") is not None else ""),
                    str(r.get("m4ri_trials") if r.get("m4ri_trials") is not None else ""),
                    r.get("code_id") or "",
                ]
            )
        )
    (best_dir / "best_by_group_k.tsv").write_text("\n".join(best_lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fix best_codes inconsistencies and rebuild website data.")
    ap.add_argument("--best-dir", default="best_codes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--publish", action="store_true", help="Run scrape_and_publish_best_codes.py after fixes.")
    args = ap.parse_args(argv)

    root = _repo_root()
    best_dir = (root / args.best_dir).resolve()
    meta_dir = best_dir / "meta"
    if not meta_dir.exists():
        raise SystemExit(f"missing meta dir: {meta_dir}")

    use_git = _is_git_repo(root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = best_dir / "archived" / f"fix_inconsistencies_{ts}"

    rename_actions: List[Tuple[str, str]] = []
    meta_files = sorted(meta_dir.glob("*.json"))
    for mp in meta_files:
        meta = _read_json(mp, verbose=args.verbose)
        if not meta:
            continue
        code_id = extract_code_id(meta, mp.stem) or mp.stem
        d_suffix = code_id_d_from_suffix(code_id)
        d_ub = extract_distance_bounds_strict(meta)[2]
        if isinstance(d_suffix, int) and isinstance(d_ub, int) and d_suffix != d_ub:
            new_id = code_id_with_d(code_id, d_ub)
            if new_id != code_id:
                rename_actions.append((code_id, new_id))

    for old_id, new_id in sorted(rename_actions):
        if (best_dir / "meta" / f"{new_id}.json").exists():
            _log(args.verbose, f"[fix] archive collision {new_id}")
            _archive_code(best_dir, new_id, archive_root / new_id, root=root, use_git=use_git, dry_run=args.dry_run, verbose=args.verbose)
        _log(args.verbose, f"[fix] rename {old_id} -> {new_id}")
        _rename_code(best_dir, old_id, new_id, root=root, use_git=use_git, dry_run=args.dry_run, verbose=args.verbose)

        meta_path = best_dir / "meta" / f"{new_id}.json"
        meta = _read_json(meta_path, verbose=args.verbose)
        if meta:
            _update_code_id_fields(meta, old_id, new_id)
            if not args.dry_run:
                atomic_write_json(meta_path, meta, sort_keys=True)

        collected_meta = best_dir / "collected" / new_id / "meta.json"
        cmeta = _read_json(collected_meta, verbose=args.verbose)
        if cmeta:
            _update_code_id_fields(cmeta, old_id, new_id)
            if not args.dry_run:
                atomic_write_json(collected_meta, cmeta, sort_keys=True)

        settings_path = best_dir / "collected" / new_id / "settings.json"
        settings = _read_json(settings_path, verbose=args.verbose)
        if settings:
            settings["code_id"] = new_id
            if not args.dry_run:
                atomic_write_json(settings_path, settings, sort_keys=True)

    current_meta_files = sorted(meta_dir.glob("*.json"))
    by_base: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for mp in current_meta_files:
        meta = _read_json(mp, verbose=args.verbose)
        if not meta:
            continue
        code_id = extract_code_id(meta, mp.stem) or mp.stem
        base = code_id_without_d(code_id)
        by_base.setdefault(base, []).append((code_id, meta))

    for base, recs in sorted(by_base.items()):
        if len(recs) <= 1:
            continue
        keep_id, _keep_meta = _choose_best_record(recs)
        for code_id, _meta in recs:
            if code_id == keep_id:
                continue
            _log(args.verbose, f"[fix] archive duplicate {code_id} (keep {keep_id})")
            _archive_code(best_dir, code_id, archive_root / code_id, root=root, use_git=use_git, dry_run=args.dry_run, verbose=args.verbose)

    _rebuild_best_codes_from_meta(best_dir, dry_run=args.dry_run, verbose=args.verbose)

    if args.publish:
        script = root / "scripts" / "scrape_and_publish_best_codes.py"
        cmd = ["python3", str(script)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.verbose:
            cmd.append("--verbose")
        _log(args.verbose, f"[info] run {' '.join(cmd)}")
        if not args.dry_run:
            return subprocess.call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
