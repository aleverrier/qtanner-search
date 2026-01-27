#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qtanner.best_codes_updater import run_best_codes_update

from _best_codes_common import (
    atomic_write_json,
    code_id_d_from_suffix,
    code_id_with_d,
    extract_distance_bounds,
    extract_group_spec,
    extract_n,
    rename_code,
)
from refine_best_codes_length import (
    find_dist_m4ri,
    find_matrix_pair,
    merge_refine,
    run_dist_m4ri,
    trials_per_side,
)


@dataclass
class RefineOutcome:
    code_id_before: str
    code_id_after: str
    status: str
    reason: str
    old_trials_per_side: Optional[int]
    new_trials_per_side: Optional[int]
    old_d_ub: Optional[int]
    new_d_ub: Optional[int]
    remained_best: Optional[bool] = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT),
            text=True,
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass
    for parent in [ROOT, *ROOT.parents]:
        if (parent / ".git").exists():
            return parent
    return ROOT


def _read_json(path: Path, *, verbose: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - best effort for messy meta
        if verbose:
            print(f"[skip] {path}: invalid json ({exc})")
        return {}


def _best_distance(*metas: Dict[str, Any]) -> Optional[int]:
    vals: List[int] = []
    for meta in metas:
        d_x, d_z, d = extract_distance_bounds(meta)
        for v in (d_x, d_z, d):
            if isinstance(v, int):
                vals.append(v)
    return min(vals) if vals else None


def _best_trials(*metas: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    per_side_vals: List[int] = []
    total_vals: List[int] = []
    for meta in metas:
        per_side, total, sx, sz = trials_per_side(meta)
        if isinstance(per_side, int):
            per_side_vals.append(per_side)
        if isinstance(total, int):
            total_vals.append(total)
        if not isinstance(per_side, int):
            side_steps = [v for v in (sx, sz) if isinstance(v, int)]
            if side_steps:
                per_side_vals.append(max(side_steps))
    per_side_best = max(per_side_vals) if per_side_vals else None
    total_best = max(total_vals) if total_vals else None
    if per_side_best is None and isinstance(total_best, int) and total_best > 0:
        per_side_best = total_best // 2
    return per_side_best, total_best


def _norm_group(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(str(s).strip().lower().split())


def _group_from_code_id(code_id: str) -> str:
    if not code_id:
        return ""
    return code_id.split("_", 1)[0]


def _group_matches(group_filter: Optional[str], code_id: str, meta: Dict[str, Any]) -> bool:
    if not group_filter:
        return True
    gf = _norm_group(group_filter)
    if not gf:
        return True
    meta_group = extract_group_spec(meta) or _group_from_code_id(code_id)
    mg = _norm_group(meta_group)
    return bool(mg) and (mg == gf or mg.startswith(gf))


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
            str(x).replace(f"/{old_code_id}/", f"/{new_code_id}/")
            for x in meta["collected_files"]
        ]
    if isinstance(meta.get("matrices_flat"), list):
        meta["matrices_flat"] = [
            str(x).replace(old_code_id, new_code_id, 1) for x in meta["matrices_flat"]
        ]


def _candidate_code_ids(meta_dir: Path, *, n: int, group: Optional[str], verbose: bool) -> List[str]:
    out: List[str] = []
    for mp in sorted(meta_dir.glob("*.json")):
        meta = _read_json(mp, verbose=verbose)
        if extract_n(meta) != n:
            continue
        if not _group_matches(group, mp.stem, meta):
            continue
        out.append(mp.stem)
    return out


def _needs_refine(trials_req: int, per_side: Optional[int], total: Optional[int]) -> bool:
    if isinstance(per_side, int) and per_side >= trials_req:
        return False
    if isinstance(total, int) and total >= 2 * trials_req:
        return False
    return True


def _fmt_int(v: Optional[int]) -> str:
    return str(v) if isinstance(v, int) else ""


def _print_summary(outcomes: Iterable[RefineOutcome]) -> None:
    rows = list(outcomes)
    refined = sum(1 for r in rows if r.status == "refined")
    skipped = sum(1 for r in rows if r.status == "skipped")
    failed = sum(1 for r in rows if r.status == "failed")
    print(f"[summary] refined={refined} skipped={skipped} failed={failed}")
    print("code_id_before\tcode_id_after\tstatus\ttrials_old\ttrials_new\td_old\td_new\tremained")
    for r in rows:
        remained = "" if r.remained_best is None else ("yes" if r.remained_best else "no")
        print(
            "\t".join(
                [
                    r.code_id_before,
                    r.code_id_after,
                    r.status if not r.reason else f"{r.status}:{r.reason}",
                    _fmt_int(r.old_trials_per_side),
                    _fmt_int(r.new_trials_per_side),
                    _fmt_int(r.old_d_ub),
                    _fmt_int(r.new_d_ub),
                    remained,
                ]
            )
        )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Refine dist_m4ri distance estimates for best_codes of a given length, "
            "then resync/publish best_codes."
        )
    )
    ap.add_argument("--n", type=int, required=True, help="Code length to refine.")
    ap.add_argument("--trials", type=int, required=True, help="dist_m4ri RW steps per side.")
    ap.add_argument("--group", type=str, default=None, help="Optional group filter.")
    ap.add_argument("--best-dir", type=str, default="best_codes", help=argparse.SUPPRESS)
    ap.add_argument("--dist-m4ri", type=str, default=None, help="Optional path to dist_m4ri.")
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) per dist_m4ri call.")
    ap.add_argument("--no-git", action="store_true", help="Skip git pull/commit/push during updater.")
    ap.add_argument("--no-publish", action="store_true", help="Skip website data updates during updater.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args(argv)

    if args.n <= 0:
        raise SystemExit("--n must be positive.")
    if args.trials <= 0:
        raise SystemExit("--trials must be positive.")

    root = _repo_root()
    best_dir = (root / args.best_dir).resolve()
    meta_dir = best_dir / "meta"
    collected_dir = best_dir / "collected"

    if not meta_dir.is_dir():
        raise SystemExit(f"ERROR: missing meta dir: {meta_dir}")

    code_ids = _candidate_code_ids(meta_dir, n=args.n, group=args.group, verbose=args.verbose)
    if not code_ids:
        group_msg = f" group={args.group}" if args.group else ""
        print(f"[done] no best_codes with n={args.n}{group_msg}")
        return 0

    print(
        f"[info] repo={root} best_dir={best_dir} n={args.n} codes={len(code_ids)} "
        f"trials={args.trials}"
    )

    outcomes: List[RefineOutcome] = []
    to_run: List[str] = []

    for code_id in code_ids:
        meta_summary = _read_json(meta_dir / f"{code_id}.json", verbose=args.verbose)
        meta_collected = _read_json(collected_dir / code_id / "meta.json", verbose=args.verbose)
        old_per_side, old_total = _best_trials(meta_summary, meta_collected)
        old_d = _best_distance(meta_summary, meta_collected)
        if _needs_refine(args.trials, old_per_side, old_total):
            to_run.append(code_id)
        else:
            outcomes.append(
                RefineOutcome(
                    code_id_before=code_id,
                    code_id_after=code_id,
                    status="skipped",
                    reason="trials>=requested",
                    old_trials_per_side=old_per_side,
                    new_trials_per_side=old_per_side,
                    old_d_ub=old_d,
                    new_d_ub=old_d,
                )
            )

    dist_exe: Optional[Path] = None
    if to_run:
        dist_exe = find_dist_m4ri(args.dist_m4ri)
        print(f"[info] dist_m4ri={dist_exe}")

    ts = _utc_stamp()
    archive_collisions_root = best_dir / "archived" / f"rename_collisions_{ts}"

    changed = False
    for idx, code_id in enumerate(to_run, start=1):
        meta_summary_path = meta_dir / f"{code_id}.json"
        meta_collected_path = collected_dir / code_id / "meta.json"
        meta_summary = _read_json(meta_summary_path, verbose=args.verbose)
        meta_collected = _read_json(meta_collected_path, verbose=args.verbose)

        old_per_side, old_total = _best_trials(meta_summary, meta_collected)
        old_d = _best_distance(meta_summary, meta_collected)

        try:
            hx, hz = find_matrix_pair(best_dir, code_id)
        except FileNotFoundError as exc:
            outcomes.append(
                RefineOutcome(
                    code_id_before=code_id,
                    code_id_after=code_id,
                    status="failed",
                    reason=f"missing_matrices:{exc}",
                    old_trials_per_side=old_per_side,
                    new_trials_per_side=old_per_side,
                    old_d_ub=old_d,
                    new_d_ub=old_d,
                )
            )
            continue

        if args.verbose:
            print(
                f"[run ] ({idx}/{len(to_run)}) {code_id} trials {old_per_side or old_total or 0} -> {args.trials}"
            )
            print(f"       Hx={hx} Hz={hz}")

        if dist_exe is None:  # pragma: no cover - guarded above
            raise RuntimeError("dist_m4ri executable not resolved")

        d_z = run_dist_m4ri(dist_exe, finH=hx, finG=hz, steps=args.trials, seed=1, timeout_s=args.timeout)
        d_x = run_dist_m4ri(dist_exe, finH=hz, finG=hx, steps=args.trials, seed=2, timeout_s=args.timeout)
        if d_x is None or d_z is None:
            outcomes.append(
                RefineOutcome(
                    code_id_before=code_id,
                    code_id_after=code_id,
                    status="failed",
                    reason="dist_m4ri_failed",
                    old_trials_per_side=old_per_side,
                    new_trials_per_side=old_per_side,
                    old_d_ub=old_d,
                    new_d_ub=old_d,
                )
            )
            continue

        meta_summary = merge_refine(meta_summary, steps=args.trials, dX=d_x, dZ=d_z)
        meta_collected = merge_refine(meta_collected, steps=args.trials, dX=d_x, dZ=d_z)

        new_d = _best_distance(meta_summary, meta_collected)
        new_per_side, _new_total = _best_trials(meta_summary, meta_collected)

        code_id_after = code_id
        old_suffix = code_id_d_from_suffix(code_id)
        if isinstance(old_suffix, int) and isinstance(new_d, int) and new_d != old_suffix:
            code_id_after = code_id_with_d(code_id, new_d)
            if code_id_after != code_id:
                rename_code(
                    best_dir,
                    code_id,
                    code_id_after,
                    archive_collisions_root=archive_collisions_root,
                )
                _update_code_id_fields(meta_summary, code_id, code_id_after)
                _update_code_id_fields(meta_collected, code_id, code_id_after)
                meta_summary_path = meta_dir / f"{code_id_after}.json"
                meta_collected_path = collected_dir / code_id_after / "meta.json"

        meta_summary["code_id"] = code_id_after
        meta_collected["code_id"] = code_id_after
        atomic_write_json(meta_summary_path, meta_summary, sort_keys=True)
        if meta_collected_path.parent.exists():
            atomic_write_json(meta_collected_path, meta_collected, sort_keys=True)

        outcomes.append(
            RefineOutcome(
                code_id_before=code_id,
                code_id_after=code_id_after,
                status="refined",
                reason="",
                old_trials_per_side=old_per_side,
                new_trials_per_side=new_per_side,
                old_d_ub=old_d,
                new_d_ub=new_d,
            )
        )
        changed = True
        print(
            f"[ok] {code_id_after}: dX_ub={d_x} dZ_ub={d_z} d_ub={min(d_x, d_z)} trials={args.trials}"
        )

    # Always resync/publish after a refinement pass over length n.
    update_result = run_best_codes_update(
        root,
        dry_run=False,
        no_git=args.no_git,
        no_publish=args.no_publish,
        verbose=args.verbose,
    )
    selected_ids = {rec.code_id for rec in update_result.selected.values()}

    for outcome in outcomes:
        outcome.remained_best = outcome.code_id_after in selected_ids

    _print_summary(outcomes)

    if not changed and args.verbose:
        print("[info] no per-code changes, but updater still ran.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
