#!/usr/bin/env python3
"""Systematic smallgroup search for [6,3,3]x[6,3,3] lifted Tanner codes."""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtanner.generators import iter_generator_sets
from qtanner.gf2 import gf2_rank
from qtanner.gap_groups import nr_smallgroups, smallgroup
from qtanner.lift_matrices import build_hx_hz
from qtanner.local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from qtanner.mtx import write_mtx_from_bitrows
from qtanner.gap_session import GapSession
from qtanner.leaderboard import (
    key_nk,
    load_best_by_nk,
    maybe_update_best,
    save_best_by_nk,
)
from qtanner.qdistrnd import dist_rand_css_mtx, dist_rand_dz_mtx, qd_stats_d_ub
from qtanner.search_utils import (
    format_slice_decision,
    parse_qd_batches,
    should_abort_after_batch,
    threshold_to_beat,
)
from qtanner.slice_codes import build_a_slice_checks_H, build_b_slice_checks_Hp
from qtanner.trials import parse_trials_schedule, trials_for_n


def _apply_variant(code: LocalCode, perm: List[int], idx: int) -> LocalCode:
    H_rows = apply_col_perm_to_rows(code.H_rows, perm, code.n)
    G_rows = apply_col_perm_to_rows(code.G_rows, perm, code.n)
    return LocalCode(
        name=f"{code.name}_var{idx}",
        n=code.n,
        k=code.k,
        H_rows=H_rows,
        G_rows=G_rows,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _target_and_mindist(n: int) -> tuple[int, int]:
    target = math.isqrt(n)
    mindist = max(target, 0)
    return target, mindist


def _hash_meta(meta: Dict[str, object]) -> str:
    encoded = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _leaderboard_entry(
    *,
    n: int,
    k: int,
    qd_stats: Dict[str, object],
    group: Dict[str, int],
    A: List[int],
    B: List[int],
    a1v: int,
    b1v: int,
    saved_path: Optional[str],
) -> Dict[str, object]:
    d_ub = int(qd_stats["d_ub"])
    kd_over_n = (k * d_ub) / n
    d_over_sqrt_n = d_ub / math.sqrt(n)
    return {
        "n": n,
        "k": k,
        "d_ub": d_ub,
        "kd_over_n": kd_over_n,
        "d_over_sqrt_n": d_over_sqrt_n,
        "group": group,
        "A": A,
        "B": B,
        "a1v": a1v,
        "b1v": b1v,
        "saved_path": saved_path,
    }


def _update_leaderboard(
    leaderboard: List[Dict[str, object]],
    entry: Dict[str, object],
    *,
    top_n: int = 10,
) -> None:
    leaderboard.append(entry)
    leaderboard.sort(
        key=lambda item: (item["kd_over_n"], item["d_over_sqrt_n"]), reverse=True
    )
    if len(leaderboard) > top_n:
        del leaderboard[top_n:]


def _write_leaderboard_snapshot(
    path: Path, leaderboard: List[Dict[str, object]]
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"top": leaderboard}, f, indent=2, sort_keys=True)


def _paper_baselines_w9() -> Dict[int, List[Tuple[int, int]]]:
    return {
        144: [(8, 12), (12, 11)],
        216: [(8, 18)],
        288: [(16, 16), (8, 19)],
        324: [(4, 26)],
    }


def _print_paper_baselines(n: int, baselines: Dict[int, List[Tuple[int, int]]]) -> None:
    entries = baselines.get(n, [])
    for k, d in entries:
        print(f"[baseline] n={n} k={k} d={d}")


def _default_slice_d_min(
    n: int, *, baselines: Dict[int, List[Tuple[int, int]]], target: int
) -> int:
    entries = baselines.get(n, [])
    if not entries:
        return target
    return max(d for _, d in entries)


def _build_meta(
    *,
    order: int,
    gid: int,
    A: List[int],
    B: List[int],
    base_code: LocalCode,
    C1: LocalCode,
    C1p: LocalCode,
    a1v: int,
    b1v: int,
    n: int,
    k: int,
    qd_batches_used: List[int],
    trials_used: int,
    threshold: int,
    qd_stats: Dict[str, object],
    slice_stats: Dict[str, Optional[Dict[str, object]]],
    mindist: int,
    slice_num: int,
    qd_debug: int,
    seed: Optional[int],
    qd_timeout: float,
    use_slice_filter: bool,
) -> Dict[str, object]:
    return {
        "group": {
            "type": "smallgroup",
            "order": order,
            "gid": gid,
        },
        "A": A,
        "B": B,
        "local_codes": {
            "C0": base_code.name,
            "C1": C1.name,
            "C0p": base_code.name,
            "C1p": C1p.name,
            "a1v": a1v,
            "b1v": b1v,
        },
        "n": n,
        "k": k,
        "distance_estimate": {
            "method": "QDistRnd",
            "num": qd_batches_used[-1] if qd_batches_used else None,
            "batches": qd_batches_used,
            "trials_used": trials_used,
            "mindist": threshold,
            "debug": qd_debug,
            "seed": seed,
            "timeout_sec": qd_timeout,
        },
        "slice_filter": {
            "enabled": use_slice_filter,
            "num": slice_num,
            "mindist": mindist,
            "stats": slice_stats,
        },
        "qdistrnd": qd_stats,
    }


def _save_best_by_nk_code(
    *,
    out_root: Path,
    n: int,
    k: int,
    hx_path: Path,
    hz_path: Path,
    tmp_dir: Path,
    meta: Dict[str, object],
) -> None:
    out_dir = out_root / f"{n}_{k}"
    _ensure_dir(out_dir)
    shutil.copyfile(hx_path, out_dir / "Hx.mtx")
    shutil.copyfile(hz_path, out_dir / "Hz.mtx")
    qd_log = tmp_dir / "qdistrnd.log"
    qd_script = tmp_dir / "qdistrnd.g"
    if qd_log.exists():
        shutil.copyfile(qd_log, out_dir / "qdistrnd.log")
    if qd_script.exists():
        shutil.copyfile(qd_script, out_dir / "qdistrnd.g")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _run_slice_filter(
    *,
    which: str,
    path: Path,
    mindist: int,
    num: int,
    seed: Optional[int],
    log_path: Path,
    session: Optional[GapSession],
    fallback_log_path: Path,
    verbose: int,
) -> tuple[Optional[Dict[str, object]], Optional[str], str]:
    try:
        stats = dist_rand_dz_mtx(
            str(path),
            num=num,
            mindist=mindist,
            seed=seed,
            log_path=str(log_path),
            session=session,
            verbose=verbose,
        )
        return stats, None, str(log_path)
    except Exception as exc:
        print(f"[slice-error] {which} {exc} (log: {log_path})")
        if session is None:
            return None, str(exc), str(log_path)
    try:
        stats = dist_rand_dz_mtx(
            str(path),
            num=num,
            mindist=mindist,
            seed=seed,
            log_path=str(fallback_log_path),
            session=None,
            verbose=verbose,
        )
        return stats, None, str(fallback_log_path)
    except Exception as exc:
        print(f"[slice-error] {which} {exc} (log: {fallback_log_path})")
        return None, str(exc), str(fallback_log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search [6,3,3]x[6,3,3] lifted Tanner codes over SmallGroup orders."
    )
    parser.add_argument("--order-max", type=int, default=27)
    parser.add_argument("--n-min", type=int, default=100)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--max-A", type=int, default=20)
    parser.add_argument("--max-B", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--a1v-max", type=int, default=5)
    parser.add_argument("--b1v-max", type=int, default=5)
    parser.add_argument("--use-slice-filter", action="store_true")
    parser.add_argument("--slice-num", type=int, default=1000)
    parser.add_argument("--slice-fast-num", type=int, default=50)
    parser.add_argument("--qd-num", type=int, default=2000)
    parser.add_argument("--qd-fast-num", type=int, default=50)
    parser.add_argument("--qd-num-schedule", type=str, default="")
    parser.add_argument("--qd-batches", type=str, default="50,200,1000")
    parser.add_argument("--qd-debug", type=int, default=2)
    parser.add_argument("--qd-timeout", type=float, default=120)
    parser.add_argument("--leaderboard-every", type=int, default=25)
    parser.add_argument("--only-group", type=str, default=None)
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--min-distinct-nonid", type=int, default=3)
    parser.add_argument("--max-multiplicity-nonid", type=int, default=None)
    parser.add_argument("--slice-d-min", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use-gap-session", dest="use_gap_session", action="store_true")
    parser.add_argument(
        "--no-gap-session", dest="use_gap_session", action="store_false"
    )
    parser.add_argument(
        "--paper-baselines", dest="paper_baselines", action="store_true"
    )
    parser.add_argument(
        "--no-paper-baselines", dest="paper_baselines", action="store_false"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.set_defaults(use_gap_session=True)
    parser.set_defaults(paper_baselines=True)
    args = parser.parse_args()

    results_path = REPO_ROOT / "results" / "search_w9_smallgroups.jsonl"
    leaderboard_path = REPO_ROOT / "results" / "best_current.json"
    best_by_nk_path = REPO_ROOT / "results" / "best_by_nk.json"
    best_by_n_path = REPO_ROOT / "results" / "best_by_n.json"
    best_by_nk_root = REPO_ROOT / "data" / "best_by_nk"
    tmp_root = REPO_ROOT / "data" / "tmp" / "search_w9_smallgroups"
    promising_root = REPO_ROOT / "data" / "promising"
    _ensure_dir(results_path.parent)
    _ensure_dir(tmp_root)
    _ensure_dir(promising_root)
    _ensure_dir(best_by_nk_root)

    base_code = hamming_6_3_3_shortened()
    perms = variants_6_3_3()
    a1v_max = min(args.a1v_max, len(perms))
    b1v_max = min(args.b1v_max, len(perms))
    a_variants = [_apply_variant(base_code, perms[i], i) for i in range(a1v_max)]
    b_variants = [_apply_variant(base_code, perms[i], i) for i in range(b1v_max)]

    schedule = []
    if args.qd_num_schedule:
        try:
            schedule = parse_trials_schedule(args.qd_num_schedule)
        except ValueError as exc:
            print(f"[error] invalid --qd-num-schedule: {exc}", file=sys.stderr)
            sys.exit(2)
    try:
        qd_batches = parse_qd_batches(args.qd_batches)
    except ValueError as exc:
        print(f"[error] invalid --qd-batches: {exc}", file=sys.stderr)
        sys.exit(2)
    if not qd_batches:
        print("[error] --qd-batches must contain at least one entry.", file=sys.stderr)
        sys.exit(2)

    leaderboard: List[Dict[str, object]] = []
    best_by_nk = load_best_by_nk(str(best_by_nk_path))
    best_by_n = load_best_by_nk(str(best_by_n_path))
    slice_cache_a: Dict[
        tuple[tuple[int, ...], int],
        tuple[Optional[Dict[str, object]], Optional[str]],
    ] = {}
    slice_cache_b: Dict[
        tuple[tuple[int, ...], int],
        tuple[Optional[Dict[str, object]], Optional[str]],
    ] = {}

    session: Optional[GapSession] = None
    if args.use_gap_session and (args.use_slice_filter or args.qd_num > 0):
        session = GapSession(gap_cmd="gap", timeout_sec=args.qd_timeout)

    try:
        with open(results_path, "a", encoding="utf-8") as results_file:
            only_group: Optional[Tuple[int, int]] = None
            if args.only_group:
                try:
                    order_str, gid_str = args.only_group.split(",", 1)
                    only_group = (int(order_str.strip()), int(gid_str.strip()))
                except ValueError as exc:
                    print(f"[error] invalid --only-group: {exc}", file=sys.stderr)
                    sys.exit(2)
                if only_group[0] < 1 or only_group[1] < 1:
                    print("[error] --only-group must be positive integers.", file=sys.stderr)
                    sys.exit(2)
            orders = [only_group[0]] if only_group else range(1, args.order_max + 1)
            baselines = _paper_baselines_w9() if args.paper_baselines else {}
            for order in orders:
                n = 36 * order
                if n < args.n_min or n > args.n_max:
                    continue
                total_groups = nr_smallgroups(order)
                if only_group and only_group[1] > total_groups:
                    print(
                        f"[error] SmallGroup({order}) has only {total_groups} groups.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                gid_range = [only_group[1]] if only_group else range(1, total_groups + 1)
                for gid in gid_range:
                    group = smallgroup(order, gid)
                    target, mindist = _target_and_mindist(n)
                    slice_d_min = (
                        args.slice_d_min
                        if args.slice_d_min is not None
                        else _default_slice_d_min(n, baselines=baselines, target=target)
                    )
                    slice_mindist = max(slice_d_min, 0)
                    qd_num_eff = args.qd_num
                    if schedule:
                        qd_num_eff = min(
                            args.qd_num, trials_for_n(n, schedule, args.qd_num)
                        )
                    qd_fast_num = min(args.qd_fast_num, qd_num_eff)
                    qd_batches_eff = [b for b in qd_batches if b <= qd_num_eff]
                    if not qd_batches_eff:
                        qd_batches_eff = [qd_num_eff]
                    print(
                        f"[group] order={order} gid={gid} n={n} target={target} "
                        f"max_pairs={args.max_pairs} qd_num={qd_num_eff}"
                    )
                    if args.paper_baselines:
                        _print_paper_baselines(n, baselines)
                    distinct = not (order < 6 or args.allow_repeats)
                    max_mult_nonid = args.max_multiplicity_nonid
                    if max_mult_nonid is None:
                        max_mult_nonid = max(0, 6 - 3)
                    A_candidates = list(
                        iter_generator_sets(
                            group,
                            size=6,
                            max_sets=args.max_A,
                            distinct=distinct,
                            min_distinct_nonid=args.min_distinct_nonid,
                            max_multiplicity_nonid=max_mult_nonid,
                            order_by_diversity=True,
                        )
                    )
                    B_candidates = list(
                        iter_generator_sets(
                            group,
                            size=6,
                            max_sets=args.max_B,
                            distinct=distinct,
                            min_distinct_nonid=args.min_distinct_nonid,
                            max_multiplicity_nonid=max_mult_nonid,
                            order_by_diversity=True,
                        )
                    )
                    if not A_candidates or not B_candidates:
                        print("[group] no A/B generator sets found, skipping.")
                        continue

                    pair_count = 0
                    for A in A_candidates:
                        for B in B_candidates:
                            if pair_count >= args.max_pairs:
                                break
                            pair_count += 1
                            for a1v in range(a1v_max):
                                for b1v in range(b1v_max):
                                    print(
                                        "[candidate] "
                                        f"pair={pair_count} A={A} B={B} a1v={a1v} "
                                        f"b1v={b1v} qd_num={qd_num_eff}"
                                    )
                                    C0 = base_code
                                    C0p = base_code
                                    C1 = a_variants[a1v]
                                    C1p = b_variants[b1v]
                                    tmp_id = f"o{order}_g{gid}_p{pair_count}_a{a1v}_b{b1v}"
                                    tmp_dir = tmp_root / tmp_id
                                    _ensure_dir(tmp_dir)

                                    slice_stats: Dict[str, Optional[Dict[str, object]]] = {}
                                    slice_error: Dict[str, Optional[Dict[str, str]]] = {
                                        "a": None,
                                        "b": None,
                                    }
                                    qd_stats = None
                                    qdistrnd_error = None
                                    reject_reason = None
                                    if args.use_slice_filter:
                                        slice_stats = {
                                            "a": None,
                                            "b": None,
                                            "a_error": None,
                                            "b_error": None,
                                            "a_log": None,
                                            "b_log": None,
                                        }
                                        a_key = (tuple(A), a1v)
                                        b_key = (tuple(B), b1v)
                                        if a_key in slice_cache_a:
                                            slice_stats["a"], slice_stats["a_error"] = slice_cache_a[
                                                a_key
                                            ]
                                        else:
                                            a_rows, a_cols = build_a_slice_checks_H(
                                                group, A, C0, C1
                                            )
                                            a_path = tmp_dir / "slice_a.mtx"
                                            write_mtx_from_bitrows(a_path, a_rows, a_cols)
                                            a_log = tmp_dir / "slice_a.log"
                                            a_fallback = tmp_dir / "slice_a_fallback.log"
                                            (
                                                slice_stats["a"],
                                                slice_stats["a_error"],
                                                slice_stats["a_log"],
                                            ) = _run_slice_filter(
                                                which="a",
                                                path=a_path,
                                                mindist=slice_mindist,
                                                num=args.slice_fast_num,
                                                seed=args.seed,
                                                log_path=a_log,
                                                session=session,
                                                fallback_log_path=a_fallback,
                                                verbose=args.verbose,
                                            )
                                            slice_cache_a[a_key] = (
                                                slice_stats["a"],
                                                slice_stats["a_error"],
                                            )
                                        if b_key in slice_cache_b:
                                            slice_stats["b"], slice_stats["b_error"] = slice_cache_b[
                                                b_key
                                            ]
                                        else:
                                            b_rows, b_cols = build_b_slice_checks_Hp(
                                                group, B, C0p, C1p
                                            )
                                            b_path = tmp_dir / "slice_b.mtx"
                                            write_mtx_from_bitrows(b_path, b_rows, b_cols)
                                            b_log = tmp_dir / "slice_b.log"
                                            b_fallback = tmp_dir / "slice_b_fallback.log"
                                            (
                                                slice_stats["b"],
                                                slice_stats["b_error"],
                                                slice_stats["b_log"],
                                            ) = _run_slice_filter(
                                                which="b",
                                                path=b_path,
                                                mindist=slice_mindist,
                                                num=args.slice_fast_num,
                                                seed=args.seed,
                                                log_path=b_log,
                                                session=session,
                                                fallback_log_path=b_fallback,
                                                verbose=args.verbose,
                                            )
                                            slice_cache_b[b_key] = (
                                                slice_stats["b"],
                                                slice_stats["b_error"],
                                            )
                                        if slice_stats["a_error"] or slice_stats["b_error"]:
                                            if slice_stats["a_error"]:
                                                slice_error["a"] = {
                                                    "error": str(slice_stats["a_error"]),
                                                    "log_path": str(
                                                        slice_stats.get("a_log") or ""
                                                    ),
                                                }
                                            if slice_stats["b_error"]:
                                                slice_error["b"] = {
                                                    "error": str(slice_stats["b_error"]),
                                                    "log_path": str(
                                                        slice_stats.get("b_log") or ""
                                                    ),
                                                }
                                            reject_reason = "slice_filter_error"
                                            record = {
                                                "group": {"order": order, "gid": gid},
                                                "A": A,
                                                "B": B,
                                                "a1v": a1v,
                                                "b1v": b1v,
                                                "n": n,
                                                "k": None,
                                                "target": target,
                                                "slice": slice_stats,
                                                "slice_error": slice_error,
                                                "qdistrnd": None,
                                                "qdistrnd_error": None,
                                                "reject_reason": reject_reason,
                                                "saved_path": None,
                                            }
                                            results_file.write(
                                                json.dumps(record, sort_keys=True) + "\n"
                                            )
                                            results_file.flush()
                                            print(
                                                "[candidate] rejected by slice filter error."
                                            )
                                            continue
                                        a_d_ub = int(slice_stats["a"]["dZ_ub"])
                                        b_d_ub = int(slice_stats["b"]["dZ_ub"])
                                        a_early = bool(slice_stats["a"]["terminated_early"])
                                        b_early = bool(slice_stats["b"]["terminated_early"])
                                        a_pass = (not a_early) and (a_d_ub >= slice_d_min)
                                        b_pass = (not b_early) and (b_d_ub >= slice_d_min)
                                        if args.verbose:
                                            print(
                                                format_slice_decision(
                                                    which="a",
                                                    n_slice=len(A) * order,
                                                    d_min=slice_d_min,
                                                    trials=int(slice_stats["a"]["num"]),
                                                    mindist=int(slice_stats["a"]["mindist"]),
                                                    d_ub=a_d_ub,
                                                    early_exit=a_early,
                                                    passed=a_pass,
                                                )
                                            )
                                            print(
                                                format_slice_decision(
                                                    which="b",
                                                    n_slice=len(B) * order,
                                                    d_min=slice_d_min,
                                                    trials=int(slice_stats["b"]["num"]),
                                                    mindist=int(slice_stats["b"]["mindist"]),
                                                    d_ub=b_d_ub,
                                                    early_exit=b_early,
                                                    passed=b_pass,
                                                )
                                            )
                                        if not a_pass or not b_pass:
                                            reject_reason = "slice_filter_reject"
                                            if args.verbose:
                                                if not a_pass:
                                                    print(
                                                        "[slice-reject] "
                                                        f"a d_ub={a_d_ub} "
                                                        f"early_exit={a_early} "
                                                        f"d_min={slice_d_min}"
                                                    )
                                                if not b_pass:
                                                    print(
                                                        "[slice-reject] "
                                                        f"b d_ub={b_d_ub} "
                                                        f"early_exit={b_early} "
                                                        f"d_min={slice_d_min}"
                                                    )
                                            record = {
                                                "group": {"order": order, "gid": gid},
                                                "A": A,
                                                "B": B,
                                                "a1v": a1v,
                                                "b1v": b1v,
                                                "n": n,
                                                "k": None,
                                                "target": target,
                                                "slice": slice_stats,
                                                "slice_error": None,
                                                "qdistrnd": None,
                                                "qdistrnd_error": None,
                                                "reject_reason": reject_reason,
                                                "saved_path": None,
                                            }
                                            results_file.write(
                                                json.dumps(record, sort_keys=True) + "\n"
                                            )
                                            results_file.flush()
                                            print("[candidate] rejected by slice filter.")
                                            continue

                                    hx_rows, hz_rows, n_cols = build_hx_hz(
                                        group, A, B, C0, C1, C0p, C1p
                                    )
                                    hx_path = tmp_dir / "Hx.mtx"
                                    hz_path = tmp_dir / "Hz.mtx"
                                    write_mtx_from_bitrows(hx_path, hx_rows, n_cols)
                                    write_mtx_from_bitrows(hz_path, hz_rows, n_cols)
                                    rank_hx = gf2_rank(hx_rows, n_cols)
                                    rank_hz = gf2_rank(hz_rows, n_cols)
                                    k = n_cols - rank_hx - rank_hz
                                    if k == 0:
                                        reject_reason = "k_zero"
                                        record = {
                                            "group": {"order": order, "gid": gid},
                                            "A": A,
                                            "B": B,
                                            "a1v": a1v,
                                            "b1v": b1v,
                                            "n": n,
                                            "k": k,
                                            "target": target,
                                            "slice": slice_stats,
                                            "slice_error": None,
                                            "qdistrnd": None,
                                            "qdistrnd_error": None,
                                            "reject_reason": reject_reason,
                                            "saved_path": None,
                                        }
                                        results_file.write(
                                            json.dumps(record, sort_keys=True) + "\n"
                                        )
                                        results_file.flush()
                                        print("[candidate] k=0, skipping.")
                                        continue
                                    best_key = key_nk(n, k)
                                    best_entry = best_by_nk.get(best_key)
                                    best_d_obs = None
                                    if isinstance(best_entry, dict):
                                        best_d_obs = best_entry.get("d_obs")
                                    threshold = threshold_to_beat(target, best_d_obs)
                                    qd_batches_used: List[int] = []
                                    trials_used = 0
                                    last_threshold_used = threshold
                                    try:
                                        for batch_num in qd_batches_eff:
                                            batch_threshold = threshold
                                            qd_stats = dist_rand_css_mtx(
                                                str(hx_path),
                                                str(hz_path),
                                                num=batch_num,
                                                mindist=batch_threshold,
                                                debug=args.qd_debug,
                                                seed=args.seed,
                                                timeout_sec=args.qd_timeout,
                                                log_path=str(tmp_dir / "qdistrnd.log"),
                                                session=session,
                                                verbose=args.verbose,
                                            )
                                            qd_batches_used.append(batch_num)
                                            trials_used += batch_num
                                            last_threshold_used = batch_threshold
                                            if qd_stats is None:
                                                break
                                            d_obs = qd_stats_d_ub(qd_stats)
                                            if d_obs is None:
                                                break
                                            best_record = {
                                                "n": n,
                                                "k": k,
                                                "d_obs": int(d_obs),
                                                "trials_used": trials_used,
                                                "group": {"order": order, "gid": gid},
                                                "A": A,
                                                "B": B,
                                                "a1v": a1v,
                                                "b1v": b1v,
                                                "timestamp": time.strftime(
                                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                                ),
                                            }
                                            updated_best = maybe_update_best(
                                                best_by_nk, best_record
                                            )
                                            if updated_best:
                                                save_best_by_nk(
                                                    str(best_by_nk_path), best_by_nk
                                                )
                                                best_by_n_key = str(n)
                                                current_best_n = best_by_n.get(best_by_n_key)
                                                best_by_n_updated = False
                                                if current_best_n is None:
                                                    best_by_n[best_by_n_key] = best_record
                                                    best_by_n_updated = True
                                                else:
                                                    current_d = current_best_n.get("d_obs")
                                                    current_trials = current_best_n.get(
                                                        "trials_used"
                                                    )
                                                    if current_d is None or current_trials is None:
                                                        best_by_n[best_by_n_key] = best_record
                                                        best_by_n_updated = True
                                                    elif int(d_obs) > int(current_d):
                                                        best_by_n[best_by_n_key] = best_record
                                                        best_by_n_updated = True
                                                    elif int(d_obs) == int(current_d) and int(
                                                        trials_used
                                                    ) < int(current_trials):
                                                        best_by_n[best_by_n_key] = best_record
                                                        best_by_n_updated = True
                                                if best_by_n_updated:
                                                    save_best_by_nk(
                                                        str(best_by_n_path), best_by_n
                                                    )
                                                best_d_obs = int(d_obs)
                                                threshold = threshold_to_beat(
                                                    target, best_d_obs
                                                )
                                                best_meta = _build_meta(
                                                    order=order,
                                                    gid=gid,
                                                    A=A,
                                                    B=B,
                                                    base_code=base_code,
                                                    C1=C1,
                                                    C1p=C1p,
                                                    a1v=a1v,
                                                    b1v=b1v,
                                                    n=n,
                                                    k=k,
                                                    qd_batches_used=list(qd_batches_used),
                                                    trials_used=trials_used,
                                                    threshold=batch_threshold,
                                                    qd_stats=qd_stats,
                                                    slice_stats=slice_stats,
                                                    mindist=slice_mindist,
                                                    slice_num=args.slice_fast_num,
                                                    qd_debug=args.qd_debug,
                                                    seed=args.seed,
                                                    qd_timeout=args.qd_timeout,
                                                    use_slice_filter=args.use_slice_filter,
                                                )
                                                _save_best_by_nk_code(
                                                    out_root=best_by_nk_root,
                                                    n=n,
                                                    k=k,
                                                    hx_path=hx_path,
                                                    hz_path=hz_path,
                                                    tmp_dir=tmp_dir,
                                                    meta=best_meta,
                                                )
                                                print(
                                                    "[best] "
                                                    f"({n},{k}) updated: d_obs={d_obs} "
                                                    f"trials={trials_used} "
                                                    f"group={order},{gid} a1v={a1v} b1v={b1v}"
                                                )
                                            if best_d_obs is not None and d_obs < best_d_obs:
                                                break
                                            if qd_stats["terminated_early_X"] or qd_stats[
                                                "terminated_early_Z"
                                            ]:
                                                break
                                            if should_abort_after_batch(
                                                d_obs=d_obs,
                                                target=target,
                                                best_d_obs=best_d_obs,
                                            ):
                                                break
                                    except Exception as exc:
                                        qdistrnd_error = str(exc)
                                    if qd_stats is None:
                                        qdistrnd_error = qdistrnd_error or "QDistRnd returned no stats."
                                    if qdistrnd_error:
                                        reject_reason = "qdistrnd_error"
                                        record = {
                                            "group": {"order": order, "gid": gid},
                                            "A": A,
                                            "B": B,
                                            "a1v": a1v,
                                            "b1v": b1v,
                                            "n": n,
                                        "k": k,
                                        "target": target,
                                        "slice": slice_stats,
                                        "slice_error": None,
                                        "qdistrnd": None,
                                        "qdistrnd_error": qdistrnd_error,
                                        "reject_reason": reject_reason,
                                        "saved_path": None,
                                        }
                                        results_file.write(
                                            json.dumps(record, sort_keys=True) + "\n"
                                        )
                                        results_file.flush()
                                        print("[candidate] rejected by QDistRnd error.")
                                        continue
                                    d_ub = qd_stats_d_ub(qd_stats)
                                    if d_ub is None:
                                        reject_reason = "qdistrnd_missing_d_ub"
                                        record = {
                                            "group": {"order": order, "gid": gid},
                                            "A": A,
                                            "B": B,
                                            "a1v": a1v,
                                            "b1v": b1v,
                                            "n": n,
                                            "k": k,
                                        "target": target,
                                        "slice": slice_stats,
                                        "slice_error": None,
                                        "qdistrnd": qd_stats,
                                        "qdistrnd_error": None,
                                        "reject_reason": reject_reason,
                                        "saved_path": None,
                                        }
                                        results_file.write(
                                            json.dumps(record, sort_keys=True) + "\n"
                                        )
                                        results_file.flush()
                                        print("[candidate] rejected by missing d_ub.")
                                        continue
                                    d_obs = int(d_ub)
                                    if d_obs < target:
                                        reject_reason = "qdistrnd_below_target"
                                    promising = d_ub >= target and k * d_ub >= n
                                    saved_path = None
                                    meta = _build_meta(
                                        order=order,
                                        gid=gid,
                                        A=A,
                                        B=B,
                                        base_code=base_code,
                                        C1=C1,
                                        C1p=C1p,
                                        a1v=a1v,
                                        b1v=b1v,
                                        n=n,
                                        k=k,
                                        qd_batches_used=list(qd_batches_used),
                                        trials_used=trials_used,
                                        threshold=last_threshold_used,
                                        qd_stats=qd_stats,
                                        slice_stats=slice_stats,
                                        mindist=slice_mindist,
                                        slice_num=args.slice_fast_num,
                                        qd_debug=args.qd_debug,
                                        seed=args.seed,
                                        qd_timeout=args.qd_timeout,
                                        use_slice_filter=args.use_slice_filter,
                                    )
                                    if promising:
                                        code_id = _hash_meta(meta)
                                        out_dir = promising_root / code_id
                                        _ensure_dir(out_dir)
                                        shutil.copyfile(hx_path, out_dir / "Hx.mtx")
                                        shutil.copyfile(hz_path, out_dir / "Hz.mtx")
                                        shutil.copyfile(
                                            tmp_dir / "qdistrnd.log",
                                            out_dir / "qdistrnd.log",
                                        )
                                        shutil.copyfile(
                                            tmp_dir / "qdistrnd.g", out_dir / "qdistrnd.g"
                                        )
                                        with open(
                                            out_dir / "meta.json", "w", encoding="utf-8"
                                        ) as f:
                                            json.dump(meta, f, indent=2, sort_keys=True)
                                        saved_path = str(out_dir)
                                        print(
                                            f"[candidate] saved promising code {code_id}"
                                        )

                                    _update_leaderboard(
                                        leaderboard,
                                        _leaderboard_entry(
                                            n=n,
                                            k=k,
                                            qd_stats=qd_stats,
                                            group={"order": order, "gid": gid},
                                            A=A,
                                            B=B,
                                            a1v=a1v,
                                            b1v=b1v,
                                            saved_path=saved_path,
                                        ),
                                    )
                                    record = {
                                        "group": {"order": order, "gid": gid},
                                        "A": A,
                                        "B": B,
                                        "a1v": a1v,
                                        "b1v": b1v,
                                        "n": n,
                                        "k": k,
                                        "target": target,
                                        "slice": slice_stats,
                                        "slice_error": None,
                                        "qdistrnd": qd_stats,
                                        "qdistrnd_error": None,
                                        "reject_reason": reject_reason,
                                        "saved_path": saved_path,
                                    }
                                    results_file.write(
                                        json.dumps(record, sort_keys=True) + "\n"
                                    )
                                    results_file.flush()
                            if (
                                args.leaderboard_every > 0
                                and pair_count % args.leaderboard_every == 0
                            ):
                                _write_leaderboard_snapshot(
                                    leaderboard_path, leaderboard
                                )
                                if leaderboard:
                                    best = leaderboard[0]
                                    group = best["group"]
                                    print(
                                        "[leaderboard] "
                                        f"best kd/n={best['kd_over_n']:.3f} "
                                        f"d/sqrt(n)={best['d_over_sqrt_n']:.3f} "
                                        f"n={best['n']} k={best['k']} d_ub={best['d_ub']} "
                                        f"group={group['order']},{group['gid']} "
                                        f"a1v={best['a1v']} b1v={best['b1v']}"
                                    )
                                else:
                                    print("[leaderboard] no candidates yet.")
                        if pair_count >= args.max_pairs:
                            print("[group] reached max-pairs limit.")
                            break
                    if only_group:
                        return
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
