"""Search for quantum Tanner codes from left-right Cayley complexes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .classical_distance import ClassicalCodeAnalysis, analyze_parity_check_bitrows
from .gf2 import gf2_rank
from .group import CyclicGroup, FiniteGroup, canonical_group_spec, group_from_spec
from .lift_matrices import build_hx_hz, css_commutes
from .local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from .mtx import write_mtx_from_bitrows
from .qdistrnd import gap_is_available, qdistrnd_is_available, write_qdistrnd_log


@dataclass(frozen=True)
class SliceMetrics:
    h: ClassicalCodeAnalysis
    g: ClassicalCodeAnalysis
    d_min: int
    k_min: int
    k_sum: int

    def as_dict(self) -> dict:
        return {
            "h": self.h.as_dict(),
            "g": self.g.as_dict(),
            "d_min": self.d_min,
            "k_min": self.k_min,
            "k_sum": self.k_sum,
        }


@dataclass(frozen=True)
class SliceCandidate:
    elements: List[int]
    perm_idx: int
    metrics: SliceMetrics

    def key(self) -> Tuple[Tuple[int, ...], int]:
        return (tuple(self.elements), int(self.perm_idx))


def _parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _ceil_sqrt(n: int) -> int:
    root = math.isqrt(n)
    return root if root * root == n else root + 1


def _slice_dist_threshold(spec: str, *, n: int) -> int:
    if spec is None:
        raise ValueError("--min-slice-dist must be provided.")
    value = spec.strip()
    if not value:
        raise ValueError("--min-slice-dist must be non-empty.")
    if value.lower() == "sqrtn":
        return _ceil_sqrt(n)
    try:
        dist = int(value)
    except ValueError as exc:
        raise ValueError(
            f"--min-slice-dist must be an integer or 'sqrtN' (got '{spec}')."
        ) from exc
    if dist < 0:
        raise ValueError("--min-slice-dist must be nonnegative.")
    return dist


def _timestamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _format_command_line() -> Tuple[str, str, str]:
    argv = list(sys.argv)
    actual = shlex.join([sys.executable, *argv])
    module = shlex.join(["python", "-m", "qtanner.search", *argv[1:]])
    raw = json.dumps(argv)
    return actual, module, raw


def _write_commands_run(outdir: Path, args: argparse.Namespace) -> None:
    actual_cmd, module_cmd, raw_argv = _format_command_line()
    outdir_str = str(outdir)
    trials = args.trials
    uniq_target = args.best_uniq_target
    gap_cmd = args.gap_cmd
    lines = [
        "# qtanner search run commands",
        "",
        "## Command used",
        f"`{actual_cmd}`",
        "",
        "## Reproducible module form",
        f"`{module_cmd}`",
        "",
        "## Raw argv",
        f"`{raw_argv}`",
        "",
        "## Monitor progress",
        f'OUT="{outdir_str}"',
        "while true; do",
        "  clear",
        "  date",
        "  echo \"=== ${OUT}/best_by_k.txt ===\"",
        "  test -f \"${OUT}/best_by_k.txt\" && cat \"${OUT}/best_by_k.txt\" || echo \"(missing)\"",
        "  echo",
        "  echo \"=== ${OUT}/coverage.txt ===\"",
        "  test -f \"${OUT}/coverage.txt\" && cat \"${OUT}/coverage.txt\" || echo \"(missing)\"",
        "  sleep 2",
        "done",
        "",
        "## Monitor by tailing the log",
        "tail -n 50 -F \"${OUT}/best_by_k.log\"",
        "",
        "## Re-check distance (QDistRnd, mindist=0)",
        "ID=\"PUT_CODE_ID_HERE\"",
        (
            "python -m qtanner.check_distance --run \"${OUT}\" "
            f"--id \"${{ID}}\" --trials {trials} --uniq-target {uniq_target} --gap-cmd {gap_cmd}"
        ),
        "",
        "Example:",
        "ID=\"EXAMPLE_CODE_ID\"",
        (
            "python -m qtanner.check_distance --run \"${OUT}\" "
            f"--id \"${{ID}}\" --trials {trials} --uniq-target {uniq_target} --gap-cmd {gap_cmd}"
        ),
        "",
        "## Generate LaTeX report",
        "python -m qtanner.report --run \"${OUT}\" --out \"${OUT}/report.tex\"",
        "",
        "Optional PDF:",
        "python -m qtanner.report --run \"${OUT}\" --out \"${OUT}/report.tex\" --pdf",
        "",
    ]
    (outdir / "COMMANDS_RUN.md").write_text("\n".join(lines), encoding="utf-8")


def _apply_variant_perm(code: LocalCode, perm: List[int], variant_idx: int) -> LocalCode:
    H_rows = apply_col_perm_to_rows(code.H_rows, perm, code.n)
    G_rows = apply_col_perm_to_rows(code.G_rows, perm, code.n)
    return LocalCode(
        name=f"{code.name}_var{variant_idx}",
        n=code.n,
        k=code.k,
        H_rows=H_rows,
        G_rows=G_rows,
    )


def _build_variant_codes(base_code: LocalCode) -> List[LocalCode]:
    perms = variants_6_3_3()
    return [
        _apply_variant_perm(base_code, perm, idx) for idx, perm in enumerate(perms)
    ]


def _comb_count(m: int) -> int:
    if m < 6:
        return 0
    return math.comb(m - 1, 5)


def _multiset_count(m: int) -> int:
    if m <= 0:
        return 0
    return math.comb(m + 4, 5)


def _sample_combinations(
    *,
    m: int,
    count: int,
    rng: random.Random,
) -> List[List[int]]:
    if count <= 0:
        return []
    if m < 6:
        return []
    pool = list(range(1, m))
    chosen: set[Tuple[int, ...]] = set()
    max_attempts = max(50, count * 50)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        comb = tuple(sorted(rng.sample(pool, 5)))
        chosen.add(comb)
        attempts += 1
    if len(chosen) < count:
        # fallback: deterministic prefix if sampling couldn't hit enough
        combs = list(combinations(pool, 5))[:count]
        chosen.update(combs)
    ordered = sorted(chosen)
    if len(ordered) > count:
        ordered = ordered[:count]
    return [[0, *comb] for comb in ordered]


def _sample_multisets(
    *,
    m: int,
    count: int,
    rng: random.Random,
) -> List[List[int]]:
    if count <= 0 or m <= 0:
        return []
    chosen: set[Tuple[int, ...]] = set()
    max_attempts = max(50, count * 50)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        sample = [rng.randrange(m) for _ in range(6)]
        if 0 not in sample:
            attempts += 1
            continue
        chosen.add(tuple(sorted(sample)))
        attempts += 1
    if len(chosen) < count:
        for comb in combinations_with_replacement(range(m), 6):
            if comb[0] != 0:
                continue
            chosen.add(comb)
            if len(chosen) >= count:
                break
    ordered = list(chosen)
    rng.shuffle(ordered)
    if len(ordered) > count:
        ordered = ordered[:count]
    return [list(comb) for comb in ordered]


def _enumerate_sets(
    *,
    m: int,
    max_sets: Optional[int],
    rng: random.Random,
    feasible_limit: int,
    allow_repeats: bool = False,
) -> List[List[int]]:
    if allow_repeats:
        total = _multiset_count(m)
        if total == 0:
            return []
        if total <= feasible_limit:
            combos = [
                list(comb)
                for comb in combinations_with_replacement(range(m), 6)
                if comb[0] == 0
            ]
            if max_sets is not None and len(combos) > max_sets:
                rng.shuffle(combos)
                combos = combos[:max_sets]
            return combos
        target = max_sets if max_sets is not None else min(feasible_limit, total)
        return _sample_multisets(m=m, count=min(target, total), rng=rng)
    total = _comb_count(m)
    if total == 0:
        return []
    if max_sets is None and total <= feasible_limit:
        return [[0, *comb] for comb in combinations(range(1, m), 5)]
    target = max_sets if max_sets is not None else min(feasible_limit, total)
    return _sample_combinations(m=m, count=target, rng=rng)


def _analyze_slice(
    rows: Sequence[int],
    ncols: int,
    *,
    rng: random.Random,
) -> ClassicalCodeAnalysis:
    return analyze_parity_check_bitrows(
        rows,
        ncols,
        max_k_exact=10,
        samples=1 << 10,
        seed=rng.randrange(1 << 31),
    )


def _score_a_slice(
    group: FiniteGroup,
    A: List[int],
    C0: LocalCode,
    C1: LocalCode,
    *,
    rng: random.Random,
) -> SliceMetrics:
    from .slice_codes import build_a_slice_checks_G, build_a_slice_checks_H

    h_rows, n_cols = build_a_slice_checks_H(group, A, C0, C1)
    g_rows, _ = build_a_slice_checks_G(group, A, C0, C1)
    h_stats = _analyze_slice(h_rows, n_cols, rng=rng)
    g_stats = _analyze_slice(g_rows, n_cols, rng=rng)
    return SliceMetrics(
        h=h_stats,
        g=g_stats,
        d_min=min(h_stats.d, g_stats.d),
        k_min=min(h_stats.k, g_stats.k),
        k_sum=h_stats.k + g_stats.k,
    )


def _score_b_slice(
    group: FiniteGroup,
    B: List[int],
    C0p: LocalCode,
    C1p: LocalCode,
    *,
    rng: random.Random,
) -> SliceMetrics:
    from .slice_codes import build_b_slice_checks_Gp, build_b_slice_checks_Hp

    h_rows, n_cols = build_b_slice_checks_Hp(group, B, C0p, C1p)
    g_rows, _ = build_b_slice_checks_Gp(group, B, C0p, C1p)
    h_stats = _analyze_slice(h_rows, n_cols, rng=rng)
    g_stats = _analyze_slice(g_rows, n_cols, rng=rng)
    return SliceMetrics(
        h=h_stats,
        g=g_stats,
        d_min=min(h_stats.d, g_stats.d),
        k_min=min(h_stats.k, g_stats.k),
        k_sum=h_stats.k + g_stats.k,
    )


def _classical_candidate_id(
    *,
    side: str,
    group: FiniteGroup,
    elements: List[int],
    perm_idx: int,
) -> str:
    elems = ",".join(str(x) for x in elements)
    return f"{group.name}:{side}:{perm_idx}:{elems}"


def _frontier_select(
    scored: List[SliceCandidate],
    *,
    group: FiniteGroup,
    side: str,
    n_quantum: int,
    frontier_max_per_point: int,
    frontier_max_total: int,
) -> Tuple[List[SliceCandidate], dict]:
    d0 = _ceil_sqrt(n_quantum)
    if not scored:
        payload = {
            "group": group.name,
            "order": group.order,
            "n_quantum": n_quantum,
            "d0": d0,
            "breakpoints": [],
            "selected_ids": [],
        }
        return [], payload

    items = []
    for cand in scored:
        items.append(
            {
                "cand": cand,
                "id": _classical_candidate_id(
                    side=side,
                    group=group,
                    elements=cand.elements,
                    perm_idx=cand.perm_idx,
                ),
                "k": cand.metrics.k_min,
                "d": cand.metrics.d_min,
            }
        )
    max_d = max(item["d"] for item in items)
    breakpoints = []
    selected_map: Dict[str, dict] = {}
    for D in range(d0, max_d + 1):
        eligible = [item for item in items if item["d"] >= D]
        if not eligible:
            continue
        k_best = max(item["k"] for item in eligible)
        best = [item for item in eligible if item["k"] == k_best]
        max_d_best = max(item["d"] for item in best)
        best = [item for item in best if item["d"] == max_d_best]
        best_sorted = sorted(best, key=lambda item: (-item["d"], -item["k"], item["id"]))
        if frontier_max_per_point > 0:
            best_sorted = best_sorted[:frontier_max_per_point]
        selected_ids = [item["id"] for item in best_sorted]
        selected_points = sorted(
            {(item["k"], item["d"]) for item in best_sorted},
            key=lambda point: (-point[0], -point[1]),
        )
        breakpoints.append(
            {
                "D": D,
                "k_best": k_best,
                "selected_ids": selected_ids,
                "selected_points": [list(point) for point in selected_points],
            }
        )
        for item in best_sorted:
            selected_map[item["id"]] = item

    selected_items = list(selected_map.values())
    if frontier_max_total > 0 and len(selected_items) > frontier_max_total:
        selected_items = sorted(
            selected_items, key=lambda item: (-item["d"], -item["k"], item["id"])
        )[:frontier_max_total]
        keep_ids = {item["id"] for item in selected_items}
        for bp in breakpoints:
            bp["selected_ids"] = [cid for cid in bp["selected_ids"] if cid in keep_ids]
            points = sorted(
                {
                    (item["k"], item["d"])
                    for item in selected_items
                    if item["id"] in bp["selected_ids"]
                },
                key=lambda point: (-point[0], -point[1]),
            )
            bp["selected_points"] = [list(point) for point in points]
    else:
        keep_ids = {item["id"] for item in selected_items}

    selected_items = sorted(
        selected_items, key=lambda item: (-item["d"], -item["k"], item["id"])
    )
    selected_ids = [item["id"] for item in selected_items if item["id"] in keep_ids]
    payload = {
        "group": group.name,
        "order": group.order,
        "n_quantum": n_quantum,
        "d0": d0,
        "breakpoints": breakpoints,
        "selected_ids": selected_ids,
    }

    points = sorted(
        {(item["k"], item["d"]) for item in selected_items},
        key=lambda point: (-point[0], -point[1]),
    )
    point_str = ", ".join(f"({k},{d})" for k, d in points) if points else "n/a"
    print(
        f"[pilot] {side} frontier kept {len(selected_items)} candidates with points: {point_str}"
    )
    return [item["cand"] for item in selected_items], payload


def _add_explore_candidates(
    scored: List[SliceCandidate],
    selected: List[SliceCandidate],
    *,
    explore_n: int,
    rng: random.Random,
    side: str,
) -> List[SliceCandidate]:
    if explore_n <= 0 or not scored:
        return selected
    selected_keys = {cand.key() for cand in selected}
    remainder = [cand for cand in scored if cand.key() not in selected_keys]
    if not remainder:
        return selected
    extra = rng.sample(remainder, min(explore_n, len(remainder)))
    if extra:
        print(f"[pilot] {side} explore added {len(extra)} candidates.")
    return selected + extra


def _select_above_sqrt(
    scored: List[SliceCandidate],
    *,
    group: FiniteGroup,
    side: str,
    d0: int,
    cap: int,
) -> Tuple[List[SliceCandidate], List[str]]:
    eligible = [cand for cand in scored if cand.metrics.d_min >= d0]
    if not eligible:
        return [], []
    if cap > 0 and len(eligible) > cap:
        ranked = sorted(
            eligible,
            key=lambda cand: (
                -cand.metrics.d_min,
                -cand.metrics.k_min,
                _classical_candidate_id(
                    side=side, group=group, elements=cand.elements, perm_idx=cand.perm_idx
                ),
            ),
        )
        eligible = ranked[:cap]
    selected_ids = [
        _classical_candidate_id(
            side=side, group=group, elements=cand.elements, perm_idx=cand.perm_idx
        )
        for cand in eligible
    ]
    return eligible, selected_ids


def _write_frontier_file(
    path: Path,
    payloads: List[dict],
    *,
    single_group: bool,
) -> None:
    if single_group:
        data = payloads[0] if payloads else {}
    else:
        data = {"groups": payloads}
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _slice_stats_summary(stats: ClassicalCodeAnalysis) -> dict:
    return {"n": stats.n, "k": stats.k, "d_est": stats.d}


def _group_record(group: FiniteGroup) -> dict:
    return {"spec": group.name, "order": group.order}


def _limiting_side(dx_ub: int, dz_ub: int, uniq_x: int, uniq_z: int) -> str:
    if dx_ub < dz_ub:
        return "x"
    if dz_ub < dx_ub:
        return "z"
    return "x" if uniq_x <= uniq_z else "z"


def _next_confirm_num(current: int, mult: int) -> int:
    if mult <= 1:
        return current + 1
    return current * mult


def _safe_id_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return cleaned.strip("_") or "group"


def _compute_base_k_table(
    *,
    base_code: LocalCode,
    variant_codes: List[LocalCode],
    outdir: Path,
) -> List[List[int]]:
    perm_total = len(variant_codes)
    group = CyclicGroup(1)
    A0 = [0] * base_code.n
    B0 = [0] * base_code.n
    table: List[List[int]] = []
    for a_idx in range(perm_total):
        row: List[int] = []
        C1 = variant_codes[a_idx]
        for b_idx in range(perm_total):
            C1p = variant_codes[b_idx]
            hx_rows, hz_rows, n_cols = build_hx_hz(
                group, A0, B0, base_code, C1, base_code, C1p
            )
            rank_hx = gf2_rank(hx_rows[:], n_cols)
            rank_hz = gf2_rank(hz_rows[:], n_cols)
            k = n_cols - rank_hx - rank_hz
            row.append(k)
        table.append(row)
    payload = {
        "n_base": 36,
        "perm_total": perm_total,
        "k_base": table,
    }
    (outdir / "base_k_table.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return table


def _mult_stats(mult: Sequence[int]) -> Tuple[int, Optional[float]]:
    uniq = len(mult)
    if uniq == 0:
        return 0, None
    return uniq, sum(mult) / uniq


def _qd_side_stats(
    *,
    d_signed: int,
    rounds_done: int,
    vec_count_total: int,
    mult: List[int],
) -> Dict[str, object]:
    uniq, avg = _mult_stats(mult)
    return {
        "d_signed": d_signed,
        "rounds_done": rounds_done,
        "vec_count_total": vec_count_total,
        "mult": mult,
        "uniq": uniq,
        "avg": avg,
        "early_stop": d_signed < 0,
    }


def _qd_summary_from_stats(
    qd_stats: Dict[str, object],
    *,
    num: int,
    mindist: int,
    seed: int,
    runtime_sec: float,
    gap_cmd: str,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], int, int, int]:
    dx_signed = int(qd_stats.get("dx_signed", 0))
    dz_signed = int(qd_stats.get("dz_signed", 0))
    dx_ub = abs(dx_signed)
    dz_ub = abs(dz_signed)
    d_ub = min(dx_ub, dz_ub)
    qd_x = _qd_side_stats(
        d_signed=dx_signed,
        rounds_done=int(qd_stats.get("rx", 0)),
        vec_count_total=int(qd_stats.get("vx", 0)),
        mult=list(qd_stats.get("mx", [])),
    )
    qd_z = _qd_side_stats(
        d_signed=dz_signed,
        rounds_done=int(qd_stats.get("rz", 0)),
        vec_count_total=int(qd_stats.get("vz", 0)),
        mult=list(qd_stats.get("mz", [])),
    )
    qd = {
        "trials_requested": num,
        "mindist": mindist,
        "dx": dx_ub,
        "dz": dz_ub,
        "d": d_ub,
        "dx_ub": dx_ub,
        "dz_ub": dz_ub,
        "d_ub": d_ub,
        "qd_x": qd_x,
        "qd_z": qd_z,
        "seed": seed,
        "runtime_sec": runtime_sec,
        "gap_cmd": gap_cmd,
    }
    return qd, qd_x, qd_z, dx_ub, dz_ub, d_ub


def _format_avg(avg: Optional[float]) -> str:
    if avg is None:
        return "n/a"
    return f"{avg:.3f}"


def _best_by_k_key(entry: Dict[str, object]) -> Tuple[str, int, int]:
    group_spec = (entry.get("group") or {}).get("spec") or "unknown"
    return (group_spec, int(entry.get("n", 0)), int(entry.get("k", 0)))


def _is_better_entry(entry: Dict[str, object], current: Dict[str, object]) -> bool:
    qd = entry.get("qdistrnd") or {}
    cur_qd = current.get("qdistrnd") or {}
    d_ub = qd.get("d_ub")
    cur_d = cur_qd.get("d_ub")
    if d_ub is None:
        return False
    if cur_d is None:
        return True
    if int(d_ub) > int(cur_d):
        return True
    if int(d_ub) == int(cur_d):
        if int(qd.get("trials_requested", 0)) < int(cur_qd.get("trials_requested", 0)):
            return True
    return False


def _recompute_best_by_k(
    summary_records: List[Dict[str, object]],
) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    best: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for entry in summary_records:
        k = int(entry.get("k", 0))
        if k <= 0:
            continue
        qd = entry.get("qdistrnd")
        if not qd or qd.get("d_ub") is None:
            continue
        key = _best_by_k_key(entry)
        current = best.get(key)
        if current is None or _is_better_entry(entry, current):
            best[key] = entry
    return best


def _best_by_k_row(entry: Dict[str, object], *, uniq_target: int) -> Dict[str, object]:
    qd = entry.get("qdistrnd") or {}
    dx_ub = int(qd.get("dx_ub", 0))
    dz_ub = int(qd.get("dz_ub", 0))
    limiting_side = "X" if dx_ub <= dz_ub else "Z"
    qd_x = qd.get("qd_x") or {}
    qd_z = qd.get("qd_z") or {}
    uniq_lim = int(qd_x.get("uniq", 0)) if limiting_side == "X" else int(qd_z.get("uniq", 0))
    avg_lim = qd_x.get("avg") if limiting_side == "X" else qd_z.get("avg")
    trials_used = entry.get("confirm_trials") or qd.get("trials_requested")
    confirmed = uniq_lim >= uniq_target
    return {
        "k": int(entry.get("k", 0)),
        "d_ub": qd.get("d_ub"),
        "trials_used": trials_used,
        "dx_ub": dx_ub,
        "dz_ub": dz_ub,
        "uniq_lim": uniq_lim,
        "avg_lim": avg_lim,
        "id": entry.get("candidate_id"),
        "confirmed": confirmed,
    }


def _best_by_k_table(
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    *,
    uniq_target: int,
) -> str:
    if not best_by_k:
        return "[best_by_k] (no entries)\n"
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for key, entry in best_by_k.items():
        group_spec, n_val, _ = key
        groups.setdefault((group_spec, n_val), []).append(entry)
    lines: List[str] = []
    for (group_spec, n_val) in sorted(groups):
        lines.append(f"[best_by_k] group={group_spec} n={n_val}")
        header = (
            "k  | d_ub | trials_used | dx_ub | dz_ub | uniq_lim | avg_lim | id | confirmed"
        )
        lines.append(header)
        rows = sorted(groups[(group_spec, n_val)], key=lambda entry: int(entry.get("k", 0)))
        for entry in rows:
            row = _best_by_k_row(entry, uniq_target=uniq_target)
            avg_lim = _format_avg(row["avg_lim"])
            line = (
                f"{row['k']:>2} | {str(row['d_ub']):>4} | {str(row['trials_used']):>11} | "
                f"{row['dx_ub']:>5} | {row['dz_ub']:>5} | {row['uniq_lim']:>8} | "
                f"{avg_lim:>7} | {row['id']} | {row['confirmed']}"
            )
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_best_by_k_outputs(
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    *,
    uniq_target: int,
    outdir: Path,
) -> None:
    table = _best_by_k_table(best_by_k, uniq_target=uniq_target)
    print(table.rstrip())
    (outdir / "best_by_k.txt").write_text(table, encoding="utf-8")
    log_path = outdir / "best_by_k.log"
    timestamp = _timestamp_utc()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] best_by_k\n")
        log_file.write(table)
        if not table.endswith("\n"):
            log_file.write("\n")
    payload: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for key, entry in best_by_k.items():
        group_spec, n_val, _ = key
        row = _best_by_k_row(entry, uniq_target=uniq_target)
        payload.setdefault(group_spec, {}).setdefault(str(n_val), []).append(row)
    for group_spec in payload:
        for n_val in payload[group_spec]:
            payload[group_spec][n_val] = sorted(
                payload[group_spec][n_val], key=lambda row: row["k"]
            )
    (outdir / "best_by_k.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _global_dir() -> Path:
    global_dir = Path("runs") / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)
    return global_dir


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_better_global(new: Dict[str, object], current: Optional[Dict[str, object]]) -> bool:
    if current is None:
        return True
    new_d = new.get("d_ub")
    cur_d = current.get("d_ub")
    if new_d is None:
        return False
    if cur_d is None:
        return True
    if int(new_d) != int(cur_d):
        return int(new_d) > int(cur_d)
    if bool(new.get("confirmed")) != bool(current.get("confirmed")):
        return bool(new.get("confirmed"))
    new_trials = int(new.get("trials_used") or 0)
    cur_trials = int(current.get("trials_used") or 0)
    if new_trials != cur_trials:
        return new_trials > cur_trials
    return str(new.get("id", "")) < str(current.get("id", ""))


def _format_global_table(best_overall: Dict[str, Dict[str, Dict[str, dict]]]) -> str:
    if not best_overall:
        return "[best_overall] (no entries)\n"
    lines: List[str] = []
    for group_spec in sorted(best_overall):
        for n_str in sorted(best_overall[group_spec], key=lambda x: int(x)):
            lines.append(f"[best_overall] group={group_spec} n={n_str}")
            header = (
                "k  | d_ub | trials_used | dx_ub | dz_ub | id | confirmed | run_dir | timestamp"
            )
            lines.append(header)
            rows = best_overall[group_spec][n_str]
            for k_str in sorted(rows, key=lambda x: int(x)):
                row = rows[k_str]
                line = (
                    f"{int(row.get('k', 0)):>2} | {str(row.get('d_ub')):>4} | "
                    f"{str(row.get('trials_used')):>11} | {int(row.get('dx_ub', 0)):>5} | "
                    f"{int(row.get('dz_ub', 0)):>5} | {row.get('id')} | "
                    f"{row.get('confirmed')} | {row.get('run_dir')} | {row.get('timestamp')}"
                )
                lines.append(line)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _escape_tex(text: str) -> str:
    return text.replace("_", "\\_")


def _write_best_overall_outputs(best_overall: Dict[str, Dict[str, Dict[str, dict]]]) -> None:
    global_dir = _global_dir()
    table = _format_global_table(best_overall)
    (global_dir / "best_overall.txt").write_text(table, encoding="utf-8")
    (global_dir / "best_overall.json").write_text(
        json.dumps(best_overall, indent=2, sort_keys=True), encoding="utf-8"
    )
    tex_lines = [
        r"\begin{tabular}{llrrrrll}",
        r"\toprule",
        r"Group & $n$ & $k$ & $d_{ub}$ & trials & $d_X$ & $d_Z$ & id \\",
        r"\midrule",
    ]
    for group_spec in sorted(best_overall):
        for n_str in sorted(best_overall[group_spec], key=lambda x: int(x)):
            for k_str in sorted(best_overall[group_spec][n_str], key=lambda x: int(x)):
                row = best_overall[group_spec][n_str][k_str]
                tex_lines.append(
                    f"{_escape_tex(group_spec)} & {n_str} & {row.get('k')} & "
                    f"{row.get('d_ub')} & {row.get('trials_used')} & "
                    f"{row.get('dx_ub')} & {row.get('dz_ub')} & "
                    f"{_escape_tex(str(row.get('id')))} \\\\"
                )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (global_dir / "best_overall.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def _update_global_best(
    *,
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    uniq_target: int,
    run_dir: Path,
    timestamp: str,
) -> None:
    global_dir = _global_dir()
    best_path = global_dir / "best_overall.json"
    best_overall = _load_json(best_path)
    history_path = global_dir / "history.jsonl"
    if not isinstance(best_overall, dict):
        best_overall = {}
    for key, entry in best_by_k.items():
        group_spec, n_val, k_val = key
        row = _best_by_k_row(entry, uniq_target=uniq_target)
        if row.get("d_ub") is None:
            continue
        new_entry = {
            "group": group_spec,
            "n": int(n_val),
            "k": int(k_val),
            "d_ub": row.get("d_ub"),
            "trials_used": row.get("trials_used"),
            "id": row.get("id"),
            "run_dir": str(run_dir),
            "timestamp": timestamp,
            "confirmed": row.get("confirmed"),
            "dx_ub": row.get("dx_ub"),
            "dz_ub": row.get("dz_ub"),
        }
        best_overall.setdefault(group_spec, {}).setdefault(str(n_val), {})
        current = best_overall[group_spec][str(n_val)].get(str(k_val))
        if _is_better_global(new_entry, current):
            best_overall[group_spec][str(n_val)][str(k_val)] = new_entry
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(new_entry, sort_keys=True) + "\n")
    _write_best_overall_outputs(best_overall)


def _coverage_ratios(entry: Dict[str, object]) -> Dict[str, Optional[float]]:
    q_possible_ge = int(entry.get("Q_possible_ge_sqrt", 0))
    q_possible_sel = int(entry.get("Q_possible_selected", 0))
    q_meas = int(entry.get("Q_meas_run", 0))
    ratio_ge = None if q_possible_ge == 0 else q_meas / q_possible_ge
    ratio_sel = None if q_possible_sel == 0 else q_meas / q_possible_sel
    return {"Q_meas_over_possible_ge_sqrt": ratio_ge, "Q_meas_over_possible_selected": ratio_sel}


def _coverage_summary(entry: Dict[str, object]) -> str:
    ratios = _coverage_ratios(entry)
    r_ge = ratios["Q_meas_over_possible_ge_sqrt"]
    r_sel = ratios["Q_meas_over_possible_selected"]
    r_ge_txt = "n/a" if r_ge is None else f"{r_ge:.3f}"
    r_sel_txt = "n/a" if r_sel is None else f"{r_sel:.3f}"
    return (
        f"[coverage] group={entry.get('group')} "
        f"A_sel={entry.get('A_candidates_selected')} "
        f"B_sel={entry.get('B_candidates_selected')} "
        f"Q_meas={entry.get('Q_meas_run')} "
        f"ratio_ge_sqrt={r_ge_txt} ratio_sel={r_sel_txt}"
    )


def _write_coverage_files(coverage_by_group: Dict[str, Dict[str, object]], outdir: Path) -> None:
    payload: Dict[str, object] = {}
    lines: List[str] = []
    for group_spec in sorted(coverage_by_group):
        entry = dict(coverage_by_group[group_spec])
        entry["ratios"] = _coverage_ratios(entry)
        payload[group_spec] = entry
        lines.append(_coverage_summary(entry))
        lines.append(
            "  "
            f"A: multisets={entry.get('A_multisets_total')} "
            f"perm_total={entry.get('A_perm_total')} "
            f"total={entry.get('A_candidates_total')} "
            f"ge_sqrt={entry.get('A_candidates_ge_sqrt')} "
            f"selected={entry.get('A_candidates_selected')}"
        )
        lines.append(
            "  "
            f"B: multisets={entry.get('B_multisets_total')} "
            f"perm_total={entry.get('B_perm_total')} "
            f"total={entry.get('B_candidates_total')} "
            f"ge_sqrt={entry.get('B_candidates_ge_sqrt')} "
            f"selected={entry.get('B_candidates_selected')}"
        )
        lines.append(
            "  "
            f"Q: possible_ge_sqrt={entry.get('Q_possible_ge_sqrt')} "
            f"possible_selected={entry.get('Q_possible_selected')} "
            f"pairs_iterated={entry.get('Q_pairs_iterated')} "
            f"k_pos={entry.get('Q_k_pos')} "
            f"filter_run={entry.get('Q_filter_run')} "
            f"filter_pass={entry.get('Q_filter_pass')} "
            f"meas_run={entry.get('Q_meas_run')} "
            f"confirm_run={entry.get('Q_confirm_run')}"
        )
        lines.append("")
    (outdir / "coverage.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (outdir / "coverage.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_meta(entry: Dict[str, object]) -> Dict[str, object]:
    qd = entry.get("qdistrnd") or {}
    meta = {
        "group": entry.get("group"),
        "A": entry.get("A_repr") or entry.get("A"),
        "A_ids": entry.get("A"),
        "B": entry.get("B_repr") or entry.get("B"),
        "B_ids": entry.get("B"),
        "local_codes": entry.get("local_codes"),
        "n": entry.get("n"),
        "k": entry.get("k"),
        "distance_estimate": {
            "method": "QDistRnd",
            "trials_requested": qd.get("trials_requested"),
            "mindist": 0,
            "rng_seed": qd.get("seed"),
            "dx_ub": qd.get("dx_ub"),
            "dz_ub": qd.get("dz_ub"),
            "d_ub": qd.get("d_ub"),
            "qd_x": qd.get("qd_x"),
            "qd_z": qd.get("qd_z"),
            "confirmed": entry.get("confirmed", False),
            "confirm_side": entry.get("confirm_side"),
            "confirm_trials": entry.get("confirm_trials"),
        },
        "classical_slices": entry.get("classical_slices"),
        "base": entry.get("base"),
        "seed": entry.get("seed"),
    }
    return meta


def _save_best_code(
    entry: Dict[str, object],
    *,
    best_root: Path,
    tmp_root: Path,
    promising_root: Path,
) -> None:
    group_spec = (entry.get("group") or {}).get("spec") or "unknown"
    n_val = int(entry.get("n", 0))
    k_val = int(entry.get("k", 0))
    dest = best_root / group_spec / f"n{n_val}" / f"k{k_val}" / str(entry.get("candidate_id"))
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = tmp_root / str(entry.get("candidate_id"))
    hx_path = src_dir / "Hx.mtx"
    hz_path = src_dir / "Hz.mtx"
    if not hx_path.exists() or not hz_path.exists():
        saved_path = entry.get("saved_path")
        if saved_path:
            src_dir = Path(saved_path)
            hx_path = src_dir / "Hx.mtx"
            hz_path = src_dir / "Hz.mtx"
        else:
            alt_dir = promising_root / str(entry.get("candidate_id"))
            hx_path = alt_dir / "Hx.mtx"
            hz_path = alt_dir / "Hz.mtx"
    if not hx_path.exists() or not hz_path.exists():
        return
    shutil.copy2(hx_path, dest / "Hx.mtx")
    shutil.copy2(hz_path, dest / "Hz.mtx")
    (dest / "meta.json").write_text(
        json.dumps(_build_meta(entry), indent=2, sort_keys=True), encoding="utf-8"
    )
    entry["best_saved_path"] = str(dest)


def _is_new_best_by_k(
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    entry: Dict[str, object],
) -> bool:
    k = int(entry.get("k", 0))
    if k <= 0:
        return False
    qd = entry.get("qdistrnd") or {}
    if qd.get("d_ub") is None:
        return False
    key = _best_by_k_key(entry)
    current = best_by_k.get(key)
    if current is None:
        return True
    return _is_better_entry(entry, current)


def _confirm_best_entries(
    entries: List[Dict[str, object]],
    *,
    args: argparse.Namespace,
    seed: int,
    batch_id: int,
    outdir: Path,
    trials_measure: int,
    best_confirm_start: int,
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    coverage_by_group: Dict[str, Dict[str, object]],
    summary_records: List[Dict[str, object]],
) -> int:
    if not entries:
        return batch_id
    pending: List[Dict[str, object]] = []
    for entry in entries:
        qd = entry.get("qdistrnd") or {}
        qd_x = qd.get("qd_x") or {}
        qd_z = qd.get("qd_z") or {}
        dx_ub = int(qd.get("dx_ub", 0))
        dz_ub = int(qd.get("dz_ub", 0))
        side = _limiting_side(dx_ub, dz_ub, int(qd_x.get("uniq", 0)), int(qd_z.get("uniq", 0)))
        entry["confirm_side"] = side
        uniq = int(qd_x.get("uniq", 0)) if side == "x" else int(qd_z.get("uniq", 0))
        if uniq >= args.best_uniq_target:
            entry["confirmed"] = True
            entry["confirm_trials"] = qd.get("trials_requested")
        else:
            entry["confirmed"] = False
            pending.append(entry)
    if not pending:
        return batch_id
    num = max(best_confirm_start, trials_measure)
    if num == trials_measure:
        num = _next_confirm_num(num, args.best_confirm_mult)
    while pending and num <= args.best_confirm_max:
        confirm_seed = (seed + batch_id) % (1 << 31)
        for entry in pending:
            group_spec = (entry.get("group") or {}).get("spec") or "unknown"
            coverage_by_group[group_spec]["Q_confirm_run"] += 1
        batch = []
        for idx, entry in enumerate(pending):
            cand_dir = Path(entry["saved_path"]) if entry.get("saved_path") else None
            hx_path = None
            hz_path = None
            if cand_dir is None:
                cand_dir = outdir / "tmp" / entry["candidate_id"]
            hx_path = cand_dir / "Hx.mtx"
            hz_path = cand_dir / "Hz.mtx"
            batch.append((idx, str(hx_path), str(hz_path)))
        qd_results, runtime_sec = _run_gap_batch(
            batch=batch,
            num=num,
            mindist=0,
            debug=args.qd_debug,
            seed=confirm_seed,
            outdir=outdir,
            batch_id=batch_id,
            gap_cmd=args.gap_cmd,
            timeout_sec=args.qd_timeout,
        )
        batch_id += 1
        still_pending: List[Dict[str, object]] = []
        for idx, entry in enumerate(pending):
            qd_stats = qd_results.get(idx)
            if not qd_stats or qd_stats.get("qd_failed"):
                entry["confirmed"] = False
                entry["confirm_trials"] = num
                entry["confirm_failed"] = True
                continue
            qd, qd_x, qd_z, dx_ub, dz_ub, _ = _qd_summary_from_stats(
                qd_stats,
                num=num,
                mindist=0,
                seed=confirm_seed,
                runtime_sec=runtime_sec,
                gap_cmd=args.gap_cmd,
            )
            entry["qdistrnd"] = qd
            side = _limiting_side(dx_ub, dz_ub, int(qd_x.get("uniq", 0)), int(qd_z.get("uniq", 0)))
            entry["confirm_side"] = side
            uniq = int(qd_x.get("uniq", 0)) if side == "x" else int(qd_z.get("uniq", 0))
            if uniq >= args.best_uniq_target:
                entry["confirmed"] = True
                entry["confirm_trials"] = num
            else:
                entry["confirmed"] = False
                still_pending.append(entry)
        pending = still_pending
        num = _next_confirm_num(num, args.best_confirm_mult)
        best_by_k.clear()
        best_by_k.update(_recompute_best_by_k(summary_records))
        _write_best_by_k_outputs(best_by_k, uniq_target=args.best_uniq_target, outdir=outdir)
    for entry in pending:
        entry["confirmed"] = False
        entry["confirm_trials"] = min(num, args.best_confirm_max)
    return batch_id


def _gap_string_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _gap_read_mmgf2_lines() -> list[str]:
    return [
        "ReadMMGF2 := function(path)",
        "  local stream, line, parts, rows, cols, nnz, idx, M, i, j, val;",
        "  stream := InputTextFile(path);",
        "  if stream = fail then",
        '    Print("QDISTERROR ReadMMGF2OpenFailed ", path, "\\n");',
        "    QuitGap(3);",
        "  fi;",
        "  line := ReadLine(stream);",
        "  while true do",
        "    line := ReadLine(stream);",
        "    if line = fail then",
        "      CloseStream(stream);",
        '      Print("QDISTERROR ReadMMGF2MissingSize ", path, "\\n");',
        "      QuitGap(3);",
        "    fi;",
        '    line := Filtered(line, c -> not c in "\\r\\n");',
        "    idx := PositionProperty(line, c -> not (c = ' ' or c = '\\t'));",
        "    if idx = fail then",
        "      continue;",
        "    fi;",
        "    if line[idx] = '%' then",
        "      continue;",
        "    fi;",
        "    break;",
        "  od;",
        '  parts := Filtered(SplitString(line, " \\t"), x -> x <> "");',
        "  if Length(parts) < 3 then",
        "    CloseStream(stream);",
        '    Print("QDISTERROR ReadMMGF2BadSize ", path, "\\n");',
        "    QuitGap(3);",
        "  fi;",
        "  rows := Int(parts[1]);",
        "  cols := Int(parts[2]);",
        "  nnz := Int(parts[3]);",
        "  M := NullMat(rows, cols, GF(2));",
        "  while true do",
        "    line := ReadLine(stream);",
        "    if line = fail then",
        "      break;",
        "    fi;",
        '    line := Filtered(line, c -> not c in "\\r\\n");',
        "    idx := PositionProperty(line, c -> not (c = ' ' or c = '\\t'));",
        "    if idx = fail then",
        "      continue;",
        "    fi;",
        "    if line[idx] = '%' then",
        "      continue;",
        "    fi;",
        '    parts := Filtered(SplitString(line, " \\t"), x -> x <> "");',
        "    if Length(parts) < 2 then",
        "      continue;",
        "    fi;",
        "    i := Int(parts[1]);",
        "    j := Int(parts[2]);",
        "    if Length(parts) >= 3 then",
        "      val := Int(parts[3]);",
        "    else",
        "      val := 1;",
        "    fi;",
        "    if (val mod 2) = 1 then",
        "      M[i][j] := M[i][j] + One(GF(2));",
        "    fi;",
        "  od;",
        "  CloseStream(stream);",
        "  return M;",
        "end;",
    ]


def _gap_dist_rand_css_stats_lines() -> list[str]:
    return [
        "DistRandCSS_stats := function(GX, GZ, num, mindist)",
        "  local DistBound, i, j, dimsWZ, rowsWZ, colsWZ, WZ, WX,",
        "        TempVec, TempWeight, per, WZ1, WZ2, VecCount,",
        "        CodeWords, mult, pos, early_stop, rounds_done,",
        "        d_signed, uniq, avg, s1, s2, x2;",
        "  WZ := NullspaceMat(TransposedMatMutable(GX));",
        "  WX := NullspaceMat(TransposedMatMutable(GZ));",
        "  dimsWZ := DimensionsMat(WZ);",
        "  rowsWZ := dimsWZ[1];",
        "  colsWZ := dimsWZ[2];",
        "  DistBound := colsWZ + 1;",
        "  VecCount := 0;",
        "  CodeWords := [];",
        "  mult := [];",
        "  early_stop := false;",
        "  rounds_done := 0;",
        "  for i in [1..num] do",
        "    rounds_done := i;",
        "    if colsWZ > 1 then",
        "      per := Random(SymmetricGroup(colsWZ));",
        "    else",
        "      per := ();",
        "    fi;",
        "    WZ1 := PermutedCols(WZ, per);",
        "    WZ2 := TriangulizedMat(WZ1);",
        "    WZ2 := PermutedCols(WZ2, Inverse(per));",
        "    for j in [1..rowsWZ] do",
        "      TempVec := WZ2[j];",
        "      TempWeight := WeightVecFFE(TempVec);",
        "      if (TempWeight > 0) and (TempWeight <= DistBound) then",
        "        if WeightVecFFE(WX * TempVec) > 0 then",
        "          if TempWeight < DistBound then",
        "            DistBound := TempWeight;",
        "            VecCount := 1;",
        "            CodeWords := [TempVec];",
        "            mult := [1];",
        "          elif TempWeight = DistBound then",
        "            VecCount := VecCount + 1;",
        "            pos := Position(CodeWords, TempVec);",
        "            if (pos = fail) and (Length(mult) < 100) then",
        "              Add(CodeWords, TempVec);",
        "              Add(mult, 1);",
        "            elif (pos <> fail) then",
        "              mult[pos] := mult[pos] + 1;",
        "            fi;",
        "          fi;",
        "        fi;",
        "      fi;",
        "      if DistBound <= mindist then",
        "        early_stop := true;",
        "        break;",
        "      fi;",
        "    od;",
        "    if early_stop then",
        "      break;",
        "    fi;",
        "  od;",
        "  if early_stop then",
        "    d_signed := -DistBound;",
        "  else",
        "    d_signed := DistBound;",
        "  fi;",
        "  uniq := Length(mult);",
        "  if uniq > 0 then",
        "    avg := QDR_AverageCalc(mult);",
        "  else",
        "    avg := fail;",
        "  fi;",
        "  if uniq > 1 then",
        "    s1 := Sum(mult);",
        "    s2 := Sum(mult, x -> x^2);",
        "    x2 := Float(s2 * uniq - s1^2) / s1;",
        "  else",
        "    x2 := fail;",
        "  fi;",
        "  return rec(",
        "    d_signed := d_signed,",
        "    rounds_done := rounds_done,",
        "    vec_count_total := VecCount,",
        "    mult := mult,",
        "    uniq := uniq,",
        "    avg := avg,",
        "    x2 := x2",
        "  );",
        "end;",
        "ToIntOrZero := function(x)",
        "  if x = fail then",
        "    return 0;",
        "  fi;",
        "  return x;",
        "end;",
        "ListToJson := function(list)",
        "  if list = fail then",
        '    return "[]";',
        "  fi;",
        "  if Length(list) = 0 then",
        '    return "[]";',
        "  fi;",
        '  return Concatenation("[", JoinStringsWithSeparator(List(list, String), ","), "]");',
        "end;",
    ]


def _build_gap_batch_script(
    batch: Sequence[Tuple[int, str, str]],
    *,
    num: int,
    mindist: int,
    debug: int,
    seed: Optional[int],
) -> str:
    lines = [
        "OnBreak := function()",
        '  Print("QDISTERROR GAPBreak\\n");',
        "  QuitGap(3);",
        "end;",
        'if LoadPackage("QDistRnd") = fail then',
        '  Print("QDISTERROR QDistRndNotLoaded\\n");',
        "  QuitGap(2);",
        "fi;",
        *_gap_read_mmgf2_lines(),
        *_gap_dist_rand_css_stats_lines(),
    ]
    if seed is not None:
        lines.append(f"Reset(GlobalMersenneTwister, {seed});")
    for idx, hx_path, hz_path in batch:
        hx_literal = _gap_string_literal(os.path.abspath(hx_path))
        hz_literal = _gap_string_literal(os.path.abspath(hz_path))
        lines.extend(
            [
                f'HX := ReadMMGF2("{hx_literal}");;',
                f'HZ := ReadMMGF2("{hz_literal}");;',
                f"dz_rec := DistRandCSS_stats(HX, HZ, {num}, {mindist});;",
                f"dx_rec := DistRandCSS_stats(HZ, HX, {num}, {mindist});;",
                (
                    f'Print("QDR|", {idx}, "|dx=", dx_rec.d_signed, "|dz=", dz_rec.d_signed, '
                    '"|rx=", dx_rec.rounds_done, "|rz=", dz_rec.rounds_done, '
                    '"|vx=", ToIntOrZero(dx_rec.vec_count_total), '
                    '"|vz=", ToIntOrZero(dz_rec.vec_count_total), '
                    '"|mx=", ListToJson(dx_rec.mult), "|mz=", ListToJson(dz_rec.mult), "\\n");'
                ),
            ]
        )
    lines.append("QuitGap(0);")
    return "\n".join(lines) + "\n"


def _extract_qdr_index(line: str) -> Optional[int]:
    parts = line.strip().split("|")
    if len(parts) < 2 or parts[0] != "QDR":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _append_qdr_parse_errors(
    log_path: Path,
    *,
    errors: Sequence[Tuple[str, Exception]],
    stdout: str,
    stderr: str,
) -> None:
    if not errors:
        return
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n[parse_errors]\n")
            for line, exc in errors:
                log_file.write(f"{line}\n")
                log_file.write(f"error {exc}\n")
            log_file.write("\n[stdout_full]\n")
            log_file.write(stdout)
            if stdout and not stdout.endswith("\n"):
                log_file.write("\n")
            log_file.write("[stderr_full]\n")
            log_file.write(stderr)
            if stderr and not stderr.endswith("\n"):
                log_file.write("\n")
    except OSError:
        pass


def _parse_qdr_line(line: str) -> Tuple[int, Dict[str, object]]:
    parts = [part for part in line.strip().split("|") if part != ""]
    if len(parts) < 3 or parts[0] != "QDR":
        raise ValueError(f"Invalid QDR line: {line}")
    try:
        idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid QDR index in line: {line}") from exc
    fields: Dict[str, str] = {}
    for part in parts[2:]:
        key, sep, value = part.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid QDR field '{part}' in line: {line}")
        fields[key] = value.strip()
    required = ("dx", "dz", "rx", "rz")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing QDR fields {missing} in line: {line}")
    vx_raw = fields.get("vx", "")
    vz_raw = fields.get("vz", "")
    mx_raw = fields.get("mx", "")
    mz_raw = fields.get("mz", "")
    return idx, {
        "dx_signed": int(fields["dx"]),
        "dz_signed": int(fields["dz"]),
        "rx": int(fields["rx"]),
        "rz": int(fields["rz"]),
        "vx": 0 if vx_raw == "" else int(vx_raw),
        "vz": 0 if vz_raw == "" else int(vz_raw),
        "mx": [] if mx_raw == "" else json.loads(mx_raw),
        "mz": [] if mz_raw == "" else json.loads(mz_raw),
    }


def _parse_batch_output(
    *,
    stdout: str,
    stderr: str,
    expected: Iterable[int],
    log_path: Path,
) -> Dict[int, Dict[str, object]]:
    errors = []
    for line in stdout.splitlines():
        if "QDISTERROR" in line or "QTANNER_GAP_ERROR" in line:
            errors.append(line.strip())
    for line in stderr.splitlines():
        if (
            "QDISTERROR" in line
            or "QTANNER_GAP_ERROR" in line
            or line.startswith("Error,")
            or line.startswith("Syntax error")
        ):
            errors.append(line.strip())
    results: Dict[int, Dict[str, object]] = {}
    parse_errors: List[Tuple[str, Exception]] = []
    for line in stdout.splitlines():
        if not line.startswith("QDR|"):
            continue
        try:
            idx, parsed = _parse_qdr_line(line)
        except Exception as exc:
            parse_errors.append((line, exc))
            idx = _extract_qdr_index(line)
            if idx is not None:
                results[idx] = {"qd_failed": True}
            continue
        results[idx] = parsed
    if parse_errors:
        _append_qdr_parse_errors(
            log_path, errors=parse_errors, stdout=stdout, stderr=stderr
        )
    missing = [idx for idx in expected if idx not in results]
    if errors:
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n[gap_errors]\n")
                for line in errors:
                    log_file.write(f"{line}\n")
        except OSError:
            pass
    for idx in missing:
        results[idx] = {"qd_failed": True}
    return results


def _run_gap_batch(
    *,
    batch: Sequence[Tuple[int, str, str]],
    num: int,
    mindist: int,
    debug: int,
    seed: Optional[int],
    outdir: Path,
    batch_id: int,
    gap_cmd: str = "gap",
    timeout_sec: Optional[float] = None,
) -> Tuple[Dict[int, Dict[str, object]], float]:
    script = _build_gap_batch_script(batch, num=num, mindist=mindist, debug=debug, seed=seed)
    script_path = outdir / f"qdistrnd_batch_{batch_id:04d}.g"
    log_path = outdir / f"qdistrnd_batch_{batch_id:04d}.log"
    script_path.write_text(script, encoding="utf-8")
    cmd = [gap_cmd, "-q", "-b", "--quitonbreak", str(script_path)]
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        runtime_sec = time.monotonic() - start
        write_qdistrnd_log(
            str(log_path),
            cmd=cmd,
            script=script,
            script_path=str(script_path),
            stdout="",
            stderr="",
            returncode=-1,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
    except subprocess.TimeoutExpired as exc:
        runtime_sec = time.monotonic() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        write_qdistrnd_log(
            str(log_path),
            cmd=cmd,
            script=script,
            script_path=str(script_path),
            stdout=stdout,
            stderr=stderr,
            returncode=-1,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(
            f"GAP timed out after {timeout_sec} seconds (runtime {runtime_sec:.2f}s)."
        ) from exc
    runtime_sec = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    write_qdistrnd_log(
        str(log_path),
        cmd=cmd,
        script=script,
        script_path=str(script_path),
        stdout=stdout,
        stderr=stderr,
        returncode=result.returncode,
        runtime_sec=runtime_sec,
        include_tail=result.returncode != 0,
    )
    expected = [idx for idx, _, _ in batch]
    if result.returncode != 0:
        results = {idx: {"qd_failed": True} for idx in expected}
        return results, runtime_sec
    results = _parse_batch_output(
        stdout=stdout, stderr=stderr, expected=expected, log_path=log_path
    )
    return results, runtime_sec


def _process_qdistrnd_batch(
    *,
    batch_items: List[Tuple[int, str, str, dict]],
    args: argparse.Namespace,
    seed: int,
    batch_id: int,
    outdir: Path,
    tmp_root: Path,
    promising_root: Path,
    best_codes_root: Path,
    trials_filter: int,
    trials_measure: int,
    best_confirm_start: int,
    results_file,
    summary_records: List[Dict[str, object]],
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]],
    coverage_by_group: Dict[str, Dict[str, object]],
) -> int:
    if not batch_items:
        return batch_id
    filter_seed = (seed + batch_id) % (1 << 31)
    qd_results, _ = _run_gap_batch(
        batch=[(i, hx, hz) for i, hx, hz, _ in batch_items],
        num=trials_filter,
        mindist=args.mindist,
        debug=args.qd_debug,
        seed=filter_seed,
        outdir=outdir,
        batch_id=batch_id,
        gap_cmd=args.gap_cmd,
        timeout_sec=args.qd_timeout,
    )
    batch_id += 1
    for _, _, _, entry in batch_items:
        group_spec = (entry.get("group") or {}).get("spec") or "unknown"
        coverage_by_group[group_spec]["Q_filter_run"] += 1
    _write_best_by_k_outputs(best_by_k, uniq_target=args.best_uniq_target, outdir=outdir)
    survivors: List[Tuple[int, str, str, dict]] = []
    for idx_key, hx, hz, entry in batch_items:
        qd_stats = qd_results.get(idx_key)
        if not qd_stats or qd_stats.get("qd_failed"):
            entry["qd_failed"] = True
            entry["failed_filter"] = True
            entry["filter_hit_dx"] = None
            entry["filter_hit_dz"] = None
            entry["filter_early_stop_dx"] = None
            entry["filter_early_stop_dz"] = None
            entry["promising"] = False
            entry["saved"] = False
            entry["save_reason"] = None
            entry["saved_path"] = None
            print(
                f"[pilot] QD_FAILED filter "
                f"id={entry['candidate_id']}"
            )
            cand_dir = tmp_root / entry["candidate_id"]
            hx_path = cand_dir / "Hx.mtx"
            hz_path = cand_dir / "Hz.mtx"
            if hx_path.exists():
                hx_path.unlink()
            if hz_path.exists():
                hz_path.unlink()
            if cand_dir.exists():
                try:
                    cand_dir.rmdir()
                except OSError:
                    pass
            results_file.write(json.dumps(entry, sort_keys=True) + "\n")
            results_file.flush()
            continue
        dx_signed = int(qd_stats["dx_signed"])
        dz_signed = int(qd_stats["dz_signed"])
        filter_early_stop_dx = dx_signed < 0
        filter_early_stop_dz = dz_signed < 0
        filter_hit_dx = abs(dx_signed)
        filter_hit_dz = abs(dz_signed)
        entry["filter_hit_dx"] = filter_hit_dx
        entry["filter_hit_dz"] = filter_hit_dz
        entry["filter_early_stop_dx"] = filter_early_stop_dx
        entry["filter_early_stop_dz"] = filter_early_stop_dz
        entry["failed_filter"] = filter_early_stop_dx or filter_early_stop_dz
        if entry["failed_filter"]:
            entry["promising"] = False
            entry["saved"] = False
            entry["save_reason"] = None
            entry["saved_path"] = None
            print(
                f"[pilot] FILTER_FAIL hit<=mindist "
                f"(dx_hit={filter_hit_dx}, dz_hit={filter_hit_dz}) "
                f"id={entry['candidate_id']}"
            )
            cand_dir = tmp_root / entry["candidate_id"]
            hx_path = cand_dir / "Hx.mtx"
            hz_path = cand_dir / "Hz.mtx"
            if hx_path.exists():
                hx_path.unlink()
            if hz_path.exists():
                hz_path.unlink()
            if cand_dir.exists():
                try:
                    cand_dir.rmdir()
                except OSError:
                    pass
            results_file.write(json.dumps(entry, sort_keys=True) + "\n")
            results_file.flush()
            continue
        survivors.append((idx_key, hx, hz, entry))
        group_spec = (entry.get("group") or {}).get("spec") or "unknown"
        coverage_by_group[group_spec]["Q_filter_pass"] += 1

    if not survivors:
        _write_best_by_k_outputs(best_by_k, uniq_target=args.best_uniq_target, outdir=outdir)
        return batch_id

    measure_seed = (seed + batch_id) % (1 << 31)
    qd_results, runtime_sec = _run_gap_batch(
        batch=[(i, hx, hz) for i, hx, hz, _ in survivors],
        num=trials_measure,
        mindist=0,
        debug=args.qd_debug,
        seed=measure_seed,
        outdir=outdir,
        batch_id=batch_id,
        gap_cmd=args.gap_cmd,
        timeout_sec=args.qd_timeout,
    )
    batch_id += 1
    for _, _, _, entry in survivors:
        group_spec = (entry.get("group") or {}).get("spec") or "unknown"
        coverage_by_group[group_spec]["Q_meas_run"] += 1
    measured_entries: List[Dict[str, object]] = []
    new_best_entries: List[Dict[str, object]] = []
    for idx_key, _, _, entry in survivors:
        qd_stats = qd_results.get(idx_key)
        cand_dir = tmp_root / entry["candidate_id"]
        hx_path = cand_dir / "Hx.mtx"
        hz_path = cand_dir / "Hz.mtx"
        if not qd_stats or qd_stats.get("qd_failed"):
            entry["qd_failed"] = True
            entry["qdistrnd"] = {
                "trials_requested": trials_measure,
                "mindist": 0,
                "seed": measure_seed,
                "runtime_sec": runtime_sec,
                "gap_cmd": args.gap_cmd,
                "qd_failed": True,
            }
            entry["promising"] = False
            entry["saved"] = False
            entry["save_reason"] = None
            entry["saved_path"] = None
            print(
                f"[pilot] QD_FAILED meas "
                f"id={entry['candidate_id']}"
            )
            if hx_path.exists():
                hx_path.unlink()
            if hz_path.exists():
                hz_path.unlink()
            if cand_dir.exists():
                try:
                    cand_dir.rmdir()
                except OSError:
                    pass
            results_file.write(json.dumps(entry, sort_keys=True) + "\n")
            results_file.flush()
            summary_records.append(entry)
            continue
        qd, qd_x, qd_z, dx_ub, dz_ub, d_ub = _qd_summary_from_stats(
            qd_stats,
            num=trials_measure,
            mindist=0,
            seed=measure_seed,
            runtime_sec=runtime_sec,
            gap_cmd=args.gap_cmd,
        )
        entry["qdistrnd"] = qd
        entry["confirmed"] = False
        entry["confirm_side"] = _limiting_side(
            dx_ub, dz_ub, int(qd_x.get("uniq", 0)), int(qd_z.get("uniq", 0))
        )
        entry["confirm_trials"] = qd.get("trials_requested")
        measured_entries.append(entry)
        summary_records.append(entry)
        if _is_new_best_by_k(best_by_k, entry):
            new_best_entries.append(entry)

        n = entry["n"]
        k = entry["k"]
        print(
            f"[pilot] MEAS n={n} k={k} dx_ub={dx_ub} "
            f"dz_ub={dz_ub} d_ub={d_ub} id={entry['candidate_id']}"
        )

    if new_best_entries:
        batch_id = _confirm_best_entries(
            new_best_entries,
            args=args,
            seed=seed,
            batch_id=batch_id,
            outdir=outdir,
            trials_measure=trials_measure,
            best_confirm_start=best_confirm_start,
            best_by_k=best_by_k,
            coverage_by_group=coverage_by_group,
            summary_records=summary_records,
        )

    if measured_entries:
        old_best = dict(best_by_k)
        best_by_k.clear()
        best_by_k.update(_recompute_best_by_k(summary_records))
        for key, entry in best_by_k.items():
            previous = old_best.get(key)
            if previous is None or previous.get("candidate_id") != entry.get("candidate_id"):
                if entry in measured_entries:
                    _save_best_code(
                        entry,
                        best_root=best_codes_root,
                        tmp_root=tmp_root,
                        promising_root=promising_root,
                    )
        _write_best_by_k_outputs(best_by_k, uniq_target=args.best_uniq_target, outdir=outdir)

    for entry in measured_entries:
        cand_dir = tmp_root / entry["candidate_id"]
        hx_path = cand_dir / "Hx.mtx"
        hz_path = cand_dir / "Hz.mtx"
        qd = entry.get("qdistrnd") or {}
        d_ub = qd.get("d_ub")
        n = entry["n"]
        k = entry["k"]
        save = False
        save_reason = None
        if d_ub is not None and k > 0:
            target = _ceil_sqrt(n)
            meets_d = int(d_ub) >= target
            meets_kd = k * int(d_ub) > n
            save = meets_d or meets_kd
            if save:
                save_reason = "d>=sqrt(n)" if meets_d else "k*d>n"
        entry["promising"] = save
        entry["saved"] = save
        entry["save_reason"] = save_reason
        entry["saved_path"] = None

        meta = _build_meta(entry)

        if save:
            out_path = promising_root / entry["candidate_id"]
            out_path.mkdir(parents=True, exist_ok=True)
            hx_path.replace(out_path / "Hx.mtx")
            hz_path.replace(out_path / "Hz.mtx")
            (out_path / "meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            entry["saved_path"] = str(out_path)
            print(
                f"[saved] id={entry['candidate_id']} -> {out_path} "
                f"(reason: {save_reason})"
            )
        else:
            if hx_path.exists():
                hx_path.unlink()
            if hz_path.exists():
                hz_path.unlink()
        if cand_dir.exists():
            try:
                cand_dir.rmdir()
            except OSError:
                pass

        results_file.write(json.dumps(entry, sort_keys=True) + "\n")
        results_file.flush()
    if not measured_entries:
        _write_best_by_k_outputs(best_by_k, uniq_target=args.best_uniq_target, outdir=outdir)
    return batch_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search for Tanner codes from left-right Cayley complexes."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma list of group specs (e.g. C4,C2xC2,SmallGroup(8,3)).",
    )
    parser.add_argument(
        "--m-list",
        default=None,
        help="Comma list of cyclic group orders (legacy; ignored if --groups is set).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=200,
        help="Skip groups with n=36*|G| above this value.",
    )
    parser.add_argument(
        "--min-slice-dist",
        type=str,
        default="sqrtN",
        help="Minimum slice score (int or 'sqrtN').",
    )
    parser.add_argument(
        "--allow-repeats",
        action="store_true",
        help="Allow repeated elements in A/B multisets.",
    )
    parser.add_argument("--maxA", type=int, default=None, help="Limit A multisets.")
    parser.add_argument("--maxB", type=int, default=None, help="Limit B multisets.")
    parser.add_argument(
        "--permH1",
        type=int,
        default=5,
        help="Deprecated (all 30 permutations are always scored).",
    )
    parser.add_argument(
        "--permH1p",
        type=int,
        default=5,
        help="Deprecated (all 30 permutations are always scored).",
    )
    parser.add_argument(
        "--topA", type=int, default=30, help="Keep top A (shorthand for --topA-d/--topA-k)."
    )
    parser.add_argument(
        "--topB", type=int, default=30, help="Keep top B (shorthand for --topB-d/--topB-k)."
    )
    parser.add_argument("--topA-d", type=int, default=None, help="Top A by d_min.")
    parser.add_argument("--topA-k", type=int, default=None, help="Top A by k_sum.")
    parser.add_argument("--topB-d", type=int, default=None, help="Top B by d_min.")
    parser.add_argument("--topB-k", type=int, default=None, help="Top B by k_sum.")
    parser.add_argument(
        "--exploreA", type=int, default=0, help="Random tail A candidates."
    )
    parser.add_argument(
        "--exploreB", type=int, default=0, help="Random tail B candidates."
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="QDistRnd measurement trials."
    )
    parser.add_argument(
        "--trials-filter",
        type=int,
        default=None,
        help="QDistRnd filter trials (default: --trials).",
    )
    parser.add_argument(
        "--mindist", type=int, default=10, help="QDistRnd filter mindist."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Candidates per GAP/QDistRnd batch.",
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for the run."
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--gap-cmd", type=str, default="gap", help="GAP command.")
    parser.add_argument(
        "--qd-timeout", type=float, default=None, help="Timeout per batch (sec)."
    )
    parser.add_argument("--qd-debug", type=int, default=2, help="QDistRnd debug.")
    parser.add_argument(
        "--best-uniq-target",
        type=int,
        default=5,
        help="Unique codewords target for confirming best-by-(n,k).",
    )
    parser.add_argument(
        "--best-confirm-start",
        type=int,
        default=None,
        help="Starting num for best confirmation (default: --trials).",
    )
    parser.add_argument(
        "--best-confirm-mult",
        type=int,
        default=10,
        help="Multiplier for best confirmation num.",
    )
    parser.add_argument(
        "--best-confirm-max",
        type=int,
        default=200000,
        help="Max num for best confirmation.",
    )
    parser.add_argument(
        "--frontier-max-per-point",
        type=int,
        default=50,
        help="Max candidates to keep per (k,d) frontier point.",
    )
    parser.add_argument(
        "--frontier-max-total",
        type=int,
        default=500,
        help="Max total candidates to keep after frontier selection.",
    )
    parser.add_argument(
        "--classical-keep",
        choices=["frontier", "above_sqrt"],
        default="frontier",
        help="Classical candidate selection mode.",
    )
    parser.add_argument(
        "--classical-cap-per-side",
        type=int,
        default=0,
        help="Cap classical candidates per side when --classical-keep=above_sqrt (0=uncapped).",
    )
    parser.add_argument(
        "--max-quantum",
        type=int,
        default=0,
        help="Max quantum pairs to sample (0 = full cartesian product).",
    )
    args = parser.parse_args()

    group_specs: List[str] = []
    if args.groups:
        group_specs = [part.strip() for part in args.groups.split(",") if part.strip()]
        if not group_specs:
            raise ValueError("--groups must contain at least one spec.")
    else:
        if args.m_list is None:
            raise ValueError("--m-list must be provided when --groups is not set.")
        m_list = _parse_int_list(args.m_list)
        if not m_list:
            raise ValueError("--m-list must contain at least one integer.")
        group_specs = [f"C{m}" for m in m_list]
    if args.max_n <= 0:
        raise ValueError("--max-n must be positive.")
    if args.min_slice_dist is None or not args.min_slice_dist.strip():
        raise ValueError("--min-slice-dist must be non-empty.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.trials_filter is not None and args.trials_filter <= 0:
        raise ValueError("--trials-filter must be positive.")
    if args.mindist < 0:
        raise ValueError("--mindist must be nonnegative.")
    for name in (
        "maxA",
        "maxB",
        "topA",
        "topB",
        "topA_d",
        "topA_k",
        "topB_d",
        "topB_k",
        "exploreA",
        "exploreB",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be nonnegative.")
    if args.best_uniq_target <= 0:
        raise ValueError("--best-uniq-target must be positive.")
    if args.best_confirm_start is not None and args.best_confirm_start <= 0:
        raise ValueError("--best-confirm-start must be positive.")
    if args.best_confirm_mult <= 0:
        raise ValueError("--best-confirm-mult must be positive.")
    if args.best_confirm_max <= 0:
        raise ValueError("--best-confirm-max must be positive.")
    if args.frontier_max_per_point <= 0:
        raise ValueError("--frontier-max-per-point must be positive.")
    if args.frontier_max_total <= 0:
        raise ValueError("--frontier-max-total must be positive.")
    if args.classical_cap_per_side < 0:
        raise ValueError("--classical-cap-per-side must be nonnegative.")
    if args.max_quantum < 0:
        raise ValueError("--max-quantum must be nonnegative.")
    if not gap_is_available(args.gap_cmd):
        raise RuntimeError(f"GAP is not available on PATH as '{args.gap_cmd}'.")
    if not qdistrnd_is_available(args.gap_cmd):
        raise RuntimeError("GAP QDistRnd package is not available.")

    trials_filter = args.trials_filter if args.trials_filter is not None else args.trials
    trials_measure = args.trials
    best_confirm_start = (
        args.best_confirm_start if args.best_confirm_start is not None else trials_measure
    )
    topA_d = args.topA_d if args.topA_d is not None else args.topA
    topA_k = args.topA_k if args.topA_k is not None else args.topA
    topB_d = args.topB_d if args.topB_d is not None else args.topB
    topB_k = args.topB_k if args.topB_k is not None else args.topB
    frontier_max_per_point = args.frontier_max_per_point
    frontier_max_total = args.frontier_max_total
    classical_keep = args.classical_keep
    classical_cap_per_side = args.classical_cap_per_side
    max_quantum = args.max_quantum

    seed = args.seed
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 31)
    rng = random.Random(seed)

    outdir = Path(args.outdir) if args.outdir else Path("runs") / f"pilot_{_timestamp_utc()}"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root = outdir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    promising_root = outdir / "promising"
    promising_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "groups": [canonical_group_spec(spec) for spec in group_specs],
        "max_n": args.max_n,
        "min_slice_dist": args.min_slice_dist,
        "allow_repeats": args.allow_repeats,
        "maxA": args.maxA,
        "maxB": args.maxB,
        "permH1": args.permH1,
        "permH1p": args.permH1p,
        "topA": args.topA,
        "topB": args.topB,
        "topA_d": topA_d,
        "topA_k": topA_k,
        "topB_d": topB_d,
        "topB_k": topB_k,
        "exploreA": args.exploreA,
        "exploreB": args.exploreB,
        "trials": trials_measure,
        "trials_filter": trials_filter,
        "mindist": args.mindist,
        "mindist_measure": 0,
        "batch_size": args.batch_size,
        "seed": seed,
        "gap_cmd": args.gap_cmd,
        "qd_timeout": args.qd_timeout,
        "qd_debug": args.qd_debug,
        "best_uniq_target": args.best_uniq_target,
        "best_confirm_start": best_confirm_start,
        "best_confirm_mult": args.best_confirm_mult,
        "best_confirm_max": args.best_confirm_max,
        "frontier_max_per_point": frontier_max_per_point,
        "frontier_max_total": frontier_max_total,
        "classical_keep": classical_keep,
        "classical_cap_per_side": classical_cap_per_side,
        "max_quantum": max_quantum,
        "slice_scoring": {"max_k_exact": 10, "samples": 1 << 10},
        "note": "slice scores use all 30 C1/C1p permutations",
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    _write_commands_run(outdir, args)

    base_code = hamming_6_3_3_shortened()
    variant_codes = _build_variant_codes(base_code)
    perm_total = len(variant_codes)
    base_k_table = _compute_base_k_table(
        base_code=base_code, variant_codes=variant_codes, outdir=outdir
    )
    best_pairs: List[Tuple[int, int, int]] = []
    for a_idx in range(perm_total):
        for b_idx in range(perm_total):
            best_pairs.append((base_k_table[a_idx][b_idx], a_idx, b_idx))
    best_pairs.sort(reverse=True)
    print("[pilot] base k_table top pairs:")
    for k_base, a_idx, b_idx in best_pairs[:5]:
        print(f"  k_base={k_base} permA={a_idx} permB={b_idx}")

    results_path = outdir / "candidates.jsonl"
    classical_a_path = outdir / "classical_A.jsonl"
    classical_b_path = outdir / "classical_B.jsonl"
    classical_a_frontier_path = outdir / "classical_A_frontier.json"
    classical_b_frontier_path = outdir / "classical_B_frontier.json"
    best_codes_root = outdir / "best_codes"
    group_cache_dir = outdir / "groups"
    summary_records: List[Dict[str, object]] = []
    best_by_k: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    coverage_by_group: Dict[str, Dict[str, object]] = {}
    candidate_counter = 0
    batch_id = 0
    a_frontier_payloads: List[dict] = []
    b_frontier_payloads: List[dict] = []
    single_group = len(group_specs) == 1

    with (
        results_path.open("w", encoding="utf-8") as results_file,
        classical_a_path.open("w", encoding="utf-8") as class_a_file,
        classical_b_path.open("w", encoding="utf-8") as class_b_file,
    ):
        for spec in group_specs:
            spec_norm = canonical_group_spec(spec)
            group = group_from_spec(
                spec_norm, gap_cmd=args.gap_cmd, cache_dir=group_cache_dir
            )
            group_tag = _safe_id_component(group.name)
            n_est = 36 * group.order
            if n_est > args.max_n:
                print(
                    f"[pilot] skipping {group.name} (n=36*|G|={n_est} exceeds --max-n={args.max_n})."
                )
                continue
            if not args.allow_repeats and group.order < 6:
                print(
                    f"[pilot] skipping {group.name} (need |G|>=6 for 5 distinct nonzero; "
                    "use --allow-repeats to permit repetitions)."
                )
                continue
            d0 = _ceil_sqrt(n_est)
            feasible_limit = 2000
            A_sets = _enumerate_sets(
                m=group.order,
                max_sets=args.maxA,
                rng=rng,
                feasible_limit=feasible_limit,
                allow_repeats=args.allow_repeats,
            )
            B_sets = _enumerate_sets(
                m=group.order,
                max_sets=args.maxB,
                rng=rng,
                feasible_limit=feasible_limit,
                allow_repeats=args.allow_repeats,
            )
            if not A_sets or not B_sets:
                print(f"[pilot] no A/B sets for {group.name}; skipping.")
                continue

            a_scored: List[SliceCandidate] = []
            for A in A_sets:
                A_repr = [group.repr(x) for x in A]
                for perm_idx, C1 in enumerate(variant_codes):
                    metrics = _score_a_slice(group, A, base_code, C1, rng=rng)
                    cand_id = _classical_candidate_id(
                        side="A",
                        group=group,
                        elements=A,
                        perm_idx=perm_idx,
                    )
                    a_scored.append(
                        SliceCandidate(elements=A, perm_idx=perm_idx, metrics=metrics)
                    )
                    record = {
                        "candidate_id": cand_id,
                        "group": _group_record(group),
                        "A": list(A),
                        "A_repr": A_repr,
                        "permA_idx": perm_idx,
                        "slice_H": _slice_stats_summary(metrics.h),
                        "slice_G": _slice_stats_summary(metrics.g),
                        "dA_min": metrics.d_min,
                        "k_min": metrics.k_min,
                        "d_min": metrics.d_min,
                        "kA_sum": metrics.k_sum,
                    }
                    class_a_file.write(json.dumps(record, sort_keys=True) + "\n")
            class_a_file.flush()

            b_scored: List[SliceCandidate] = []
            for B in B_sets:
                B_repr = [group.repr(x) for x in B]
                for perm_idx, C1p in enumerate(variant_codes):
                    metrics = _score_b_slice(group, B, base_code, C1p, rng=rng)
                    cand_id = _classical_candidate_id(
                        side="B",
                        group=group,
                        elements=B,
                        perm_idx=perm_idx,
                    )
                    b_scored.append(
                        SliceCandidate(elements=B, perm_idx=perm_idx, metrics=metrics)
                    )
                    record = {
                        "candidate_id": cand_id,
                        "group": _group_record(group),
                        "B": list(B),
                        "B_repr": B_repr,
                        "permB_idx": perm_idx,
                        "slice_Hp": _slice_stats_summary(metrics.h),
                        "slice_Gp": _slice_stats_summary(metrics.g),
                        "dB_min": metrics.d_min,
                        "k_min": metrics.k_min,
                        "d_min": metrics.d_min,
                        "kB_sum": metrics.k_sum,
                    }
                    class_b_file.write(json.dumps(record, sort_keys=True) + "\n")
            class_b_file.flush()

            A_frontier, payloadA = _frontier_select(
                a_scored,
                group=group,
                side="A",
                n_quantum=n_est,
                frontier_max_per_point=frontier_max_per_point,
                frontier_max_total=frontier_max_total,
            )
            if classical_keep == "above_sqrt":
                A_keep_base, selected_ids = _select_above_sqrt(
                    a_scored,
                    group=group,
                    side="A",
                    d0=d0,
                    cap=classical_cap_per_side,
                )
                payloadA["selected_ids"] = selected_ids
                payloadA["selection_mode"] = "above_sqrt"
                payloadA["selection_cap"] = classical_cap_per_side
                A_keep = _add_explore_candidates(
                    a_scored,
                    A_keep_base,
                    explore_n=args.exploreA,
                    rng=rng,
                    side="A",
                )
            else:
                payloadA["selection_mode"] = "frontier"
                payloadA["selection_cap"] = frontier_max_total
                A_keep = _add_explore_candidates(
                    a_scored,
                    A_frontier,
                    explore_n=args.exploreA,
                    rng=rng,
                    side="A",
                )
            payloadA["selected_ids"] = [
                _classical_candidate_id(
                    side="A",
                    group=group,
                    elements=cand.elements,
                    perm_idx=cand.perm_idx,
                )
                for cand in A_keep
            ]
            a_frontier_payloads.append(payloadA)
            _write_frontier_file(
                classical_a_frontier_path,
                a_frontier_payloads,
                single_group=single_group,
            )

            B_frontier, payloadB = _frontier_select(
                b_scored,
                group=group,
                side="B",
                n_quantum=n_est,
                frontier_max_per_point=frontier_max_per_point,
                frontier_max_total=frontier_max_total,
            )
            if classical_keep == "above_sqrt":
                B_keep_base, selected_ids = _select_above_sqrt(
                    b_scored,
                    group=group,
                    side="B",
                    d0=d0,
                    cap=classical_cap_per_side,
                )
                payloadB["selected_ids"] = selected_ids
                payloadB["selection_mode"] = "above_sqrt"
                payloadB["selection_cap"] = classical_cap_per_side
                B_keep = _add_explore_candidates(
                    b_scored,
                    B_keep_base,
                    explore_n=args.exploreB,
                    rng=rng,
                    side="B",
                )
            else:
                payloadB["selection_mode"] = "frontier"
                payloadB["selection_cap"] = frontier_max_total
                B_keep = _add_explore_candidates(
                    b_scored,
                    B_frontier,
                    explore_n=args.exploreB,
                    rng=rng,
                    side="B",
                )
            payloadB["selected_ids"] = [
                _classical_candidate_id(
                    side="B",
                    group=group,
                    elements=cand.elements,
                    perm_idx=cand.perm_idx,
                )
                for cand in B_keep
            ]
            b_frontier_payloads.append(payloadB)
            _write_frontier_file(
                classical_b_frontier_path,
                b_frontier_payloads,
                single_group=single_group,
            )

            group_spec = group.name
            coverage_by_group[group_spec] = {
                "group": group_spec,
                "order": group.order,
                "n_quantum": n_est,
                "A_multisets_total": len(A_sets),
                "A_perm_total": perm_total,
                "A_candidates_total": len(A_sets) * perm_total,
                "A_candidates_ge_sqrt": sum(
                    1 for cand in a_scored if cand.metrics.d_min >= d0
                ),
                "A_candidates_selected": len(A_keep),
                "B_multisets_total": len(B_sets),
                "B_perm_total": perm_total,
                "B_candidates_total": len(B_sets) * perm_total,
                "B_candidates_ge_sqrt": sum(
                    1 for cand in b_scored if cand.metrics.d_min >= d0
                ),
                "B_candidates_selected": len(B_keep),
                "Q_possible_ge_sqrt": 0,
                "Q_possible_selected": 0,
                "Q_pairs_iterated": 0,
                "Q_k_pos": 0,
                "Q_filter_run": 0,
                "Q_filter_pass": 0,
                "Q_meas_run": 0,
                "Q_confirm_run": 0,
            }
            coverage_by_group[group_spec]["Q_possible_ge_sqrt"] = (
                coverage_by_group[group_spec]["A_candidates_ge_sqrt"]
                * coverage_by_group[group_spec]["B_candidates_ge_sqrt"]
            )
            coverage_by_group[group_spec]["Q_possible_selected"] = (
                coverage_by_group[group_spec]["A_candidates_selected"]
                * coverage_by_group[group_spec]["B_candidates_selected"]
            )

            print(
                f"[pilot] {group.name} n={n_est} d0={d0} "
                f"A_sets={len(A_sets)}*{perm_total} -> keep {len(A_keep)}; "
                f"B_sets={len(B_sets)}*{perm_total} -> keep {len(B_keep)}"
            )

            batch_items: List[Tuple[int, str, str, dict]] = []
            total_pairs = len(A_keep) * len(B_keep)
            pair_indices = None
            if max_quantum > 0 and total_pairs > max_quantum:
                sampled = rng.sample(range(total_pairs), max_quantum)
                pair_indices = [divmod(idx, len(B_keep)) for idx in sampled]
            coverage_by_group[group_spec]["Q_pairs_iterated"] = (
                len(pair_indices) if pair_indices is not None else total_pairs
            )

            def _pair_iter():
                if pair_indices is not None:
                    for a_idx, b_idx in pair_indices:
                        yield A_keep[a_idx], B_keep[b_idx]
                else:
                    for A_info in A_keep:
                        for B_info in B_keep:
                            yield A_info, B_info

            for A_info, B_info in _pair_iter():
                permA = A_info.perm_idx
                permB = B_info.perm_idx
                candidate_id = f"{group_tag}_c{candidate_counter:05d}"
                candidate_counter += 1
                C1 = variant_codes[permA]
                C1p = variant_codes[permB]
                hx_rows, hz_rows, n_cols = build_hx_hz(
                    group,
                    A_info.elements,
                    B_info.elements,
                    base_code,
                    C1,
                    base_code,
                    C1p,
                )
                if not css_commutes(hx_rows, hz_rows):
                    raise RuntimeError(f"HX/HZ do not commute for {candidate_id}.")
                rank_hx = gf2_rank(hx_rows[:], n_cols)
                rank_hz = gf2_rank(hz_rows[:], n_cols)
                k = n_cols - rank_hx - rank_hz
                entry = {
                    "candidate_id": candidate_id,
                    "group": _group_record(group),
                    "A": list(A_info.elements),
                    "A_repr": [group.repr(x) for x in A_info.elements],
                    "B": list(B_info.elements),
                    "B_repr": [group.repr(x) for x in B_info.elements],
                    "local_codes": {
                        "C0": base_code.name,
                        "C1": C1.name,
                        "C0p": base_code.name,
                        "C1p": C1p.name,
                        "permA_idx": permA,
                        "permB_idx": permB,
                    },
                    "classical_slices": {
                        "A": {
                            "perm_idx": permA,
                            "H": _slice_stats_summary(A_info.metrics.h),
                            "G": _slice_stats_summary(A_info.metrics.g),
                            "d_min": A_info.metrics.d_min,
                            "k_sum": A_info.metrics.k_sum,
                            "k_min": A_info.metrics.k_min,
                        },
                        "B": {
                            "perm_idx": permB,
                            "Hp": _slice_stats_summary(B_info.metrics.h),
                            "Gp": _slice_stats_summary(B_info.metrics.g),
                            "d_min": B_info.metrics.d_min,
                            "k_sum": B_info.metrics.k_sum,
                            "k_min": B_info.metrics.k_min,
                        },
                    },
                    "base": {
                        "n_base": 36,
                        "k_base": base_k_table[permA][permB],
                    },
                    "n": n_cols,
                    "k": k,
                    "seed": seed,
                    "column_order": "col = ((i*nB + j)*|G| + g), g fastest",
                }
                if k == 0:
                    entry["skipped_reason"] = "k=0"
                    entry["saved"] = False
                    entry["save_reason"] = None
                    entry["saved_path"] = None
                    entry["promising"] = False
                    results_file.write(json.dumps(entry, sort_keys=True) + "\n")
                    results_file.flush()
                    continue

                coverage_by_group[group_spec]["Q_k_pos"] += 1
                cand_dir = tmp_root / candidate_id
                cand_dir.mkdir(parents=True, exist_ok=True)
                hx_path = cand_dir / "Hx.mtx"
                hz_path = cand_dir / "Hz.mtx"
                write_mtx_from_bitrows(str(hx_path), hx_rows, n_cols)
                write_mtx_from_bitrows(str(hz_path), hz_rows, n_cols)

                idx = len(batch_items)
                batch_items.append((idx, str(hx_path), str(hz_path), entry))

                if len(batch_items) >= args.batch_size:
                    batch_id = _process_qdistrnd_batch(
                        batch_items=batch_items,
                        args=args,
                        seed=seed,
                        batch_id=batch_id,
                        outdir=outdir,
                        tmp_root=tmp_root,
                        promising_root=promising_root,
                        best_codes_root=best_codes_root,
                        trials_filter=trials_filter,
                        trials_measure=trials_measure,
                        best_confirm_start=best_confirm_start,
                        results_file=results_file,
                        summary_records=summary_records,
                        best_by_k=best_by_k,
                        coverage_by_group=coverage_by_group,
                    )
                    batch_items = []

            if batch_items:
                batch_id = _process_qdistrnd_batch(
                    batch_items=batch_items,
                    args=args,
                    seed=seed,
                    batch_id=batch_id,
                    outdir=outdir,
                    tmp_root=tmp_root,
                    promising_root=promising_root,
                    best_codes_root=best_codes_root,
                    trials_filter=trials_filter,
                    trials_measure=trials_measure,
                    best_confirm_start=best_confirm_start,
                    results_file=results_file,
                    summary_records=summary_records,
                    best_by_k=best_by_k,
                    coverage_by_group=coverage_by_group,
                )

            _write_coverage_files(coverage_by_group, outdir)
            print(_coverage_summary(coverage_by_group[group_spec]))

    if coverage_by_group:
        _write_coverage_files(coverage_by_group, outdir)
        print("[coverage] run summary:")
        for group_spec in sorted(coverage_by_group):
            print(_coverage_summary(coverage_by_group[group_spec]))

    if tmp_root.exists():
        try:
            tmp_root.rmdir()
        except OSError:
            pass

    ranked = [
        rec
        for rec in summary_records
        if rec.get("k", 0) > 0
        and rec.get("qdistrnd")
        and rec["qdistrnd"].get("d_ub") is not None
    ]
    ranked.sort(
        key=lambda rec: (
            rec["qdistrnd"]["d_ub"],
            rec["k"] * rec["qdistrnd"]["d_ub"],
        ),
        reverse=True,
    )
    print("[pilot] top results:")
    for idx, rec in enumerate(ranked[:10], start=1):
        qd = rec["qdistrnd"]
        group_spec = (rec.get("group") or {}).get("spec")
        permA = (rec.get("local_codes") or {}).get("permA_idx")
        permB = (rec.get("local_codes") or {}).get("permB_idx")
        print(
            f"  {idx:02d} group={group_spec} n={rec['n']} k={rec['k']} "
            f"d_ub={qd['d_ub']} permA={permA} permB={permB} "
            f"saved={bool(rec.get('saved_path'))}"
        )

    _update_global_best(
        best_by_k=best_by_k,
        uniq_target=args.best_uniq_target,
        run_dir=outdir,
        timestamp=_timestamp_utc(),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
