"""Progressive exhaustive classical-first search mode."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .dist_m4ri import (
    dist_m4ri_is_available,
    run_dist_m4ri_classical_rw,
    run_dist_m4ri_css_rw,
)
from .gf2 import gf2_rank
from .group import FiniteGroup, canonical_group_spec, group_from_spec
from .lift_matrices import build_hx_hz, css_commutes
from .local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from .mtx import write_mtx_from_bitrows
from .qdistrnd import gap_is_available
from .slice_codes import (
    build_a_slice_checks_G,
    build_a_slice_checks_H,
    build_b_slice_checks_Gp,
    build_b_slice_checks_Hp,
)

SIDE_LEN = 6
SIDE_PICK = SIDE_LEN - 1


@dataclass(frozen=True)
class ProgressiveSetting:
    side: str
    elements: List[int]
    elements_repr: List[str]
    perm_idx: int
    k_classical: int
    d_est: int
    record: Dict[str, object]
    setting_id: str


@dataclass(frozen=True)
class TwoStageDistanceResult:
    passed: bool
    d_final_ub: Optional[int]
    d_fast_ub: Optional[int]
    d_slow_ub: Optional[int]
    sqrt_n: int
    fast: Dict[str, object]
    slow: Optional[Dict[str, object]]
    ran_slow: bool


def _timestamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_id_component(text: str) -> str:
    cleaned = []
    for ch in str(text):
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    joined = "".join(cleaned).strip("_")
    return joined or "group"


def _ceil_sqrt(n: int) -> int:
    root = math.isqrt(n)
    return root if root * root == n else root + 1


def _progressive_setting_id(side: str, elements: Sequence[int], perm_idx: int) -> str:
    elems = "-".join(str(int(x)) for x in elements)
    return f"{side}p{int(perm_idx)}_{elems}"


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


def _enumerate_multisets_with_identity(order: int) -> List[List[int]]:
    if order <= 0:
        return []
    return [[0, *comb] for comb in combinations_with_replacement(range(order), SIDE_PICK)]


def _filter_min_distinct(sets: Iterable[List[int]], min_distinct: int) -> List[List[int]]:
    if min_distinct <= 1:
        return list(sets)
    return [values for values in sets if len(set(values)) >= min_distinct]


def canonical_multiset(
    group: FiniteGroup,
    multiset: Sequence[int],
    *,
    automorphisms: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[int, ...]:
    base = tuple(sorted(int(x) for x in multiset))
    if automorphisms is None:
        automorphisms = group.automorphisms()
    best = base
    for perm in automorphisms:
        mapped = tuple(sorted(perm[x] for x in base))
        if mapped < best:
            best = mapped
    return best


def _dedup_multisets(
    group: FiniteGroup,
    sets: List[List[int]],
    *,
    automorphisms: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], int, int]:
    raw_count = len(sets)
    if raw_count == 0:
        return [], 0, 0
    deduped: List[List[int]] = []
    for mult in sets:
        key = tuple(sorted(int(x) for x in mult))
        if key == canonical_multiset(group, key, automorphisms=automorphisms):
            deduped.append(list(key))
    return deduped, raw_count, len(deduped)


def _classical_eval(
    rows: Sequence[int],
    n_cols: int,
    *,
    steps: int,
    wmin: int,
    rng: random.Random,
    dist_m4ri_cmd: str,
) -> Dict[str, object]:
    rank = gf2_rank(list(rows), n_cols)
    k_val = n_cols - rank
    seed = rng.randrange(1 << 31)
    d_signed = run_dist_m4ri_classical_rw(
        rows,
        n_cols,
        steps,
        wmin,
        seed=seed,
        dist_m4ri_cmd=dist_m4ri_cmd,
    )
    return {
        "k": k_val,
        "rank": rank,
        "d_signed": d_signed,
        "d_est": abs(d_signed),
        "early_stop": d_signed < 0,
        "seed": seed,
    }


def _estimate_css_distance_rw(
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    *,
    steps: int,
    wmin: int,
    target_distance: int,
    rng: random.Random,
    dist_m4ri_cmd: str,
    estimator: Callable[..., int] = run_dist_m4ri_css_rw,
) -> Tuple[Dict[str, object], bool]:
    seed_z = rng.randrange(1 << 31)
    dz_signed = estimator(
        hx_rows,
        hz_rows,
        n_cols,
        steps,
        wmin,
        seed=seed_z,
        dist_m4ri_cmd=dist_m4ri_cmd,
    )
    dz_ub = abs(dz_signed)
    early_stop_z = dz_signed < 0
    if early_stop_z or dz_ub < target_distance:
        return (
            {
                "steps": steps,
                "seed_z": seed_z,
                "seed_x": None,
                "dz_signed": dz_signed,
                "dx_signed": None,
                "dz_ub": dz_ub,
                "dx_ub": None,
                "d_ub": None,
                "early_stop_z": early_stop_z,
                "early_stop_x": False,
            },
            False,
        )

    seed_x = rng.randrange(1 << 31)
    dx_signed = estimator(
        hz_rows,
        hx_rows,
        n_cols,
        steps,
        wmin,
        seed=seed_x,
        dist_m4ri_cmd=dist_m4ri_cmd,
    )
    dx_ub = abs(dx_signed)
    early_stop_x = dx_signed < 0
    d_ub = min(dz_ub, dx_ub)
    passed = (not early_stop_x) and d_ub >= target_distance
    return (
        {
            "steps": steps,
            "seed_z": seed_z,
            "seed_x": seed_x,
            "dz_signed": dz_signed,
            "dx_signed": dx_signed,
            "dz_ub": dz_ub,
            "dx_ub": dx_ub,
            "d_ub": d_ub,
            "early_stop_z": early_stop_z,
            "early_stop_x": early_stop_x,
        },
        passed,
    )


def _two_stage_css_distance(
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    *,
    target_distance: int,
    wmin: int,
    steps_fast: int,
    steps_slow: int,
    current_best: float,
    rng: random.Random,
    dist_m4ri_cmd: str,
    estimator: Callable[..., int] = run_dist_m4ri_css_rw,
    on_refine: Optional[Callable[[int, int, float], None]] = None,
) -> TwoStageDistanceResult:
    sqrt_n = math.isqrt(n_cols)
    if sqrt_n * sqrt_n < n_cols:
        sqrt_n += 1

    fast_eval, fast_ok = _estimate_css_distance_rw(
        hx_rows,
        hz_rows,
        n_cols,
        steps=steps_fast,
        wmin=wmin,
        target_distance=target_distance,
        rng=rng,
        dist_m4ri_cmd=dist_m4ri_cmd,
        estimator=estimator,
    )
    d_fast_ub = (
        None if fast_eval.get("d_ub") is None else int(fast_eval["d_ub"])
    )
    if not fast_ok or d_fast_ub is None:
        return TwoStageDistanceResult(
            passed=False,
            d_final_ub=None,
            d_fast_ub=d_fast_ub,
            d_slow_ub=None,
            sqrt_n=sqrt_n,
            fast=fast_eval,
            slow=None,
            ran_slow=False,
        )

    should_slow = d_fast_ub >= sqrt_n and d_fast_ub > current_best
    slow_eval: Optional[Dict[str, object]] = None
    d_slow_ub: Optional[int] = None
    if should_slow:
        if on_refine is not None:
            on_refine(d_fast_ub, sqrt_n, current_best)
        slow_eval, slow_ok = _estimate_css_distance_rw(
            hx_rows,
            hz_rows,
            n_cols,
            steps=steps_slow,
            wmin=wmin,
            target_distance=target_distance,
            rng=rng,
            dist_m4ri_cmd=dist_m4ri_cmd,
            estimator=estimator,
        )
        if slow_eval.get("d_ub") is not None:
            d_slow_ub = int(slow_eval["d_ub"])
        if not slow_ok or d_slow_ub is None:
            return TwoStageDistanceResult(
                passed=False,
                d_final_ub=None,
                d_fast_ub=d_fast_ub,
                d_slow_ub=d_slow_ub,
                sqrt_n=sqrt_n,
                fast=fast_eval,
                slow=slow_eval,
                ran_slow=True,
            )
        d_final_ub = d_slow_ub
    else:
        d_final_ub = d_fast_ub

    return TwoStageDistanceResult(
        passed=True,
        d_final_ub=d_final_ub,
        d_fast_ub=d_fast_ub,
        d_slow_ub=d_slow_ub,
        sqrt_n=sqrt_n,
        fast=fast_eval,
        slow=slow_eval,
        ran_slow=should_slow,
    )


def _histogram_payload(hist: Dict[Tuple[int, int], int]) -> List[Dict[str, int]]:
    entries = []
    for (k_val, d_val), count in hist.items():
        entries.append(
            {
                "k_classical": int(k_val),
                "d_est": int(d_val),
                "count": int(count),
            }
        )
    entries.sort(
        key=lambda entry: (-entry["count"], -entry["k_classical"], -entry["d_est"])
    )
    return entries


def _print_histogram_top(
    *,
    side: str,
    hist: Dict[Tuple[int, int], int],
    top_n: int = 10,
) -> None:
    entries = _histogram_payload(hist)
    print(f"[progressive] {side} histogram top {min(top_n, len(entries))}:")
    if not entries:
        print("  (none)")
        return
    for entry in entries[:top_n]:
        print(
            f"  k={entry['k_classical']} d_est={entry['d_est']} count={entry['count']}"
        )


def _write_histogram(
    *,
    path: Path,
    side: str,
    group: FiniteGroup,
    hist: Dict[Tuple[int, int], int],
    classical_target: int,
) -> None:
    payload = {
        "side": side,
        "group": {"spec": group.name, "order": group.order},
        "classical_target": classical_target,
        "entries": _histogram_payload(hist),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _precompute_classical_side(
    *,
    side: str,
    group: FiniteGroup,
    multisets: List[List[int]],
    variant_codes: List[LocalCode],
    base_code: LocalCode,
    steps: int,
    classical_target: int,
    dist_m4ri_cmd: str,
    rng: random.Random,
    out_path: Path,
) -> Tuple[List[ProgressiveSetting], Dict[Tuple[int, int], int], int]:
    if classical_target <= 0:
        raise ValueError("--classical-target must be positive.")
    wmin = max(0, classical_target - 1)
    kept: List[ProgressiveSetting] = []
    hist: Dict[Tuple[int, int], int] = {}
    total_settings = 0

    if side == "A":
        build_h = build_a_slice_checks_H
        build_g = build_a_slice_checks_G
        h_label = "H"
        g_label = "G"
    elif side == "B":
        build_h = build_b_slice_checks_Hp
        build_g = build_b_slice_checks_Gp
        h_label = "Hp"
        g_label = "Gp"
    else:
        raise ValueError("side must be 'A' or 'B'.")

    with out_path.open("w", encoding="utf-8") as out_file:
        for elements in multisets:
            elements_repr = [group.repr(x) for x in elements]
            for perm_idx, C1 in enumerate(variant_codes):
                rows_h, n_cols = build_h(group, elements, base_code, C1)
                rows_g, _ = build_g(group, elements, base_code, C1)
                h_eval = _classical_eval(
                    rows_h,
                    n_cols,
                    steps=steps,
                    wmin=wmin,
                    rng=rng,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                )
                g_eval = _classical_eval(
                    rows_g,
                    n_cols,
                    steps=steps,
                    wmin=wmin,
                    rng=rng,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                )
                if side == "A":
                    slice_codes = {
                        "A_H": {
                            "n": n_cols,
                            "k": int(h_eval["k"]),
                            "d_ub": int(h_eval["d_est"]),
                        },
                        "A_G": {
                            "n": n_cols,
                            "k": int(g_eval["k"]),
                            "d_ub": int(g_eval["d_est"]),
                        },
                    }
                else:
                    slice_codes = {
                        "B_G": {
                            "n": n_cols,
                            "k": int(g_eval["k"]),
                            "d_ub": int(g_eval["d_est"]),
                        },
                        "B_H": {
                            "n": n_cols,
                            "k": int(h_eval["k"]),
                            "d_ub": int(h_eval["d_est"]),
                        },
                    }
                k_min = min(int(h_eval["k"]), int(g_eval["k"]))
                d_min = min(int(h_eval["d_est"]), int(g_eval["d_est"]))
                hist[(k_min, d_min)] = hist.get((k_min, d_min), 0) + 1
                total_settings += 1

                passes = (int(h_eval["d_signed"]) >= 0) and (
                    int(g_eval["d_signed"]) >= 0
                )
                record = {
                    "side": side,
                    "group": {"spec": group.name, "order": group.order},
                    "elements": list(elements),
                    "elements_repr": elements_repr,
                    "perm_idx": perm_idx,
                    "k_classical": k_min,
                    "d_est": d_min,
                    "slice_n": n_cols,
                    "slice_codes": slice_codes,
                    "checks": {
                        h_label: h_eval,
                        g_label: g_eval,
                    },
                    "passes": passes,
                }
                if passes:
                    setting_id = _progressive_setting_id(side, elements, perm_idx)
                    record["id"] = setting_id
                    kept.append(
                        ProgressiveSetting(
                            side=side,
                            elements=list(elements),
                            elements_repr=elements_repr,
                            perm_idx=perm_idx,
                            k_classical=k_min,
                            d_est=d_min,
                            record=record,
                            setting_id=setting_id,
                        )
                    )
                    out_file.write(json.dumps(record, sort_keys=True) + "\n")
        out_file.flush()
    return kept, hist, total_settings


def _group_settings_by_k(
    settings: List[ProgressiveSetting],
) -> Dict[int, List[ProgressiveSetting]]:
    grouped: Dict[int, List[ProgressiveSetting]] = {}
    for setting in settings:
        grouped.setdefault(setting.k_classical, []).append(setting)
    for k_val in grouped:
        grouped[k_val].sort(
            key=lambda item: (-item.d_est, item.setting_id)
        )
    return grouped


def _interleaved_rounds(
    settings: List[ProgressiveSetting],
) -> List[List[ProgressiveSetting]]:
    grouped = _group_settings_by_k(settings)
    if not grouped:
        return []
    k_values = sorted(grouped.keys(), reverse=True)
    max_len = max(len(items) for items in grouped.values())
    rounds: List[List[ProgressiveSetting]] = []
    for r in range(max_len):
        round_items: List[ProgressiveSetting] = []
        for k_val in k_values:
            items = grouped[k_val]
            if r < len(items):
                round_items.append(items[r])
        if round_items:
            rounds.append(round_items)
    return rounds


def _iter_progressive_pairs(
    rounds_a: List[List[ProgressiveSetting]],
    rounds_b: List[List[ProgressiveSetting]],
) -> Iterable[Tuple[ProgressiveSetting, ProgressiveSetting]]:
    if not rounds_a or not rounds_b:
        return
    max_sum = (len(rounds_a) - 1) + (len(rounds_b) - 1)
    for rank_sum in range(max_sum + 1):
        pairs: List[Tuple[ProgressiveSetting, ProgressiveSetting]] = []
        for r_a, round_a in enumerate(rounds_a):
            r_b = rank_sum - r_a
            if r_b < 0 or r_b >= len(rounds_b):
                continue
            round_b = rounds_b[r_b]
            for a_setting in round_a:
                for b_setting in round_b:
                    pairs.append((a_setting, b_setting))
        if not pairs:
            continue
        pairs.sort(
            key=lambda pair: (
                -min(pair[0].d_est, pair[1].d_est),
                pair[0].setting_id,
                pair[1].setting_id,
            )
        )
        for pair in pairs:
            yield pair


def _format_best_by_k_table(best_by_k: Dict[int, Dict[str, object]]) -> str:
    lines = ["k | best d_ub | steps | when_found(eval) | A_id | B_id"]
    for k_val in sorted(best_by_k):
        entry = best_by_k[k_val]
        lines.append(
            f"{k_val} | {entry['d_ub']} | {entry['steps']} | "
            f"{entry['when']}({entry['eval']}) | {entry['A_id']} | {entry['B_id']}"
        )
    return "\n".join(lines)


def _save_best_code(
    *,
    outdir: Path,
    code_id: str,
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    meta: Dict[str, object],
) -> None:
    code_dir = outdir / "best_codes" / code_id
    code_dir.mkdir(parents=True, exist_ok=True)
    write_mtx_from_bitrows(str(code_dir / "Hx.mtx"), list(hx_rows), n_cols)
    write_mtx_from_bitrows(str(code_dir / "Hz.mtx"), list(hz_rows), n_cols)
    (code_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def progressive_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Progressive exhaustive classical-first search."
    )
    parser.add_argument("--group", type=str, required=True, help="Group spec.")
    parser.add_argument(
        "--target-distance",
        type=int,
        required=True,
        help="Reject quantum candidates below this target distance.",
    )
    parser.add_argument(
        "--classical-target",
        type=int,
        default=None,
        help="Classical filter target distance (default: ceil(sqrt(n_slice))).",
    )
    parser.add_argument(
        "--classical-steps",
        type=int,
        default=100,
        help="dist_m4ri RW steps for classical prefilter.",
    )
    parser.add_argument(
        "--quantum-steps-fast",
        type=int,
        default=2000,
        help="dist_m4ri RW steps for fast quantum distance estimates.",
    )
    parser.add_argument(
        "--quantum-steps-slow",
        type=int,
        default=50000,
        help="dist_m4ri RW steps for slow quantum distance estimates.",
    )
    parser.add_argument(
        "--quantum-steps",
        type=int,
        default=None,
        help=(
            "Deprecated alias: sets both --quantum-steps-fast and "
            "--quantum-steps-slow."
        ),
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Report best-by-k every N quantum evals.",
    )
    parser.add_argument(
        "--max-quantum-evals",
        type=int,
        default=0,
        help="Stop after this many quantum distance evaluations (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: results/progressive_<group>_<ts>).",
    )
    parser.add_argument(
        "--dist-m4ri-cmd",
        type=str,
        default="dist_m4ri",
        help="dist_m4ri command.",
    )
    parser.add_argument(
        "--gap-cmd",
        type=str,
        default="gap",
        help="GAP command (needed for SmallGroup or --dedup-cayley).",
    )
    parser.add_argument(
        "--dedup-cayley",
        action="store_true",
        help="Deduplicate multisets up to Aut(G).",
    )
    parser.add_argument(
        "--min-distinct",
        type=int,
        default=None,
        help="Require at least this many distinct elements per multiset.",
    )
    args = parser.parse_args(argv)

    if args.target_distance <= 0:
        raise ValueError("--target-distance must be positive.")
    if args.classical_steps <= 0:
        raise ValueError("--classical-steps must be positive.")
    if args.quantum_steps is not None:
        if args.quantum_steps <= 0:
            raise ValueError("--quantum-steps must be positive.")
        if (
            args.quantum_steps_fast != 2000
            or args.quantum_steps_slow != 50000
        ):
            raise ValueError(
                "--quantum-steps is deprecated; do not mix with "
                "--quantum-steps-fast/--quantum-steps-slow."
            )
        args.quantum_steps_fast = args.quantum_steps
        args.quantum_steps_slow = args.quantum_steps
    if args.quantum_steps_fast <= 0:
        raise ValueError("--quantum-steps-fast must be positive.")
    if args.quantum_steps_slow <= 0:
        raise ValueError("--quantum-steps-slow must be positive.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be positive.")
    if args.max_quantum_evals < 0:
        raise ValueError("--max-quantum-evals must be nonnegative.")
    if args.min_distinct is not None and args.min_distinct <= 0:
        raise ValueError("--min-distinct must be positive.")
    if args.min_distinct is not None and args.min_distinct > SIDE_LEN:
        raise ValueError("--min-distinct cannot exceed 6.")

    group_spec = canonical_group_spec(args.group)
    needs_gap = args.dedup_cayley or group_spec.startswith("SmallGroup")
    if needs_gap and not gap_is_available(args.gap_cmd):
        raise RuntimeError(
            f"GAP is required for group data or automorphisms, but '{args.gap_cmd}' "
            "was not found on PATH."
        )
    if not dist_m4ri_is_available(args.dist_m4ri_cmd):
        raise RuntimeError(
            "dist_m4ri not found on PATH; install dist-m4ri and ensure the dist_m4ri "
            "binary is available (see README.md#dist-m4ri)."
        )

    seed = args.seed
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 31)
    rng = random.Random(seed)

    group_tag = _safe_id_component(group_spec)
    if args.results_dir:
        outdir = Path(args.results_dir)
    else:
        outdir = Path("results") / f"progressive_{group_tag}_{_timestamp_utc()}"
    outdir.mkdir(parents=True, exist_ok=True)

    group_cache_dir = outdir / "groups"
    aut_cache_dir = outdir / "cache" / "aut"
    group = group_from_spec(group_spec, gap_cmd=args.gap_cmd, cache_dir=group_cache_dir)

    n_slice = SIDE_LEN * group.order
    classical_target = args.classical_target
    if classical_target is None:
        classical_target = _ceil_sqrt(n_slice)

    run_meta = {
        "mode": "progressive",
        "group": {"spec": group.name, "order": group.order},
        "target_distance": args.target_distance,
        "classical_target": classical_target,
        "classical_steps": args.classical_steps,
        "quantum_steps_fast": args.quantum_steps_fast,
        "quantum_steps_slow": args.quantum_steps_slow,
        "report_every": args.report_every,
        "max_quantum_evals": args.max_quantum_evals,
        "seed": seed,
        "dist_m4ri_cmd": args.dist_m4ri_cmd,
        "dedup_cayley": bool(args.dedup_cayley),
        "min_distinct": args.min_distinct,
    }
    (outdir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    base_code = hamming_6_3_3_shortened()
    variant_codes = _build_variant_codes(base_code)
    perm_total = len(variant_codes)

    multisets = _enumerate_multisets_with_identity(group.order)
    raw_multiset_count = len(multisets)
    if args.min_distinct is not None:
        multisets = _filter_min_distinct(multisets, args.min_distinct)
    after_distinct_count = len(multisets)
    if args.dedup_cayley:
        automorphisms = group.automorphisms(
            gap_cmd=args.gap_cmd,
            cache_dir=aut_cache_dir,
        )
        multisets, raw_count, uniq_count = _dedup_multisets(
            group, multisets, automorphisms=automorphisms
        )
        print(
            f"[progressive] dedup-cayley raw={raw_count} unique={uniq_count}"
        )
    print(
        f"[progressive] multisets enumerated={raw_multiset_count} "
        f"after_min_distinct={after_distinct_count} "
        f"after_dedup={len(multisets)}"
    )
    if not multisets:
        print("[progressive] no multisets to score after filtering; exiting.")
        return 0

    total_settings = len(multisets) * perm_total
    print(
        f"[progressive] perms={perm_total} settings_per_side={total_settings}"
    )

    a_keep, a_hist, a_total = _precompute_classical_side(
        side="A",
        group=group,
        multisets=multisets,
        variant_codes=variant_codes,
        base_code=base_code,
        steps=args.classical_steps,
        classical_target=classical_target,
        dist_m4ri_cmd=args.dist_m4ri_cmd,
        rng=rng,
        out_path=outdir / "classical_A_kept.jsonl",
    )
    _print_histogram_top(side="A", hist=a_hist)
    _write_histogram(
        path=outdir / "classical_A_histogram.json",
        side="A",
        group=group,
        hist=a_hist,
        classical_target=classical_target,
    )
    print(
        f"[progressive] A settings={a_total} kept={len(a_keep)}"
    )

    b_keep, b_hist, b_total = _precompute_classical_side(
        side="B",
        group=group,
        multisets=multisets,
        variant_codes=variant_codes,
        base_code=base_code,
        steps=args.classical_steps,
        classical_target=classical_target,
        dist_m4ri_cmd=args.dist_m4ri_cmd,
        rng=rng,
        out_path=outdir / "classical_B_kept.jsonl",
    )
    _print_histogram_top(side="B", hist=b_hist)
    _write_histogram(
        path=outdir / "classical_B_histogram.json",
        side="B",
        group=group,
        hist=b_hist,
        classical_target=classical_target,
    )
    print(
        f"[progressive] B settings={b_total} kept={len(b_keep)}"
    )

    if not a_keep or not b_keep:
        print("[progressive] no settings passed classical filter; exiting.")
        return 0

    a_rounds = _interleaved_rounds(a_keep)
    b_rounds = _interleaved_rounds(b_keep)
    print(
        f"[progressive] A rounds={len(a_rounds)} B rounds={len(b_rounds)}"
    )

    best_by_k: Dict[int, Dict[str, object]] = {}
    milestones_path = outdir / "milestones.jsonl"
    eval_index = 0
    pairs_seen = 0
    last_report_time = time.monotonic()
    last_report_eval = 0
    wmin_q = max(0, args.target_distance - 1)

    interrupted = False
    try:
        for a_setting, b_setting in _iter_progressive_pairs(a_rounds, b_rounds):
            pairs_seen += 1
            permA = a_setting.perm_idx
            permB = b_setting.perm_idx
            C1 = variant_codes[permA]
            C1p = variant_codes[permB]
            hx_rows, hz_rows, n_cols = build_hx_hz(
                group,
                a_setting.elements,
                b_setting.elements,
                base_code,
                C1,
                base_code,
                C1p,
            )
            if not css_commutes(hx_rows, hz_rows):
                raise RuntimeError(
                    f"HX/HZ do not commute for A={a_setting.setting_id} "
                    f"B={b_setting.setting_id}."
                )
            rank_hx = gf2_rank(hx_rows[:], n_cols)
            rank_hz = gf2_rank(hz_rows[:], n_cols)
            k_val = n_cols - rank_hx - rank_hz
            if k_val <= 0:
                continue

            if args.max_quantum_evals and eval_index >= args.max_quantum_evals:
                break
            eval_index += 1

            current = best_by_k.get(k_val)
            current_best: float = (
                float("-inf") if current is None else int(current["d_ub"])
            )

            def _log_refine(d_fast: int, sqrt_n: int, best_d: float) -> None:
                best_text = "-inf" if best_d == float("-inf") else str(int(best_d))
                print(
                    f"[refine] eval={eval_index} k={k_val} d_fast={d_fast} "
                    f">= sqrt(n)={sqrt_n} and beats best={best_text} -> "
                    f"running {args.quantum_steps_slow} steps"
                )

            distance_result = _two_stage_css_distance(
                hx_rows,
                hz_rows,
                n_cols,
                target_distance=args.target_distance,
                wmin=wmin_q,
                steps_fast=args.quantum_steps_fast,
                steps_slow=args.quantum_steps_slow,
                current_best=current_best,
                rng=rng,
                dist_m4ri_cmd=args.dist_m4ri_cmd,
                on_refine=_log_refine,
            )
            if (
                not distance_result.passed
                or distance_result.d_final_ub is None
                or not distance_result.ran_slow
            ):
                continue

            if distance_result.d_slow_ub is None:
                continue
            d_final_ub = int(distance_result.d_slow_ub)
            if d_final_ub > current_best:
                timestamp = _timestamp_utc()
                code_id_raw = (
                    f"{group.name}_A{a_setting.setting_id}_B{b_setting.setting_id}_"
                    f"k{k_val}_d{d_final_ub}"
                )
                code_id = _safe_id_component(code_id_raw)
                classical_slices = {
                    **a_setting.record.get("slice_codes", {}),
                    **b_setting.record.get("slice_codes", {}),
                }
                meta = {
                    "code_id": code_id,
                    "group": {"spec": group.name, "order": group.order},
                    "A": a_setting.record,
                    "B": b_setting.record,
                    "local_codes": {
                        "C0": base_code.name,
                        "C1": C1.name,
                        "C0p": base_code.name,
                        "C1p": C1p.name,
                        "permA_idx": permA,
                        "permB_idx": permB,
                    },
                    "classical_slices": classical_slices,
                    "n": n_cols,
                    "k": k_val,
                    "distance": {
                        "method": "dist_m4ri_rw",
                        "wmin": wmin_q,
                        "steps_fast": args.quantum_steps_fast,
                        "steps_slow": args.quantum_steps_slow,
                        "fast": distance_result.fast,
                        "slow": distance_result.slow,
                        "d_fast_ub": distance_result.d_fast_ub,
                        "d_slow_ub": distance_result.d_slow_ub,
                        "d_ub": d_final_ub,
                        "sqrt_n": distance_result.sqrt_n,
                    },
                    "target_distance": args.target_distance,
                    "seed": seed,
                }
                _save_best_code(
                    outdir=outdir,
                    code_id=code_id,
                    hx_rows=hx_rows,
                    hz_rows=hz_rows,
                    n_cols=n_cols,
                    meta=meta,
                )
                best_by_k[k_val] = {
                    "d_ub": d_final_ub,
                    "steps": args.quantum_steps_slow,
                    "eval": eval_index,
                    "when": timestamp,
                    "A_id": a_setting.setting_id,
                    "B_id": b_setting.setting_id,
                    "code_id": code_id,
                }
                with milestones_path.open("a", encoding="utf-8") as mfile:
                    mfile.write(
                        json.dumps(
                            {
                                "timestamp": timestamp,
                                "eval": eval_index,
                                "k": k_val,
                                "d_ub": d_final_ub,
                                "A_id": a_setting.setting_id,
                                "B_id": b_setting.setting_id,
                                "code_id": code_id,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                print(
                    f"NEW_BEST {timestamp} eval={eval_index} n={n_cols} "
                    f"k={k_val} d_fast={distance_result.d_fast_ub} "
                    f"d_slow={distance_result.d_slow_ub} d_ub={d_final_ub} "
                    f"steps={args.quantum_steps_slow} "
                    f"A={a_setting.setting_id} B={b_setting.setting_id} "
                    f"sqrt_n={distance_result.sqrt_n}"
                )
                if classical_slices:
                    print(f"classical slices (steps={args.classical_steps}):")
                    for key in ("A_H", "A_G", "B_G", "B_H"):
                        entry = classical_slices.get(key)
                        if entry is None:
                            continue
                        print(
                            f"  {key}: n={entry['n']} k={entry['k']} "
                            f"d_ub={entry['d_ub']}"
                        )

            now = time.monotonic()
            if (
                eval_index % args.report_every == 0
                or now - last_report_time >= 60
            ) and eval_index != last_report_eval:
                print(_format_best_by_k_table(best_by_k))
                last_report_time = now
                last_report_eval = eval_index
    except KeyboardInterrupt:
        interrupted = True
        print("[progressive] interrupted by Ctrl-C; printing summary.")

    print(_format_best_by_k_table(best_by_k))
    summary = {
        "pairs_seen": pairs_seen,
        "quantum_evals": eval_index,
        "best_by_k": best_by_k,
        "interrupted": interrupted,
    }
    (outdir / "best_by_k.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


__all__ = [
    "ProgressiveSetting",
    "_enumerate_multisets_with_identity",
    "_interleaved_rounds",
    "_iter_progressive_pairs",
    "progressive_main",
]
