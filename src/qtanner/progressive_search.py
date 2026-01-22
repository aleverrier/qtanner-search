"""Progressive exhaustive classical-first search mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .classical_dist_fast import _estimate_classical_distance_fast_details
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
class ProgressiveDistanceResult:
    passed_fast: bool
    d_x_best: int
    d_z_best: int
    steps_x: int
    steps_z: int
    fast: Dict[str, object]
    refine_chunks: List[Dict[str, object]]
    ran_refine: bool
    aborted: bool
    abort_reason: Optional[str]


@dataclass(frozen=True)
class ClassicalEvalResult:
    elements: List[int]
    perm_idx: int
    n_cols: int
    h_eval: Dict[str, object]
    g_eval: Dict[str, object]
    passes: bool
    k_min: int
    d_min: int


def _add_timing(
    timings: Optional[Dict[str, float]],
    key: str,
    delta: float,
) -> None:
    if timings is None:
        return
    timings[key] = timings.get(key, 0.0) + float(delta)


def _merge_timings(
    dest: Optional[Dict[str, float]],
    src: Optional[Dict[str, float]],
) -> None:
    if dest is None or src is None:
        return
    for key, value in src.items():
        dest[key] = dest.get(key, 0.0) + float(value)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _print_timing_summary(
    label: str,
    timings: Optional[Dict[str, float]],
    *,
    wall_seconds: Optional[float] = None,
) -> None:
    if timings is None and wall_seconds is None:
        return
    data = timings or {}
    total = sum(data.values())
    line = f"[timings] {label} total={_format_seconds(total)}"
    if wall_seconds is not None:
        other = max(0.0, wall_seconds - total)
        line += (
            f" wall={_format_seconds(wall_seconds)} "
            f"other={_format_seconds(other)}"
        )
    print(line)
    for key, value in sorted(data.items(), key=lambda item: -item[1]):
        print(f"  {key}: {_format_seconds(value)}")


def _skipped_eval(backend: str) -> Dict[str, object]:
    return {
        "k": None,
        "rank": None,
        "d_signed": None,
        "d_witness": None,
        "d_est": None,
        "exact": False,
        "early_stop": None,
        "seed": None,
        "backend": backend,
        "codewords_checked": None,
        "skipped": True,
    }


def _timestamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stable_seed(seed: int, label: str) -> int:
    payload = f"{int(seed)}:{label}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)


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


def _adaptive_quantum_steps_slow(n_cols: int) -> int:
    if n_cols <= 200:
        return 50000
    if n_cols <= 600:
        return 200000
    return 500000


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in argv)


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


def _local_code_signature(
    base_code: LocalCode,
    variant_codes: Sequence[LocalCode],
) -> Tuple[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]], ...]]:
    base_sig = (
        int(base_code.n),
        int(base_code.k),
        tuple(int(x) for x in base_code.H_rows),
        tuple(int(x) for x in base_code.G_rows),
    )
    variant_sig = tuple(
        (
            int(code.n),
            int(code.k),
            tuple(int(x) for x in code.H_rows),
            tuple(int(x) for x in code.G_rows),
        )
        for code in variant_codes
    )
    return base_sig, variant_sig


def _local_code_specs_match(
    base_a: LocalCode,
    variants_a: Sequence[LocalCode],
    base_b: LocalCode,
    variants_b: Sequence[LocalCode],
) -> bool:
    return _local_code_signature(base_a, variants_a) == _local_code_signature(
        base_b, variants_b
    )


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


def _canonical_multiset_key(
    multiset: Sequence[int],
    *,
    canonicalize: Optional[Callable[[Sequence[int]], Tuple[int, ...]]] = None,
) -> Tuple[int, ...]:
    base = tuple(sorted(int(x) for x in multiset))
    if canonicalize is None:
        return base
    return canonicalize(base)


def _inverse_multiset_key(
    group: FiniteGroup,
    multiset: Sequence[int],
    *,
    canonicalize: Optional[Callable[[Sequence[int]], Tuple[int, ...]]] = None,
) -> Tuple[int, ...]:
    inv_base = tuple(sorted(int(group.inv(x)) for x in multiset))
    if canonicalize is None:
        return inv_base
    return canonicalize(inv_base)


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
    rng: Optional[random.Random],
    dist_m4ri_cmd: str,
    backend: str,
    exhaustive_k_max: int,
    sample_count: int,
    seed_override: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    if seed_override is None:
        if rng is None:
            raise ValueError("rng is required when seed_override is not set.")
        seed = rng.randrange(1 << 31)
    else:
        seed = int(seed_override)
    if backend == "fast":
        witness, k_val, exact, checked = _estimate_classical_distance_fast_details(
            rows,
            n_cols,
            wmin,
            exhaustive_k_max,
            sample_count,
            seed,
        )
        rank = n_cols - k_val
        d_est = n_cols + 1 if witness is None else int(witness)
        early_stop = witness is not None and int(witness) <= wmin
        return {
            "k": k_val,
            "rank": rank,
            "d_signed": None,
            "d_witness": witness,
            "d_est": d_est,
            "exact": exact,
            "early_stop": early_stop,
            "seed": seed,
            "backend": "fast",
            "codewords_checked": checked,
        }
    if backend != "dist-m4ri":
        raise ValueError(f"Unknown classical backend: {backend}")
    rank_start = time.perf_counter()
    rank = gf2_rank(list(rows), n_cols)
    _add_timing(timings, "rank", time.perf_counter() - rank_start)
    k_val = n_cols - rank
    d_signed = run_dist_m4ri_classical_rw(
        rows,
        n_cols,
        steps,
        wmin,
        seed=seed,
        dist_m4ri_cmd=dist_m4ri_cmd,
        timings=timings,
    )
    witness = abs(d_signed)
    early_stop = witness <= wmin
    return {
        "k": k_val,
        "rank": rank,
        "d_signed": d_signed,
        "d_witness": witness,
        "d_est": witness,
        "exact": False,
        "early_stop": early_stop,
        "seed": seed,
        "backend": "dist-m4ri",
        "codewords_checked": None,
    }


def _evaluate_classical_chunk(
    settings: List[Tuple[List[int], int]],
    *,
    side: str,
    group: FiniteGroup,
    variant_codes: Sequence[LocalCode],
    base_code: LocalCode,
    steps: int,
    wmin: int,
    backend: str,
    exhaustive_k_max: int,
    sample_count: int,
    dist_m4ri_cmd: str,
    first_label: str,
    timings_enabled: bool,
    classical_jobs: int,
    seed_base: int,
    eval_fn: Callable[..., Dict[str, object]] = _classical_eval,
) -> Tuple[List[ClassicalEvalResult], Dict[str, float], int]:
    if side == "A":
        build_h = build_a_slice_checks_H
        build_g = build_a_slice_checks_G
        slice_h_key = "A_H"
        slice_g_key = "A_G"
    elif side == "B":
        build_h = build_b_slice_checks_Hp
        build_g = build_b_slice_checks_Gp
        slice_h_key = "B_H"
        slice_g_key = "B_G"
    else:
        raise ValueError("side must be 'A' or 'B'.")

    timings: Optional[Dict[str, float]] = {} if timings_enabled else None
    dummy_rng = random.Random(0)
    results: List[ClassicalEvalResult] = []
    calls = 0
    first_is_h = first_label.upper().startswith("H")
    thread_pool: Optional[ThreadPoolExecutor] = None
    if classical_jobs > 1:
        thread_pool = ThreadPoolExecutor(max_workers=classical_jobs)

    def _passes(eval_entry: Dict[str, object]) -> bool:
        witness = eval_entry.get("d_witness")
        if witness is None:
            return True
        return int(witness) > wmin

    def _slice_seed(slice_key: str, setting_id: str) -> int:
        return _stable_seed(seed_base, f"{slice_key}:{setting_id}")

    def _eval_rows(
        rows: Sequence[int],
        n_cols: int,
        seed_override: int,
    ) -> Tuple[Dict[str, object], Optional[Dict[str, float]]]:
        local_timings: Optional[Dict[str, float]] = (
            {} if timings_enabled else None
        )
        eval_entry = eval_fn(
            rows,
            n_cols,
            steps=steps,
            wmin=wmin,
            rng=dummy_rng,
            dist_m4ri_cmd=dist_m4ri_cmd,
            backend=backend,
            exhaustive_k_max=exhaustive_k_max,
            sample_count=sample_count,
            seed_override=seed_override,
            timings=local_timings,
        )
        return eval_entry, local_timings

    try:
        for elements, perm_idx in settings:
            C1 = variant_codes[perm_idx]
            setting_id = _progressive_setting_id(side, elements, perm_idx)
            seed_h = _slice_seed(slice_h_key, setting_id)
            seed_g = _slice_seed(slice_g_key, setting_id)
            if thread_pool is not None:
                build_start = time.perf_counter()
                rows_h, n_cols = build_h(group, elements, base_code, C1)
                rows_g, _ = build_g(group, elements, base_code, C1)
                _add_timing(
                    timings, "build_slices", time.perf_counter() - build_start
                )
                future_h = thread_pool.submit(_eval_rows, rows_h, n_cols, seed_h)
                future_g = thread_pool.submit(_eval_rows, rows_g, n_cols, seed_g)
                h_eval, h_timings = future_h.result()
                g_eval, g_timings = future_g.result()
                calls += 2
                _merge_timings(timings, h_timings)
                _merge_timings(timings, g_timings)
                passes = _passes(h_eval) and _passes(g_eval)
            elif first_is_h:
                build_start = time.perf_counter()
                rows_h, n_cols = build_h(group, elements, base_code, C1)
                _add_timing(
                    timings, "build_slices", time.perf_counter() - build_start
                )
                h_eval = eval_fn(
                    rows_h,
                    n_cols,
                    steps=steps,
                    wmin=wmin,
                    rng=dummy_rng,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                    backend=backend,
                    exhaustive_k_max=exhaustive_k_max,
                    sample_count=sample_count,
                    seed_override=seed_h,
                    timings=timings,
                )
                calls += 1
                passes = _passes(h_eval)
                if passes:
                    build_start = time.perf_counter()
                    rows_g, _ = build_g(group, elements, base_code, C1)
                    _add_timing(
                        timings, "build_slices", time.perf_counter() - build_start
                    )
                    g_eval = eval_fn(
                        rows_g,
                        n_cols,
                        steps=steps,
                        wmin=wmin,
                        rng=dummy_rng,
                        dist_m4ri_cmd=dist_m4ri_cmd,
                        backend=backend,
                        exhaustive_k_max=exhaustive_k_max,
                        sample_count=sample_count,
                        seed_override=seed_g,
                        timings=timings,
                    )
                    calls += 1
                    passes = passes and _passes(g_eval)
                else:
                    g_eval = _skipped_eval(backend)
            else:
                build_start = time.perf_counter()
                rows_g, n_cols = build_g(group, elements, base_code, C1)
                _add_timing(
                    timings, "build_slices", time.perf_counter() - build_start
                )
                g_eval = eval_fn(
                    rows_g,
                    n_cols,
                    steps=steps,
                    wmin=wmin,
                    rng=dummy_rng,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                    backend=backend,
                    exhaustive_k_max=exhaustive_k_max,
                    sample_count=sample_count,
                    seed_override=seed_g,
                    timings=timings,
                )
                calls += 1
                passes = _passes(g_eval)
                if passes:
                    build_start = time.perf_counter()
                    rows_h, _ = build_h(group, elements, base_code, C1)
                    _add_timing(
                        timings, "build_slices", time.perf_counter() - build_start
                    )
                    h_eval = eval_fn(
                        rows_h,
                        n_cols,
                        steps=steps,
                        wmin=wmin,
                        rng=dummy_rng,
                        dist_m4ri_cmd=dist_m4ri_cmd,
                        backend=backend,
                        exhaustive_k_max=exhaustive_k_max,
                        sample_count=sample_count,
                        seed_override=seed_h,
                        timings=timings,
                    )
                    calls += 1
                    passes = passes and _passes(h_eval)
                else:
                    h_eval = _skipped_eval(backend)

            k_vals = [
                val for val in (h_eval.get("k"), g_eval.get("k")) if val is not None
            ]
            d_vals = [
                val
                for val in (h_eval.get("d_est"), g_eval.get("d_est"))
                if val is not None
            ]
            k_min = min(int(val) for val in k_vals) if k_vals else 0
            d_min = min(int(val) for val in d_vals) if d_vals else 0
            results.append(
                ClassicalEvalResult(
                    elements=list(elements),
                    perm_idx=perm_idx,
                    n_cols=n_cols,
                    h_eval=h_eval,
                    g_eval=g_eval,
                    passes=passes,
                    k_min=k_min,
                    d_min=d_min,
                )
            )
    finally:
        if thread_pool is not None:
            thread_pool.shutdown()

    return results, (timings or {}), calls


def _estimate_css_distance_direction(
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    *,
    steps: int,
    wmin: int,
    rng: random.Random,
    dist_m4ri_cmd: str,
    estimator: Callable[..., int] = run_dist_m4ri_css_rw,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    seed = rng.randrange(1 << 31)
    if timings is None:
        signed = estimator(
            hx_rows,
            hz_rows,
            n_cols,
            steps,
            wmin,
            seed=seed,
            dist_m4ri_cmd=dist_m4ri_cmd,
        )
    else:
        signed = estimator(
            hx_rows,
            hz_rows,
            n_cols,
            steps,
            wmin,
            seed=seed,
            dist_m4ri_cmd=dist_m4ri_cmd,
            timings=timings,
        )
    d_ub = abs(signed)
    return {
        "steps": steps,
        "seed": seed,
        "signed": signed,
        "d_ub": d_ub,
        "early_stop": signed < 0,
    }


def _progressive_css_distance(
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    *,
    must_exceed: int,
    steps_fast: int,
    steps_slow: int,
    refine_chunk: int,
    rng: random.Random,
    dist_m4ri_cmd: str,
    estimator: Callable[..., int] = run_dist_m4ri_css_rw,
    on_chunk: Optional[
        Callable[[int, int, int, int, int, Optional[str]], None]
    ] = None,
    timings: Optional[Dict[str, float]] = None,
) -> ProgressiveDistanceResult:
    wmin = max(0, int(must_exceed))
    dz_fast = _estimate_css_distance_direction(
        hx_rows,
        hz_rows,
        n_cols,
        steps=steps_fast,
        wmin=wmin,
        rng=rng,
        dist_m4ri_cmd=dist_m4ri_cmd,
        estimator=estimator,
        timings=timings,
    )
    dx_fast = _estimate_css_distance_direction(
        hz_rows,
        hx_rows,
        n_cols,
        steps=steps_fast,
        wmin=wmin,
        rng=rng,
        dist_m4ri_cmd=dist_m4ri_cmd,
        estimator=estimator,
        timings=timings,
    )
    d_z_best = int(dz_fast["d_ub"])
    d_x_best = int(dx_fast["d_ub"])
    steps_z = steps_fast
    steps_x = steps_fast
    passed_fast = d_x_best > must_exceed and d_z_best > must_exceed
    refine_chunks: List[Dict[str, object]] = []
    ran_refine = False
    aborted = False
    abort_reason: Optional[str] = None

    if passed_fast and steps_slow > steps_fast:
        ran_refine = True
        steps_used = steps_fast
        chunk_index = 0
        while steps_used < steps_slow:
            chunk_steps = min(refine_chunk, steps_slow - steps_used)
            chunk_index += 1
            dx_chunk = _estimate_css_distance_direction(
                hz_rows,
                hx_rows,
                n_cols,
                steps=chunk_steps,
                wmin=wmin,
                rng=rng,
                dist_m4ri_cmd=dist_m4ri_cmd,
                estimator=estimator,
                timings=timings,
            )
            steps_x += chunk_steps
            d_x_best = min(d_x_best, int(dx_chunk["d_ub"]))
            chunk_record: Dict[str, object] = {
                "chunk": chunk_index,
                "steps": chunk_steps,
                "dx": dx_chunk,
                "d_x_best": d_x_best,
            }
            if d_x_best <= must_exceed:
                aborted = True
                abort_reason = "dx<=must_exceed"
                chunk_record["abort_reason"] = abort_reason
                refine_chunks.append(chunk_record)
                if on_chunk is not None:
                    on_chunk(
                        chunk_index,
                        steps_x,
                        steps_z,
                        d_x_best,
                        d_z_best,
                        abort_reason,
                    )
                break
            dz_chunk = _estimate_css_distance_direction(
                hx_rows,
                hz_rows,
                n_cols,
                steps=chunk_steps,
                wmin=wmin,
                rng=rng,
                dist_m4ri_cmd=dist_m4ri_cmd,
                estimator=estimator,
                timings=timings,
            )
            steps_z += chunk_steps
            d_z_best = min(d_z_best, int(dz_chunk["d_ub"]))
            chunk_record["dz"] = dz_chunk
            chunk_record["d_z_best"] = d_z_best
            if d_z_best <= must_exceed:
                aborted = True
                abort_reason = "dz<=must_exceed"
                chunk_record["abort_reason"] = abort_reason
            refine_chunks.append(chunk_record)
            if on_chunk is not None:
                on_chunk(
                    chunk_index,
                    steps_x,
                    steps_z,
                    d_x_best,
                    d_z_best,
                    abort_reason,
                )
            if aborted:
                break
            steps_used += chunk_steps

    return ProgressiveDistanceResult(
        passed_fast=passed_fast,
        d_x_best=d_x_best,
        d_z_best=d_z_best,
        steps_x=steps_x,
        steps_z=steps_z,
        fast={"dx": dx_fast, "dz": dz_fast},
        refine_chunks=refine_chunks,
        ran_refine=ran_refine,
        aborted=aborted,
        abort_reason=abort_reason,
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


def _print_classical_summary(
    *,
    side: str,
    total: int,
    kept: int,
    hist: Dict[Tuple[int, int], int],
) -> None:
    print(
        f"[classical] summary {side} settings={total} kept={kept} "
        f"hist_pairs={len(hist)}"
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
    classical_backend: str,
    classical_exhaustive_k_max: int,
    classical_sample_count: int,
    dist_m4ri_cmd: str,
    seed_base: Optional[int] = None,
    rng: Optional[random.Random] = None,
    out_path: Path,
    classical_workers: int = 1,
    classical_jobs: int = 1,
    lookup: Optional[Dict[Tuple[Tuple[int, ...], int], Dict[str, object]]] = None,
    canonicalize: Optional[Callable[[Sequence[int]], Tuple[int, ...]]] = None,
    progress_every: int = 500,
    progress_seconds: float = 5.0,
    timings: Optional[Dict[str, float]] = None,
    classical_eval_fn: Optional[Callable[..., Dict[str, object]]] = None,
) -> Tuple[List[ProgressiveSetting], Dict[Tuple[int, int], int], int]:
    if classical_target <= 0:
        raise ValueError("--classical-target must be positive.")
    if lookup is not None and side != "A":
        raise ValueError("lookup can only be populated for side 'A'.")
    if classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")
    if classical_jobs <= 0:
        raise ValueError("--classical-jobs must be positive.")
    if seed_base is None:
        if rng is None:
            raise ValueError("seed_base or rng is required for classical seeding.")
        seed_base = rng.randrange(1 << 31)
    if classical_eval_fn is None:
        classical_eval_fn = _classical_eval
    wmin = max(0, classical_target - 1)
    kept: List[ProgressiveSetting] = []
    hist: Dict[Tuple[int, int], int] = {}
    total_settings = len(multisets) * len(variant_codes)
    processed = 0
    calls = 0
    kept_count = 0
    exact_evals = 0
    sampled_evals = 0
    codewords_checked = 0
    start_time = time.monotonic()
    last_report_time = start_time
    last_report_processed = 0

    def _report(force: bool = False) -> None:
        nonlocal last_report_time, last_report_processed
        if total_settings == 0:
            return
        if force and processed == last_report_processed:
            return
        now = time.monotonic()
        if (
            not force
            and processed - last_report_processed < progress_every
            and now - last_report_time < progress_seconds
        ):
            return
        pct = 100.0 * processed / total_settings
        elapsed = _format_elapsed(now - start_time)
        elapsed_seconds = now - start_time
        avg_settings = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        line = (
            f"[classical] {side} {processed}/{total_settings} "
            f"({pct:.1f}%) kept={kept_count} calls={calls} elapsed={elapsed} "
            f"avg_settings_per_sec={avg_settings:.2f}"
        )
        if classical_backend == "fast":
            avg_codewords = (
                codewords_checked / elapsed_seconds if elapsed_seconds > 0 else 0.0
            )
            line += (
                f" avg_codewords_checked_per_sec={avg_codewords:.2f} "
                f"exact={exact_evals} sampled={sampled_evals}"
            )
        print(
            line
        )
        last_report_time = now
        last_report_processed = processed

    if side == "A":
        h_label = "H"
        g_label = "G"
        slice_h_key = "A_H"
        slice_g_key = "A_G"
    elif side == "B":
        h_label = "Hp"
        g_label = "Gp"
        slice_h_key = "B_H"
        slice_g_key = "B_G"
    else:
        raise ValueError("side must be 'A' or 'B'.")

    task_chunk_size = 200
    batch_size = max(task_chunk_size, task_chunk_size * 4)
    fail_stats = {"H": {"fails": 0, "total": 0}, "G": {"fails": 0, "total": 0}}

    def _pick_first_label() -> str:
        h_total = fail_stats["H"]["total"]
        g_total = fail_stats["G"]["total"]
        h_rate = fail_stats["H"]["fails"] / h_total if h_total else 0.0
        g_rate = fail_stats["G"]["fails"] / g_total if g_total else 0.0
        return "H" if h_rate >= g_rate else "G"

    def _update_fail_stats(eval_entry: Dict[str, object], label: str) -> None:
        if eval_entry.get("skipped"):
            return
        fail_stats[label]["total"] += 1
        witness = eval_entry.get("d_witness")
        if witness is not None and int(witness) <= wmin:
            fail_stats[label]["fails"] += 1

    def _slice_payload(
        eval_entry: Dict[str, object],
        n_cols: int,
    ) -> Dict[str, object]:
        if eval_entry.get("skipped"):
            return {
                "n": n_cols,
                "k": None,
                "d_witness": None,
                "d_ub": None,
                "exact": False,
                "backend": eval_entry.get("backend"),
                "skipped": True,
            }
        k_val = eval_entry.get("k")
        d_est = eval_entry.get("d_est")
        return {
            "n": n_cols,
            "k": None if k_val is None else int(k_val),
            "d_witness": eval_entry.get("d_witness"),
            "d_ub": None if d_est is None else int(d_est),
            "exact": bool(eval_entry.get("exact")),
            "backend": eval_entry.get("backend"),
        }

    executor: Optional[ProcessPoolExecutor] = None
    if classical_workers > 1:
        ctx = mp.get_context("spawn")
        try:
            executor = ProcessPoolExecutor(
                max_workers=classical_workers, mp_context=ctx
            )
        except (NotImplementedError, PermissionError, OSError) as exc:
            print(
                "[classical] parallel workers unavailable; "
                f"falling back to serial ({exc})"
            )
            executor = None

    def _consume_batch(
        batch: List[Tuple[List[int], int]],
        out_file,
    ) -> None:
        nonlocal calls, processed, kept_count, exact_evals, sampled_evals
        nonlocal codewords_checked
        if not batch:
            return
        first_label = _pick_first_label()
        chunks = [
            batch[i : i + task_chunk_size]
            for i in range(0, len(batch), task_chunk_size)
        ]
        results_by_idx: Dict[int, Tuple[List[ClassicalEvalResult], Dict[str, float], int]] = {}
        if executor is None:
            for idx, chunk in enumerate(chunks):
                results_by_idx[idx] = _evaluate_classical_chunk(
                    chunk,
                    side=side,
                    group=group,
                    variant_codes=variant_codes,
                    base_code=base_code,
                    steps=steps,
                    wmin=wmin,
                    backend=classical_backend,
                    exhaustive_k_max=classical_exhaustive_k_max,
                    sample_count=classical_sample_count,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                    first_label=first_label,
                    timings_enabled=timings is not None,
                    classical_jobs=classical_jobs,
                    seed_base=seed_base,
                    eval_fn=classical_eval_fn,
                )
        else:
            futures = {
                idx: executor.submit(
                    _evaluate_classical_chunk,
                    chunk,
                    side=side,
                    group=group,
                    variant_codes=variant_codes,
                    base_code=base_code,
                    steps=steps,
                    wmin=wmin,
                    backend=classical_backend,
                    exhaustive_k_max=classical_exhaustive_k_max,
                    sample_count=classical_sample_count,
                    dist_m4ri_cmd=dist_m4ri_cmd,
                    first_label=first_label,
                    timings_enabled=timings is not None,
                    classical_jobs=classical_jobs,
                    seed_base=seed_base,
                    eval_fn=classical_eval_fn,
                )
                for idx, chunk in enumerate(chunks)
            }
            for idx, future in futures.items():
                results_by_idx[idx] = future.result()

        for idx in range(len(chunks)):
            chunk_results, chunk_timings, chunk_calls = results_by_idx[idx]
            calls += chunk_calls
            _merge_timings(timings, chunk_timings)
            for result in chunk_results:
                _update_fail_stats(result.h_eval, "H")
                _update_fail_stats(result.g_eval, "G")
                if classical_backend == "fast":
                    for eval_entry in (result.h_eval, result.g_eval):
                        if eval_entry.get("skipped"):
                            continue
                        if eval_entry.get("exact"):
                            exact_evals += 1
                        else:
                            sampled_evals += 1
                        checked = eval_entry.get("codewords_checked")
                        if checked is not None:
                            codewords_checked += int(checked)

                bookkeeping_start = time.perf_counter()
                elements = result.elements
                elements_repr = [group.repr(x) for x in elements]
                k_min = int(result.k_min)
                d_min = int(result.d_min)
                hist[(k_min, d_min)] = hist.get((k_min, d_min), 0) + 1
                processed += 1
                slice_codes = {
                    slice_h_key: _slice_payload(result.h_eval, result.n_cols),
                    slice_g_key: _slice_payload(result.g_eval, result.n_cols),
                }
                passes = bool(result.passes)
                if lookup is not None:
                    lookup_key = (
                        _canonical_multiset_key(
                            elements, canonicalize=canonicalize
                        ),
                        result.perm_idx,
                    )
                    lookup[lookup_key] = {
                        "k_min": k_min,
                        "d_min": d_min,
                        "slice_n": result.n_cols,
                        "slice_codes": dict(slice_codes),
                        "h_eval": dict(result.h_eval),
                        "g_eval": dict(result.g_eval),
                        "passes": passes,
                    }
                record = {
                    "side": side,
                    "group": {"spec": group.name, "order": group.order},
                    "elements": list(elements),
                    "elements_repr": elements_repr,
                    "perm_idx": result.perm_idx,
                    "k_classical": k_min,
                    "d_est": d_min,
                    "slice_n": result.n_cols,
                    "slice_codes": slice_codes,
                    "checks": {
                        h_label: result.h_eval,
                        g_label: result.g_eval,
                    },
                    "passes": passes,
                }
                if passes:
                    setting_id = _progressive_setting_id(
                        side, elements, result.perm_idx
                    )
                    record["id"] = setting_id
                    kept.append(
                        ProgressiveSetting(
                            side=side,
                            elements=list(elements),
                            elements_repr=elements_repr,
                            perm_idx=result.perm_idx,
                            k_classical=k_min,
                            d_est=d_min,
                            record=record,
                            setting_id=setting_id,
                        )
                    )
                    kept_count += 1
                    out_file.write(json.dumps(record, sort_keys=True) + "\n")
                _add_timing(
                    timings, "bookkeeping", time.perf_counter() - bookkeeping_start
                )
                _report()

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            batch: List[Tuple[List[int], int]] = []
            for elements in multisets:
                for perm_idx in range(len(variant_codes)):
                    batch.append((list(elements), perm_idx))
                    if len(batch) >= batch_size:
                        _consume_batch(batch, out_file)
                        batch = []
            if batch:
                _consume_batch(batch, out_file)
            out_file.flush()
    finally:
        if executor is not None:
            executor.shutdown()
    _report(force=True)
    return kept, hist, processed


def _precompute_classical_side_from_lookup(
    *,
    side: str,
    group: FiniteGroup,
    multisets: List[List[int]],
    variant_codes: List[LocalCode],
    lookup: Dict[Tuple[Tuple[int, ...], int], Dict[str, object]],
    classical_backend: str,
    out_path: Path,
    canonicalize: Optional[Callable[[Sequence[int]], Tuple[int, ...]]] = None,
    progress_every: int = 500,
    progress_seconds: float = 5.0,
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[List[ProgressiveSetting], Dict[Tuple[int, int], int], int]:
    if side != "B":
        raise ValueError("lookup reuse is only supported for side 'B'.")
    kept: List[ProgressiveSetting] = []
    hist: Dict[Tuple[int, int], int] = {}
    total_settings = len(multisets) * len(variant_codes)
    processed = 0
    kept_count = 0
    calls = 0
    exact_evals = 0
    sampled_evals = 0
    codewords_checked = 0
    start_time = time.monotonic()
    last_report_time = start_time
    last_report_processed = 0

    def _report(force: bool = False) -> None:
        nonlocal last_report_time, last_report_processed
        if total_settings == 0:
            return
        if force and processed == last_report_processed:
            return
        now = time.monotonic()
        if (
            not force
            and processed - last_report_processed < progress_every
            and now - last_report_time < progress_seconds
        ):
            return
        pct = 100.0 * processed / total_settings
        elapsed = _format_elapsed(now - start_time)
        elapsed_seconds = now - start_time
        avg_settings = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        line = (
            f"[classical] {side} {processed}/{total_settings} "
            f"({pct:.1f}%) kept={kept_count} calls={calls} elapsed={elapsed} "
            f"avg_settings_per_sec={avg_settings:.2f}"
        )
        if classical_backend == "fast":
            avg_codewords = (
                codewords_checked / elapsed_seconds if elapsed_seconds > 0 else 0.0
            )
            line += (
                f" avg_codewords_checked_per_sec={avg_codewords:.2f} "
                f"exact={exact_evals} sampled={sampled_evals}"
            )
        print(
            line
        )
        last_report_time = now
        last_report_processed = processed

    with out_path.open("w", encoding="utf-8") as out_file:
        for elements in multisets:
            elements_repr = [group.repr(x) for x in elements]
            inv_key = _inverse_multiset_key(
                group, elements, canonicalize=canonicalize
            )
            for perm_idx, _ in enumerate(variant_codes):
                lookup_key = (inv_key, perm_idx)
                params = lookup.get(lookup_key)
                if params is None:
                    raise KeyError(
                        f"Missing A lookup for multiset={inv_key} perm={perm_idx}."
                    )
                k_min = int(params["k_min"])
                d_min = int(params["d_min"])
                slice_n = int(params["slice_n"])
                h_eval = dict(params["h_eval"])
                g_eval = dict(params["g_eval"])
                if classical_backend == "fast":
                    for eval_entry in (h_eval, g_eval):
                        if eval_entry.get("skipped"):
                            continue
                        if eval_entry.get("exact"):
                            exact_evals += 1
                        else:
                            sampled_evals += 1
                        checked = eval_entry.get("codewords_checked")
                        if checked is not None:
                            codewords_checked += int(checked)
                passes = bool(params["passes"])
                slice_codes = params["slice_codes"]
                slice_codes_b = {
                    "B_G": dict(slice_codes["A_G"]),
                    "B_H": dict(slice_codes["A_H"]),
                }
                bookkeeping_start = time.perf_counter()
                hist[(k_min, d_min)] = hist.get((k_min, d_min), 0) + 1
                processed += 1
                record = {
                    "side": side,
                    "group": {"spec": group.name, "order": group.order},
                    "elements": list(elements),
                    "elements_repr": elements_repr,
                    "perm_idx": perm_idx,
                    "k_classical": k_min,
                    "d_est": d_min,
                    "slice_n": slice_n,
                    "slice_codes": slice_codes_b,
                    "checks": {
                        "Hp": h_eval,
                        "Gp": g_eval,
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
                    kept_count += 1
                    out_file.write(json.dumps(record, sort_keys=True) + "\n")
                _add_timing(
                    timings, "bookkeeping", time.perf_counter() - bookkeeping_start
                )
                _report()
        out_file.flush()
    _report(force=True)
    return kept, hist, processed


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
    timings: Optional[Dict[str, float]] = None,
) -> None:
    code_dir = outdir / "best_codes" / code_id
    code_dir.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    write_mtx_from_bitrows(str(code_dir / "Hx.mtx"), list(hx_rows), n_cols)
    write_mtx_from_bitrows(str(code_dir / "Hz.mtx"), list(hz_rows), n_cols)
    _add_timing(timings, "write_mtx", time.perf_counter() - write_start)
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
        "--classical-distance-backend",
        type=str,
        choices=("dist-m4ri", "fast"),
        default="fast",
        help="Backend for classical slice distance estimation.",
    )
    parser.add_argument(
        "--classical-exhaustive-k-max",
        type=int,
        default=8,
        help="Exact enumeration cutoff for the fast classical backend.",
    )
    parser.add_argument(
        "--classical-sample-count",
        type=int,
        default=256,
        help="Random samples for the fast classical backend.",
    )
    parser.add_argument(
        "--classical-workers",
        type=int,
        default=1,
        help="Process count for classical precompute (default: 1).",
    )
    parser.add_argument(
        "--classical-jobs",
        type=int,
        default=0,
        help="Thread count for per-setting classical slice eval (0=auto).",
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
        default=None,
        help=(
            "dist_m4ri RW steps for slow quantum distance estimates "
            "(default: adaptive based on n)."
        ),
    )
    parser.add_argument(
        "--quantum-refine-chunk",
        type=int,
        default=20000,
        help="dist_m4ri RW steps per refinement chunk.",
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
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print timing breakdowns for progressive runs.",
    )
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    if args.target_distance <= 0:
        raise ValueError("--target-distance must be positive.")
    if args.classical_steps <= 0:
        raise ValueError("--classical-steps must be positive.")
    if args.classical_exhaustive_k_max < 0:
        raise ValueError("--classical-exhaustive-k-max must be nonnegative.")
    if args.classical_sample_count <= 0:
        raise ValueError("--classical-sample-count must be positive.")
    if args.classical_workers <= 0:
        raise ValueError("--classical-workers must be positive.")
    if args.classical_jobs < 0:
        raise ValueError("--classical-jobs must be nonnegative.")
    if args.quantum_steps is not None:
        if args.quantum_steps <= 0:
            raise ValueError("--quantum-steps must be positive.")
        if _argv_has_flag(raw_argv, "--quantum-steps-fast") or _argv_has_flag(
            raw_argv, "--quantum-steps-slow"
        ):
            raise ValueError(
                "--quantum-steps is deprecated; do not mix with "
                "--quantum-steps-fast/--quantum-steps-slow."
            )
        args.quantum_steps_fast = args.quantum_steps
        args.quantum_steps_slow = args.quantum_steps
        quantum_steps_slow_set = True
    else:
        quantum_steps_slow_set = _argv_has_flag(raw_argv, "--quantum-steps-slow")
    if args.quantum_steps_fast <= 0:
        raise ValueError("--quantum-steps-fast must be positive.")
    if args.quantum_steps_slow is not None and args.quantum_steps_slow <= 0:
        raise ValueError("--quantum-steps-slow must be positive.")
    if args.quantum_refine_chunk <= 0:
        raise ValueError("--quantum-refine-chunk must be positive.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be positive.")
    if args.max_quantum_evals < 0:
        raise ValueError("--max-quantum-evals must be nonnegative.")
    if args.min_distinct is not None and args.min_distinct <= 0:
        raise ValueError("--min-distinct must be positive.")
    if args.min_distinct is not None and args.min_distinct > SIDE_LEN:
        raise ValueError("--min-distinct cannot exceed 6.")

    if args.classical_jobs == 0:
        cpu_count = os.cpu_count() or 1
        if args.classical_distance_backend == "dist-m4ri":
            args.classical_jobs = min(4, cpu_count)
        else:
            args.classical_jobs = 1

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

    timings_enabled = bool(args.timings)
    run_timings: Optional[Dict[str, float]] = {} if timings_enabled else None
    classical_timings: Optional[Dict[str, float]] = (
        {} if timings_enabled else None
    )
    run_start = time.monotonic() if timings_enabled else None
    classical_start: Optional[float] = None
    classical_finalized = False

    def _finalize_classical_timings() -> None:
        nonlocal classical_finalized
        if not timings_enabled or classical_finalized:
            return
        wall = None
        if classical_start is not None:
            wall = time.monotonic() - classical_start
        _print_timing_summary(
            "classical_precompute", classical_timings, wall_seconds=wall
        )
        _merge_timings(run_timings, classical_timings)
        classical_finalized = True

    def _finalize_run_timings() -> None:
        if not timings_enabled:
            return
        _finalize_classical_timings()
        wall = None
        if run_start is not None:
            wall = time.monotonic() - run_start
        _print_timing_summary(
            "run_total", run_timings, wall_seconds=wall
        )

    group_tag = _safe_id_component(group_spec)
    if args.results_dir:
        outdir = Path(args.results_dir)
    else:
        outdir = Path("results") / f"progressive_{group_tag}_{_timestamp_utc()}"
    outdir.mkdir(parents=True, exist_ok=True)

    group_cache_dir = outdir / "groups"
    aut_cache_dir = outdir / "cache" / "aut"
    group = group_from_spec(group_spec, gap_cmd=args.gap_cmd, cache_dir=group_cache_dir)

    base_code = hamming_6_3_3_shortened()
    variant_codes = _build_variant_codes(base_code)
    perm_total = len(variant_codes)
    local_codes_match = _local_code_specs_match(
        base_code, variant_codes, base_code, variant_codes
    )

    n_slice = SIDE_LEN * group.order
    classical_target = args.classical_target
    if classical_target is None:
        classical_target = _ceil_sqrt(n_slice)
    if not quantum_steps_slow_set:
        n_cols_default = base_code.n * base_code.n * group.order
        args.quantum_steps_slow = _adaptive_quantum_steps_slow(n_cols_default)
    if args.quantum_steps_slow is None or args.quantum_steps_slow <= 0:
        raise ValueError("--quantum-steps-slow must be positive.")

    run_meta = {
        "mode": "progressive",
        "group": {"spec": group.name, "order": group.order},
        "target_distance": args.target_distance,
        "classical_target": classical_target,
        "classical_steps": args.classical_steps,
        "classical_distance_backend": args.classical_distance_backend,
        "classical_exhaustive_k_max": args.classical_exhaustive_k_max,
        "classical_sample_count": args.classical_sample_count,
        "classical_workers": args.classical_workers,
        "classical_jobs": args.classical_jobs,
        "quantum_steps_fast": args.quantum_steps_fast,
        "quantum_steps_slow": args.quantum_steps_slow,
        "quantum_refine_chunk": args.quantum_refine_chunk,
        "report_every": args.report_every,
        "max_quantum_evals": args.max_quantum_evals,
        "seed": seed,
        "dist_m4ri_cmd": args.dist_m4ri_cmd,
        "dedup_cayley": bool(args.dedup_cayley),
        "min_distinct": args.min_distinct,
        "timings": bool(args.timings),
    }
    (outdir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    if timings_enabled:
        classical_start = time.monotonic()
        gen_start = time.perf_counter()

    multisets = _enumerate_multisets_with_identity(group.order)
    raw_multiset_count = len(multisets)
    if args.min_distinct is not None:
        multisets = _filter_min_distinct(multisets, args.min_distinct)
    after_distinct_count = len(multisets)
    automorphisms = None
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
    if timings_enabled:
        _add_timing(
            classical_timings,
            "generate_multisets_perms",
            time.perf_counter() - gen_start,
        )
    print(
        f"[progressive] multisets enumerated={raw_multiset_count} "
        f"after_min_distinct={after_distinct_count} "
        f"after_dedup={len(multisets)}"
    )
    if not multisets:
        print("[progressive] no multisets to score after filtering; exiting.")
        _finalize_run_timings()
        return 0

    total_settings = len(multisets) * perm_total
    print(
        f"[progressive] perms={perm_total} settings_per_side={total_settings}"
    )

    canonicalize = None
    if automorphisms is not None:
        canonicalize = lambda mult: canonical_multiset(  # noqa: E731
            group, mult, automorphisms=automorphisms
        )

    use_abelian_opt = bool(group.is_abelian and local_codes_match)
    a_lookup = {} if use_abelian_opt else None

    a_keep, a_hist, a_total = _precompute_classical_side(
        side="A",
        group=group,
        multisets=multisets,
        variant_codes=variant_codes,
        base_code=base_code,
        steps=args.classical_steps,
        classical_target=classical_target,
        classical_backend=args.classical_distance_backend,
        classical_exhaustive_k_max=args.classical_exhaustive_k_max,
        classical_sample_count=args.classical_sample_count,
        dist_m4ri_cmd=args.dist_m4ri_cmd,
        seed_base=seed,
        out_path=outdir / "classical_A_kept.jsonl",
        classical_workers=args.classical_workers,
        classical_jobs=args.classical_jobs,
        lookup=a_lookup,
        canonicalize=canonicalize,
        timings=classical_timings,
    )
    _print_histogram_top(side="A", hist=a_hist)
    _write_histogram(
        path=outdir / "classical_A_histogram.json",
        side="A",
        group=group,
        hist=a_hist,
        classical_target=classical_target,
    )
    _print_classical_summary(
        side="A",
        total=a_total,
        kept=len(a_keep),
        hist=a_hist,
    )
    print(
        f"[progressive] A settings={a_total} kept={len(a_keep)}"
    )

    if use_abelian_opt:
        print(
            "[classical] abelian optimization: skipping B dist-m4ri calls; "
            "reusing A via inversion mapping"
        )
        if a_lookup is None:
            raise RuntimeError("A lookup is missing for abelian optimization.")
        b_keep, b_hist, b_total = _precompute_classical_side_from_lookup(
            side="B",
            group=group,
            multisets=multisets,
            variant_codes=variant_codes,
            lookup=a_lookup,
            classical_backend=args.classical_distance_backend,
            out_path=outdir / "classical_B_kept.jsonl",
            canonicalize=canonicalize,
            timings=classical_timings,
        )
    else:
        b_keep, b_hist, b_total = _precompute_classical_side(
            side="B",
            group=group,
            multisets=multisets,
            variant_codes=variant_codes,
            base_code=base_code,
            steps=args.classical_steps,
            classical_target=classical_target,
            classical_backend=args.classical_distance_backend,
            classical_exhaustive_k_max=args.classical_exhaustive_k_max,
            classical_sample_count=args.classical_sample_count,
            dist_m4ri_cmd=args.dist_m4ri_cmd,
            seed_base=seed,
            out_path=outdir / "classical_B_kept.jsonl",
            classical_workers=args.classical_workers,
            classical_jobs=args.classical_jobs,
            timings=classical_timings,
        )
    _print_histogram_top(side="B", hist=b_hist)
    _write_histogram(
        path=outdir / "classical_B_histogram.json",
        side="B",
        group=group,
        hist=b_hist,
        classical_target=classical_target,
    )
    _print_classical_summary(
        side="B",
        total=b_total,
        kept=len(b_keep),
        hist=b_hist,
    )
    print(
        f"[progressive] B settings={b_total} kept={len(b_keep)}"
    )

    _finalize_classical_timings()

    if not a_keep or not b_keep:
        print("[progressive] no settings passed classical filter; exiting.")
        _finalize_run_timings()
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

    interrupted = False
    try:
        for a_setting, b_setting in _iter_progressive_pairs(a_rounds, b_rounds):
            pairs_seen += 1
            permA = a_setting.perm_idx
            permB = b_setting.perm_idx
            C1 = variant_codes[permA]
            C1p = variant_codes[permB]
            build_start = time.perf_counter()
            hx_rows, hz_rows, n_cols = build_hx_hz(
                group,
                a_setting.elements,
                b_setting.elements,
                base_code,
                C1,
                base_code,
                C1p,
            )
            _add_timing(
                run_timings, "build_slices", time.perf_counter() - build_start
            )
            if not css_commutes(hx_rows, hz_rows):
                raise RuntimeError(
                    f"HX/HZ do not commute for A={a_setting.setting_id} "
                    f"B={b_setting.setting_id}."
                )
            rank_start = time.perf_counter()
            rank_hx = gf2_rank(hx_rows[:], n_cols)
            rank_hz = gf2_rank(hz_rows[:], n_cols)
            _add_timing(
                run_timings, "rank", time.perf_counter() - rank_start
            )
            k_val = n_cols - rank_hx - rank_hz
            if k_val <= 0:
                continue

            if args.max_quantum_evals and eval_index >= args.max_quantum_evals:
                break
            eval_index += 1

            current = best_by_k.get(k_val)
            best = 0 if current is None else int(current["d_ub"])
            must_exceed = max(best, args.target_distance - 1)

            def _log_refine(
                chunk_index: int,
                steps_x: int,
                steps_z: int,
                d_x_best: int,
                d_z_best: int,
                abort_reason: Optional[str],
            ) -> None:
                note = f" abort={abort_reason}" if abort_reason else ""
                print(
                    f"[refine] eval={eval_index} k={k_val} chunk={chunk_index} "
                    f"steps_x={steps_x} steps_z={steps_z} "
                    f"dX_best={d_x_best} dZ_best={d_z_best}{note}"
                )

            distance_result = _progressive_css_distance(
                hx_rows,
                hz_rows,
                n_cols,
                must_exceed=must_exceed,
                steps_fast=args.quantum_steps_fast,
                steps_slow=args.quantum_steps_slow,
                refine_chunk=args.quantum_refine_chunk,
                rng=rng,
                dist_m4ri_cmd=args.dist_m4ri_cmd,
                on_chunk=_log_refine,
                timings=run_timings,
            )
            d_x_best = distance_result.d_x_best
            d_z_best = distance_result.d_z_best
            steps_used = distance_result.steps_x + distance_result.steps_z
            d_final_ub = min(d_x_best, d_z_best)
            is_new_best = False
            decision = "reject"
            if distance_result.passed_fast:
                if distance_result.aborted:
                    decision = "refine"
                else:
                    is_new_best = (
                        d_final_ub > best
                        and d_final_ub >= args.target_distance
                    )
                    decision = "new_best" if is_new_best else "refine"
            print(
                f"[eval] eval={eval_index} n={n_cols} k={k_val} best={best} "
                f"target_distance={args.target_distance} "
                f"must_exceed={must_exceed} steps_used={steps_used} "
                f"dX_best={d_x_best} dZ_best={d_z_best} decision={decision}"
            )
            if not is_new_best:
                continue

            bookkeeping_start = time.perf_counter()
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
                    "method": "dist_m4ri_rw_progressive",
                    "wmin": must_exceed,
                    "steps_fast": args.quantum_steps_fast,
                    "steps_slow": args.quantum_steps_slow,
                    "refine_chunk": args.quantum_refine_chunk,
                    "fast": distance_result.fast,
                    "refine_chunks": distance_result.refine_chunks,
                    "steps_used_x": distance_result.steps_x,
                    "steps_used_z": distance_result.steps_z,
                    "steps_used_total": steps_used,
                    "dX_best": d_x_best,
                    "dZ_best": d_z_best,
                    "d_ub": d_final_ub,
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
                timings=run_timings,
            )
            best_by_k[k_val] = {
                "d_ub": d_final_ub,
                "steps": max(distance_result.steps_x, distance_result.steps_z),
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
                f"k={k_val} dX_best={d_x_best} dZ_best={d_z_best} "
                f"d_ub={d_final_ub} steps_used={steps_used} "
                f"A={a_setting.setting_id} B={b_setting.setting_id}"
            )
            if classical_slices:
                print("classical slices:")
                for key in ("A_H", "A_G", "B_G", "B_H"):
                    entry = classical_slices.get(key)
                    if entry is None:
                        continue
                    backend = entry.get("backend", "?")
                    exact = entry.get("exact")
                    mode = "exact" if exact else "sampled"
                    print(
                        f"  {key}: n={entry['n']} k={entry['k']} "
                        f"d_ub={entry['d_ub']} backend={backend} mode={mode}"
                    )
            _add_timing(
                run_timings,
                "bookkeeping",
                time.perf_counter() - bookkeeping_start,
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
    _finalize_run_timings()
    return 0


__all__ = [
    "ProgressiveSetting",
    "_enumerate_multisets_with_identity",
    "_interleaved_rounds",
    "_iter_progressive_pairs",
    "progressive_main",
]
