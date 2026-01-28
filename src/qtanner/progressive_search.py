"""Progressive exhaustive classical-first search mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
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
MIN_SLOW_TRIALS = 50_000


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
    slow_skipped: bool
    slow_skip_reason: Optional[str]
    slow_steps_target: Optional[int]


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


@dataclass(frozen=True)
class BestCodesEntry:
    n: int
    k: int
    d: Optional[int]
    m4ri_trials: Optional[int]
    code_id: str


@dataclass(frozen=True)
class SlowDistanceDecision:
    run_slow: bool
    steps_slow: int
    reason: str
    best_d: Optional[int] = None
    best_trials: Optional[int] = None
    best_code_id: Optional[str] = None


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


def _is_promising_code(n: int, k: int, d: int) -> bool:
    if n <= 0 or k <= 0 or d <= 0:
        return False
    if d < math.sqrt(n):
        return False
    if k * d < n:
        return False
    return True


def _adaptive_quantum_steps_slow(n_cols: int) -> int:
    if n_cols <= 200:
        return 50000
    if n_cols <= 600:
        return 200000
    return 500000


def compute_slow_trials(best_trials: Optional[int], override: Optional[int]) -> int:
    if override is not None:
        return max(MIN_SLOW_TRIALS, int(override))
    if best_trials is not None:
        return max(MIN_SLOW_TRIALS, int(best_trials))
    return MIN_SLOW_TRIALS


def should_abort_refine(d_x_best: int, d_z_best: int, best_d: Optional[int]) -> bool:
    if best_d is None:
        return False
    return min(int(d_x_best), int(d_z_best)) <= int(best_d)


def _to_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _get_first_int(obj: Dict[str, object], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        if key in obj:
            value = _to_int(obj.get(key))
            if value is not None:
                return value
    return None


def _parse_best_codes_json(text: str) -> List[Dict[str, object]]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    codes = data.get("codes")
    if not isinstance(codes, list):
        return []
    out: List[Dict[str, object]] = []
    for entry in codes:
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _parse_best_codes_index_tsv(text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    reader = csv.reader(text.splitlines(), delimiter="\t")
    headers = None
    for row in reader:
        if not row:
            continue
        if headers is None:
            headers = [h.strip() for h in row]
            continue
        if headers:
            item = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
            rows.append(item)
    return rows


def _select_best_codes_by_nk(records: Sequence[Dict[str, object]]) -> Dict[Tuple[int, int], BestCodesEntry]:
    grouped: Dict[Tuple[int, int], List[BestCodesEntry]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        n = _get_first_int(record, ["n"])
        k = _get_first_int(record, ["k"])
        if n is None or k is None:
            continue
        d = _get_first_int(record, ["d_ub", "d_recorded", "d_obs", "d"])
        trials = _get_first_int(record, ["m4ri_trials", "trials", "steps_used_total", "steps"])
        code_id = str(record.get("code_id") or record.get("id") or "")
        entry = BestCodesEntry(
            n=int(n),
            k=int(k),
            d=d if d is not None else None,
            m4ri_trials=trials if trials is not None else None,
            code_id=code_id,
        )
        grouped.setdefault((entry.n, entry.k), []).append(entry)

    selected: Dict[Tuple[int, int], BestCodesEntry] = {}
    for key, entries in grouped.items():
        max_trials = max(
            (e.m4ri_trials for e in entries if isinstance(e.m4ri_trials, int)),
            default=None,
        )
        if max_trials is None:
            candidates = list(entries)
        else:
            candidates = [e for e in entries if e.m4ri_trials == max_trials]
        if not candidates:
            continue
        candidates.sort(
            key=lambda e: (
                -(e.d if isinstance(e.d, int) else -1),
                e.code_id or "",
            )
        )
        selected[key] = candidates[0]
    return selected


def _git_run(repo_root: Path, args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )


def _git_has_origin(repo_root: Path) -> bool:
    result = _git_run(repo_root, ["remote", "get-url", "origin"])
    return result.returncode == 0


def _git_fetch_origin(repo_root: Path) -> bool:
    result = _git_run(repo_root, ["fetch", "origin"])
    return result.returncode == 0


def _git_show_text(repo_root: Path, ref: str, path: str) -> Optional[str]:
    result = _git_run(repo_root, ["show", f"{ref}:{path}"])
    if result.returncode != 0:
        return None
    return result.stdout


def _read_working_tree(repo_root: Path, path: str) -> Optional[str]:
    try:
        full_path = repo_root / path
        if full_path.exists():
            return full_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return None


def _load_best_codes_records_from_text(text: str, path: str) -> List[Dict[str, object]]:
    if path.endswith(".json"):
        return _parse_best_codes_json(text)
    if path.endswith(".tsv"):
        return _parse_best_codes_index_tsv(text)
    return []


def load_best_codes_index(repo_root: Path, source: str = "auto") -> Dict[Tuple[int, int], BestCodesEntry]:
    records, _info = _load_best_codes_index_with_meta(repo_root, source=source)
    return records


def _load_best_codes_index_with_meta(
    repo_root: Path,
    *,
    source: str = "auto",
) -> Tuple[Dict[Tuple[int, int], BestCodesEntry], Dict[str, str]]:
    paths = ["best_codes/data.json", "best_codes/index.tsv"]
    info: Dict[str, str] = {"source": "none", "path": ""}
    source = (source or "auto").strip().lower()

    def parse_text(text: str, path: str) -> Dict[Tuple[int, int], BestCodesEntry]:
        records = _load_best_codes_records_from_text(text, path)
        return _select_best_codes_by_nk(records)

    if source in {"origin", "auto"} and _git_has_origin(repo_root):
        if _git_fetch_origin(repo_root):
            for path in paths:
                text = _git_show_text(repo_root, "origin/main", path)
                if text:
                    info["source"] = "origin"
                    info["path"] = path
                    return parse_text(text, path), info

    if source in {"working-tree", "auto", "origin"}:
        for path in paths:
            text = _read_working_tree(repo_root, path)
            if text:
                info["source"] = "working-tree"
                info["path"] = path
                return parse_text(text, path), info

    if source == "website":
        path = "best_codes/data.json"
        text = _read_working_tree(repo_root, path)
        if text:
            info["source"] = "website"
            info["path"] = path
            return parse_text(text, path), info

    return {}, info


def decide_slow_quantum_plan(
    *,
    d_fast: int,
    fast_trials: int,
    best_entry: Optional[BestCodesEntry],
    override: Optional[int] = None,
) -> SlowDistanceDecision:
    best_d = best_entry.d if best_entry else None
    best_trials = best_entry.m4ri_trials if best_entry else None
    best_code_id = best_entry.code_id if best_entry else None

    steps_slow = compute_slow_trials(best_trials, override)

    if best_d is None:
        return SlowDistanceDecision(
            run_slow=True,
            steps_slow=steps_slow,
            reason="no_best_entry",
            best_d=None,
            best_trials=best_trials,
            best_code_id=best_code_id,
        )
    if d_fast <= best_d:
        return SlowDistanceDecision(
            run_slow=False,
            steps_slow=steps_slow,
            reason="d_fast<=best",
            best_d=best_d,
            best_trials=best_trials,
            best_code_id=best_code_id,
        )
    return SlowDistanceDecision(
        run_slow=True,
        steps_slow=steps_slow,
        reason="d_fast>best",
        best_d=best_d,
        best_trials=best_trials,
        best_code_id=best_code_id,
    )


@dataclass
class BestCodesIndexCache:
    repo_root: Path
    source: str
    refresh_seconds: int
    verbose: bool = False
    index: Dict[Tuple[int, int], BestCodesEntry] = field(default_factory=dict)
    last_refresh: float = 0.0
    last_info: Dict[str, str] = field(default_factory=dict)

    def refresh(self) -> None:
        index, info = _load_best_codes_index_with_meta(
            self.repo_root,
            source=self.source,
        )
        self.index = index
        self.last_info = info
        self.last_refresh = time.monotonic()
        source_label = info.get("source", "unknown")
        path_label = info.get("path", "")
        if self.source in {"origin", "auto"} and source_label != "origin":
            print(
                f"[best_codes] WARNING: origin/main unavailable; using "
                f"{source_label}:{path_label or '(unknown)'}"
            )
        if source_label == "origin":
            print(
                f"[best_codes] loaded {len(index)} entries from "
                f"origin/main:{path_label or '(unknown)'}"
            )
        elif self.verbose:
            print(
                f"[best_codes] loaded {len(index)} entries from "
                f"{source_label}:{path_label or '(unknown)'}"
            )

    def get_index(self) -> Dict[Tuple[int, int], BestCodesEntry]:
        now = time.monotonic()
        if (
            self.index
            and self.refresh_seconds > 0
            and now - self.last_refresh < self.refresh_seconds
        ):
            return self.index
        self.refresh()
        return self.index

    def info(self) -> Dict[str, str]:
        return dict(self.last_info)


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
    slow_decider: Optional[
        Callable[[int, int, int, int], SlowDistanceDecision]
    ] = None,
    refine_abort_check: Optional[
        Callable[[int, int], Optional[str]]
    ] = None,
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
    slow_skipped = False
    slow_skip_reason: Optional[str] = None
    slow_steps_target: Optional[int] = steps_slow

    if slow_decider is not None:
        decision = slow_decider(d_x_best, d_z_best, steps_fast, steps_slow)
        slow_steps_target = int(decision.steps_slow)
        steps_slow = int(decision.steps_slow)
        if not decision.run_slow:
            slow_skipped = True
            slow_skip_reason = decision.reason

    if passed_fast and not slow_skipped and steps_slow > steps_fast:
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
            if refine_abort_check is not None:
                maybe_abort = refine_abort_check(d_x_best, d_z_best)
                if maybe_abort:
                    aborted = True
                    abort_reason = maybe_abort
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
            elif refine_abort_check is not None:
                maybe_abort = refine_abort_check(d_x_best, d_z_best)
                if maybe_abort:
                    aborted = True
                    abort_reason = maybe_abort
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
        slow_skipped=slow_skipped,
        slow_skip_reason=slow_skip_reason,
        slow_steps_target=slow_steps_target,
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


def _histogram_by_k(
    hist: Dict[Tuple[int, int], int]
) -> Dict[int, Counter[int]]:
    by_k: Dict[int, Counter[int]] = defaultdict(Counter)
    for (k_val, d_val), count in hist.items():
        by_k[int(k_val)][int(d_val)] += int(count)
    return dict(by_k)



def _print_histogram_top(
    *,
    side: str,
    hist: Dict[Tuple[int, int], int],
    top_n: int = 10,
) -> None:
    entries = _histogram_payload(hist)
    print(
        f"[progressive] {side} histogram top {min(top_n, len(entries))} "
        "by frequency (k, d_est):"
    )
    if not entries:
        print("  (none)")
        return
    for entry in entries[:top_n]:
        print(
            f"  k={entry['k_classical']} d_est={entry['d_est']} count={entry['count']}"
        )


def _print_histogram_distribution(
    *,
    side: str,
    hist: Dict[Tuple[int, int], int],
) -> None:
    print(f"[progressive] {side} distribution of d_est per k (counts):")
    if not hist:
        print("  (none)")
        return
    by_k = _histogram_by_k(hist)
    for k_val in sorted(by_k):
        counter = by_k[k_val]
        if not counter:
            continue
        total = sum(counter.values())
        best_d = max(counter)
        d_items = sorted(counter.items(), key=lambda item: item[0], reverse=True)
        d_parts = " ".join(
            f"d={d_val}:{count}" for d_val, count in d_items
        )
        print(
            f"  k={k_val} total={total} best_d_est={best_d} | {d_parts}"
        )


def _print_histogram_report(
    *,
    side: str,
    hist: Dict[Tuple[int, int], int],
    top_n: int = 10,
) -> None:
    _print_histogram_top(side=side, hist=hist, top_n=top_n)
    _print_histogram_distribution(side=side, hist=hist)


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


def _relpath_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_json(obj: Dict[str, object]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _new_best_artifact_filename(artifact: Dict[str, object]) -> str:
    group = artifact.get("group")
    if isinstance(group, dict):
        group_label = group.get("spec") or group.get("name") or group.get("id")
    else:
        group_label = group
    group_label = _safe_id_component(group_label or "group")
    n = artifact.get("n")
    k = artifact.get("k")
    d_ub = artifact.get("d_ub")
    eval_index = artifact.get("eval")
    digest = hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()
    short_hash = digest[:8]
    return f"{group_label}_n{n}_k{k}_dUB{d_ub}_eval{eval_index}_{short_hash}.json"


def _write_new_best_artifact(
    *,
    save_dir: Path,
    artifact: Dict[str, object],
) -> Optional[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = _new_best_artifact_filename(artifact)
    out_path = save_dir / filename
    if out_path.exists():
        stem = out_path.stem
        suffix = out_path.suffix
        for idx in range(1, 1000):
            candidate = out_path.with_name(f"{stem}_{idx}{suffix}")
            if not candidate.exists():
                out_path = candidate
                break
        else:
            print(
                f"[progressive] new_best artifact exists, skipping: {out_path}"
            )
            return None
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, out_path)
    print(f"[progressive] new_best artifact saved: {out_path}")
    return out_path


def _maybe_save_new_best_artifact(
    *,
    decision: str,
    save_dir: Optional[Path],
    artifact: Dict[str, object],
) -> Optional[Path]:
    if decision != "new_best":
        return None
    if save_dir is None:
        print("[progressive] new_best artifact disabled (--no-save-new-bests)")
        return None
    return _write_new_best_artifact(save_dir=save_dir, artifact=artifact)


def _repo_root_for_best_codes(start: Path) -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            text=True,
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def _run_best_codes_update_after_progressive(
    *,
    outdir: Path,
    group_tag: str,
    no_git: bool,
    no_publish: bool,
    include_git_history: bool,
    max_attempts: int,
) -> bool:
    # Skip updater when an unhandled exception is bubbling up.
    if sys.exc_info()[0] is not None:
        print(
            "[best_codes] skipping update due to active exception.",
            file=sys.stderr,
        )
        return False

    repo_root = _repo_root_for_best_codes(outdir)
    history_label = "on" if include_git_history else "off"
    publish_label = "off" if no_publish else "on"
    git_label = "off" if no_git else "on"
    print(
        "[best_codes] updating best_codes "
        f"(history={history_label} publish={publish_label} git={git_label})"
    )

    try:
        # Lazy import so disabling the updater keeps the search fast.
        from .best_codes_updater import run_best_codes_update

        context = f"progressive {group_tag}"
        commit_message = (
            f"best_codes: refresh best-by-nk after {context} "
            f"({_timestamp_utc()})"
        )
        result = run_best_codes_update(
            repo_root,
            dry_run=False,
            no_git=no_git,
            no_publish=no_publish,
            verbose=False,
            include_git_history=include_git_history,
            max_attempts=max_attempts,
            commit_message=commit_message,
        )
    except Exception as exc:
        print(f"[best_codes] update failed: {exc}", file=sys.stderr)
        return False

    committed_label = "yes" if result.committed else "no"
    print(
        "[best_codes] done "
        f"scanned={len(result.records)} selected={len(result.selected)} "
        f"attempts={result.attempts} committed={committed_label}"
    )
    return True


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
            "Deprecated (use --slow-quantum-trials-override). "
            "Ignored unless no override is provided."
        ),
    )
    parser.add_argument(
        "--slow-quantum-trials-override",
        type=int,
        default=None,
        help=(
            "Override slow quantum distance trials (debugging only). "
            "Default: derived from best_codes for each (n,k)."
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
            "Deprecated alias: sets --quantum-steps-fast and "
            "--slow-quantum-trials-override."
        ),
    )
    parser.add_argument(
        "--best-codes-source",
        choices=("auto", "origin", "working-tree", "website"),
        default="auto",
        help="Source for best_codes index (default: auto).",
    )
    parser.add_argument(
        "--best-codes-refresh-seconds",
        type=int,
        default=600,
        help="Minimum seconds between best_codes refreshes (default: 600).",
    )
    parser.add_argument(
        "--best-codes-verbose",
        action="store_true",
        help="Log best_codes thresholds for each (n,k).",
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
        "--save-new-bests-dir",
        type=str,
        default="codes/pending",
        help="Directory for new_best JSON artifacts (default: codes/pending).",
    )
    parser.add_argument(
        "--no-save-new-bests",
        action="store_true",
        help="Disable saving new_best JSON artifacts.",
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
    parser.add_argument(
        "--no-best-codes-update",
        action="store_true",
        help="Skip best_codes sync/publish/git update at the end.",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="When updating best_codes, skip git pull/commit/push.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="When updating best_codes, skip website data updates.",
    )
    parser.add_argument(
        "--best-codes-no-history",
        action="store_true",
        help="Skip git history scan when updating best_codes (faster).",
    )
    parser.add_argument(
        "--best-codes-max-attempts",
        type=int,
        default=3,
        help="Max push attempts for best_codes updates.",
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
        if (
            _argv_has_flag(raw_argv, "--quantum-steps-fast")
            or _argv_has_flag(raw_argv, "--quantum-steps-slow")
            or _argv_has_flag(raw_argv, "--slow-quantum-trials-override")
        ):
            raise ValueError(
                "--quantum-steps is deprecated; do not mix with "
                "--quantum-steps-fast/--quantum-steps-slow/--slow-quantum-trials-override."
            )
        args.quantum_steps_fast = args.quantum_steps
        args.slow_quantum_trials_override = args.quantum_steps
    if args.quantum_steps_fast <= 0:
        raise ValueError("--quantum-steps-fast must be positive.")
    if args.quantum_steps_slow is not None:
        if args.quantum_steps_slow <= 0:
            raise ValueError("--quantum-steps-slow must be positive.")
        if args.slow_quantum_trials_override is None:
            print(
                "[progressive] WARNING: --quantum-steps-slow is deprecated; "
                "using it as --slow-quantum-trials-override.",
                file=sys.stderr,
            )
            args.slow_quantum_trials_override = args.quantum_steps_slow
        else:
            print(
                "[progressive] WARNING: --quantum-steps-slow is deprecated and "
                "ignored because --slow-quantum-trials-override is set.",
                file=sys.stderr,
            )
    if args.slow_quantum_trials_override is not None and args.slow_quantum_trials_override <= 0:
        raise ValueError("--slow-quantum-trials-override must be positive.")
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
    if args.best_codes_max_attempts <= 0:
        raise ValueError("--best-codes-max-attempts must be positive.")
    if args.best_codes_refresh_seconds <= 0:
        raise ValueError("--best-codes-refresh-seconds must be positive.")

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
    repo_root = _repo_root_for_best_codes(outdir)
    save_new_bests_dir: Optional[Path]
    if args.no_save_new_bests:
        save_new_bests_dir = None
    else:
        candidate = Path(args.save_new_bests_dir)
        save_new_bests_dir = (
            candidate if candidate.is_absolute() else repo_root / candidate
        )

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
        "quantum_steps_slow_policy": "best_codes",
        "slow_quantum_trials_override": args.slow_quantum_trials_override,
        "quantum_refine_chunk": args.quantum_refine_chunk,
        "report_every": args.report_every,
        "max_quantum_evals": args.max_quantum_evals,
        "seed": seed,
        "dist_m4ri_cmd": args.dist_m4ri_cmd,
        "dedup_cayley": bool(args.dedup_cayley),
        "min_distinct": args.min_distinct,
        "timings": bool(args.timings),
        "best_codes_source": args.best_codes_source,
        "best_codes_refresh_seconds": args.best_codes_refresh_seconds,
        "best_codes_verbose": bool(args.best_codes_verbose),
        "save_new_bests": save_new_bests_dir is not None,
        "save_new_bests_dir": (
            _relpath_or_abs(save_new_bests_dir, repo_root)
            if save_new_bests_dir is not None
            else None
        ),
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
    _print_histogram_report(side="A", hist=a_hist)
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
    _print_histogram_report(side="B", hist=b_hist)
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

    best_codes_cache = BestCodesIndexCache(
        repo_root=repo_root,
        source=args.best_codes_source,
        refresh_seconds=args.best_codes_refresh_seconds,
        verbose=args.best_codes_verbose,
    )
    best_codes_cache.refresh()

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
            best_codes_index = best_codes_cache.get_index()
            best_entry = best_codes_index.get((n_cols, k_val))
            best_codes_info = best_codes_cache.info()

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

            plan_holder: Dict[str, SlowDistanceDecision] = {}

            def _decide_slow(
                d_x_best: int,
                d_z_best: int,
                steps_fast: int,
                steps_slow: int,
            ) -> SlowDistanceDecision:
                d_fast = min(int(d_x_best), int(d_z_best))
                plan = decide_slow_quantum_plan(
                    d_fast=d_fast,
                    fast_trials=args.quantum_steps_fast,
                    best_entry=best_entry,
                    override=args.slow_quantum_trials_override,
                )
                if (
                    args.slow_quantum_trials_override is not None
                    and args.slow_quantum_trials_override < MIN_SLOW_TRIALS
                ):
                    print(
                        "[progressive] slow-quantum-trials-override "
                        f"{args.slow_quantum_trials_override} below "
                        f"{MIN_SLOW_TRIALS}; enforcing minimum."
                    )
                plan_holder["plan"] = plan
                return plan

            def _refine_abort_check(
                d_x_best: int,
                d_z_best: int,
            ) -> Optional[str]:
                if best_entry is None:
                    return None
                best_d = best_entry.d
                if best_d is None:
                    return None
                if should_abort_refine(d_x_best, d_z_best, best_d):
                    d_ub = min(int(d_x_best), int(d_z_best))
                    best_trials = best_entry.m4ri_trials
                    print(
                        "Abort refine: "
                        f"(n,k)=({n_cols},{k_val}) d_ub={d_ub} "
                        f"<= best_d={best_d} (best_trials={best_trials})"
                    )
                    return "best_codes"
                return None

            distance_result = _progressive_css_distance(
                hx_rows,
                hz_rows,
                n_cols,
                must_exceed=must_exceed,
                steps_fast=args.quantum_steps_fast,
                steps_slow=args.quantum_steps_fast,
                refine_chunk=args.quantum_refine_chunk,
                rng=rng,
                dist_m4ri_cmd=args.dist_m4ri_cmd,
                on_chunk=_log_refine,
                timings=run_timings,
                slow_decider=_decide_slow,
                refine_abort_check=_refine_abort_check,
            )
            plan = plan_holder.get("plan")
            d_x_best = distance_result.d_x_best
            d_z_best = distance_result.d_z_best
            d_fast = min(d_x_best, d_z_best)
            steps_used = distance_result.steps_x + distance_result.steps_z
            d_final_ub = min(d_x_best, d_z_best)
            is_new_best_by_k = False
            decision = "reject"
            if distance_result.passed_fast:
                if distance_result.slow_skipped:
                    decision = "skip_slow"
                elif distance_result.aborted:
                    decision = "refine"
                else:
                    is_new_best_by_k = (
                        d_final_ub > best
                        and d_final_ub >= args.target_distance
                    )
                    decision = "new_best" if is_new_best_by_k else "refine"
            if plan and plan.best_d is not None and d_fast <= plan.best_d:
                print(
                    "Skip slow quantum: "
                    f"(n,k)=({n_cols},{k_val}) d_fast={d_fast} "
                    f"current best d={plan.best_d} best trials={plan.best_trials}"
                )
            if args.best_codes_verbose and plan is not None:
                source_label = best_codes_info.get("source", "?")
                path_label = best_codes_info.get("path", "?")
                if args.slow_quantum_trials_override is not None:
                    trials_source = "override"
                elif best_entry is not None and best_entry.m4ri_trials is not None:
                    trials_source = "best_codes"
                else:
                    trials_source = "default_50k"
                if plan.best_d is None:
                    print(
                        f"[best_codes] (n,k)=({n_cols},{k_val}) "
                        f"no entry source={source_label}:{path_label} "
                        f"slow_trials={plan.steps_slow} "
                        f"slow_trials_source={trials_source}"
                    )
                else:
                    print(
                        f"[best_codes] (n,k)=({n_cols},{k_val}) "
                        f"best_d={plan.best_d} best_trials={plan.best_trials} "
                        f"slow_trials={plan.steps_slow} "
                        f"slow_trials_source={trials_source} "
                        f"source={source_label}:{path_label}"
                    )
            print(
                f"[eval] eval={eval_index} n={n_cols} k={k_val} best={best} "
                f"target_distance={args.target_distance} "
                f"must_exceed={must_exceed} steps_used={steps_used} "
                f"dX_best={d_x_best} dZ_best={d_z_best} decision={decision}"
            )

            bookkeeping_start = time.perf_counter()
            timestamp = _timestamp_utc()
            code_id_raw = (
                f"{group.name}_A{a_setting.setting_id}_B{b_setting.setting_id}_"
                f"k{k_val}_d{d_final_ub}"
            )
            code_id = _safe_id_component(code_id_raw)
            slow_steps_target = distance_result.slow_steps_target
            if slow_steps_target is None and plan is not None:
                slow_steps_target = plan.steps_slow
            if slow_steps_target is None:
                slow_steps_target = args.quantum_steps_fast
            best_d = plan.best_d if plan else None
            best_trials = plan.best_trials if plan else None
            best_code_id = plan.best_code_id if plan else None

            best_codes_candidate = False
            if best_d is None:
                best_codes_candidate = True
            else:
                if best_trials is not None and steps_used < best_trials:
                    best_codes_candidate = False
                elif d_final_ub > best_d:
                    best_codes_candidate = True
                elif (
                    d_final_ub == best_d
                    and best_trials is not None
                    and steps_used == best_trials
                    and best_code_id
                ):
                    best_codes_candidate = code_id < best_code_id
            if distance_result.aborted and distance_result.abort_reason == "best_codes":
                best_codes_candidate = False

            promising = _is_promising_code(n_cols, k_val, d_final_ub)
            save_candidate = promising and (
                distance_result.slow_skipped or best_codes_candidate
            )
            if not promising and distance_result.slow_skipped:
                print(
                    f"[progressive] skip save (non-promising): "
                    f"n={n_cols} k={k_val} d_ub={d_final_ub}"
                )

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
                    "steps_slow": slow_steps_target,
                    "refine_chunk": args.quantum_refine_chunk,
                    "fast": distance_result.fast,
                    "refine_chunks": distance_result.refine_chunks,
                    "steps_used_x": distance_result.steps_x,
                    "steps_used_z": distance_result.steps_z,
                    "steps_used_total": steps_used,
                    "dX_best": d_x_best,
                    "dZ_best": d_z_best,
                    "d_ub": d_final_ub,
                    "slow": {
                        "skipped": distance_result.slow_skipped,
                        "skip_reason": distance_result.slow_skip_reason,
                        "steps_target": slow_steps_target,
                    },
                },
                "best_codes": {
                    "source": best_codes_info.get("source"),
                    "path": best_codes_info.get("path"),
                    "best_d": best_d,
                    "best_trials": best_trials,
                    "best_code_id": best_code_id,
                },
                "best_codes_candidate": best_codes_candidate,
                "target_distance": args.target_distance,
                "seed": seed,
            }
            saved_best_code = False
            if save_candidate:
                _save_best_code(
                    outdir=outdir,
                    code_id=code_id,
                    hx_rows=hx_rows,
                    hz_rows=hz_rows,
                    n_cols=n_cols,
                    meta=meta,
                    timings=run_timings,
                )
                saved_best_code = True
            if decision == "new_best" and not saved_best_code:
                _save_best_code(
                    outdir=outdir,
                    code_id=code_id,
                    hx_rows=hx_rows,
                    hz_rows=hz_rows,
                    n_cols=n_cols,
                    meta=meta,
                    timings=run_timings,
                )
                saved_best_code = True
            if decision == "new_best":
                code_dir = outdir / "best_codes" / code_id
                artifact = {
                    "schema": "qtanner.new_best.v1",
                    "timestamp": timestamp,
                    "decision": decision,
                    "code_id": code_id,
                    "group": {"spec": group.name, "order": group.order},
                    "n": n_cols,
                    "k": k_val,
                    "A_id": a_setting.setting_id,
                    "B_id": b_setting.setting_id,
                    "A": a_setting.record,
                    "B": b_setting.record,
                    "local_codes": meta["local_codes"],
                    "distance": meta["distance"],
                    "dX_best": d_x_best,
                    "dZ_best": d_z_best,
                    "d_ub": d_final_ub,
                    "steps_used": steps_used,
                    "eval": eval_index,
                    "seed": seed,
                    "target_distance": args.target_distance,
                    "promising": promising,
                    "best_codes_candidate": best_codes_candidate,
                    "artifacts": {
                        "results_dir": _relpath_or_abs(outdir, repo_root),
                        "code_dir": _relpath_or_abs(code_dir, repo_root),
                        "hx_path": _relpath_or_abs(code_dir / "Hx.mtx", repo_root),
                        "hz_path": _relpath_or_abs(code_dir / "Hz.mtx", repo_root),
                        "meta_path": _relpath_or_abs(code_dir / "meta.json", repo_root),
                    },
                }
                _maybe_save_new_best_artifact(
                    decision=decision,
                    save_dir=save_new_bests_dir,
                    artifact=artifact,
                )
                if is_new_best_by_k:
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
    "BestCodesEntry",
    "ProgressiveSetting",
    "_maybe_save_new_best_artifact",
    "_enumerate_multisets_with_identity",
    "_interleaved_rounds",
    "_iter_progressive_pairs",
    "compute_slow_trials",
    "decide_slow_quantum_plan",
    "should_abort_refine",
    "progressive_main",
]
