"""Extract group and local-code structure from LRZ paper matrices."""

from __future__ import annotations

import argparse
import json
import itertools
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .classical_distance import read_mtx_coordinate_binary
from .group import CyclicGroup, FiniteGroup, group_from_spec
from .lift_matrices import build_hx_hz
from .local_codes import LocalCode, apply_col_perm_to_rows, row_from_bits
from .qdistrnd import gap_is_available


_H_6_3_3 = [
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 0],
]

_G_6_3_3 = [
    [0, 1, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 1],
]

_H_8_4_4 = [
    [1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
]

_G_8_4_4 = [
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 1],
]


def _canonical_supports(n: int) -> List[frozenset[int]]:
    if n == 2:
        return [frozenset({0, 1})]
    if n == 6:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _H_6_3_3]
    if n == 8:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _H_8_4_4]
    raise ValueError(f"Unsupported local code length {n}.")


def _canonical_supports_g(n: int) -> List[frozenset[int]]:
    if n == 2:
        return [frozenset({0, 1})]
    if n == 6:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _G_6_3_3]
    if n == 8:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _G_8_4_4]
    raise ValueError(f"Unsupported local code length {n}.")


def _local_code_canonical(n: int) -> LocalCode:
    if n == 2:
        H = [row_from_bits([1, 1])]
        G = [row_from_bits([1, 1])]
        return LocalCode(name="rep_2_1_2", n=2, k=1, H_rows=H, G_rows=G)
    if n == 6:
        H = [row_from_bits(row) for row in _H_6_3_3]
        G = [row_from_bits(row) for row in _G_6_3_3]
        return LocalCode(name="hamming_6_3_3", n=6, k=3, H_rows=H, G_rows=G)
    if n == 8:
        H = [row_from_bits(row) for row in _H_8_4_4]
        G = [row_from_bits(row) for row in _G_8_4_4]
        return LocalCode(name="hamming_8_4_4", n=8, k=4, H_rows=H, G_rows=G)
    raise ValueError(f"Unsupported local code length {n}.")


def _permute_local_code(code: LocalCode, perm: Sequence[int], name: str) -> LocalCode:
    perm_list = [int(x) for x in perm]
    return LocalCode(
        name=name,
        n=code.n,
        k=code.k,
        H_rows=apply_col_perm_to_rows(code.H_rows, perm_list, code.n),
        G_rows=apply_col_perm_to_rows(code.G_rows, perm_list, code.n),
    )


def _perm_inverse(perm: Sequence[int]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[int(j)] = int(i)
    return tuple(inv)


def _perm_comp(p: Sequence[int], q: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(p[int(i)]) for i in q)


def _perm_order(perm: Sequence[int]) -> int:
    m = len(perm)
    seen = [False] * m
    order = 1
    for i in range(m):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = int(perm[j])
            length += 1
        if length > 0:
            order = math.lcm(order, length)
    return order


def _perm_group_closure(
    generators: Sequence[Tuple[int, ...]],
    *,
    m: int,
    max_size: Optional[int] = None,
) -> set[Tuple[int, ...]]:
    identity = tuple(range(m))
    gens = list(generators)
    gens.extend([_perm_inverse(g) for g in generators])
    seen = {identity}
    frontier = [identity]
    while frontier:
        cur = frontier.pop()
        for gen in gens:
            nxt = _perm_comp(gen, cur)
            if nxt in seen:
                continue
            seen.add(nxt)
            if max_size is not None and len(seen) > max_size:
                return seen
            frontier.append(nxt)
    return seen


def _permute_supports(
    supports: Iterable[frozenset[int]],
    perm: Sequence[int],
) -> set[frozenset[int]]:
    return {frozenset(int(perm[i]) for i in s) for s in supports}


def _perm_candidates_from_supports(
    n: int,
    support_sets: Iterable[frozenset[int]],
    *,
    canonical_supports: Optional[Sequence[frozenset[int]]] = None,
) -> List[Tuple[int, ...]]:
    if n == 2:
        return [tuple(range(n))]
    canonical = canonical_supports or _canonical_supports(n)
    support_set = set(support_sets)
    candidates: List[Tuple[int, ...]] = []
    for perm in itertools.permutations(range(n)):
        permuted = _permute_supports(canonical, perm)
        if permuted.issubset(support_set):
            candidates.append(tuple(perm))
    return list(dict.fromkeys(candidates))


def _select_perm_generators(perms: Sequence[Tuple[int, ...]], *, m: int) -> Tuple[List[Tuple[int, ...]], set]:
    unique = list(dict.fromkeys(perms))
    gens: List[Tuple[int, ...]] = []
    group = _perm_group_closure(gens, m=m, max_size=m)
    for perm in unique:
        candidate = _perm_group_closure(gens + [perm], m=m, max_size=m)
        if len(candidate) > len(group):
            gens.append(perm)
            group = candidate
            if len(group) == m:
                break
    return gens, group


def _conjugate_perm(
    perm: Sequence[int],
    *,
    sigma: Sequence[int],
    sigma_inv: Sequence[int],
) -> Tuple[int, ...]:
    m = len(perm)
    return tuple(int(sigma[int(perm[int(sigma_inv[i])])]) for i in range(m))


def _encode3(vals: Sequence[int], sizes: Sequence[int], order: Sequence[int]) -> int:
    idx = 0
    for axis in order:
        idx = idx * int(sizes[axis]) + int(vals[axis])
    return idx


def _perm_to_element_left(
    perm: Sequence[int],
    *,
    left_perm_map: Optional[Dict[Tuple[int, ...], int]] = None,
) -> Optional[int]:
    perm_t = tuple(int(x) for x in perm)
    if left_perm_map is None:
        return perm_t[0]
    return left_perm_map.get(perm_t)


def _perm_to_element_right(
    perm: Sequence[int],
    *,
    right_perm_map: Optional[Dict[Tuple[int, ...], int]] = None,
    group: Optional[FiniteGroup] = None,
) -> Optional[int]:
    perm_t = tuple(int(x) for x in perm)
    if right_perm_map is not None and group is not None:
        t = right_perm_map.get(perm_t)
        if t is None:
            return None
        return int(group.inv(t))
    inv = _perm_inverse(perm_t)
    return inv[0]


def _decompose_perm_lr(perm: Sequence[int], group: FiniteGroup) -> Optional[Tuple[int, int]]:
    perm_t = tuple(int(x) for x in perm)
    if not perm_t:
        return None
    for x in group.elements():
        y = group.mul(group.inv(x), perm_t[0])
        ok = True
        for g in group.elements():
            if group.mul(group.mul(x, g), y) != perm_t[g]:
                ok = False
                break
        if ok:
            return int(x), int(y)
    return None


def _match_perm(
    perm: Sequence[int],
    *,
    left_map: Dict[Tuple[int, ...], int],
    left_inv_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
    right_inv_map: Dict[Tuple[int, ...], int],
    prefer: Sequence[str],
) -> Optional[Tuple[str, int]]:
    perm_t = tuple(int(x) for x in perm)
    lookup = {
        "L": left_map,
        "L_inv": left_inv_map,
        "R": right_map,
        "R_inv": right_inv_map,
    }
    for key in prefer:
        table = lookup.get(key)
        if table is None:
            continue
        elem = table.get(perm_t)
        if elem is not None:
            return key, int(elem)
    return None


def _match_perm_with_inverse(
    perm: Sequence[int],
    *,
    left_map: Dict[Tuple[int, ...], int],
    left_inv_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
    right_inv_map: Dict[Tuple[int, ...], int],
    prefer: Sequence[str],
) -> Optional[Tuple[str, int, bool]]:
    match = _match_perm(
        perm,
        left_map=left_map,
        left_inv_map=left_inv_map,
        right_map=right_map,
        right_inv_map=right_inv_map,
        prefer=prefer,
    )
    if match:
        return match[0], match[1], False
    inv_perm = _perm_inverse(perm)
    match = _match_perm(
        inv_perm,
        left_map=left_map,
        left_inv_map=left_inv_map,
        right_map=right_map,
        right_inv_map=right_inv_map,
        prefer=prefer,
    )
    if match:
        return match[0], match[1], True
    return None


def _element_from_match_left(match_type: str, elem: int, inverse_used: bool, group: FiniteGroup) -> int:
    if match_type == "L":
        return int(group.inv(elem)) if inverse_used else int(elem)
    if match_type == "L_inv":
        return int(elem) if inverse_used else int(group.inv(elem))
    if match_type == "R":
        return int(group.inv(elem)) if inverse_used else int(elem)
    if match_type == "R_inv":
        return int(elem) if inverse_used else int(group.inv(elem))
    return int(elem)


def _element_from_match_right(match_type: str, elem: int, inverse_used: bool, group: FiniteGroup) -> int:
    if match_type == "R":
        return int(elem) if inverse_used else int(group.inv(elem))
    if match_type == "R_inv":
        return int(group.inv(elem)) if inverse_used else int(elem)
    if match_type == "L":
        return int(elem) if inverse_used else int(group.inv(elem))
    if match_type == "L_inv":
        return int(group.inv(elem)) if inverse_used else int(elem)
    return int(elem)


def _build_left_right_maps(group: FiniteGroup) -> Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], int]]:
    left_map: Dict[Tuple[int, ...], int] = {}
    right_map: Dict[Tuple[int, ...], int] = {}
    for elem in group.elements():
        left_perm = tuple(group.mul(elem, g) for g in group.elements())
        right_perm = tuple(group.mul(g, elem) for g in group.elements())
        left_map[left_perm] = elem
        right_map[right_perm] = elem
    return left_map, right_map


def _build_perm_maps(
    group: FiniteGroup,
) -> Tuple[
    Dict[Tuple[int, ...], int],
    Dict[Tuple[int, ...], int],
    Dict[Tuple[int, ...], int],
    Dict[Tuple[int, ...], int],
]:
    left_map: Dict[Tuple[int, ...], int] = {}
    left_inv_map: Dict[Tuple[int, ...], int] = {}
    right_map: Dict[Tuple[int, ...], int] = {}
    right_inv_map: Dict[Tuple[int, ...], int] = {}
    for elem in group.elements():
        inv_elem = group.inv(elem)
        left_map[tuple(group.mul(elem, g) for g in group.elements())] = elem
        left_inv_map[tuple(group.mul(inv_elem, g) for g in group.elements())] = elem
        right_map[tuple(group.mul(g, elem) for g in group.elements())] = elem
        right_inv_map[tuple(group.mul(g, inv_elem) for g in group.elements())] = elem
    return left_map, left_inv_map, right_map, right_inv_map


def _is_abelian(group: FiniteGroup) -> bool:
    for a in group.elements():
        for b in group.elements():
            if group.mul(a, b) != group.mul(b, a):
                return False
    return True


def _sigma_cache_path(code_id: str, cache_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", code_id)
    return cache_dir / f"paper_extract_sigma_{safe}.json"


def _load_sigma_cache(code_id: str, cache_dir: Path) -> Optional[dict]:
    path = _sigma_cache_path(code_id, cache_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("sigma"):
        return payload
    return None


def _write_sigma_cache(code_id: str, cache_dir: Path, payload: dict) -> None:
    path = _sigma_cache_path(code_id, cache_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _score_sigma_matches(
    perms: Sequence[Tuple[int, ...]],
    *,
    sigma: Sequence[int],
    left_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
) -> Tuple[int, int, int]:
    sigma_inv = _perm_inverse(sigma)
    left_hits = 0
    right_hits = 0
    for perm in perms:
        conj = _conjugate_perm(perm, sigma=sigma, sigma_inv=sigma_inv)
        if conj in left_map:
            left_hits += 1
        if conj in right_map:
            right_hits += 1
    return left_hits + right_hits, left_hits, right_hits


def _infer_sigma_cyclic(
    perms: Sequence[Tuple[int, ...]],
    *,
    m: int,
    left_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
) -> Optional[Dict[str, object]]:
    generators = [perm for perm in perms if _perm_order(perm) == m]
    if not generators:
        return None
    best = None
    for perm in generators[:5]:
        for start in range(m):
            for direction in (1, -1):
                sigma = [0] * m
                cur = start
                for step in range(m):
                    sigma[cur] = step if direction == 1 else (-step) % m
                    cur = perm[cur]
                score, left_hits, right_hits = _score_sigma_matches(
                    perms, sigma=sigma, left_map=left_map, right_map=right_map
                )
                record = {
                    "sigma": sigma,
                    "score": score,
                    "left_hits": left_hits,
                    "right_hits": right_hits,
                    "method": "cyclic_cycle",
                    "generator_index": start,
                    "direction": direction,
                }
                if best is None or record["score"] > best["score"]:
                    best = record
    if best is None:
        return None
    return best


def _gap_list(perms: Sequence[Sequence[int]]) -> str:
    parts = []
    for perm in perms:
        parts.append("[" + ",".join(str(int(x) + 1) for x in perm) + "]")
    return "[" + ",".join(parts) + "]"


def _infer_sigma_gap(
    observed_perms: Sequence[Tuple[int, ...]],
    canonical_perms: Sequence[Tuple[int, ...]],
    *,
    m: int,
    gap_cmd: str,
) -> Tuple[Optional[List[int]], Optional[str]]:
    if not observed_perms or not canonical_perms:
        return None, "empty_perms"
    if not gap_is_available(gap_cmd):
        return None, "gap_missing"
    script = "\n".join(
        [
            f"obs := {_gap_list(observed_perms)};",
            f"can := {_gap_list(canonical_perms)};",
            "P := Group(List(obs, PermList));",
            "Q := Group(List(can, PermList));",
            f"if Size(P) <> {m} then Print(\"QTANNER_GAP_ERROR P_size\\n\"); QuitGap(2); fi;",
            f"if Size(Q) <> {m} then Print(\"QTANNER_GAP_ERROR Q_size\\n\"); QuitGap(2); fi;",
            "iso := IsomorphismGroups(P, Q);",
            "if iso = fail then Print(\"QTANNER_GAP_ERROR iso_fail\\n\"); QuitGap(2); fi;",
            "sigma := [];",
            "for x in [1..Size(P)] do",
            "  p := RepresentativeAction(P, 1, x);",
            "  if p = fail then Print(\"QTANNER_GAP_ERROR rep_fail\\n\"); QuitGap(2); fi;",
            "  y := Image(iso, p)(1);",
            "  sigma[x] := y;",
            "od;",
            "Print(\"QTANNER_SIGMA \", sigma, \"\\n\");",
            "QuitGap(0);",
        ]
    )
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".g", delete=False) as tmp:
            tmp.write(script)
            script_path = tmp.name
        result = subprocess.run(
            [gap_cmd, "-q", "-b", "--quitonbreak", script_path],
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
    finally:
        if "script_path" in locals():
            try:
                os.remove(script_path)
            except OSError:
                pass
    if returncode != 0:
        return None, f"gap_error:{returncode}"
    for line in stdout.splitlines():
        if line.startswith("QTANNER_SIGMA"):
            payload = line.split("QTANNER_SIGMA", 1)[1].strip()
            if payload.startswith("[") and payload.endswith("]"):
                vals = payload.strip("[]").strip()
                if not vals:
                    return None, "sigma_empty"
                sigma = [int(x.strip()) - 1 for x in vals.split(",")]
                if len(sigma) != m:
                    return None, "sigma_len"
                return sigma, None
    return None, "sigma_missing"


def _infer_sigma_bruteforce(
    perms: Sequence[Tuple[int, ...]],
    *,
    left_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
) -> Optional[Dict[str, object]]:
    if not perms:
        return None
    m = len(next(iter(perms)))
    best = None
    for sigma in itertools.permutations(range(m)):
        score, left_hits, right_hits = _score_sigma_matches(
            perms, sigma=sigma, left_map=left_map, right_map=right_map
        )
        record = {
            "sigma": list(sigma),
            "score": score,
            "left_hits": left_hits,
            "right_hits": right_hits,
            "method": "bruteforce",
        }
        if best is None or record["score"] > best["score"]:
            best = record
    return best


def _infer_sigma(
    perms: Sequence[Tuple[int, ...]],
    *,
    group: FiniteGroup,
    left_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
    gap_cmd: str,
    cache_dir: Path,
    code_id: str,
) -> Tuple[Optional[List[int]], Dict[str, object]]:
    cache = _load_sigma_cache(code_id, cache_dir)
    if cache and cache.get("sigma"):
        return list(cache["sigma"]), {"source": "cache", "cached": True}

    m = group.order
    if isinstance(group, CyclicGroup):
        result = _infer_sigma_cyclic(perms, m=m, left_map=left_map, right_map=right_map)
        if result and result.get("sigma"):
            sigma = list(result["sigma"])
            payload = {"sigma": sigma, "source": "cyclic", "details": result}
            _write_sigma_cache(code_id, cache_dir, payload)
            return sigma, payload

    sigma, error = _infer_sigma_gap(
        perms,
        list(left_map.keys()),
        m=m,
        gap_cmd=gap_cmd,
    )
    info: Dict[str, object] = {"source": "gap", "error": error}
    if sigma is None and m <= 8:
        brute = _infer_sigma_bruteforce(perms, left_map=left_map, right_map=right_map)
        if brute and brute.get("sigma"):
            sigma = list(brute["sigma"])
            info = {"source": "bruteforce", "details": brute}
    if sigma is None:
        return None, info
    payload = {"sigma": sigma, "source": "gap", "details": info}
    _write_sigma_cache(code_id, cache_dir, payload)
    return sigma, payload


def _sigma_candidates_cyclic_affine(
    *,
    m: int,
    perms: Sequence[Tuple[int, ...]],
    left_map: Dict[Tuple[int, ...], int],
    right_map: Dict[Tuple[int, ...], int],
) -> Tuple[List[List[int]], int]:
    units = [u for u in range(m) if math.gcd(u, m) == 1]
    best_score = -1
    best: List[List[int]] = []
    for u in units:
        for c in range(m):
            sigma = [(c + u * x) % m for x in range(m)]
            score, _left_hits, _right_hits = _score_sigma_matches(
                perms, sigma=sigma, left_map=left_map, right_map=right_map
            )
            if score > best_score:
                best_score = score
                best = [sigma]
            elif score == best_score:
                best.append(sigma)
    return best, best_score


def _iter_nonzero_entries(rows: Sequence[int]) -> Iterable[Tuple[int, int]]:
    for r, bits in enumerate(rows):
        row_bits = int(bits)
        while row_bits:
            lsb = row_bits & -row_bits
            c = lsb.bit_length() - 1
            yield r, c
            row_bits ^= lsb


def _map_index(idx: int, *, m: int, num_blocks: int, mode: str) -> Tuple[int, int]:
    if mode == "fiber-major":
        return idx // m, idx % m
    if mode == "group-major":
        return idx % num_blocks, idx // num_blocks
    raise ValueError(f"Unknown mapping mode {mode!r}.")


def _decode3(idx: int, sizes: Sequence[int], order: Sequence[int]) -> List[int]:
    coords = [0, 0, 0]
    rem = int(idx)
    for axis in reversed(order):
        size = int(sizes[axis])
        rem, val = divmod(rem, size)
        coords[axis] = val
    if rem != 0:
        raise ValueError(f"Index {idx} out of range for sizes {sizes} with order {order}.")
    return coords


def _build_block_maps3(
    entries: Iterable[Tuple[int, int]],
    *,
    row_sizes: Sequence[int],
    row_axes: Sequence[str],
    row_order: Sequence[int],
    col_sizes: Sequence[int],
    col_axes: Sequence[str],
    col_order: Sequence[int],
) -> Tuple[Dict[Tuple[int, int], Dict[int, set[int]]], Dict[Tuple[int, int], int], List[set[int]]]:
    row_non_g = [name for name in row_axes if name != "g"]
    row_sizes_map = {name: size for name, size in zip(row_axes, row_sizes)}
    col_sizes_map = {name: size for name, size in zip(col_axes, col_sizes)}
    row_block_count = int(row_sizes_map[row_non_g[0]] * row_sizes_map[row_non_g[1]])
    row_blocks = [set() for _ in range(row_block_count)]
    block_perm: Dict[Tuple[int, int], Dict[int, set[int]]] = {}
    block_nnz: Dict[Tuple[int, int], int] = {}
    for r, c in entries:
        row_vals = _decode3(r, row_sizes, row_order)
        col_vals = _decode3(c, col_sizes, col_order)
        row_val_map = {name: row_vals[idx] for idx, name in enumerate(row_axes)}
        col_val_map = {name: col_vals[idx] for idx, name in enumerate(col_axes)}
        row_block = (
            row_val_map[row_non_g[0]] * row_sizes_map[row_non_g[1]]
            + row_val_map[row_non_g[1]]
        )
        col_block = col_val_map["i"] * col_sizes_map["j"] + col_val_map["j"]
        row_blocks[int(row_block)].add(int(col_block))
        gr = row_val_map["g"]
        gc = col_val_map["g"]
        key = (int(row_block), int(col_block))
        block_perm.setdefault(key, {}).setdefault(int(gr), set()).add(int(gc))
        block_nnz[key] = block_nnz.get(key, 0) + 1
    return block_perm, block_nnz, row_blocks


def _score_block_maps(
    block_perm: Dict[Tuple[int, int], Dict[int, set[int]]],
    block_nnz: Dict[Tuple[int, int], int],
    *,
    m: int,
) -> Tuple[int, int, int, int]:
    full = 0
    eq_m = 0
    good = 0
    total = 0
    for key, gr_map in block_perm.items():
        nnz = block_nnz.get(key, 0)
        if nnz <= 0:
            continue
        total += 1
        if nnz == m:
            eq_m += 1
        gc_seen: set[int] = set()
        is_good = True
        for gcs in gr_map.values():
            if len(gcs) != 1:
                is_good = False
                break
            gc = next(iter(gcs))
            if gc in gc_seen:
                is_good = False
                break
            gc_seen.add(gc)
        if is_good:
            good += 1
            if len(gr_map) == m:
                full += 1
    return full, eq_m, good, total


def _order_label(order: Sequence[int], axes: Sequence[str]) -> str:
    return ",".join(axes[idx] for idx in order)


def _select_axes_mapping(
    *,
    hx_entries: Iterable[Tuple[int, int]],
    hz_entries: Iterable[Tuple[int, int]],
    hx_row_axes: Sequence[str],
    hx_row_sizes: Sequence[int],
    hz_row_axes: Sequence[str],
    hz_row_sizes: Sequence[int],
    col_axes: Sequence[str],
    col_sizes: Sequence[int],
    nA: int,
    nB: int,
    m: int,
) -> Dict[str, object]:
    perms = list(itertools.permutations(range(3)))
    best: Optional[Dict[str, object]] = None
    for col_order in perms:
        best_hx: Optional[Dict[str, object]] = None
        best_hz: Optional[Dict[str, object]] = None
        for row_order_hx in perms:
            hx_block_perm, hx_block_nnz, hx_row_blocks = _build_block_maps3(
                hx_entries,
                row_sizes=hx_row_sizes,
                row_axes=hx_row_axes,
                row_order=row_order_hx,
                col_sizes=col_sizes,
                col_axes=col_axes,
                col_order=col_order,
            )
            hx_score = _score_block_maps(hx_block_perm, hx_block_nnz, m=m)
            record = {
                "row_order_hx": tuple(row_order_hx),
                "hx_score": hx_score,
                "hx_block_perm": hx_block_perm,
                "hx_block_nnz": hx_block_nnz,
                "hx_row_blocks": hx_row_blocks,
            }
            if best_hx is None or record["hx_score"] > best_hx["hx_score"]:
                best_hx = record
        for row_order_hz in perms:
            hz_block_perm, hz_block_nnz, hz_row_blocks = _build_block_maps3(
                hz_entries,
                row_sizes=hz_row_sizes,
                row_axes=hz_row_axes,
                row_order=row_order_hz,
                col_sizes=col_sizes,
                col_axes=col_axes,
                col_order=col_order,
            )
            hz_score = _score_block_maps(hz_block_perm, hz_block_nnz, m=m)
            record = {
                "row_order_hz": tuple(row_order_hz),
                "hz_score": hz_score,
                "hz_block_perm": hz_block_perm,
                "hz_block_nnz": hz_block_nnz,
                "hz_row_blocks": hz_row_blocks,
            }
            if best_hz is None or record["hz_score"] > best_hz["hz_score"]:
                best_hz = record
        if best_hx is None or best_hz is None:
            continue
        hx_score = best_hx["hx_score"]
        hz_score = best_hz["hz_score"]
        full = hx_score[0] + hz_score[0]
        eq_m = hx_score[1] + hz_score[1]
        good = hx_score[2] + hz_score[2]
        total = hx_score[3] + hz_score[3]
        score = (full, eq_m, good, -total)
        cb_to_ij = _cb_to_ij(nA, nB)
        hx_const_j, _ = _count_const_rows(best_hx["hx_row_blocks"], cb_to_ij=cb_to_ij)
        _, hz_const_i = _count_const_rows(best_hz["hz_row_blocks"], cb_to_ij=cb_to_ij)
        class_score = hx_const_j + hz_const_i
        record = {
            "row_order_hx": best_hx["row_order_hx"],
            "row_order_hz": best_hz["row_order_hz"],
            "col_order": tuple(col_order),
            "score": score,
            "hx_score": hx_score,
            "hz_score": hz_score,
            "class_score": class_score,
            "hx_block_perm": best_hx["hx_block_perm"],
            "hx_block_nnz": best_hx["hx_block_nnz"],
            "hx_row_blocks": best_hx["hx_row_blocks"],
            "hz_block_perm": best_hz["hz_block_perm"],
            "hz_block_nnz": best_hz["hz_block_nnz"],
            "hz_row_blocks": best_hz["hz_row_blocks"],
        }
        if best is None or record["score"] > best["score"]:
            best = record
        elif record["score"] == best["score"] and record["class_score"] > best["class_score"]:
            best = record
    if best is None:
        raise RuntimeError("Unable to select axis mapping.")
    return best


def _select_axes_mapping_candidates(
    *,
    hx_entries: Iterable[Tuple[int, int]],
    hz_entries: Iterable[Tuple[int, int]],
    hx_row_axes: Sequence[str],
    hx_row_sizes: Sequence[int],
    hz_candidates: Sequence[Tuple[Sequence[str], Sequence[int], str]],
    col_axes: Sequence[str],
    col_sizes: Sequence[int],
    nA: int,
    nB: int,
    m: int,
) -> Dict[str, object]:
    best: Optional[Dict[str, object]] = None
    for hz_row_axes, hz_row_sizes, hz_label in hz_candidates:
        choice = _select_axes_mapping(
            hx_entries=hx_entries,
            hz_entries=hz_entries,
            hx_row_axes=hx_row_axes,
            hx_row_sizes=hx_row_sizes,
            hz_row_axes=hz_row_axes,
            hz_row_sizes=hz_row_sizes,
            col_axes=col_axes,
            col_sizes=col_sizes,
            nA=nA,
            nB=nB,
            m=m,
        )
        choice = {
            **choice,
            "hz_row_axes": tuple(hz_row_axes),
            "hz_row_sizes": list(hz_row_sizes),
            "hz_row_label": hz_label,
        }
        if best is None or choice["score"] > best["score"]:
            best = choice
        elif choice["score"] == best["score"] and choice["class_score"] > best["class_score"]:
            best = choice
    if best is None:
        raise RuntimeError("Unable to determine axis mapping.")
    return best


def _build_block_maps(
    entries: Iterable[Tuple[int, int]],
    *,
    n_rows: int,
    n_cols: int,
    m: int,
    row_mode: str,
    col_mode: str,
) -> Tuple[Dict[Tuple[int, int], Dict[int, set[int]]], List[set[int]]]:
    num_row_blocks = n_rows // m
    num_col_blocks = n_cols // m
    row_blocks = [set() for _ in range(num_row_blocks)]
    block_perm: Dict[Tuple[int, int], Dict[int, set[int]]] = {}
    for r, c in entries:
        rb, gr = _map_index(r, m=m, num_blocks=num_row_blocks, mode=row_mode)
        cb, gc = _map_index(c, m=m, num_blocks=num_col_blocks, mode=col_mode)
        row_blocks[rb].add(cb)
        key = (rb, cb)
        if key not in block_perm:
            block_perm[key] = {}
        block_perm[key].setdefault(gr, set()).add(gc)
    return block_perm, row_blocks


def _score_block_perm(block_perm: Dict[Tuple[int, int], Dict[int, set[int]]], *, m: int) -> Tuple[int, int, int]:
    full_good = 0
    good = 0
    bad = 0
    for gr_map in block_perm.values():
        if not gr_map:
            continue
        gc_seen: set[int] = set()
        is_good = True
        for gcs in gr_map.values():
            if len(gcs) != 1:
                is_good = False
                break
            gc = next(iter(gcs))
            if gc in gc_seen:
                is_good = False
                break
            gc_seen.add(gc)
        if is_good:
            good += 1
            if len(gr_map) == m:
                full_good += 1
        else:
            bad += 1
    return full_good, good, bad


def _select_fiber_mappings(
    *,
    hx_entries: Iterable[Tuple[int, int]],
    hz_entries: Iterable[Tuple[int, int]],
    hx_shape: Tuple[int, int],
    hz_shape: Tuple[int, int],
    m: int,
) -> Dict[str, object]:
    options = []
    row_modes = ["fiber-major", "group-major"]
    col_modes = ["fiber-major", "group-major"]
    for row_mode in row_modes:
        for col_mode in col_modes:
            hx_block_perm, _ = _build_block_maps(
                hx_entries,
                n_rows=hx_shape[0],
                n_cols=hx_shape[1],
                m=m,
                row_mode=row_mode,
                col_mode=col_mode,
            )
            hz_block_perm, _ = _build_block_maps(
                hz_entries,
                n_rows=hz_shape[0],
                n_cols=hz_shape[1],
                m=m,
                row_mode=row_mode,
                col_mode=col_mode,
            )
            hx_score = _score_block_perm(hx_block_perm, m=m)
            hz_score = _score_block_perm(hz_block_perm, m=m)
            full_good = hx_score[0] + hz_score[0]
            good = hx_score[1] + hz_score[1]
            bad = hx_score[2] + hz_score[2]
            score = (full_good, good, -bad)
            options.append(
                {
                    "row_mode": row_mode,
                    "col_mode": col_mode,
                    "score": score,
                    "hx_score": hx_score,
                    "hz_score": hz_score,
                }
            )
    options.sort(key=lambda x: x["score"], reverse=True)
    return options[0]


def _block_masks(m: int, total_blocks: int) -> List[int]:
    base_mask = (1 << m) - 1
    return [base_mask << (cb * m) for cb in range(total_blocks)]


def _row_blocks_touched(
    rows: Sequence[int],
    *,
    m: int,
    n_blocks: int,
) -> List[set[int]]:
    masks = _block_masks(m, n_blocks)
    n_rows = len(rows)
    blocks = []
    total_row_blocks = n_rows // m
    for rb in range(total_row_blocks):
        touched: set[int] = set()
        row_start = rb * m
        for r in range(row_start, min(row_start + m, n_rows)):
            row_bits = rows[r]
            if row_bits == 0:
                continue
            for cb, mask in enumerate(masks):
                if row_bits & mask:
                    touched.add(cb)
        blocks.append(touched)
    return blocks


def _extract_block_perm(
    rows: Sequence[int],
    *,
    rb: int,
    cb: int,
    m: int,
) -> Tuple[Optional[Tuple[int, ...]], bool]:
    n_rows = len(rows)
    row_start = rb * m
    col_start = cb * m
    mask = ((1 << m) - 1) << col_start
    perm: List[int] = []
    seen_cols = [0] * m
    any_nonzero = False
    for r_in in range(m):
        r = row_start + r_in
        if r >= n_rows:
            return None, False
        row_bits = rows[r] & mask
        if row_bits == 0:
            perm.append(-1)
            continue
        any_nonzero = True
        if row_bits & (row_bits - 1):
            return None, False
        c_in = (row_bits.bit_length() - 1) - col_start
        if c_in < 0 or c_in >= m:
            return None, False
        perm.append(c_in)
        seen_cols[c_in] += 1
    if not any_nonzero:
        return None, True
    if any(p < 0 for p in perm):
        return None, False
    if any(count != 1 for count in seen_cols):
        return None, False
    return tuple(perm), True


def _perm_from_block_map(gr_map: Dict[int, set[int]], *, m: int) -> Tuple[Optional[Tuple[int, ...]], bool]:
    if not gr_map:
        return None, True
    if len(gr_map) != m:
        return None, False
    perm = [-1] * m
    seen_gc: set[int] = set()
    for gr, gcs in gr_map.items():
        if len(gcs) != 1:
            return None, False
        gc = next(iter(gcs))
        if gc in seen_gc:
            return None, False
        seen_gc.add(gc)
        if gr < 0 or gr >= m:
            return None, False
        perm[gr] = gc
    if any(p < 0 for p in perm):
        return None, False
    return tuple(perm), True


def _cb_to_ij(nA: int, nB: int):
    return lambda cb: (cb // nB, cb % nB)


def _cb_to_ij_option(nA: int, nB: int, option: str):
    if option == "i_nB_j":
        return lambda cb: (cb // nB, cb % nB)
    if option == "j_nA_i":
        return lambda cb: (cb % nA, cb // nA)
    raise ValueError(f"Unknown option {option!r}.")


def _cb_to_ij_from_col_order(
    *,
    col_order: Sequence[int],
    col_axes: Sequence[str],
    nA: int,
    nB: int,
):
    order_names = [col_axes[idx] for idx in col_order]
    non_g = [name for name in order_names if name != "g"]
    if non_g == ["i", "j"]:
        return _cb_to_ij_option(nA, nB, "i_nB_j")
    if non_g == ["j", "i"]:
        return _cb_to_ij_option(nA, nB, "j_nA_i")
    raise ValueError(f"Unexpected column axes order {non_g!r}.")


def _classify_row_blocks(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
) -> Tuple[List[Optional[Tuple[str, int]]], Dict[str, int]]:
    classifications: List[Optional[Tuple[str, int]]] = []
    counts = {"A": 0, "B": 0, "unknown": 0}
    for touched in row_blocks:
        if not touched:
            classifications.append(None)
            counts["unknown"] += 1
            continue
        pairs = [cb_to_ij(cb) for cb in touched]
        i_vals = {i for i, _ in pairs}
        j_vals = {j for _, j in pairs}
        if len(j_vals) == 1:
            j_val = next(iter(j_vals))
            classifications.append(("A", int(j_val)))
            counts["A"] += 1
        elif len(i_vals) == 1:
            i_val = next(iter(i_vals))
            classifications.append(("B", int(i_val)))
            counts["B"] += 1
        else:
            classifications.append(None)
            counts["unknown"] += 1
    return classifications, counts


def _count_const_rows(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
) -> Tuple[int, int]:
    const_j = 0
    const_i = 0
    for touched in row_blocks:
        if not touched:
            continue
        pairs = [cb_to_ij(cb) for cb in touched]
        if len({j for _, j in pairs}) == 1:
            const_j += 1
        if len({i for i, _ in pairs}) == 1:
            const_i += 1
    return const_j, const_i


def _select_cb_mapping(
    *,
    nA: int,
    nB: int,
    hx_row_blocks: Sequence[set[int]],
    hz_row_blocks: Sequence[set[int]],
) -> Dict[str, object]:
    best = None
    for option in ("i_nB_j", "j_nA_i"):
        cb_to_ij = _cb_to_ij_option(nA, nB, option)
        hx_const_j, hx_const_i = _count_const_rows(hx_row_blocks, cb_to_ij=cb_to_ij)
        hz_const_j, hz_const_i = _count_const_rows(hz_row_blocks, cb_to_ij=cb_to_ij)
        score_a = hx_const_j + hz_const_i
        score_b = hx_const_i + hz_const_j
        score = max(score_a, score_b)
        orientation = "HX:j/HZ:i" if score_a >= score_b else "HX:i/HZ:j"
        record = {
            "option": option,
            "cb_to_ij": cb_to_ij,
            "score": score,
            "score_a": score_a,
            "score_b": score_b,
            "orientation": orientation,
            "hx_const_j": hx_const_j,
            "hx_const_i": hx_const_i,
            "hz_const_j": hz_const_j,
            "hz_const_i": hz_const_i,
        }
        if best is None or record["score"] > best["score"]:
            best = record
    if best is None:
        raise RuntimeError("Unable to determine column block ordering.")
    return best


def _choose_block_mapping(
    row_blocks: Sequence[set[int]],
    *,
    nA: int,
    nB: int,
) -> Tuple[str, callable, List[Optional[Tuple[str, int]]], Dict[str, int]]:
    best = None
    for option in ("i_nB_j", "j_nA_i"):
        cb_to_ij = _cb_to_ij_option(nA, nB, option)
        classifications, counts = _classify_row_blocks(row_blocks, cb_to_ij=cb_to_ij)
        score = counts["A"] + counts["B"]
        if best is None or score > best[0]:
            best = (score, option, cb_to_ij, classifications, counts)
    if best is None:
        raise RuntimeError("Unable to determine block mapping.")
    _, option, cb_to_ij, classifications, counts = best
    return option, cb_to_ij, classifications, counts


def _support_sets_for_a(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
    nB: int,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_j: Dict[int, set[frozenset[int]]] = {}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        j_val = rb % nB
        i_set = {cb_to_ij(cb)[0] for cb in touched}
        supports_by_j.setdefault(j_val, set()).add(frozenset(i_set))
    return supports_by_j


def _row_block_half_hx(rb: int, *, nB: int, kB: int) -> int:
    return 0 if (rb % nB) < kB else 1


def _row_block_half_hz(
    rb: int,
    *,
    label: str,
    kA: int,
    rB: int,
    nB: int,
) -> int:
    if label == "ir":
        return 0 if (rb // rB) < kA else 1
    if label == "rj":
        return 0 if (rb % nB) < rB else 1
    raise ValueError(f"Unknown HZ row label {label!r}.")


def _support_sets_for_b_by_j(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
    nB: int,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_j: Dict[int, set[frozenset[int]]] = {}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        j_val = rb % nB
        j_set = {cb_to_ij(cb)[1] for cb in touched}
        supports_by_j.setdefault(j_val, set()).add(frozenset(j_set))
    return supports_by_j


def _support_sets_for_b_by_half(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
    label: str,
    kA: int,
    rB: int,
    nB: int,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_half: Dict[int, set[frozenset[int]]] = {0: set(), 1: set()}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        half = _row_block_half_hz(rb, label=label, kA=kA, rB=rB, nB=nB)
        j_set = {cb_to_ij(cb)[1] for cb in touched}
        supports_by_half[half].add(frozenset(j_set))
    return supports_by_half


def _support_sets_for_b(
    row_blocks: Sequence[set[int]],
    *,
    cb_to_ij,
    rB: int,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_i: Dict[int, set[frozenset[int]]] = {}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        i_val = rb // rB
        j_set = {cb_to_ij(cb)[1] for cb in touched}
        supports_by_i.setdefault(i_val, set()).add(frozenset(j_set))
    return supports_by_i


def _find_perm_from_supports(
    n: int,
    *,
    canonical_supports: Iterable[frozenset[int]],
    target_supports: Iterable[frozenset[int]],
) -> Optional[Tuple[int, ...]]:
    canonical_set = {frozenset(s) for s in canonical_supports}
    target_set = {frozenset(s) for s in target_supports}
    if target_set == canonical_set:
        return tuple(range(n))
    if len(target_set) != len(canonical_set):
        return None
    for perm in itertools.permutations(range(n)):
        permuted = {frozenset(perm[i] for i in s) for s in canonical_set}
        if permuted == target_set:
            return tuple(perm)
    return None


def _perm_from_supports_by_index(
    n: int,
    *,
    supports_by_index: Dict[int, set[frozenset[int]]],
    indices: Iterable[int],
) -> Optional[Tuple[int, ...]]:
    canonical = _canonical_supports(n)
    candidates: List[Tuple[int, ...]] = []
    for idx in indices:
        supports = supports_by_index.get(int(idx))
        if not supports:
            continue
        perm = _find_perm_from_supports(
            n, canonical_supports=canonical, target_supports=supports
        )
        if perm is not None:
            candidates.append(perm)
    if not candidates:
        return None
    counts = Counter(candidates)
    best_perm, _ = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best_perm


def _perms_from_supports_by_index(
    n: int,
    *,
    supports_by_index: Dict[int, set[frozenset[int]]],
    indices: Iterable[int],
    canonical_supports: Optional[Sequence[frozenset[int]]] = None,
) -> List[Tuple[int, ...]]:
    canonical = canonical_supports or _canonical_supports(n)
    perms: List[Tuple[int, ...]] = []
    for idx in indices:
        supports = supports_by_index.get(int(idx))
        if not supports:
            continue
        perm = _find_perm_from_supports(
            n, canonical_supports=canonical, target_supports=supports
        )
        if perm is not None:
            perms.append(perm)
    return list(dict.fromkeys(perms))


def _classify_by_supports(
    supports: Sequence[frozenset[int]],
    *,
    perm0: Sequence[int],
    perm1: Sequence[int],
    canonical_supports: Sequence[frozenset[int]],
) -> List[Optional[str]]:
    perm0_set = _permute_supports(canonical_supports, perm0)
    perm1_set = _permute_supports(canonical_supports, perm1)
    types: List[Optional[str]] = []
    for supp in supports:
        if supp in perm1_set:
            types.append("1")
        elif supp in perm0_set:
            types.append("0")
        else:
            types.append(None)
    return types


def _apply_col_perm_to_bitrows(rows: Sequence[int], perm: Sequence[int]) -> List[int]:
    out: List[int] = []
    for row in rows:
        new_row = 0
        x = int(row)
        while x:
            lsb = x & -x
            idx = lsb.bit_length() - 1
            new_row |= 1 << int(perm[idx])
            x -= lsb
        out.append(new_row)
    return out


def _reorder_rows(rows: Sequence[int], row_perm: Sequence[int]) -> List[int]:
    out = [0] * len(rows)
    for old_idx, row in enumerate(rows):
        out[int(row_perm[old_idx])] = int(row)
    return out


def _hx_row_index(r: int, j: int, g: int, *, rA: int, kB: int, m: int) -> int:
    if j < kB:
        block = int(r) * kB + int(j)
    else:
        block = int(r) * kB + int(j - kB) + rA * kB
    return block * m + int(g)


def _hz_row_index(i: int, r: int, g: int, *, kA: int, rB: int, m: int) -> int:
    if i < kA:
        block = int(i) * rB + int(r)
    else:
        block = int(i - kA) * rB + int(r) + kA * rB
    return block * m + int(g)


def _build_col_perm(
    *,
    nA: int,
    nB: int,
    m: int,
    col_order: Sequence[int],
    sigma_inv: Optional[Sequence[int]],
) -> List[int]:
    sizes = [nA, nB, m]
    n_cols = nA * nB * m
    perm = [0] * n_cols
    for i in range(nA):
        for j in range(nB):
            for g_can in range(m):
                g_obs = int(sigma_inv[g_can]) if sigma_inv is not None else int(g_can)
                new_idx = _encode3([i, j, g_obs], sizes, col_order)
                old_idx = ((i * nB + j) * m + g_can)
                perm[old_idx] = new_idx
    return perm


def _build_row_perm_hx(
    *,
    rA: int,
    nB: int,
    kB: int,
    m: int,
    row_order: Sequence[int],
    sigma_inv: Optional[Sequence[int]],
    r_perm: Optional[Sequence[int]] = None,
    j_perm: Optional[Sequence[int]] = None,
) -> List[int]:
    sizes = [rA, nB, m]
    n_rows = rA * nB * m
    perm = [0] * n_rows
    for r in range(rA):
        for j in range(nB):
            for g_can in range(m):
                g_obs = int(sigma_inv[g_can]) if sigma_inv is not None else int(g_can)
                r_val = int(r_perm[r]) if r_perm is not None else int(r)
                j_val = int(j_perm[j]) if j_perm is not None else int(j)
                new_idx = _encode3([r_val, j_val, g_obs], sizes, row_order)
                old_idx = _hx_row_index(r, j, g_can, rA=rA, kB=kB, m=m)
                perm[old_idx] = new_idx
    return perm


def _build_row_perm_hz(
    *,
    nA: int,
    rB: int,
    kA: int,
    m: int,
    row_order: Sequence[int],
    sigma_inv: Optional[Sequence[int]],
    row_axes_hz: str = "ir",
    rA: Optional[int] = None,
    nB: Optional[int] = None,
    i_perm: Optional[Sequence[int]] = None,
    r_perm: Optional[Sequence[int]] = None,
    j_perm: Optional[Sequence[int]] = None,
) -> List[int]:
    if row_axes_hz == "ir":
        sizes = [nA, rB, m]
        n_rows = nA * rB * m
        perm = [0] * n_rows
        for i in range(nA):
            for r in range(rB):
                for g_can in range(m):
                    g_obs = (
                        int(sigma_inv[g_can]) if sigma_inv is not None else int(g_can)
                    )
                    i_val = int(i_perm[i]) if i_perm is not None else int(i)
                    r_val = int(r_perm[r]) if r_perm is not None else int(r)
                    new_idx = _encode3([i_val, r_val, g_obs], sizes, row_order)
                    old_idx = _hz_row_index(i, r, g_can, kA=kA, rB=rB, m=m)
                    perm[old_idx] = new_idx
        return perm
    if row_axes_hz == "rj":
        if rA is None or nB is None:
            raise ValueError("rA and nB are required for HZ row_axes 'rj'.")
        sizes = [rA, nB, m]
        n_rows = rA * nB * m
        perm = [0] * n_rows
        for i in range(nA):
            half = 0 if i < kA else 1
            r_idx = i if half == 0 else i - kA
            for r in range(rB):
                j_idx = r if half == 0 else r + rB
                for g_can in range(m):
                    g_obs = (
                        int(sigma_inv[g_can]) if sigma_inv is not None else int(g_can)
                    )
                    r_val = int(r_perm[r_idx]) if r_perm is not None else int(r_idx)
                    j_val = int(j_perm[j_idx]) if j_perm is not None else int(j_idx)
                    new_idx = _encode3([r_val, j_val, g_obs], sizes, row_order)
                    old_idx = _hz_row_index(i, r, g_can, kA=kA, rB=rB, m=m)
                    perm[old_idx] = new_idx
        return perm
    raise ValueError(f"Unknown HZ row_axes {row_axes_hz!r}.")


def _iter_half_perms(size: int) -> Iterable[Tuple[int, ...]]:
    return itertools.permutations(range(size))


def _iter_block_perms(total: int, half: int, *, allow_swap: bool = False) -> Iterable[List[int]]:
    for left in _iter_half_perms(half):
        for right in _iter_half_perms(half):
            perm = [0] * total
            for i, val in enumerate(left):
                perm[i] = val
            for i, val in enumerate(right):
                perm[half + i] = half + val
            yield perm
            if allow_swap:
                perm_swap = [0] * total
                for i, val in enumerate(left):
                    perm_swap[half + i] = val
                for i, val in enumerate(right):
                    perm_swap[i] = half + val
                yield perm_swap


def _find_row_perm_hx(
    rows: Sequence[int],
    rows_in: Sequence[int],
    *,
    rA: int,
    nB: int,
    kB: int,
    m: int,
    row_order: Sequence[int],
    sigma_inv: Optional[Sequence[int]],
) -> Optional[List[int]]:
    for r_perm in _iter_half_perms(rA):
        for j_perm in _iter_block_perms(nB, kB, allow_swap=True):
            row_perm = _build_row_perm_hx(
                rA=rA,
                nB=nB,
                kB=kB,
                m=m,
                row_order=row_order,
                sigma_inv=sigma_inv,
                r_perm=r_perm,
                j_perm=j_perm,
            )
            if _reorder_rows(rows, row_perm) == list(rows_in):
                return row_perm
    return None


def _find_row_perm_hz(
    rows: Sequence[int],
    rows_in: Sequence[int],
    *,
    nA: int,
    rB: int,
    kA: int,
    rA: int,
    nB: int,
    m: int,
    row_order: Sequence[int],
    sigma_inv: Optional[Sequence[int]],
    row_axes_hz: str = "ir",
) -> Optional[List[int]]:
    if row_axes_hz == "ir":
        for i_perm in _iter_block_perms(nA, kA, allow_swap=True):
            for r_perm in _iter_half_perms(rB):
                row_perm = _build_row_perm_hz(
                    nA=nA,
                    rB=rB,
                    kA=kA,
                    m=m,
                    row_order=row_order,
                    sigma_inv=sigma_inv,
                    row_axes_hz=row_axes_hz,
                    i_perm=i_perm,
                    r_perm=r_perm,
                )
                if _reorder_rows(rows, row_perm) == list(rows_in):
                    return row_perm
        return None
    if row_axes_hz == "rj":
        for r_perm in _iter_half_perms(rA):
            for j_perm in _iter_block_perms(nB, rB, allow_swap=True):
                row_perm = _build_row_perm_hz(
                    nA=nA,
                    rB=rB,
                    kA=kA,
                    rA=rA,
                    nB=nB,
                    m=m,
                    row_order=row_order,
                    sigma_inv=sigma_inv,
                    row_axes_hz=row_axes_hz,
                    r_perm=r_perm,
                    j_perm=j_perm,
                )
                if _reorder_rows(rows, row_perm) == list(rows_in):
                    return row_perm
        return None
    raise ValueError(f"Unknown HZ row_axes {row_axes_hz!r}.")


def _reconstruct_and_compare(
    *,
    group: FiniteGroup,
    A_ids: Sequence[int],
    B_ids: Sequence[int],
    permA0: Optional[Sequence[int]],
    permA1: Optional[Sequence[int]],
    permB0: Optional[Sequence[int]],
    permB1: Optional[Sequence[int]],
    nA: int,
    nB: int,
    rA: int,
    rB: int,
    m: int,
    row_order_hx: Sequence[int],
    row_order_hz: Sequence[int],
    col_order: Sequence[int],
    row_axes_hz: str,
    sigma_inv: Optional[Sequence[int]],
    hx_rows_in: Sequence[int],
    hz_rows_in: Sequence[int],
) -> Tuple[bool, str]:
    if permA0 is None or permA1 is None or permB0 is None or permB1 is None:
        return False, "perm_missing"
    if any(a is None for a in A_ids) or any(b is None for b in B_ids):
        return False, "A_or_B_missing"

    canonA = _local_code_canonical(nA)
    canonB = _local_code_canonical(nB)
    C0 = _permute_local_code(canonA, permA0, name=f"{canonA.name}_permA0")
    C1 = _permute_local_code(canonA, permA1, name=f"{canonA.name}_permA1")
    C0p = _permute_local_code(canonB, permB0, name=f"{canonB.name}_permB0")
    C1p = _permute_local_code(canonB, permB1, name=f"{canonB.name}_permB1")

    kA = len(C0.G_rows)
    kB = len(C0p.G_rows)
    if 2 * kA != nA or 2 * kB != nB:
        return False, "local_code_dim_mismatch"

    hx_rows, hz_rows, n_cols = build_hx_hz(
        group,
        list(A_ids),
        list(B_ids),
        C0,
        C1,
        C0p,
        C1p,
    )
    if n_cols != nA * nB * m:
        return False, "n_cols_mismatch"

    col_perm = _build_col_perm(
        nA=nA, nB=nB, m=m, col_order=col_order, sigma_inv=sigma_inv
    )
    hx_rows = _apply_col_perm_to_bitrows(hx_rows, col_perm)
    hz_rows = _apply_col_perm_to_bitrows(hz_rows, col_perm)

    row_perm_hx = _find_row_perm_hx(
        hx_rows,
        hx_rows_in,
        rA=rA,
        nB=nB,
        kB=kB,
        m=m,
        row_order=row_order_hx,
        sigma_inv=sigma_inv,
    )
    if row_perm_hx is None:
        return False, "hx_row_perm"
    hx_rows = _reorder_rows(hx_rows, row_perm_hx)

    row_perm_hz = _find_row_perm_hz(
        hz_rows,
        hz_rows_in,
        nA=nA,
        rB=rB,
        kA=kA,
        rA=rA,
        nB=nB,
        m=m,
        row_order=row_order_hz,
        sigma_inv=sigma_inv,
        row_axes_hz=row_axes_hz,
    )
    if row_perm_hz is None:
        return False, "hz_row_perm"
    hz_rows = _reorder_rows(hz_rows, row_perm_hz)

    if list(hx_rows) != list(hx_rows_in):
        return False, "hx_mismatch"
    if list(hz_rows) != list(hz_rows_in):
        return False, "hz_mismatch"
    return True, "ok"


def _parse_stem(stem: str) -> Tuple[Optional[str], int, int, int]:
    parts = stem.split("_")
    if len(parts) == 5:
        _, group_tag, n_str, k_str, d_str = parts
        return group_tag, int(n_str), int(k_str), int(d_str)
    if len(parts) == 4:
        _, n_str, k_str, d_str = parts
        return None, int(n_str), int(k_str), int(d_str)
    raise ValueError(f"Unrecognized filename stem {stem!r}.")


def _group_spec_from_tag(
    tag: Optional[str],
    *,
    m: int,
    gap_cmd: str,
    cache_dir: Path,
) -> Tuple[str, str, Optional[Dict[str, object]]]:
    if tag is None:
        return f"C{m}", "inferred_cyclic", None
    clean = tag.strip()
    if re.fullmatch(r"C\d+", clean):
        return clean, "tag_cyclic", None
    if re.fullmatch(r"C\d+C\d+", clean):
        left = clean[1:].split("C", 1)[0]
        right = clean.split("C", 2)[2]
        return f"C{left}xC{right}", "tag_direct_product", None
    if clean == "C2C2":
        return "C2xC2", "tag_direct_product", None
    if clean == "C6C2":
        return "C6xC2", "tag_direct_product", None
    if clean == "Q8":
        spec = _resolve_tag_via_gap(
            tag="Q8",
            order=8,
            gap_cmd=gap_cmd,
            cache_dir=cache_dir,
        )
        if spec:
            return spec, "tag_structure_Q8", {"tag": "Q8"}
        return "Q8", "tag_unresolved_no_gap", {"tag": "Q8"}
    if clean == "C4rC4":
        spec = _resolve_tag_via_gap(
            tag="C4rC4",
            order=16,
            gap_cmd=gap_cmd,
            cache_dir=cache_dir,
        )
        if spec:
            return spec, "tag_structure_C4rC4", {"tag": "C4rC4"}
        return "C4rC4", "tag_unresolved_no_gap", {"tag": "C4rC4"}
    return clean, "tag_raw", {"tag": clean}


def _structure_cache_path(order: int, cache_dir: Path) -> Path:
    return cache_dir / f"paper_extract_structure_{order}.json"


def _tag_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "paper_extract_tag_map.json"


def _load_tag_cache(cache_dir: Path) -> Dict[str, object]:
    path = _tag_cache_path(cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tag_cache(cache_dir: Path, payload: Dict[str, object]) -> None:
    path = _tag_cache_path(cache_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _gap_structure_map(order: int, *, gap_cmd: str, cache_dir: Path) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _structure_cache_path(order, cache_dir)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("structures"):
            return {str(k): str(v) for k, v in payload["structures"].items()}
    if not gap_is_available(gap_cmd):
        return {}
    script = "\n".join(
        [
            'if LoadPackage("smallgrp") = fail then',
            '  Print("QTANNER_GAP_ERROR smallgrp_missing\\n");',
            "  QuitGap(2);",
            "fi;",
            f"n := NrSmallGroups({order});;",
            "for i in [1..n] do",
            f"  Print(i, \":\", StructureDescription(SmallGroup({order}, i)), \"\\n\");",
            "od;",
            "QuitGap(0);",
        ]
    )
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".g", delete=False) as tmp:
            tmp.write(script)
            script_path = tmp.name
        result = subprocess.run(
            [gap_cmd, "-q", "-b", "--quitonbreak", script_path],
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
    finally:
        if "script_path" in locals():
            try:
                os.remove(script_path)
            except OSError:
                pass
    if returncode != 0:
        return {}
    structures: Dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        idx, desc = line.split(":", 1)
        idx = idx.strip()
        desc = desc.strip()
        if idx.isdigit() and desc:
            structures[idx] = desc
    cache_path.write_text(
        json.dumps({"order": order, "structures": structures}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return structures


def _resolve_tag_via_gap(
    *,
    tag: str,
    order: int,
    gap_cmd: str,
    cache_dir: Path,
) -> Optional[str]:
    cache = _load_tag_cache(cache_dir)
    cache_key = f"{tag}_{order}"
    if cache_key in cache:
        return cache[cache_key].get("spec")
    structures = _gap_structure_map(order, gap_cmd=gap_cmd, cache_dir=cache_dir)
    if not structures:
        return None
    spec = None
    if tag == "Q8":
        for gid, desc in structures.items():
            if desc == "Q8":
                spec = f"SmallGroup({order},{gid})"
                break
    elif tag == "C4rC4":
        for gid, desc in structures.items():
            if "C4" in desc and ":" in desc:
                spec = f"SmallGroup({order},{gid})"
                break
    if spec:
        cache[cache_key] = {"spec": spec, "tag": tag, "order": order}
        _write_tag_cache(cache_dir, cache)
    return spec


def _smallgroup_count(order: int) -> Optional[int]:
    counts = {1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 5, 9: 2, 10: 2, 11: 1, 12: 5, 13: 1, 14: 2, 15: 1, 16: 14}
    return counts.get(order)


def _select_smallgroup_spec(
    *,
    order: int,
    left_perms: Sequence[Tuple[int, ...]],
    right_perms: Sequence[Tuple[int, ...]],
    gap_cmd: str,
    cache_dir: Path,
) -> Tuple[str, Dict[str, int]]:
    count = _smallgroup_count(order)
    if count is None:
        raise ValueError(f"No SmallGroup count known for order {order}.")
    best_spec = f"SmallGroup({order},1)"
    best_score = -1
    best_stats = {"left_matches": 0, "right_matches": 0, "total_left": len(left_perms), "total_right": len(right_perms)}
    for gid in range(1, count + 1):
        spec = f"SmallGroup({order},{gid})"
        try:
            group = group_from_spec(spec, gap_cmd=gap_cmd, cache_dir=cache_dir)
        except Exception:
            continue
        left_map, right_map = _build_left_right_maps(group)
        left_matches = sum(1 for perm in left_perms if perm in left_map)
        right_matches = sum(1 for perm in right_perms if perm in right_map)
        score = left_matches + right_matches
        if score > best_score:
            best_spec = spec
            best_score = score
            best_stats = {
                "left_matches": left_matches,
                "right_matches": right_matches,
                "total_left": len(left_perms),
                "total_right": len(right_perms),
            }
    return best_spec, best_stats


def _load_group(
    spec: str,
    *,
    gap_cmd: str,
    cache_dir: Path,
) -> Tuple[Optional[FiniteGroup], Optional[str]]:
    try:
        group = group_from_spec(spec, gap_cmd=gap_cmd, cache_dir=cache_dir)
    except Exception as exc:
        return None, str(exc)
    return group, None


def _collect_pairs(root: Path) -> List[dict]:
    pairs: Dict[Tuple[str, Optional[str], int, int, int], dict] = {}
    for path in sorted(root.rglob("*.mtx")):
        stem = path.stem
        if not (stem.startswith("HX_") or stem.startswith("HZ_")):
            continue
        group_tag, n_val, k_val, d_val = _parse_stem(stem)
        folder = path.parent.name
        key = (folder, group_tag, n_val, k_val, d_val)
        entry = pairs.setdefault(
            key,
            {
                "folder": folder,
                "group_tag": group_tag,
                "n": n_val,
                "k": k_val,
                "d": d_val,
                "hx": None,
                "hz": None,
            },
        )
        if stem.startswith("HX_"):
            entry["hx"] = path
        else:
            entry["hz"] = path
    results = []
    for entry in pairs.values():
        if entry["hx"] is None or entry["hz"] is None:
            continue
        results.append(entry)
    return sorted(results, key=lambda e: (e["folder"], e["group_tag"] or "", e["n"], e["k"], e["d"]))


def _local_dims_from_folder(folder: str) -> Tuple[int, int, int, int]:
    if folder == "633x633":
        return 6, 3, 6, 3
    if folder == "633x212":
        return 6, 3, 2, 1
    if folder == "844x212":
        return 8, 4, 2, 1
    raise ValueError(f"Unrecognized local code folder {folder!r}.")


def _extract_from_pair(
    *,
    hx_path: Path,
    hz_path: Path,
    folder: str,
    group_tag: Optional[str],
    n_val: int,
    k_val: int,
    d_val: int,
    gap_cmd: str,
    cache_dir: Path,
    debug: bool = False,
) -> dict:
    nA, rA, nB, rB = _local_dims_from_folder(folder)
    hx_m, hx_n, hx_rows = read_mtx_coordinate_binary(hx_path)
    hz_m, hz_n, hz_rows = read_mtx_coordinate_binary(hz_path)
    if hx_n != hz_n:
        raise ValueError(f"Column mismatch: {hx_path} has n={hx_n}, {hz_path} has n={hz_n}.")
    if hx_n % (nA * nB) != 0:
        raise ValueError(f"n={hx_n} is not divisible by nA*nB={nA*nB}.")
    m = hx_n // (nA * nB)
    if hx_n != nA * nB * m:
        raise ValueError(f"Unexpected n={hx_n}; expected nA*nB*m={nA*nB*m}.")
    if hx_m != rA * nB * m:
        raise ValueError(f"HX rows {hx_m} != rA*nB*m={rA*nB*m}.")
    if hz_m != nA * rB * m:
        raise ValueError(f"HZ rows {hz_m} != nA*rB*m={nA*rB*m}.")
    code_id = f"{group_tag or 'C' + str(m)}_{n_val}_{k_val}_{d_val}"
    kA = nA // 2
    kB = nB // 2

    hx_entries = list(_iter_nonzero_entries(hx_rows))
    hz_entries = list(_iter_nonzero_entries(hz_rows))

    hx_row_axes = ["r", "j", "g"]
    hx_row_sizes = [rA, nB, m]
    col_axes = ["i", "j", "g"]
    col_sizes = [nA, nB, m]
    hz_candidates = [
        (["i", "r", "g"], [nA, rB, m], "ir"),
        (["r", "j", "g"], [rA, nB, m], "rj"),
    ]

    mapping_choice = _select_axes_mapping_candidates(
        hx_entries=hx_entries,
        hz_entries=hz_entries,
        hx_row_axes=hx_row_axes,
        hx_row_sizes=hx_row_sizes,
        hz_candidates=hz_candidates,
        col_axes=col_axes,
        col_sizes=col_sizes,
        nA=nA,
        nB=nB,
        m=m,
    )

    row_order_hx = mapping_choice["row_order_hx"]
    row_order_hz = mapping_choice["row_order_hz"]
    col_order = mapping_choice["col_order"]
    hz_row_axes = mapping_choice["hz_row_axes"]

    hx_block_perm = mapping_choice["hx_block_perm"]
    hx_block_nnz = mapping_choice["hx_block_nnz"]
    hx_row_blocks = mapping_choice["hx_row_blocks"]
    hz_block_perm = mapping_choice["hz_block_perm"]
    hz_block_nnz = mapping_choice["hz_block_nnz"]
    hz_row_blocks = mapping_choice["hz_row_blocks"]

    cb_to_ij = _cb_to_ij(nA, nB)
    cb_option = "i_nB_j"

    hx_types, hx_counts = _classify_row_blocks(hx_row_blocks, cb_to_ij=cb_to_ij)
    hz_types, hz_counts = _classify_row_blocks(hz_row_blocks, cb_to_ij=cb_to_ij)

    def collect_perms(block_perm, block_nnz):
        perms = {}
        stats = {"blocks": 0, "eq_m": 0, "valid": 0, "invalid": 0}
        for key, gr_map in block_perm.items():
            nnz = block_nnz.get(key, 0)
            if nnz <= 0:
                continue
            stats["blocks"] += 1
            if nnz == m:
                stats["eq_m"] += 1
            if nnz != m:
                continue
            perm, ok = _perm_from_block_map(gr_map, m=m)
            if perm is None or not ok:
                stats["invalid"] += 1
                continue
            perms[key] = perm
            stats["valid"] += 1
        return perms, stats

    hx_perms, hx_perm_stats = collect_perms(hx_block_perm, hx_block_nnz)
    hz_perms, hz_perm_stats = collect_perms(hz_block_perm, hz_block_nnz)

    hz_row_label = mapping_choice["hz_row_label"]

    def hz_half(rb: int) -> int:
        return _row_block_half_hz(rb, label=hz_row_label, kA=kA, rB=rB, nB=nB)

    def hx_half(rb: int) -> int:
        return _row_block_half_hx(rb, nB=nB, kB=kB)

    default_b_half_perm1 = 0
    left_perms: List[Tuple[int, ...]] = []
    right_perms: List[Tuple[int, ...]] = []
    for (rb, _), perm in hz_perms.items():
        half = hz_half(rb)
        if half == default_b_half_perm1:
            right_perms.append(perm)
        else:
            left_perms.append(perm)
    if not left_perms:
        left_perms = list(hx_perms.values())
    if not right_perms:
        right_perms = list(hx_perms.values())

    group_spec, spec_source, tag_info = _group_spec_from_tag(
        group_tag, m=m, gap_cmd=gap_cmd, cache_dir=cache_dir
    )
    group = None
    group_error = None
    group_spec_final = group_spec
    smallgroup_stats = None
    if group_spec.startswith("SmallGroup") and spec_source not in (
        "tag_structure_Q8",
        "tag_structure_C4rC4",
    ):
        if gap_is_available(gap_cmd):
            match_spec, match_stats = _select_smallgroup_spec(
                order=m,
                left_perms=left_perms,
                right_perms=right_perms,
                gap_cmd=gap_cmd,
                cache_dir=cache_dir,
            )
            group_spec_final = match_spec
            spec_source = "smallgroup_match"
            smallgroup_stats = match_stats
        else:
            spec_source = "smallgroup_gap_missing"
    group, group_error = _load_group(group_spec_final, gap_cmd=gap_cmd, cache_dir=cache_dir)

    left_map = {}
    left_inv_map = {}
    right_map = {}
    right_inv_map = {}
    if group is not None:
        left_map, left_inv_map, right_map, right_inv_map = _build_perm_maps(group)
    is_abelian = group is not None and _is_abelian(group)

    a_supports_by_j = _support_sets_for_a(hx_row_blocks, cb_to_ij=cb_to_ij, nB=nB)
    permA_halves: Dict[int, List[Tuple[int, ...]]] = {0: [], 1: []}
    if nA == 2:
        permA_halves[0] = [tuple(range(nA))]
        permA_halves[1] = [tuple(range(nA))]
    else:
        permA_halves[0] = _perms_from_supports_by_index(
            nA,
            supports_by_index=a_supports_by_j,
            indices=range(0, kB),
            canonical_supports=_canonical_supports(nA),
        )
        permA_halves[0] += _perms_from_supports_by_index(
            nA,
            supports_by_index=a_supports_by_j,
            indices=range(0, kB),
            canonical_supports=_canonical_supports_g(nA),
        )
        permA_halves[0] = list(dict.fromkeys(permA_halves[0]))
        permA_halves[1] = _perms_from_supports_by_index(
            nA,
            supports_by_index=a_supports_by_j,
            indices=range(kB, nB),
            canonical_supports=_canonical_supports(nA),
        )
        permA_halves[1] += _perms_from_supports_by_index(
            nA,
            supports_by_index=a_supports_by_j,
            indices=range(kB, nB),
            canonical_supports=_canonical_supports_g(nA),
        )
        permA_halves[1] = list(dict.fromkeys(permA_halves[1]))
    if not permA_halves[0]:
        permA_halves[0] = [tuple(range(nA))]
    if not permA_halves[1]:
        permA_halves[1] = [tuple(range(nA))]

    permB_halves: Dict[int, List[Tuple[int, ...]]] = {0: [], 1: []}
    if nB == 2:
        permB_halves[0] = [tuple(range(nB))]
        permB_halves[1] = [tuple(range(nB))]
    else:
        if hz_row_label == "rj":
            b_supports_by_j = _support_sets_for_b_by_j(
                hz_row_blocks, cb_to_ij=cb_to_ij, nB=nB
            )
            permB_halves[0] = _perms_from_supports_by_index(
                nB,
                supports_by_index=b_supports_by_j,
                indices=range(0, rB),
                canonical_supports=_canonical_supports(nB),
            )
            permB_halves[0] += _perms_from_supports_by_index(
                nB,
                supports_by_index=b_supports_by_j,
                indices=range(0, rB),
                canonical_supports=_canonical_supports_g(nB),
            )
            permB_halves[0] = list(dict.fromkeys(permB_halves[0]))
            permB_halves[1] = _perms_from_supports_by_index(
                nB,
                supports_by_index=b_supports_by_j,
                indices=range(rB, nB),
                canonical_supports=_canonical_supports(nB),
            )
            permB_halves[1] += _perms_from_supports_by_index(
                nB,
                supports_by_index=b_supports_by_j,
                indices=range(rB, nB),
                canonical_supports=_canonical_supports_g(nB),
            )
            permB_halves[1] = list(dict.fromkeys(permB_halves[1]))
        else:
            b_supports_by_half = _support_sets_for_b_by_half(
                hz_row_blocks,
                cb_to_ij=cb_to_ij,
                label=hz_row_label,
                kA=kA,
                rB=rB,
                nB=nB,
            )
            permB_halves[0] = _perm_candidates_from_supports(
                nB, b_supports_by_half[0], canonical_supports=_canonical_supports(nB)
            )
            permB_halves[0] += _perm_candidates_from_supports(
                nB, b_supports_by_half[0], canonical_supports=_canonical_supports_g(nB)
            )
            permB_halves[0] = list(dict.fromkeys(permB_halves[0]))
            permB_halves[1] = _perm_candidates_from_supports(
                nB, b_supports_by_half[1], canonical_supports=_canonical_supports(nB)
            )
            permB_halves[1] += _perm_candidates_from_supports(
                nB, b_supports_by_half[1], canonical_supports=_canonical_supports_g(nB)
            )
            permB_halves[1] = list(dict.fromkeys(permB_halves[1]))
    if not permB_halves[0]:
        permB_halves[0] = [tuple(range(nB))]
    if not permB_halves[1]:
        permB_halves[1] = [tuple(range(nB))]

    def finalize_votes(votes):
        result = []
        margins = []
        for counter in votes:
            if not counter:
                result.append(None)
                margins.append(None)
                continue
            items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            top_elem, top_count = items[0]
            second = items[1][1] if len(items) > 1 else 0
            result.append(int(top_elem))
            margins.append(int(top_count - second))
        return result, margins

    def _attempt_reconstruct(
        pA0,
        pA1,
        pB0,
        pB1,
        *,
        A_use,
        B_use,
        sigma_inv,
        row_axes_hz,
    ):
        return _reconstruct_and_compare(
            group=group,
            A_ids=[int(a) if a is not None else None for a in A_use],
            B_ids=[int(b) if b is not None else None for b in B_use],
            permA0=pA0,
            permA1=pA1,
            permB0=pB0,
            permB1=pB1,
            nA=nA,
            nB=nB,
            rA=rA,
            rB=rB,
            m=m,
            row_order_hx=row_order_hx,
            row_order_hz=row_order_hz,
            col_order=col_order,
            row_axes_hz=row_axes_hz,
            sigma_inv=sigma_inv,
            hx_rows_in=hx_rows,
            hz_rows_in=hz_rows,
        )

    best_attempt: Optional[Dict[str, object]] = None
    best_score = -1

    def attempt(
        *,
        permA0: Tuple[int, ...],
        permA1: Tuple[int, ...],
        permB0: Tuple[int, ...],
        permB1: Tuple[int, ...],
        a_half_perm1: int,
        b_half_perm1: int,
    ) -> Dict[str, object]:
        left_perms_local: List[Tuple[int, ...]] = []
        right_perms_local: List[Tuple[int, ...]] = []
        for (rb, _), perm in hz_perms.items():
            half = hz_half(rb)
            if half == b_half_perm1:
                right_perms_local.append(perm)
            else:
                left_perms_local.append(perm)

        sigma_candidates: List[Tuple[Optional[List[int]], Dict[str, object]]] = [
            (None, {"source": "none"})
        ]
        sigma_base_perms = left_perms_local or right_perms_local
        if not sigma_base_perms:
            sigma_base_perms = list(dict.fromkeys(list(hx_perms.values()) + list(hz_perms.values())))
        if group is not None and left_map:
            if isinstance(group, CyclicGroup):
                sigmas, best_score = _sigma_candidates_cyclic_affine(
                    m=m,
                    perms=sigma_base_perms,
                    left_map=left_map,
                    right_map=right_map,
                )
                if sigmas:
                    sigma_candidates = [
                        (
                            sigma,
                            {
                                "source": "cyclic_affine",
                                "best_score": best_score,
                                "candidate_count": len(sigmas),
                            },
                        )
                        for sigma in sigmas
                    ]
            else:
                sigma_candidates = []
                sigma_candidates_raw = None
                sigma_source = "none"
                sigma_closure = None
                if left_perms_local:
                    gens, closure = _select_perm_generators(left_perms_local, m=m)
                    sigma_closure = len(closure)
                    if len(closure) == m:
                        sigma_candidates_raw = gens
                        sigma_source = "hz_left"
                if sigma_candidates_raw is None and right_perms_local:
                    gens, closure = _select_perm_generators(right_perms_local, m=m)
                    sigma_closure = len(closure)
                    if len(closure) == m:
                        sigma_candidates_raw = gens
                        sigma_source = "hz_right"
                if sigma_candidates_raw is None:
                    gens, closure = _select_perm_generators(sigma_base_perms, m=m)
                    sigma_closure = len(closure)
                    sigma_candidates_raw = gens if gens else sigma_base_perms
                    sigma_source = "all_perms"
                sigma, sigma_info = _infer_sigma(
                    sigma_candidates_raw,
                    group=group,
                    left_map=left_map,
                    right_map=right_map,
                    gap_cmd=gap_cmd,
                    cache_dir=cache_dir,
                    code_id=code_id,
                )
                sigma_info = {
                    **sigma_info,
                    "candidates": sigma_source,
                    "closure_size": sigma_closure,
                }
                sigma_candidates = [(sigma, sigma_info)]

        def run_with_sigma(
            sigma: Optional[List[int]], sigma_info: Dict[str, object]
        ) -> Dict[str, object]:
            a_votes = [Counter() for _ in range(nA)]
            b_votes = [Counter() for _ in range(nB)]
            c_votes: Dict[Tuple[int, int], Counter] = {}
            a_stats = {"total": 0, "mismatches": 0, "match_counts": Counter()}
            b_stats = {"total": 0, "mismatches": 0, "match_counts": Counter()}

            sigma_inv = _perm_inverse(tuple(sigma)) if sigma is not None else None

            for (rb, cb), perm in hz_perms.items():
                half = hz_half(rb)
                i_val, j_val = cb_to_ij(cb)
                perm_use = perm
                if sigma is not None and sigma_inv is not None:
                    perm_use = _conjugate_perm(
                        perm, sigma=sigma, sigma_inv=sigma_inv
                    )
                if half == b_half_perm1:
                    b_stats["total"] += 1
                    matchB = None
                    if group is not None:
                        matchB = _match_perm_with_inverse(
                            perm_use,
                            left_map=left_map,
                            left_inv_map=left_inv_map,
                            right_map=right_map,
                            right_inv_map=right_inv_map,
                            prefer=("R", "R_inv", "L", "L_inv"),
                        )
                    if matchB and group is not None:
                        match_type, elem, inverse_used = matchB
                        elem_val = _element_from_match_right(
                            match_type, elem, inverse_used, group
                        )
                        b_votes[j_val][int(elem_val)] += 1
                        key = f"{match_type}{'_T' if inverse_used else ''}"
                        b_stats["match_counts"][key] += 1
                    else:
                        b_stats["mismatches"] += 1
                else:
                    a_stats["total"] += 1
                    matchA = None
                    if group is not None:
                        matchA = _match_perm_with_inverse(
                            perm_use,
                            left_map=left_map,
                            left_inv_map=left_inv_map,
                            right_map=right_map,
                            right_inv_map=right_inv_map,
                            prefer=("L", "L_inv", "R", "R_inv"),
                        )
                    if matchA and group is not None:
                        match_type, elem, inverse_used = matchA
                        elem_val = _element_from_match_left(
                            match_type, elem, inverse_used, group
                        )
                        a_votes[i_val][int(elem_val)] += 1
                        key = f"{match_type}{'_T' if inverse_used else ''}"
                        a_stats["match_counts"][key] += 1
                    else:
                        a_stats["mismatches"] += 1

            for (rb, cb), perm in hx_perms.items():
                if hx_half(rb) != a_half_perm1:
                    continue
                i_val, j_val = cb_to_ij(cb)
                perm_use = perm
                if sigma is not None and sigma_inv is not None:
                    perm_use = _conjugate_perm(
                        perm, sigma=sigma, sigma_inv=sigma_inv
                    )
                if group is None:
                    continue
                if is_abelian:
                    c_votes.setdefault((i_val, j_val), Counter())[
                        int(perm_use[0])
                    ] += 1
                    continue
                lr = _decompose_perm_lr(perm_use, group)
                a_stats["total"] += 1
                b_stats["total"] += 1
                if lr:
                    a_elem, b_elem_inv = lr
                    a_votes[i_val][int(a_elem)] += 1
                    b_votes[j_val][int(group.inv(b_elem_inv))] += 1
                    a_stats["match_counts"]["LR"] += 1
                    b_stats["match_counts"]["LR"] += 1
                else:
                    a_stats["mismatches"] += 1
                    b_stats["mismatches"] += 1

            if is_abelian and c_votes and group is not None:
                c_map = {}
                for key, counter in c_votes.items():
                    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
                    c_map[key] = int(items[0][0])
                A_seed = [None] * nA
                B_seed = [None] * nB
                if c_map:
                    i0 = min(i for (i, _) in c_map.keys())
                    A_seed[i0] = int(group.id())
                    for (i_val, j_val), c_val in c_map.items():
                        if i_val == i0 and B_seed[j_val] is None:
                            B_seed[j_val] = int(
                                group.mul(group.inv(c_val), A_seed[i0])
                            )
                    changed = True
                    while changed:
                        changed = False
                        for (i_val, j_val), c_val in c_map.items():
                            if A_seed[i_val] is None and B_seed[j_val] is not None:
                                A_seed[i_val] = int(group.mul(c_val, B_seed[j_val]))
                                changed = True
                            if B_seed[j_val] is None and A_seed[i_val] is not None:
                                B_seed[j_val] = int(
                                    group.mul(group.inv(c_val), A_seed[i_val])
                                )
                                changed = True
                for i_val, val in enumerate(A_seed):
                    if val is None and a_votes[i_val]:
                        A_seed[i_val] = int(
                            max(a_votes[i_val].items(), key=lambda kv: (kv[1], -kv[0]))[0]
                        )
                for j_val, val in enumerate(B_seed):
                    if val is None and b_votes[j_val]:
                        B_seed[j_val] = int(
                            max(b_votes[j_val].items(), key=lambda kv: (kv[1], -kv[0]))[0]
                        )
                best_t = int(group.id())
                best_score = -1
                for t in group.elements():
                    score = 0
                    for i_val, val in enumerate(A_seed):
                        if val is None:
                            continue
                        score += a_votes[i_val].get(group.mul(val, t), 0)
                    for j_val, val in enumerate(B_seed):
                        if val is None:
                            continue
                        score += b_votes[j_val].get(group.mul(val, t), 0)
                    if score > best_score:
                        best_score = score
                        best_t = int(t)
                if best_score >= 0:
                    A_seed = [
                        group.mul(val, best_t) if val is not None else None
                        for val in A_seed
                    ]
                    B_seed = [
                        group.mul(val, best_t) if val is not None else None
                        for val in B_seed
                    ]
                boost = 1000
                for i_val, val in enumerate(A_seed):
                    if val is not None:
                        a_votes[i_val][int(val)] += boost
                for j_val, val in enumerate(B_seed):
                    if val is not None:
                        b_votes[j_val][int(val)] += boost

            A_ids, A_margins = finalize_votes(a_votes)
            B_ids, B_margins = finalize_votes(b_votes)

            perm_match = {
                "left_matches": int(
                    sum(
                        count
                        for key, count in a_stats["match_counts"].items()
                        if key.startswith("L")
                    )
                ),
                "right_matches": int(
                    sum(
                        count
                        for key, count in b_stats["match_counts"].items()
                        if key.startswith("R")
                    )
                ),
                "total_left": int(a_stats.get("total", 0)),
                "total_right": int(b_stats.get("total", 0)),
                "match_counts_A": dict(Counter(a_stats["match_counts"])),
                "match_counts_B": dict(Counter(b_stats["match_counts"])),
            }
            vote_histograms = {
                "A": [dict(counter) for counter in a_votes],
                "B": [dict(counter) for counter in b_votes],
            }

            recon_ok = False
            recon_reason = "group_missing" if group is None else "unattempted"
            A_ids_use = A_ids
            B_ids_use = B_ids
            if group is not None and all(a is not None for a in A_ids) and all(
                b is not None for b in B_ids
            ):
                variants = [(A_ids, B_ids)]
                invA = [group.inv(a) for a in A_ids]
                invB = [group.inv(b) for b in B_ids]
                variants.extend(
                    [
                        (invA, B_ids),
                        (A_ids, invB),
                        (invA, invB),
                    ]
                )
                for A_var, B_var in variants:
                    if is_abelian:
                        for t in group.elements():
                            A_shift = [group.mul(a, t) for a in A_var]
                            B_shift = [group.mul(b, t) for b in B_var]
                            recon_ok, recon_reason = _attempt_reconstruct(
                                permA0,
                                permA1,
                                permB0,
                                permB1,
                                A_use=A_shift,
                                B_use=B_shift,
                                sigma_inv=sigma_inv,
                                row_axes_hz=hz_row_label,
                            )
                            if recon_ok:
                                A_ids_use = A_shift
                                B_ids_use = B_shift
                                break
                        if recon_ok:
                            break
                    else:
                        recon_ok, recon_reason = _attempt_reconstruct(
                            permA0,
                            permA1,
                            permB0,
                            permB1,
                            A_use=A_var,
                            B_use=B_var,
                            sigma_inv=sigma_inv,
                            row_axes_hz=hz_row_label,
                        )
                        if recon_ok:
                            A_ids_use = A_var
                            B_ids_use = B_var
                            break
            if recon_ok:
                A_ids_out = [int(a) for a in A_ids_use]
                B_ids_out = [int(b) for b in B_ids_use]
            else:
                A_ids_out = [None] * nA
                B_ids_out = [None] * nB

            return {
                "permA0": permA0,
                "permA1": permA1,
                "permB0": permB0,
                "permB1": permB1,
                "A_ids": A_ids_out,
                "B_ids": B_ids_out,
                "A_margins": A_margins,
                "B_margins": B_margins,
                "vote_histograms": vote_histograms,
                "vote_stats": {"A": a_stats, "B": b_stats},
                "perm_match": perm_match,
                "sigma": sigma,
                "sigma_info": sigma_info,
                "recon_ok": recon_ok,
                "recon_reason": recon_reason,
            }

        best_local = None
        best_local_score = -1
        for sigma, sigma_info in sigma_candidates:
            result = run_with_sigma(sigma, sigma_info)
            score = int(result["perm_match"]["left_matches"]) + int(
                result["perm_match"]["right_matches"]
            )
            if result["recon_ok"]:
                return result
            if score > best_local_score:
                best_local_score = score
                best_local = result
        if best_local is not None:
            return best_local
        return run_with_sigma(None, {"source": "none"})

    found = False
    for a_assign in (0, 1):
        permA0_candidates = permA_halves[a_assign]
        permA1_candidates = permA_halves[1 - a_assign]
        a_half_perm1 = 1 - a_assign
        for b_assign in (0, 1):
            permB0_candidates = permB_halves[b_assign]
            permB1_candidates = permB_halves[1 - b_assign]
            b_half_perm1 = 1 - b_assign
            for pA0 in permA0_candidates:
                for pA1 in permA1_candidates:
                    for pB0 in permB0_candidates:
                        for pB1 in permB1_candidates:
                            attempt_rec = attempt(
                                permA0=pA0,
                                permA1=pA1,
                                permB0=pB0,
                                permB1=pB1,
                                a_half_perm1=a_half_perm1,
                                b_half_perm1=b_half_perm1,
                            )
                            score = int(attempt_rec["perm_match"]["left_matches"]) + int(
                                attempt_rec["perm_match"]["right_matches"]
                            )
                            if attempt_rec["recon_ok"]:
                                best_attempt = attempt_rec
                                found = True
                                break
                            if score > best_score:
                                best_score = score
                                best_attempt = attempt_rec
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    if best_attempt is None:
        best_attempt = {
            "permA0": None,
            "permA1": None,
            "permB0": None,
            "permB1": None,
            "A_ids": [None] * nA,
            "B_ids": [None] * nB,
            "A_margins": [None] * nA,
            "B_margins": [None] * nB,
            "vote_histograms": {"A": [], "B": []},
            "vote_stats": {"A": {}, "B": {}},
            "perm_match": {
                "left_matches": 0,
                "right_matches": 0,
                "total_left": 0,
                "total_right": 0,
                "match_counts_A": {},
                "match_counts_B": {},
            },
            "sigma": None,
            "sigma_info": {"source": "none"},
            "recon_ok": False,
            "recon_reason": "unattempted",
        }

    permA0 = best_attempt["permA0"]
    permA1 = best_attempt["permA1"]
    permB0 = best_attempt["permB0"]
    permB1 = best_attempt["permB1"]
    permA = permA1
    permB = permB1
    A_ids = best_attempt["A_ids"]
    B_ids = best_attempt["B_ids"]
    A_margins = best_attempt["A_margins"]
    B_margins = best_attempt["B_margins"]
    vote_histograms = best_attempt["vote_histograms"]
    vote_stats = best_attempt["vote_stats"]
    perm_match = best_attempt["perm_match"]
    sigma = best_attempt["sigma"]
    sigma_info = best_attempt["sigma_info"]
    recon_ok = best_attempt["recon_ok"]
    recon_reason = best_attempt["recon_reason"]

    def perm_one_based(perm):
        if perm is None:
            return None
        return [int(x) + 1 for x in perm]

    if group is None:
        group = CyclicGroup(m, name=group_spec_final)

    perm_stats = {
        "hx": hx_perm_stats,
        "hz": hz_perm_stats,
    }

    A_repr = [group.repr(a) if a is not None else "?" for a in A_ids]
    B_repr = [group.repr(b) if b is not None else "?" for b in B_ids]

    debug_examples = []
    if debug and group is not None:
        def add_examples(perms, label):
            sigma_inv_local = _perm_inverse(tuple(sigma)) if sigma is not None else None
            for (rb, cb), perm in list(perms.items()):
                if len(debug_examples) >= 8:
                    return
                i_val, j_val = cb_to_ij(cb)
                perm_use = perm
                if sigma is not None and sigma_inv_local is not None:
                    perm_use = _conjugate_perm(perm, sigma=sigma, sigma_inv=sigma_inv_local)
                matchA = _match_perm_with_inverse(
                    perm_use,
                    left_map=left_map,
                    left_inv_map=left_inv_map,
                    right_map=right_map,
                    right_inv_map=right_inv_map,
                    prefer=("L", "L_inv", "R", "R_inv"),
                )
                matchB = _match_perm_with_inverse(
                    perm_use,
                    left_map=left_map,
                    left_inv_map=left_inv_map,
                    right_map=right_map,
                    right_inv_map=right_inv_map,
                    prefer=("R", "R_inv", "L", "L_inv"),
                )
                if matchA:
                    match_type, elem, inverse_used = matchA
                    elem_val = _element_from_match_left(match_type, elem, inverse_used, group)
                    debug_examples.append(
                        {
                            "src": label,
                            "rb": rb,
                            "cb": cb,
                            "kind": "A",
                            "i": i_val,
                            "j": j_val,
                            "match": f"{match_type}{'_T' if inverse_used else ''}",
                            "elem": group.repr(elem_val),
                        }
                    )
                elif matchB:
                    match_type, elem, inverse_used = matchB
                    elem_val = _element_from_match_right(match_type, elem, inverse_used, group)
                    debug_examples.append(
                        {
                            "src": label,
                            "rb": rb,
                            "cb": cb,
                            "kind": "B",
                            "i": i_val,
                            "j": j_val,
                            "match": f"{match_type}{'_T' if inverse_used else ''}",
                            "elem": group.repr(elem_val),
                        }
                    )

        add_examples(hx_perms, "HX")
        add_examples(hz_perms, "HZ")

    if debug:
        print(f"[paper_extract] debug {hx_path.stem}")
        print(
            "  row_order_hx="
            f"{_order_label(row_order_hx, hx_row_axes)} col_order="
            f"{_order_label(col_order, col_axes)}"
        )
        print(
            "  row_order_hz="
            f"{_order_label(row_order_hz, hz_row_axes)} (axes={hz_row_label})"
        )
        print(
            "  mapping score="
            f"{mapping_choice['score']} hx_score={mapping_choice['hx_score']} "
            f"hz_score={mapping_choice['hz_score']} class_score={mapping_choice.get('class_score')}"
        )
        full_blocks, eq_m_blocks = mapping_choice["score"][:2]
        print(f"  full_blocks={full_blocks} eq_m_blocks={eq_m_blocks}")
        print(f"  cb mapping={cb_option}")
        print(f"  row blocks HX={hx_counts} HZ={hz_counts}")
        print(f"  sigma={sigma if sigma is not None else 'None'}")
        print(
            f"  perm matches left {perm_match['left_matches']}/{perm_match['total_left']} "
            f"right {perm_match['right_matches']}/{perm_match['total_right']}"
        )
        print(f"  reconstruction_ok={recon_ok} reason={recon_reason}")
        if debug_examples:
            print("  examples:")
            for ex in debug_examples[:5]:
                print(
                    f"    src={ex['src']} rb={ex['rb']} cb={ex['cb']} kind={ex['kind']} "
                    f"i={ex['i']} j={ex['j']} match={ex['match']} elem={ex['elem']}"
                )

    return {
        "code_id": code_id,
        "folder": folder,
        "hx_path": str(hx_path),
        "hz_path": str(hz_path),
        "group_tag": group_tag,
        "group_spec": group_spec_final,
        "group_spec_source": spec_source,
        "group_order": m,
        "group_load_error": group_error,
        "smallgroup_match": smallgroup_stats,
        "n": n_val,
        "k": k_val,
        "d": d_val,
        "nA": nA,
        "rA": rA,
        "nB": nB,
        "rB": rB,
        "mapping_choice": {
            "row_order_hx": _order_label(row_order_hx, hx_row_axes),
            "row_order_hz": _order_label(row_order_hz, hz_row_axes),
            "col_order": _order_label(col_order, col_axes),
            "hz_row_axes": hz_row_label,
            "score": mapping_choice["score"],
            "hx_score": mapping_choice["hx_score"],
            "hz_score": mapping_choice["hz_score"],
            "class_score": mapping_choice.get("class_score"),
            "cb_option": cb_option,
        },
        "row_block_counts": {"hx": hx_counts, "hz": hz_counts},
        "A_ids": A_ids,
        "A_repr": A_repr,
        "B_ids": B_ids,
        "B_repr": B_repr,
        "permA0_0based": list(permA0) if permA0 is not None else None,
        "permA0_1based": perm_one_based(permA0),
        "permA_0based": list(permA) if permA is not None else None,
        "permA_1based": perm_one_based(permA),
        "permB0_0based": list(permB0) if permB0 is not None else None,
        "permB0_1based": perm_one_based(permB0),
        "permB_0based": list(permB) if permB is not None else None,
        "permB_1based": perm_one_based(permB),
        "perm_stats": perm_stats,
        "perm_match": perm_match,
        "sigma": sigma,
        "sigma_info": sigma_info,
        "reconstruction_ok": recon_ok,
        "reconstruction_error": recon_reason,
        "vote_margins": {"A": A_margins, "B": B_margins},
        "vote_histograms": vote_histograms,
        "vote_stats": vote_stats,
    }


def _render_markdown(results: List[dict]) -> str:
    lines = [
        "# LRZ paper code extraction",
        "",
        "## Index",
        "",
        "| code | group | n | k | d | folder |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in results:
        lines.append(
            f"| {rec['code_id']} | {rec['group_spec']} | {rec['n']} | {rec['k']} | {rec['d']} | {rec['folder']} |"
        )
    lines.append("")
    for rec in results:
        lines.append(f"## {rec['code_id']}")
        lines.append("")
        lines.append(f"- Files: HX=`{rec['hx_path']}`, HZ=`{rec['hz_path']}`")
        lines.append(f"- Group: {rec['group_spec']} (tag={rec['group_tag']}, |G|={rec['group_order']})")
        if rec.get("group_load_error"):
            lines.append(f"- Group load error: {rec['group_load_error']}")
        mapping = rec.get("mapping_choice", {})
        lines.append(
            f"- Local dims: nA={rec['nA']}, rA={rec.get('rA')}, nB={rec['nB']}, rB={rec.get('rB')}, "
            f"row_hx={mapping.get('row_order_hx')}, row_hz={mapping.get('row_order_hz')}, "
            f"col={mapping.get('col_order')}, cb_map={mapping.get('cb_option')}"
        )
        lines.append(f"- A: {rec['A_repr']}")
        lines.append(f"- B: {rec['B_repr']}")
        lines.append(
            f"- piA (0-based): {rec['permA_0based']} ; (1-based): {rec['permA_1based']}"
        )
        if "permA0_0based" in rec:
            lines.append(
                f"- piA0 (0-based): {rec['permA0_0based']} ; (1-based): {rec['permA0_1based']}"
            )
        lines.append(
            f"- piB (0-based): {rec['permB_0based']} ; (1-based): {rec['permB_1based']}"
        )
        if "permB0_0based" in rec:
            lines.append(
                f"- piB0 (0-based): {rec['permB0_0based']} ; (1-based): {rec['permB0_1based']}"
            )
        lines.append(
            f"- Row blocks (HX): {rec['row_block_counts']['hx']} ; (HZ): {rec['row_block_counts']['hz']}"
        )
        if rec.get("perm_match"):
            match = rec["perm_match"]
            lines.append(
                f"- Perm matches: left {match['left_matches']}/{match['total_left']}, "
                f"right {match['right_matches']}/{match['total_right']}"
            )
        if "reconstruction_ok" in rec:
            lines.append(
                f"- Reconstruction: {rec['reconstruction_ok']} ({rec.get('reconstruction_error')})"
            )
        lines.append(f"- Vote margins A: {rec['vote_margins']['A']}")
        lines.append(f"- Vote margins B: {rec['vote_margins']['B']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract LRZ paper codes from MatrixMarket files.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input root directory.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output Markdown path.")
    parser.add_argument("--json", dest="json_path", required=True, help="Output JSON path.")
    parser.add_argument("--gap-cmd", default="gap", help="GAP command (for SmallGroup).")
    parser.add_argument(
        "--debug-one",
        dest="debug_one",
        default=None,
        help="HX/HZ stem to debug (e.g. HX_60_2_8).",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    out_path = Path(args.out_path)
    json_path = Path(args.json_path)
    cache_dir = Path("runs") / "_group_cache"

    pairs = _collect_pairs(root)
    if not pairs:
        print(f"[paper_extract] no HX/HZ pairs found under {root}.", file=sys.stderr)
        return 1

    results = []
    for pair in pairs:
        debug = False
        if args.debug_one:
            stem_match = args.debug_one.strip()
            if stem_match not in (pair["hx"].stem, pair["hz"].stem):
                continue
            debug = True
        rec = _extract_from_pair(
            hx_path=pair["hx"],
            hz_path=pair["hz"],
            folder=pair["folder"],
            group_tag=pair["group_tag"],
            n_val=pair["n"],
            k_val=pair["k"],
            d_val=pair["d"],
            gap_cmd=args.gap_cmd,
            cache_dir=cache_dir,
            debug=debug,
        )
        results.append(rec)

    out_path.write_text(_render_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[paper_extract] wrote {out_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
