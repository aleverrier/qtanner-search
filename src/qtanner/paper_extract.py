"""Extract group and local-code structure from LRZ paper matrices."""

from __future__ import annotations

import argparse
import json
import itertools
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
from .qdistrnd import gap_is_available


_H_6_3_3 = [
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 0],
]

_H_8_4_4 = [
    [1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
]


def _canonical_supports(n: int) -> List[frozenset[int]]:
    if n == 2:
        return [frozenset({0, 1})]
    if n == 6:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _H_6_3_3]
    if n == 8:
        return [frozenset(idx for idx, v in enumerate(row) if v) for row in _H_8_4_4]
    raise ValueError(f"Unsupported local code length {n}.")


def _perm_inverse(perm: Sequence[int]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[int(j)] = int(i)
    return tuple(inv)


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


def _cb_to_ij_option(nA: int, nB: int, option: str):
    if option == "i_nB_j":
        return lambda cb: (cb // nB, cb % nB)
    if option == "j_nA_i":
        return lambda cb: (cb % nA, cb // nA)
    raise ValueError(f"Unknown option {option!r}.")


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
    row_types: Sequence[Optional[Tuple[str, int]]],
    *,
    cb_to_ij,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_j: Dict[int, set[frozenset[int]]] = {}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        classification = row_types[rb]
        if not classification or classification[0] != "A":
            continue
        j_val = classification[1]
        i_set = {cb_to_ij(cb)[0] for cb in touched}
        supports_by_j.setdefault(j_val, set()).add(frozenset(i_set))
    return supports_by_j


def _support_sets_for_b(
    row_blocks: Sequence[set[int]],
    row_types: Sequence[Optional[Tuple[str, int]]],
    *,
    cb_to_ij,
) -> Dict[int, set[frozenset[int]]]:
    supports_by_i: Dict[int, set[frozenset[int]]] = {}
    for rb, touched in enumerate(row_blocks):
        if not touched:
            continue
        classification = row_types[rb]
        if not classification or classification[0] != "B":
            continue
        i_val = classification[1]
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


def _local_dims_from_folder(folder: str) -> Tuple[int, int]:
    if folder == "633x633":
        return 6, 6
    if folder == "633x212":
        return 6, 2
    if folder == "844x212":
        return 8, 2
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
    nA, nB = _local_dims_from_folder(folder)
    hx_m, hx_n, hx_rows = read_mtx_coordinate_binary(hx_path)
    hz_m, hz_n, hz_rows = read_mtx_coordinate_binary(hz_path)
    if hx_n != hz_n:
        raise ValueError(f"Column mismatch: {hx_path} has n={hx_n}, {hz_path} has n={hz_n}.")
    if hx_n % (nA * nB) != 0:
        raise ValueError(f"n={hx_n} is not divisible by nA*nB={nA*nB}.")
    m = hx_n // (nA * nB)
    n_blocks = nA * nB

    hx_entries = list(_iter_nonzero_entries(hx_rows))
    hz_entries = list(_iter_nonzero_entries(hz_rows))

    mapping_choice = _select_fiber_mappings(
        hx_entries=hx_entries,
        hz_entries=hz_entries,
        hx_shape=(hx_m, hx_n),
        hz_shape=(hz_m, hz_n),
        m=m,
    )
    row_mode = mapping_choice["row_mode"]
    col_mode = mapping_choice["col_mode"]

    hx_block_perm, hx_row_blocks = _build_block_maps(
        hx_entries, n_rows=hx_m, n_cols=hx_n, m=m, row_mode=row_mode, col_mode=col_mode
    )
    hz_block_perm, hz_row_blocks = _build_block_maps(
        hz_entries, n_rows=hz_m, n_cols=hz_n, m=m, row_mode=row_mode, col_mode=col_mode
    )

    cb_choice = _select_cb_mapping(
        nA=nA,
        nB=nB,
        hx_row_blocks=hx_row_blocks,
        hz_row_blocks=hz_row_blocks,
    )
    cb_to_ij = cb_choice["cb_to_ij"]

    hx_types, hx_counts = _classify_row_blocks(hx_row_blocks, cb_to_ij=cb_to_ij)
    hz_types, hz_counts = _classify_row_blocks(hz_row_blocks, cb_to_ij=cb_to_ij)

    def collect_perms(block_perm):
        perms = {}
        stats = {"nonzero": 0, "valid": 0, "invalid": 0}
        for key, gr_map in block_perm.items():
            if not gr_map:
                continue
            stats["nonzero"] += 1
            perm, ok = _perm_from_block_map(gr_map, m=m)
            if perm is None or not ok:
                stats["invalid"] += 1
                continue
            perms[key] = perm
            stats["valid"] += 1
        return perms, stats

    hx_perms, hx_perm_stats = collect_perms(hx_block_perm)
    hz_perms, hz_perm_stats = collect_perms(hz_block_perm)

    left_perms = [perm for (rb, _), perm in hx_perms.items() if hx_types[rb] and hx_types[rb][0] == "A"]
    right_perms = [perm for (rb, _), perm in hx_perms.items() if hx_types[rb] and hx_types[rb][0] == "B"]

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

    def accumulate_votes(perms, row_types, *, use_left: bool):
        votes = [Counter() for _ in range(nA if use_left else nB)]
        mismatches = 0
        total = 0
        match_counts = Counter()
        for (rb, cb), perm in perms.items():
            classification = row_types[rb]
            if not classification:
                continue
            kind, _ = classification
            if use_left and kind != "A":
                continue
            if not use_left and kind != "B":
                continue
            i_val, j_val = cb_to_ij(cb)
            total += 1
            if use_left:
                prefer = ("L", "L_inv", "R", "R_inv")
                match = _match_perm(
                    perm,
                    left_map=left_map,
                    left_inv_map=left_inv_map,
                    right_map=right_map,
                    right_inv_map=right_inv_map,
                    prefer=prefer,
                )
                target_idx = i_val
                elem = match[1] if match else None
            else:
                prefer = ("R_inv", "R", "L_inv", "L")
                match = _match_perm(
                    perm,
                    left_map=left_map,
                    left_inv_map=left_inv_map,
                    right_map=right_map,
                    right_inv_map=right_inv_map,
                    prefer=prefer,
                )
                target_idx = j_val
                if match is None:
                    elem = None
                else:
                    match_type, elem = match
                    if match_type == "R" and group is not None:
                        elem = int(group.inv(elem))
            if elem is None:
                mismatches += 1
                continue
            if match:
                match_counts[match[0]] += 1
            votes[target_idx][int(elem)] += 1
        return votes, {"total": total, "mismatches": mismatches, "match_counts": dict(match_counts)}

    a_votes_hx, a_stats_hx = accumulate_votes(hx_perms, hx_types, use_left=True)
    b_votes_hx, b_stats_hx = accumulate_votes(hx_perms, hx_types, use_left=False)
    a_votes_hz, a_stats_hz = accumulate_votes(hz_perms, hz_types, use_left=True)
    b_votes_hz, b_stats_hz = accumulate_votes(hz_perms, hz_types, use_left=False)

    def count_blocks(block_perm, row_types, kind: str) -> int:
        total = 0
        for (rb, _), gr_map in block_perm.items():
            if not gr_map:
                continue
            classification = row_types[rb]
            if classification and classification[0] == kind:
                total += 1
        return total

    def merge_votes(v1, v2):
        merged = []
        for c1, c2 in zip(v1, v2):
            out = Counter()
            out.update(c1)
            out.update(c2)
            merged.append(out)
        return merged

    a_votes = merge_votes(a_votes_hx, a_votes_hz)
    b_votes = merge_votes(b_votes_hx, b_votes_hz)

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

    A_ids, A_margins = finalize_votes(a_votes)
    B_ids, B_margins = finalize_votes(b_votes)

    if group is None:
        group = CyclicGroup(m, name=group_spec_final)

    A_repr = [group.repr(a) if a is not None else "?" for a in A_ids]
    B_repr = [group.repr(b) if b is not None else "?" for b in B_ids]

    a_supports_by_j = _support_sets_for_a(hx_row_blocks, hx_types, cb_to_ij=cb_to_ij)
    b_supports_by_i = _support_sets_for_b(hz_row_blocks, hz_types, cb_to_ij=cb_to_ij)
    a_supports = set().union(*a_supports_by_j.values()) if a_supports_by_j else set()
    b_supports = set().union(*b_supports_by_i.values()) if b_supports_by_i else set()

    permA = None
    permB = None
    if nA == 2:
        permA = tuple(range(nA))
    else:
        canonical_a = _canonical_supports(nA)
        canonical_set = set(canonical_a)
        other_a = a_supports - canonical_set
        if not other_a:
            permA = tuple(range(nA))
        else:
            permA = _find_perm_from_supports(nA, canonical_supports=canonical_a, target_supports=other_a)
    if nB == 2:
        permB = tuple(range(nB))
    else:
        canonical_b = _canonical_supports(nB)
        canonical_set_b = set(canonical_b)
        other_b = b_supports - canonical_set_b
        if not other_b:
            permB = tuple(range(nB))
        else:
            permB = _find_perm_from_supports(nB, canonical_supports=canonical_b, target_supports=other_b)

    def perm_one_based(perm):
        if perm is None:
            return None
        return [int(x) + 1 for x in perm]

    perm_stats = {
        "hx": hx_perm_stats,
        "hz": hz_perm_stats,
    }
    a_match_counts = Counter()
    b_match_counts = Counter()
    for stats in (a_stats_hx, a_stats_hz):
        a_match_counts.update(stats.get("match_counts", {}))
    for stats in (b_stats_hx, b_stats_hz):
        b_match_counts.update(stats.get("match_counts", {}))
    total_left = count_blocks(hx_block_perm, hx_types, "A") + count_blocks(
        hz_block_perm, hz_types, "A"
    )
    total_right = count_blocks(hx_block_perm, hx_types, "B") + count_blocks(
        hz_block_perm, hz_types, "B"
    )
    perm_match = {
        "left_matches": int(a_match_counts.get("L", 0) + a_match_counts.get("L_inv", 0)),
        "right_matches": int(b_match_counts.get("R", 0) + b_match_counts.get("R_inv", 0)),
        "total_left": total_left,
        "total_right": total_right,
        "match_counts_A": dict(a_match_counts),
        "match_counts_B": dict(b_match_counts),
    }

    debug_examples = []
    if debug and group is not None:
        for (rb, cb), perm in list(hx_perms.items())[:8]:
            classification = hx_types[rb]
            if not classification:
                continue
            kind, _ = classification
            prefer = ("L", "L_inv", "R", "R_inv") if kind == "A" else ("R_inv", "R", "L_inv", "L")
            match = _match_perm(
                perm,
                left_map=left_map,
                left_inv_map=left_inv_map,
                right_map=right_map,
                right_inv_map=right_inv_map,
                prefer=prefer,
            )
            i_val, j_val = cb_to_ij(cb)
            if match:
                debug_examples.append(
                    {
                        "rb": rb,
                        "cb": cb,
                        "kind": kind,
                        "i": i_val,
                        "j": j_val,
                        "match": match[0],
                        "elem": group.repr(match[1]),
                    }
                )

    if debug:
        print(f"[paper_extract] debug {hx_path.stem}")
        print(
            f"  fiber map row={row_mode} col={col_mode} score={mapping_choice['score']}"
        )
        print(f"  cb mapping={cb_choice['option']} orientation={cb_choice['orientation']}")
        print(
            f"  row blocks HX={hx_counts} HZ={hz_counts}"
        )
        if debug_examples:
            print("  examples:")
            for ex in debug_examples[:5]:
                print(
                    f"    rb={ex['rb']} cb={ex['cb']} kind={ex['kind']} "
                    f"i={ex['i']} j={ex['j']} match={ex['match']} elem={ex['elem']}"
                )

    return {
        "code_id": f"{group_tag or 'C' + str(m)}_{n_val}_{k_val}_{d_val}",
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
        "nB": nB,
        "mapping_choice": {
            "row_mode": row_mode,
            "col_mode": col_mode,
            "score": mapping_choice["score"],
            "hx_score": mapping_choice["hx_score"],
            "hz_score": mapping_choice["hz_score"],
            "cb_option": cb_choice["option"],
            "cb_orientation": cb_choice["orientation"],
        },
        "row_block_counts": {"hx": hx_counts, "hz": hz_counts},
        "A_ids": A_ids,
        "A_repr": A_repr,
        "B_ids": B_ids,
        "B_repr": B_repr,
        "permA_0based": list(permA) if permA is not None else None,
        "permA_1based": perm_one_based(permA),
        "permB_0based": list(permB) if permB is not None else None,
        "permB_1based": perm_one_based(permB),
        "perm_stats": perm_stats,
        "perm_match": perm_match,
        "vote_margins": {"A": A_margins, "B": B_margins},
        "vote_stats": {
            "A": {"hx": a_stats_hx, "hz": a_stats_hz},
            "B": {"hx": b_stats_hx, "hz": b_stats_hz},
        },
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
            f"- Local dims: nA={rec['nA']}, nB={rec['nB']}, "
            f"row_map={mapping.get('row_mode')}, col_map={mapping.get('col_mode')}, "
            f"cb_map={mapping.get('cb_option')} ({mapping.get('cb_orientation')})"
        )
        lines.append(f"- A: {rec['A_repr']}")
        lines.append(f"- B: {rec['B_repr']}")
        lines.append(
            f"- piA (0-based): {rec['permA_0based']} ; (1-based): {rec['permA_1based']}"
        )
        lines.append(
            f"- piB (0-based): {rec['permB_0based']} ; (1-based): {rec['permB_1based']}"
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
