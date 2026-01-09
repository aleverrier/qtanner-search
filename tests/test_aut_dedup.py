from itertools import combinations_with_replacement

import pytest

from qtanner.group import group_from_spec
from qtanner.qdistrnd import gap_is_available
from qtanner.search import canonical_multiset


def _raw_multisets(order: int) -> list[list[int]]:
    return [
        list(comb)
        for comb in combinations_with_replacement(range(order), 6)
        if comb[0] == 0
    ]


def _check_group_dedup(group_spec: str, tmp_path) -> None:
    group = group_from_spec(group_spec)
    auts = group.automorphisms(gap_cmd="gap", cache_dir=tmp_path)
    raw = _raw_multisets(group.order)
    dedup = [
        comb
        for comb in raw
        if tuple(comb) == canonical_multiset(group, comb, automorphisms=auts)
    ]
    assert len(dedup) <= len(raw)
    sample = raw[len(raw) // 2]
    canon = canonical_multiset(group, sample, automorphisms=auts)
    assert canon == canonical_multiset(group, canon, automorphisms=auts)


def test_aut_dedup_c4_c2x2(tmp_path):
    if not gap_is_available():
        pytest.skip("GAP is not available on PATH.")
    _check_group_dedup("C4", tmp_path)
    _check_group_dedup("C2xC2", tmp_path)


def test_aut_identity_fixed_c2x2(tmp_path):
    if not gap_is_available():
        pytest.skip("GAP is not available on PATH.")
    group = group_from_spec("C2xC2")
    auts = group.automorphisms(gap_cmd="gap", cache_dir=tmp_path)
    assert auts
    assert all(perm[0] == 0 for perm in auts)
