from pathlib import Path

import pytest

from qtanner.group import CyclicGroup
from qtanner import paper_extract


def test_extract_sample_pair_if_present(tmp_path):
    hx = Path("data/lrz_paper_mtx/633x212/HX_60_2_8.mtx")
    hz = Path("data/lrz_paper_mtx/633x212/HZ_60_2_8.mtx")
    if not hx.exists() or not hz.exists():
        pytest.skip("LRZ paper matrices not present in workspace.")
    rec = paper_extract._extract_from_pair(
        hx_path=hx,
        hz_path=hz,
        folder="633x212",
        group_tag=None,
        n_val=60,
        k_val=2,
        d_val=8,
        gap_cmd="gap",
        cache_dir=tmp_path,
    )
    assert rec["nA"] == 6
    assert rec["nB"] == 2
    assert len(rec["A_ids"]) == 6
    assert len(rec["B_ids"]) == 2
    assert rec["permB_0based"] == [0, 1]


def test_perm_mapping_synthetic_cyclic():
    group = CyclicGroup(5)
    left_map, right_map = paper_extract._build_left_right_maps(group)
    left_perm = tuple(group.mul(2, g) for g in group.elements())
    right_perm = tuple(group.mul(g, 3) for g in group.elements())
    assert paper_extract._perm_to_element_left(left_perm, left_perm_map=left_map) == 2
    assert paper_extract._perm_to_element_right(
        right_perm, right_perm_map=right_map, group=group
    ) == group.inv(3)
    assert paper_extract._perm_to_element_right(right_perm) == paper_extract._perm_inverse(
        right_perm
    )[0]


def test_select_fiber_mapping_prefers_fiber_major():
    m = 3
    num_row_blocks = 2
    num_col_blocks = 4
    n_rows = num_row_blocks * m
    n_cols = num_col_blocks * m
    rows = [0] * n_rows
    for rb in range(num_row_blocks):
        for gr in range(m):
            r = rb * m + gr
            bits = 0
            for cb in range(num_col_blocks):
                c = cb * m + gr
                bits |= 1 << c
            rows[r] = bits
    entries = list(paper_extract._iter_nonzero_entries(rows))
    choice = paper_extract._select_fiber_mappings(
        hx_entries=entries,
        hz_entries=entries,
        hx_shape=(n_rows, n_cols),
        hz_shape=(n_rows, n_cols),
        m=m,
    )
    assert choice["row_mode"] == "fiber-major"
    assert choice["col_mode"] == "fiber-major"
