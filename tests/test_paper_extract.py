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


def test_decode3_scoring_synthetic_c3():
    m = 3
    nA, nB = 2, 2
    rA, rB = 2, 2
    hx_row_axes = ["r", "j", "g"]
    hz_row_axes = ["i", "r", "g"]
    col_axes = ["i", "j", "g"]
    hx_row_sizes = [rA, nB, m]
    hz_row_sizes = [nA, rB, m]
    col_sizes = [nA, nB, m]
    row_order = (0, 1, 2)
    col_order = (0, 1, 2)

    def encode3(vals, sizes, order):
        idx = 0
        for axis in order:
            idx = idx * sizes[axis] + vals[axis]
        return idx

    hx_entries = []
    for r in range(rA):
        for j in range(nB):
            for i in range(nA):
                for g in range(m):
                    row_vals = [r, j, g]
                    col_vals = [i, j, g]
                    hx_entries.append(
                        (
                            encode3(row_vals, hx_row_sizes, row_order),
                            encode3(col_vals, col_sizes, col_order),
                        )
                    )
    hz_entries = []
    for i in range(nA):
        for r in range(rB):
            for j in range(nB):
                for g in range(m):
                    row_vals = [i, r, g]
                    col_vals = [i, j, g]
                    hz_entries.append(
                        (
                            encode3(row_vals, hz_row_sizes, row_order),
                            encode3(col_vals, col_sizes, col_order),
                        )
                    )

    hx_block_perm, hx_block_nnz, _ = paper_extract._build_block_maps3(
        hx_entries,
        row_sizes=hx_row_sizes,
        row_axes=hx_row_axes,
        row_order=row_order,
        col_sizes=col_sizes,
        col_axes=col_axes,
        col_order=col_order,
    )
    hx_score = paper_extract._score_block_maps(hx_block_perm, hx_block_nnz, m=m)
    assert hx_score[0] > 0
    assert hx_score[1] == hx_score[0]

    choice = paper_extract._select_axes_mapping(
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
    assert choice["score"][0] >= hx_score[0]
