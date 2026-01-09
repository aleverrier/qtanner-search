import random
from pathlib import Path

import pytest

from qtanner.group import CyclicGroup, group_from_spec
from qtanner.mtx import write_mtx_from_bitrows
from qtanner.qdistrnd import gap_is_available
from qtanner.lift_matrices import build_hx_hz
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


def test_sigma_recovery_c2x2_synthetic(tmp_path):
    if not gap_is_available("gap"):
        pytest.skip("GAP not available for sigma recovery.")
    group = group_from_spec("C2xC2")
    nA, nB = 6, 2
    rA, rB = 3, 1
    m = group.order

    C0 = paper_extract._local_code_canonical(nA)
    C1 = paper_extract._permute_local_code(C0, list(range(nA)), name="C1")
    C0p = paper_extract._local_code_canonical(nB)
    C1p = paper_extract._permute_local_code(C0p, list(range(nB)), name="C1p")

    A = [0, 1, 2, 3, 1, 2]
    B = [1, 2]
    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)

    rng = random.Random(0)
    sigma = list(range(m))
    rng.shuffle(sigma)
    sigma_inv = paper_extract._perm_inverse(sigma)

    col_perm = paper_extract._build_col_perm(
        nA=nA, nB=nB, m=m, col_order=(0, 1, 2), sigma_inv=sigma_inv
    )
    hx_obs = paper_extract._apply_col_perm_to_bitrows(hx_rows, col_perm)
    hz_obs = paper_extract._apply_col_perm_to_bitrows(hz_rows, col_perm)

    kA = len(C0.G_rows)
    kB = len(C0p.G_rows)
    row_perm_hx = paper_extract._build_row_perm_hx(
        rA=rA, nB=nB, kB=kB, m=m, row_order=(0, 1, 2), sigma_inv=sigma_inv
    )
    row_perm_hz = paper_extract._build_row_perm_hz(
        nA=nA, rB=rB, kA=kA, m=m, row_order=(0, 1, 2), sigma_inv=sigma_inv
    )
    hx_obs = paper_extract._reorder_rows(hx_obs, row_perm_hx)
    hz_obs = paper_extract._reorder_rows(hz_obs, row_perm_hz)

    folder = tmp_path / "633x212"
    folder.mkdir()
    hx_path = folder / "HX_C2C2_48_0_0.mtx"
    hz_path = folder / "HZ_C2C2_48_0_0.mtx"
    write_mtx_from_bitrows(hx_path, hx_obs, n_cols)
    write_mtx_from_bitrows(hz_path, hz_obs, n_cols)

    rec = paper_extract._extract_from_pair(
        hx_path=hx_path,
        hz_path=hz_path,
        folder="633x212",
        group_tag="C2C2",
        n_val=n_cols,
        k_val=0,
        d_val=0,
        gap_cmd="gap",
        cache_dir=tmp_path,
    )
    assert rec["reconstruction_ok"] is True
    assert all(a is not None for a in rec["A_ids"])
    assert all(b is not None for b in rec["B_ids"])
