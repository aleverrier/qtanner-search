"""Tests for lifted Tanner matrices."""

from qtanner.gf2 import gf2_rank
from qtanner.group import FiniteGroup
from qtanner.lift_matrices import build_hx_hz, css_commutes
from qtanner.local_codes import hamming_6_3_3_shortened


def test_lifted_6_3_3_cyclic_commutes():
    group = FiniteGroup.cyclic(1)
    A = [0, 0, 0, 0, 0, 0]
    B = [0, 0, 0, 0, 0, 0]
    C0 = hamming_6_3_3_shortened()
    C1 = hamming_6_3_3_shortened()
    C0p = hamming_6_3_3_shortened()
    C1p = hamming_6_3_3_shortened()
    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)
    assert css_commutes(hx_rows, hz_rows)
    assert n_cols == 36
    assert len(hx_rows) == 18
    assert len(hz_rows) == 18
    k = n_cols - gf2_rank(hx_rows, n_cols) - gf2_rank(hz_rows, n_cols)
    assert k >= 0
