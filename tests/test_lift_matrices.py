"""Tests for lifted Tanner matrices."""

from qtanner.gf2 import gf2_rank
from qtanner.group import FiniteGroup
from qtanner.lift_matrices import build_hx_hz, css_commutes
from qtanner.local_codes import repetition_2


def test_lifted_repetition_cyclic_commutes():
    group = FiniteGroup.cyclic(3)
    A = [0, 1]
    B = [0, 1]
    C0 = repetition_2()
    C1 = repetition_2()
    C0p = repetition_2()
    C1p = repetition_2()
    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)
    assert css_commutes(hx_rows, hz_rows)
    assert n_cols == 2 * 2 * 3
    k = n_cols - gf2_rank(hx_rows, n_cols) - gf2_rank(hz_rows, n_cols)
    assert k >= 0
