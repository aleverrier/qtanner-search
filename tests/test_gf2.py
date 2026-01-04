"""Tests for GF(2) linear algebra helpers."""

from qtanner.gf2 import gf2_is_in_rowspan, gf2_rank


def test_rank_identity():
    rows = [0b001, 0b010, 0b100]
    assert gf2_rank(rows, 3) == 3


def test_rank_duplicate_rows():
    rows = [0b101, 0b101, 0b101]
    assert gf2_rank(rows, 3) == 1


def test_rank_with_zero_row():
    rows = [0b1100, 0b0110, 0]
    assert gf2_rank(rows, 4) == 2


def test_is_in_rowspan():
    rows = [0b1101, 0b0110]
    assert gf2_is_in_rowspan(0b1011, rows, 4)
    assert not gf2_is_in_rowspan(0b1111, rows, 4)
