"""Optional tests for GAP SmallGroup import."""

import os

import pytest

from qtanner.gap_groups import smallgroup
from qtanner.qdistrnd import gap_is_available


def test_smallgroup_optional():
    if os.environ.get("QTANNER_RUN_GAP_TESTS") != "1":
        pytest.skip("Set QTANNER_RUN_GAP_TESTS=1 to run GAP SmallGroup tests.")
    if not gap_is_available():
        pytest.skip("GAP is not available on PATH.")

    group = smallgroup(3, 1)
    n = group.order
    mul_table = group.mul_table
    inv = group.inv_table

    for x in range(n):
        assert mul_table[0][x] == x
        assert mul_table[x][0] == x
        assert mul_table[x][inv[x]] == 0
        assert mul_table[inv[x]][x] == 0
        assert group.inv(x) == inv[x]
    assert inv[0] == 0
