import pytest

from qtanner.group import list_small_groups
from qtanner.qdistrnd import gap_is_available


def test_list_small_groups_gap() -> None:
    if not gap_is_available():
        pytest.skip("GAP is not available on PATH.")
    records = list_small_groups(8)
    assert records
    assert all(record.order <= 8 for record in records)
    assert any(record.spec == "SG(8,4)" for record in records)
