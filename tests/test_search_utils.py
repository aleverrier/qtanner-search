from __future__ import annotations

import pytest

from qtanner.search_utils import parse_qd_batches, should_abort_after_batch


def test_parse_qd_batches() -> None:
    assert parse_qd_batches("50,200,1000") == [50, 200, 1000]
    assert parse_qd_batches(" 10 , 20 ") == [10, 20]


def test_parse_qd_batches_errors() -> None:
    with pytest.raises(ValueError):
        parse_qd_batches("")
    with pytest.raises(ValueError):
        parse_qd_batches("10,")
    with pytest.raises(ValueError):
        parse_qd_batches("foo")
    with pytest.raises(ValueError):
        parse_qd_batches("-5")


def test_should_abort_after_batch() -> None:
    assert should_abort_after_batch(d_obs=5, target=10, best_d_obs=None) is True
    assert should_abort_after_batch(d_obs=9, target=8, best_d_obs=10) is True
    assert should_abort_after_batch(d_obs=10, target=10, best_d_obs=10) is False
    assert should_abort_after_batch(d_obs=12, target=10, best_d_obs=11) is False
