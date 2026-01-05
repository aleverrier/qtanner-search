from __future__ import annotations

import pytest

from qtanner.search_utils import (
    format_slice_decision,
    parse_qd_batches,
    should_abort_after_batch,
)


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


def test_format_slice_decision() -> None:
    msg = format_slice_decision(
        which="a",
        n_slice=144,
        d_min=12,
        trials=50,
        mindist=12,
        d_ub=14,
        early_exit=False,
        passed=True,
    )
    assert msg.startswith("[slice-a]")
    assert "n_slice=144" in msg
    assert "want d>=12" in msg
    assert "trials=50" in msg
    assert "mindist=12" in msg
    assert "got d_ub=14" in msg
    assert "early_exit=False" in msg
    assert msg.endswith("PASS")
