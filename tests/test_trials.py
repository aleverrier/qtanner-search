from __future__ import annotations

import pytest

from qtanner.trials import parse_trials_schedule, trials_for_n


def test_parse_trials_schedule() -> None:
    schedule = parse_trials_schedule("150:1000,300:2000,1000:50000")
    assert schedule == [(150, 1000), (300, 2000), (1000, 50000)]


def test_parse_trials_schedule_errors() -> None:
    with pytest.raises(ValueError):
        parse_trials_schedule("150-1000")
    with pytest.raises(ValueError):
        parse_trials_schedule("150:foo")
    with pytest.raises(ValueError):
        parse_trials_schedule("150:1000,100:2000")


def test_trials_for_n() -> None:
    schedule = [(150, 1000), (300, 2000), (1000, 50000)]
    assert trials_for_n(100, schedule, 999) == 1000
    assert trials_for_n(150, schedule, 999) == 1000
    assert trials_for_n(151, schedule, 999) == 2000
    assert trials_for_n(1000, schedule, 999) == 50000
    assert trials_for_n(1001, schedule, 999) == 999
