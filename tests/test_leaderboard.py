from __future__ import annotations

import pytest

from qtanner.leaderboard import key_nk, maybe_update_best


def test_key_nk() -> None:
    assert key_nk(324, 12) == "324,12"


def test_maybe_update_best() -> None:
    data: dict[str, dict[str, object]] = {}
    record = {
        "n": 324,
        "k": 12,
        "d_obs": 8,
        "trials_used": 200,
        "group": {"order": 27, "gid": 3},
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "a1v": 0,
        "b1v": 1,
        "timestamp": "2025-01-01T00:00:00Z",
    }
    assert maybe_update_best(data, record) is True
    assert data["324,12"]["d_obs"] == 8

    worse_trials = dict(record)
    worse_trials["trials_used"] = 300
    assert maybe_update_best(data, worse_trials) is False

    better_trials = dict(record)
    better_trials["trials_used"] = 100
    assert maybe_update_best(data, better_trials) is True

    better_distance = dict(record)
    better_distance["d_obs"] = 9
    assert maybe_update_best(data, better_distance) is True


def test_maybe_update_best_requires_fields() -> None:
    data: dict[str, dict[str, object]] = {}
    with pytest.raises(ValueError):
        maybe_update_best(data, {"n": 10, "k": 2})
