import itertools
import math
import random

from qtanner.progressive_search import (
    ProgressiveSetting,
    _enumerate_multisets_with_identity,
    _interleaved_rounds,
    _iter_progressive_pairs,
    _two_stage_css_distance,
)


def _make_setting(name: str, *, side: str, k: int, d: int) -> ProgressiveSetting:
    record = {"id": name}
    return ProgressiveSetting(
        side=side,
        elements=[0],
        elements_repr=["0"],
        perm_idx=0,
        k_classical=k,
        d_est=d,
        record=record,
        setting_id=name,
    )


def test_progressive_multiset_count_identity() -> None:
    combos = _enumerate_multisets_with_identity(8)
    assert len(combos) == math.comb(12, 5)
    assert all(0 in combo for combo in combos)


def test_progressive_interleaving_diagonals() -> None:
    a1 = _make_setting("A1", side="A", k=2, d=9)
    a2 = _make_setting("A2", side="A", k=2, d=7)
    a3 = _make_setting("A3", side="A", k=1, d=8)
    rounds_a = _interleaved_rounds([a1, a2, a3])
    assert [s.setting_id for s in rounds_a[0]] == ["A1", "A3"]
    assert [s.setting_id for s in rounds_a[1]] == ["A2"]

    b1 = _make_setting("B1", side="B", k=2, d=10)
    b2 = _make_setting("B2", side="B", k=2, d=5)
    b3 = _make_setting("B3", side="B", k=1, d=6)
    rounds_b = _interleaved_rounds([b1, b2, b3])
    assert [s.setting_id for s in rounds_b[0]] == ["B1", "B3"]
    assert [s.setting_id for s in rounds_b[1]] == ["B2"]

    pairs = list(itertools.islice(_iter_progressive_pairs(rounds_a, rounds_b), 6))
    assert pairs[0][0].setting_id == "A1"
    assert pairs[0][1].setting_id == "B1"

    rank_a = {
        setting.setting_id: r
        for r, round_items in enumerate(rounds_a)
        for setting in round_items
    }
    rank_b = {
        setting.setting_id: r
        for r, round_items in enumerate(rounds_b)
        for setting in round_items
    }
    rank_sums = [rank_a[a.setting_id] + rank_b[b.setting_id] for a, b in pairs]
    assert rank_sums[:4] == [0, 0, 0, 0]
    assert rank_sums[4] == 1


def test_two_stage_skips_slow_when_fast_below_sqrt() -> None:
    calls = []
    values = [3, 4]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(0)
    result = _two_stage_css_distance(
        [0b1],
        [0b1],
        10,
        target_distance=1,
        wmin=0,
        steps_fast=10,
        steps_slow=100,
        current_best=0,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed is True
    assert result.ran_slow is False
    assert result.d_final_ub == 3
    assert calls == [10, 10]


def test_two_stage_skips_slow_when_not_improving_best() -> None:
    calls = []
    values = [5, 6]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(1)
    result = _two_stage_css_distance(
        [0b1],
        [0b1],
        10,
        target_distance=1,
        wmin=0,
        steps_fast=10,
        steps_slow=100,
        current_best=5,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed is True
    assert result.ran_slow is False
    assert result.d_final_ub == 5
    assert calls == [10, 10]


def test_two_stage_runs_slow_and_uses_slow_bound() -> None:
    calls = []
    values = [7, 6, 4, 5]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(2)
    result = _two_stage_css_distance(
        [0b1],
        [0b1],
        16,
        target_distance=1,
        wmin=0,
        steps_fast=10,
        steps_slow=100,
        current_best=3,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed is True
    assert result.ran_slow is True
    assert result.d_final_ub == 4
    assert calls == [10, 10, 100, 100]
