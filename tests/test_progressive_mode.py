import itertools
import math
import random

from qtanner.progressive_search import (
    ProgressiveSetting,
    _enumerate_multisets_with_identity,
    _interleaved_rounds,
    _iter_progressive_pairs,
    _progressive_css_distance,
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


def test_progressive_rejects_fast_below_threshold() -> None:
    calls = []
    values = [4, 7]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(0)
    result = _progressive_css_distance(
        [0b1],
        [0b1],
        10,
        must_exceed=5,
        steps_fast=10,
        steps_slow=100,
        refine_chunk=10,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed_fast is False
    assert result.ran_refine is False
    assert result.d_z_best == 4
    assert result.d_x_best == 7
    assert calls == [10, 10]


def test_progressive_refine_aborts_on_dx_chunk() -> None:
    calls = []
    values = [8, 9, 5]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(1)
    result = _progressive_css_distance(
        [0b1],
        [0b1],
        10,
        must_exceed=5,
        steps_fast=10,
        steps_slow=30,
        refine_chunk=10,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed_fast is True
    assert result.ran_refine is True
    assert result.aborted is True
    assert result.abort_reason == "dx<=must_exceed"
    assert result.d_z_best == 8
    assert result.d_x_best == 5
    assert result.steps_z == 10
    assert result.steps_x == 20
    assert calls == [10, 10, 10]


def test_progressive_refine_runs_to_completion() -> None:
    calls = []
    values = [9, 8, 7, 8, 6, 7]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(2)
    result = _progressive_css_distance(
        [0b1],
        [0b1],
        16,
        must_exceed=5,
        steps_fast=10,
        steps_slow=20,
        refine_chunk=5,
        rng=rng,
        dist_m4ri_cmd="dist_m4ri",
        estimator=fake_estimator,
    )
    assert result.passed_fast is True
    assert result.ran_refine is True
    assert result.aborted is False
    assert result.d_z_best == 7
    assert result.d_x_best == 6
    assert result.steps_z == 20
    assert result.steps_x == 20
    assert calls == [10, 10, 5, 5, 5, 5]
