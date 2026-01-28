import itertools
import math
import random

from qtanner.progressive_search import (
    BestCodesEntry,
    ProgressiveSetting,
    _enumerate_multisets_with_identity,
    _interleaved_rounds,
    _iter_progressive_pairs,
    _maybe_save_new_best_artifact,
    _progressive_css_distance,
    compute_slow_trials,
    decide_slow_quantum_plan,
    should_abort_refine,
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


def test_slow_decision_skips_when_fast_leq_best() -> None:
    entry = BestCodesEntry(n=100, k=10, d=12, m4ri_trials=50000, code_id="best")
    decision = decide_slow_quantum_plan(
        d_fast=12,
        fast_trials=2000,
        best_entry=entry,
        override=None,
    )
    assert decision.run_slow is False
    assert decision.steps_slow == 50000
    assert decision.reason == "d_fast<=best"
    assert decision.best_d == 12


def test_slow_decision_runs_when_fast_beats_best() -> None:
    entry = BestCodesEntry(n=100, k=10, d=12, m4ri_trials=30000, code_id="best")
    decision = decide_slow_quantum_plan(
        d_fast=13,
        fast_trials=2000,
        best_entry=entry,
        override=None,
    )
    assert decision.run_slow is True
    assert decision.steps_slow == 50000
    assert decision.reason == "d_fast>best"


def test_slow_decision_runs_when_no_best_entry() -> None:
    decision = decide_slow_quantum_plan(
        d_fast=8,
        fast_trials=2000,
        best_entry=None,
        override=None,
    )
    assert decision.run_slow is True
    assert decision.steps_slow == 50000
    assert decision.reason == "no_best_entry"


def test_slow_decision_override_trials() -> None:
    entry = BestCodesEntry(n=100, k=10, d=12, m4ri_trials=30000, code_id="best")
    decision = decide_slow_quantum_plan(
        d_fast=13,
        fast_trials=2000,
        best_entry=entry,
        override=12345,
    )
    assert decision.run_slow is True
    assert decision.steps_slow == 50000


def test_compute_slow_trials_minimums() -> None:
    assert compute_slow_trials(None, None) == 50000
    assert compute_slow_trials(10000, None) == 50000
    assert compute_slow_trials(200000, None) == 200000
    assert compute_slow_trials(None, 30000) == 50000
    assert compute_slow_trials(None, 80000) == 80000


def test_progressive_new_best_writes_artifact(tmp_path) -> None:
    artifact = {
        "schema": "qtanner.new_best.v1",
        "timestamp": "20260128T000000Z",
        "decision": "new_best",
        "code_id": "C2xC2_A1_B1_k4_d6",
        "group": {"spec": "C2xC2", "order": 4},
        "n": 12,
        "k": 4,
        "A_id": "A1",
        "B_id": "B1",
        "A": {"elements": [0, 1, 2], "perm_idx": 0},
        "B": {"elements": [0, 1, 3], "perm_idx": 1},
        "local_codes": {"C0": "H63", "C1": "H63v1", "permA_idx": 0, "permB_idx": 1},
        "distance": {"dX_best": 7, "dZ_best": 6, "d_ub": 6},
        "dX_best": 7,
        "dZ_best": 6,
        "d_ub": 6,
        "steps_used": 200,
        "eval": 3,
        "seed": 123,
        "target_distance": 6,
        "promising": True,
        "best_codes_candidate": True,
        "artifacts": {"results_dir": "results/run1", "code_dir": "results/run1/best_codes/C2xC2_A1_B1_k4_d6"},
    }
    out_path = _maybe_save_new_best_artifact(
        decision="new_best",
        save_dir=tmp_path,
        artifact=artifact,
    )
    assert out_path is not None
    assert out_path.exists()
    data = out_path.read_text(encoding="utf-8")
    assert "C2xC2_A1_B1_k4_d6" in data


def test_should_abort_refine_best_d() -> None:
    assert should_abort_refine(30, 30, None) is False
    assert should_abort_refine(28, 35, 29) is True
    assert should_abort_refine(31, 29, 29) is True
    assert should_abort_refine(30, 31, 29) is False


def test_progressive_refine_aborts_on_best_codes_bound() -> None:
    calls = []
    values = [12, 12, 7]

    def fake_estimator(hx, hz, n_cols, steps, wmin, seed, dist_m4ri_cmd):
        calls.append(steps)
        return values.pop(0)

    rng = random.Random(3)

    def abort_check(d_x_best: int, d_z_best: int):
        if should_abort_refine(d_x_best, d_z_best, 7):
            return "best_codes"
        return None

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
        refine_abort_check=abort_check,
    )
    assert result.passed_fast is True
    assert result.ran_refine is True
    assert result.aborted is True
    assert result.abort_reason == "best_codes"
    assert result.d_x_best == 7
    assert result.d_z_best == 12
    assert result.steps_x == 20
    assert result.steps_z == 10
    assert calls == [10, 10, 10]
