from qtanner.classical_distance import ClassicalCodeAnalysis
from qtanner.group import CyclicGroup
from qtanner.search import SliceCandidate, SliceMetrics, _frontier_select


def _analysis(k: int, d: int) -> ClassicalCodeAnalysis:
    return ClassicalCodeAnalysis(
        m=0,
        n=0,
        rank=0,
        k=k,
        d=d,
        exact=True,
        codewords_checked=0,
        method="test",
    )


def _candidate(idx: int, k: int, d: int) -> SliceCandidate:
    metrics = SliceMetrics(
        h=_analysis(k, d),
        g=_analysis(k, d),
        d_min=d,
        k_min=k,
        k_sum=2 * k,
    )
    return SliceCandidate(elements=[0, idx], perm_idx=idx, metrics=metrics)


def test_frontier_select_tradeoff() -> None:
    group = CyclicGroup(1)
    scored = [
        _candidate(0, 3, 3),
        _candidate(1, 2, 5),
        _candidate(2, 4, 2),
        _candidate(3, 4, 4),
        _candidate(4, 5, 1),
    ]
    selected, payload = _frontier_select(
        scored,
        group=group,
        side="A",
        n_quantum=3,
        frontier_max_per_point=10,
        frontier_max_total=10,
    )
    points = {(cand.metrics.k_min, cand.metrics.d_min) for cand in selected}
    assert points == {(4, 4), (2, 5)}
    assert payload["d0"] == 2
    assert payload["selected_ids"]
