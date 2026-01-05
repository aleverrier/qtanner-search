from __future__ import annotations

from collections import Counter

from qtanner.generators import iter_generator_sets
from qtanner.group import FiniteGroup


def test_iter_generator_sets_allows_repeats_for_small_group() -> None:
    group = FiniteGroup.cyclic(4)
    gens = list(iter_generator_sets(group, size=6, distinct=False, max_sets=50))
    assert gens
    for candidate in gens:
        assert candidate[0] == 0
        assert candidate == sorted(candidate)
    assert candidate.count(0) == 1
    assert any(len(set(candidate)) < len(candidate) for candidate in gens)


def test_iter_generator_sets_filters_repeats_with_diversity_ordering() -> None:
    group = FiniteGroup.cyclic(4)
    gens = list(
        iter_generator_sets(
            group,
            size=6,
            distinct=False,
            max_sets=50,
            min_distinct_nonid=3,
            max_multiplicity_nonid=3,
            order_by_diversity=True,
        )
    )
    assert gens
    assert gens[0] != [0, 1, 1, 1, 1, 1]
    for candidate in gens:
        nonid = [val for val in candidate if val != 0]
        assert len(set(nonid)) >= 3
        counts = Counter(nonid)
        assert max(counts.values()) <= 3
