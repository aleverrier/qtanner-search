import random

from qtanner.group import group_from_spec
from qtanner.search import _enumerate_sets


def test_enum_modes_c2xc2xc2_counts() -> None:
    group = group_from_spec("C2xC2xC2")
    feasible_limit = 100000
    subset = _enumerate_sets(
        m=group.order,
        max_sets=None,
        rng=random.Random(0),
        feasible_limit=feasible_limit,
        enum_mode="subset",
    )
    assert len(subset) == 21
    multiset = _enumerate_sets(
        m=group.order,
        max_sets=None,
        rng=random.Random(0),
        feasible_limit=feasible_limit,
        enum_mode="multiset",
    )
    assert len(multiset) == 792
    ordered = _enumerate_sets(
        m=group.order,
        max_sets=None,
        rng=random.Random(0),
        feasible_limit=feasible_limit,
        enum_mode="ordered",
    )
    assert len(ordered) == 32768
