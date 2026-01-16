from qtanner.group import CyclicGroup, DirectProductGroup, group_from_spec


def test_cyclic_group_c4_ops() -> None:
    group = CyclicGroup(4)
    assert group.order == 4
    assert group.id() == 0
    assert list(group.elements()) == [0, 1, 2, 3]
    assert group.mul(1, 3) == 0
    assert group.inv(1) == 3
    assert group.repr(2) == "2"


def test_direct_product_c2x_c2_ops() -> None:
    group = group_from_spec("C2xC2")
    assert group.order == 4
    assert group.mul(1, 2) == 3
    assert group.inv(3) == 3
    assert group.repr(3) == "(1,1)"


def test_direct_product_c2x_c2x_c2_ops() -> None:
    group = group_from_spec("C2xC2xC2")
    assert group.order == 8
    for g in group.elements():
        assert group.inv(g) == g


def test_la_rb_commute_for_c4_and_c2x_c2() -> None:
    groups = [
        CyclicGroup(4),
        DirectProductGroup(CyclicGroup(2), CyclicGroup(2)),
    ]
    for group in groups:
        for a in group.elements():
            for b in group.elements():
                for g in group.elements():
                    la_rb = group.mul(a, group.mul(g, group.inv(b)))
                    rb_la = group.mul(group.mul(a, g), group.inv(b))
                    assert la_rb == rb_la
