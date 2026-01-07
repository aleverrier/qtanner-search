from qtanner.group import FiniteGroup
from qtanner.local_codes import hamming_6_3_3_shortened
from qtanner.slice_codes import build_a_slice_checks_H, build_b_slice_checks_Hp


def test_slice_code_sizes_6_3_3_cyclic1() -> None:
    group = FiniteGroup.cyclic(1)
    C0 = hamming_6_3_3_shortened()
    C1 = hamming_6_3_3_shortened()
    C0p = hamming_6_3_3_shortened()
    C1p = hamming_6_3_3_shortened()
    A = [0, 0, 0, 0, 0, 0]
    B = [0, 0, 0, 0, 0, 0]

    rows_a, n_cols_a = build_a_slice_checks_H(group, A, C0, C1)
    assert n_cols_a == 6
    assert len(rows_a) == 6

    rows_b, n_cols_b = build_b_slice_checks_Hp(group, B, C0p, C1p)
    assert n_cols_b == 6
    assert len(rows_b) == 6
