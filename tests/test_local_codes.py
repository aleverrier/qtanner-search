"""Tests for local code definitions and variants."""

from qtanner.local_codes import (
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    hamming_8_4_4_extended,
    is_orthogonal,
    variants_6_3_3,
    variants_8_4_4,
)


def test_canonical_codes_are_orthogonal():
    code_6 = hamming_6_3_3_shortened()
    code_8 = hamming_8_4_4_extended()
    assert is_orthogonal(code_6.H_rows, code_6.G_rows)
    assert is_orthogonal(code_8.H_rows, code_8.G_rows)


def test_variant_counts():
    assert len(variants_6_3_3()) == 30
    assert len(variants_8_4_4()) == 30


def test_representative_perms_preserve_orthogonality():
    code_6 = hamming_6_3_3_shortened()
    code_8 = hamming_8_4_4_extended()

    for perm in variants_6_3_3()[:5]:
        permuted_H = apply_col_perm_to_rows(code_6.H_rows, perm, code_6.n)
        permuted_G = apply_col_perm_to_rows(code_6.G_rows, perm, code_6.n)
        assert is_orthogonal(permuted_H, permuted_G)

    for perm in variants_8_4_4()[:5]:
        permuted_H = apply_col_perm_to_rows(code_8.H_rows, perm, code_8.n)
        permuted_G = apply_col_perm_to_rows(code_8.G_rows, perm, code_8.n)
        assert is_orthogonal(permuted_H, permuted_G)
