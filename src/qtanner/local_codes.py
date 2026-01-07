"""Local binary linear codes and column-permutation variants."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class LocalCode:
    """Binary linear code with parity-check and generator rows as int bitsets."""

    name: str
    n: int
    k: int
    H_rows: List[int]
    G_rows: List[int]


def row_from_bits(bits: List[int]) -> int:
    """Pack a list of bits into an int bitset (LSB = column 0)."""
    row = 0
    for i, b in enumerate(bits):
        if b:
            row |= 1 << i
    return row


def apply_col_perm_to_row(row: int, perm: List[int], n: int) -> int:
    """Apply a column permutation to a row bitset.

    perm maps old column index -> new column index.
    """
    new_row = 0
    for i in range(n):
        if (row >> i) & 1:
            new_row |= 1 << perm[i]
    return new_row


def apply_col_perm_to_rows(rows: List[int], perm: List[int], n: int) -> List[int]:
    """Apply a column permutation to a list of row bitsets."""
    return [apply_col_perm_to_row(r, perm, n) for r in rows]


def span_codewords(G_rows: List[int]) -> List[int]:
    """Enumerate all codewords in the row span of G_rows, sorted."""
    k = len(G_rows)
    codewords = []
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if (mask >> i) & 1:
                word ^= G_rows[i]
        codewords.append(word)
    return sorted(codewords)


def is_orthogonal(H_rows: Iterable[int], G_rows: Iterable[int]) -> bool:
    """Check H * G^T = 0 over GF(2)."""
    for h in H_rows:
        for g in G_rows:
            if ((h & g).bit_count() % 2) != 0:
                return False
    return True


def hamming_6_3_3_shortened() -> LocalCode:
    """Canonical shortened Hamming [6,3,3] code."""
    G = [
        row_from_bits([1, 0, 0, 1, 1, 0]),
        row_from_bits([0, 1, 0, 1, 0, 1]),
        row_from_bits([0, 0, 1, 0, 1, 1]),
    ]
    H = [
        row_from_bits([1, 1, 0, 1, 0, 0]),
        row_from_bits([1, 0, 1, 0, 1, 0]),
        row_from_bits([0, 1, 1, 0, 0, 1]),
    ]
    return LocalCode(name="hamming_6_3_3_shortened", n=6, k=3, H_rows=H, G_rows=G)


def distinct_column_permutation_representatives(code: LocalCode) -> List[List[int]]:
    """Enumerate distinct column permutations by codeword-span signature."""
    n = code.n
    reps = {}
    for perm in permutations(range(n)):
        permuted_G = apply_col_perm_to_rows(code.G_rows, list(perm), n)
        signature = tuple(span_codewords(permuted_G))
        if signature not in reps:
            reps[signature] = list(perm)
    ordered = sorted(reps.items(), key=lambda item: item[0])
    return [perm for _, perm in ordered]


def variants_6_3_3() -> List[List[int]]:
    """Column-permutation representatives for [6,3,3] (expected 30)."""
    return distinct_column_permutation_representatives(hamming_6_3_3_shortened())


__all__ = [
    "LocalCode",
    "row_from_bits",
    "apply_col_perm_to_row",
    "apply_col_perm_to_rows",
    "span_codewords",
    "is_orthogonal",
    "hamming_6_3_3_shortened",
    "distinct_column_permutation_representatives",
    "variants_6_3_3",
]
