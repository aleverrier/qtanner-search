"""Classical slice-code parity-check and generator matrices as bit rows."""

from __future__ import annotations

from typing import List, Tuple

from .group import FiniteGroup
from .local_codes import LocalCode


def _bit_indices(row: int) -> List[int]:
    indices: List[int] = []
    while row:
        lsb = row & -row
        indices.append(lsb.bit_length() - 1)
        row -= lsb
    return indices


def _rows_direct(rows: List[int], order: int) -> List[int]:
    out: List[int] = []
    for row in rows:
        support = _bit_indices(row)
        for g in range(order):
            packed = 0
            for i in support:
                col = i * order + g
                packed |= 1 << col
            out.append(packed)
    return out


def _rows_la(rows: List[int], order: int, group: FiniteGroup, A: List[int]) -> List[int]:
    out: List[int] = []
    for row in rows:
        support = _bit_indices(row)
        for g in range(order):
            packed = 0
            for i in support:
                g2 = group.mul(A[i], g)
                col = i * order + g2
                packed |= 1 << col
            out.append(packed)
    return out


def _rows_rb(rows: List[int], order: int, group: FiniteGroup, B: List[int]) -> List[int]:
    out: List[int] = []
    b_inv = [group.inv_of(b) for b in B]
    for row in rows:
        support = _bit_indices(row)
        for g in range(order):
            packed = 0
            for j in support:
                g2 = group.mul(g, b_inv[j])
                col = j * order + g2
                packed |= 1 << col
            out.append(packed)
    return out


def build_a_slice_checks_H(
    group: FiniteGroup, A: List[int], C0: LocalCode, C1: LocalCode
) -> Tuple[List[int], int]:
    """Build A-slice parity-check rows for (H0 ⊗ I) and (H1 ⊗ I)*LA."""
    nA = C0.n
    if C1.n != nA or len(A) != nA:
        raise ValueError("A-slice dimensions do not match local code lengths.")
    order = group.order
    rows = _rows_direct(C0.H_rows, order) + _rows_la(
        C1.H_rows, order, group, A
    )
    return rows, nA * order


def build_a_slice_checks_G(
    group: FiniteGroup, A: List[int], C0: LocalCode, C1: LocalCode
) -> Tuple[List[int], int]:
    """Build A-slice generator rows for (G0 ⊗ I) and (G1 ⊗ I)*LA."""
    nA = C0.n
    if C1.n != nA or len(A) != nA:
        raise ValueError("A-slice dimensions do not match local code lengths.")
    order = group.order
    rows = _rows_direct(C0.G_rows, order) + _rows_la(
        C1.G_rows, order, group, A
    )
    return rows, nA * order


def build_b_slice_checks_Gp(
    group: FiniteGroup, B: List[int], C0p: LocalCode, C1p: LocalCode
) -> Tuple[List[int], int]:
    """Build B-slice generator rows for (G0' ⊗ I) and (G1' ⊗ I)*RB."""
    nB = C0p.n
    if C1p.n != nB or len(B) != nB:
        raise ValueError("B-slice dimensions do not match local code lengths.")
    order = group.order
    rows = _rows_direct(C0p.G_rows, order) + _rows_rb(
        C1p.G_rows, order, group, B
    )
    return rows, nB * order


def build_b_slice_checks_Hp(
    group: FiniteGroup, B: List[int], C0p: LocalCode, C1p: LocalCode
) -> Tuple[List[int], int]:
    """Build B-slice parity-check rows for (H0' ⊗ I) and (H1' ⊗ I)*RB."""
    nB = C0p.n
    if C1p.n != nB or len(B) != nB:
        raise ValueError("B-slice dimensions do not match local code lengths.")
    order = group.order
    rows = _rows_direct(C0p.H_rows, order) + _rows_rb(
        C1p.H_rows, order, group, B
    )
    return rows, nB * order


__all__ = [
    "build_a_slice_checks_H",
    "build_a_slice_checks_G",
    "build_b_slice_checks_Gp",
    "build_b_slice_checks_Hp",
]
