"""Lifted quantum Tanner check matrices (Leverrier–Rozendaal–Zémor)."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .group import FiniteGroup
from .local_codes import LocalCode

from .group_ops import ensure_group_ops


def _bit_indices(row: int) -> List[int]:
    indices: List[int] = []
    while row:
        lsb = row & -row
        indices.append(lsb.bit_length() - 1)
        row -= lsb
    return indices


def kron_support(
    rowsA: List[int], nA: int, rowsB: List[int], nB: int
) -> List[List[Tuple[int, int]]]:
    """Return supports of rows in A ⊗ B as (i,j) index pairs."""
    suppA = [_bit_indices(row) for row in rowsA]
    suppB = [_bit_indices(row) for row in rowsB]
    supports: List[List[Tuple[int, int]]] = []
    for a in suppA:
        for b in suppB:
            row_support: List[Tuple[int, int]] = []
            for i in a:
                for j in b:
                    row_support.append((i, j))
            supports.append(row_support)
    return supports


def _lift_block(
    supports: List[List[Tuple[int, int]]],
    nA: int,
    nB: int,
    group: FiniteGroup,
    A: List[int],
    B: List[int],
    perm: str,
) -> List[int]:
    group = ensure_group_ops(group)
    order = group.order
    b_inv = [group.inv_of(b) for b in B]
    rows: List[int] = []
    for support in supports:
        for g0 in range(order):
            row = 0
            for i, j in support:
                a_i = A[i]
                b_inv_j = b_inv[j]
                g1 = g0
                if perm in ("RB", "LARB"):
                    g1 = group.mul(g1, b_inv_j)
                if perm in ("LA", "LARB"):
                    g1 = group.mul(a_i, g1)
                col = ((i * nB + j) * order + g1)
                row |= 1 << col
            rows.append(row)
    return rows


def build_hx_hz(
    group: FiniteGroup,
    A: List[int],
    B: List[int],
    C0: LocalCode,
    C1: LocalCode,
    C0p: LocalCode,
    C1p: LocalCode,
) -> Tuple[List[int], List[int], int]:
    """Build lifted CSS check matrices HX and HZ as bitset rows."""
    group = ensure_group_ops(group)
    # qtanner-search compat: accept dict group
    if isinstance(group, dict):
        from types import SimpleNamespace
        group = SimpleNamespace(**group)

    nA = C0.n
    nB = C0p.n
    if C1.n != nA or C1p.n != nB:
        raise ValueError("Local code dimensions do not match A/B sizes.")
    if len(A) != nA or len(B) != nB:
        raise ValueError("A/B sizes do not match local code lengths.")

    order = group.order
    n_cols = nA * nB * order

    hx_rows: List[int] = []
    supports = kron_support(C0.H_rows, nA, C0p.G_rows, nB)
    hx_rows.extend(_lift_block(supports, nA, nB, group, A, B, perm="none"))
    supports = kron_support(C1.H_rows, nA, C1p.G_rows, nB)
    hx_rows.extend(_lift_block(supports, nA, nB, group, A, B, perm="LARB"))

    hz_rows: List[int] = []
    supports = kron_support(C0.G_rows, nA, C1p.H_rows, nB)
    hz_rows.extend(_lift_block(supports, nA, nB, group, A, B, perm="RB"))
    supports = kron_support(C1.G_rows, nA, C0p.H_rows, nB)
    hz_rows.extend(_lift_block(supports, nA, nB, group, A, B, perm="LA"))

    expected_hx_rows = (
        len(C0.H_rows) * len(C0p.G_rows) + len(C1.H_rows) * len(C1p.G_rows)
    ) * order
    expected_hz_rows = (
        len(C0.G_rows) * len(C1p.H_rows) + len(C1.G_rows) * len(C0p.H_rows)
    ) * order
    assert len(hx_rows) == expected_hx_rows
    assert len(hz_rows) == expected_hz_rows

    return hx_rows, hz_rows, n_cols


def css_commutes(hx_rows: Iterable[int], hz_rows: Iterable[int]) -> bool:
    """Check HX * HZ^T = 0 over GF(2) using bitset intersections."""
    for h in hx_rows:
        for z in hz_rows:
            if ((h & z).bit_count() % 2) != 0:
                return False
    return True


__all__ = ["build_hx_hz", "css_commutes", "kron_support"]
