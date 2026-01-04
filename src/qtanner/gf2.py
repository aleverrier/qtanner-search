"""Small GF(2) linear algebra helpers using int bitsets."""

from __future__ import annotations

from typing import List


def gf2_rank(rows: List[int], n_cols: int) -> int:
    """Compute rank over GF(2) via Gaussian elimination."""
    work = rows[:]
    rank = 0
    row_idx = 0
    for col in range(n_cols):
        pivot = None
        for r in range(row_idx, len(work)):
            if (work[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        work[row_idx], work[pivot] = work[pivot], work[row_idx]
        for r in range(len(work)):
            if r != row_idx and ((work[r] >> col) & 1):
                work[r] ^= work[row_idx]
        rank += 1
        row_idx += 1
        if row_idx == len(work):
            break
    return rank


def gf2_is_in_rowspan(vec: int, rows: List[int], n_cols: int) -> bool:
    """Check whether vec is in the rowspan of rows over GF(2)."""
    base_rank = gf2_rank(rows, n_cols)
    augmented_rank = gf2_rank(rows + [vec], n_cols)
    return augmented_rank == base_rank


__all__ = ["gf2_rank", "gf2_is_in_rowspan"]
