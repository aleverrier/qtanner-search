"""Fast in-Python classical distance estimator for slice codes."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple


def _rref_bitrows(rows: Sequence[int], ncols: int) -> Tuple[List[int], List[int]]:
    if ncols < 0:
        raise ValueError("ncols must be nonnegative")
    mat = [int(r) for r in rows]
    m = len(mat)
    pivots: List[int] = []
    r = 0
    for c in range(ncols):
        if r >= m:
            break
        bit = 1 << c
        pivot_row = None
        for i in range(r, m):
            if mat[i] & bit:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
        pivots.append(c)
        pivot_val = mat[r]
        for i in range(m):
            if i != r and (mat[i] & bit):
                mat[i] ^= pivot_val
        r += 1
    rref_rows = [mat[i] for i in range(r) if mat[i] != 0]
    pivots = pivots[: len(rref_rows)]
    return rref_rows, pivots


def _nullspace_basis_from_rref(
    rref_rows: Sequence[int], pivots: Sequence[int], ncols: int
) -> List[int]:
    pivot_set = set(pivots)
    basis: List[int] = []
    for f in range(ncols):
        if f in pivot_set:
            continue
        v = 1 << f
        bit_f = 1 << f
        for row, pcol in zip(rref_rows, pivots):
            if row & bit_f:
                v |= 1 << pcol
        basis.append(v)
    return basis


def _min_weight_gray(
    basis: Sequence[int],
    *,
    wmin: int,
) -> Tuple[int, int, bool]:
    k = len(basis)
    best = None
    cw = 0
    prev_gray = 0
    checked = 0
    limit = 1 << k
    for t in range(1, limit):
        gray = t ^ (t >> 1)
        diff = gray ^ prev_gray
        idx = (diff & -diff).bit_length() - 1
        cw ^= basis[idx]
        prev_gray = gray
        checked += 1
        w = cw.bit_count()
        if best is None or w < best:
            best = w
            if wmin >= 0 and w <= wmin:
                return w, checked, True
    if best is None:
        raise RuntimeError("no nonzero codewords found in exhaustive search")
    return best, checked, False


def _min_weight_sample(
    basis: Sequence[int],
    *,
    sample_count: int,
    rng: random.Random,
    wmin: int,
) -> Tuple[int, int, bool]:
    k = len(basis)
    best = None
    checked = 0
    for _ in range(sample_count):
        mask = 0
        while mask == 0:
            mask = rng.getrandbits(k)
        cw = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cw ^= basis[idx]
            m ^= lsb
        checked += 1
        w = cw.bit_count()
        if best is None or w < best:
            best = w
            if wmin >= 0 and w <= wmin:
                return w, checked, True
    if best is None:
        raise RuntimeError("no nonzero codewords found in sampling search")
    return best, checked, False


def _estimate_classical_distance_fast_details(
    h_rows_bits: Sequence[int],
    n: int,
    wmin: int,
    exhaustive_k_max: int,
    sample_count: int,
    rng_seed: int,
) -> Tuple[Optional[int], int, bool, int]:
    if n < 0:
        raise ValueError("n must be nonnegative")
    if exhaustive_k_max < 0:
        raise ValueError("exhaustive_k_max must be nonnegative")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rref_rows, pivots = _rref_bitrows(h_rows_bits, n)
    rank = len(pivots)
    k = n - rank
    if k == 0:
        return None, 0, True, 0
    basis = _nullspace_basis_from_rref(rref_rows, pivots, n)
    if k <= exhaustive_k_max:
        witness, checked, _ = _min_weight_gray(basis, wmin=wmin)
        return witness, k, True, checked
    max_samples = 1 << min(k, 8)
    if k <= 8:
        max_samples = (1 << k) - 1
    sample_count = min(sample_count, max_samples)
    rng = random.Random(rng_seed)
    witness, checked, _ = _min_weight_sample(
        basis, sample_count=sample_count, rng=rng, wmin=wmin
    )
    return witness, k, False, checked


def estimate_classical_distance_fast(
    h_rows_bits: Sequence[int],
    n: int,
    wmin: int,
    exhaustive_k_max: int,
    sample_count: int,
    rng_seed: int,
) -> Tuple[Optional[int], int, bool]:
    witness, k, exact, _ = _estimate_classical_distance_fast_details(
        h_rows_bits,
        n,
        wmin,
        exhaustive_k_max,
        sample_count,
        rng_seed,
    )
    return witness, k, exact


__all__ = ["estimate_classical_distance_fast"]
