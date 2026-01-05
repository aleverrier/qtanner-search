"""
Fast (often exact) minimum-distance estimation for small binary *classical* linear codes.

This is intended to replace GAP/QDistRnd calls for *small* Tanner slice codes, typically:
  - n <= 100
  - k <= ~10 (so 2^k enumeration is cheap)

We assume matrices are over GF(2) and stored as MatrixMarket coordinate files (.mtx)
produced by this repo.

Core use-cases:
  - Given a parity-check matrix H, compute:
      n = number of columns
      k = dim ker(H) = n - rank(H)
      d = minimum Hamming weight of a nonzero codeword in ker(H)
    exactly when k is small; otherwise, sample codewords.

Notes:
  - For k=0 (only the all-zero codeword), "distance" is undefined; we return d = n+1
    so that it behaves as "infinite" for threshold comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClassicalCodeAnalysis:
    """Summary statistics for a binary linear code."""

    # Parity-check matrix shape
    m: int  # rows of H
    n: int  # cols of H (code length)

    # Code parameters
    rank: int  # rank(H)
    k: int  # dim ker(H) = n - rank(H)

    # Distance info (minimum weight among nonzero codewords)
    d: int  # exact if `exact=True`, else best found upper bound
    exact: bool  # True if exhaustive enumeration performed

    # Work performed
    codewords_checked: int
    method: str  # "exact_gray", "sample", "trivial"

    def as_dict(self) -> dict:
        return {
            "m": self.m,
            "n": self.n,
            "rank": self.rank,
            "k": self.k,
            "d": self.d,
            "exact": self.exact,
            "codewords_checked": self.codewords_checked,
            "method": self.method,
        }


def _noncomment_lines(f) -> Iterator[str]:
    for line in f:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        yield s


def read_mtx_coordinate_binary(path: str | Path) -> Tuple[int, int, List[int]]:
    """
    Read a MatrixMarket 'coordinate' matrix as a binary (GF(2)) matrix.

    Returns (m, n, rows_bitmask) where rows_bitmask is a list of length m and each
    element is a Python int whose bit j (0-based) is the entry at column j.

    Supported headers:
      - %%MatrixMarket matrix coordinate integer general
      - %%MatrixMarket matrix coordinate pattern general
    Values are reduced mod 2; duplicate entries toggle.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"{p}: not a MatrixMarket file (missing %%MatrixMarket header)")
        header_l = header.lower()
        if "matrix" not in header_l or "coordinate" not in header_l:
            raise ValueError(f"{p}: only coordinate MatrixMarket is supported, got: {header}")
        is_pattern = "pattern" in header_l

        it = _noncomment_lines(f)
        try:
            dims = next(it)
        except StopIteration as e:
            raise ValueError(f"{p}: missing dimension line") from e

        parts = dims.split()
        if len(parts) < 3:
            raise ValueError(f"{p}: invalid dimension line: {dims!r}")
        m, n, nnz = (int(parts[0]), int(parts[1]), int(parts[2]))
        if m < 0 or n < 0 or nnz < 0:
            raise ValueError(f"{p}: negative dimension/nnz in: {dims!r}")

        rows = [0] * m
        read = 0
        for line in it:
            if read >= nnz:
                # Some writers may append extra whitespace/comments; ignore after nnz.
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            if i < 0 or i >= m or j < 0 or j >= n:
                raise ValueError(f"{p}: index out of bounds at entry {read+1}/{nnz}: {line!r}")
            if is_pattern:
                val = 1
            else:
                if len(parts) < 3:
                    raise ValueError(f"{p}: expected value in coordinate integer format: {line!r}")
                val = int(parts[2]) & 1
            if val:
                rows[i] ^= 1 << j  # mod-2 accumulation
            read += 1

        if read < nnz:
            raise ValueError(f"{p}: expected {nnz} entries, found {read}")
        return m, n, rows


def rref_bitrows(rows: Sequence[int], ncols: int) -> Tuple[List[int], List[int]]:
    """
    Reduced row echelon form over GF(2) for a matrix represented as bit-rows.

    Returns (rref_rows, pivots), where:
      - rref_rows is a list of nonzero rows (bitmasks), in pivot order
      - pivots is a list of pivot column indices (0-based), same length as rref_rows
    """
    if ncols < 0:
        raise ValueError("ncols must be nonnegative")
    # Work on a copy
    mat = [int(r) for r in rows]
    m = len(mat)

    pivots: List[int] = []
    r = 0  # current pivot row
    for c in range(ncols):
        if r >= m:
            break
        # Find pivot row with bit c set
        pivot_row = None
        bit = 1 << c
        for i in range(r, m):
            if mat[i] & bit:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        # Swap into position r
        if pivot_row != r:
            mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
        pivots.append(c)

        # Eliminate this column in all other rows (RREF directly)
        pivot_val = mat[r]
        for i in range(m):
            if i != r and (mat[i] & bit):
                mat[i] ^= pivot_val

        r += 1

    # Keep only nonzero pivot rows
    rref_rows = [mat[i] for i in range(r) if mat[i] != 0]
    pivots = pivots[: len(rref_rows)]
    return rref_rows, pivots


def nullspace_basis_from_rref(rref_rows: Sequence[int], pivots: Sequence[int], ncols: int) -> List[int]:
    """
    Given an RREF(H) over GF(2), return a basis of ker(H) as bitmasks of length ncols.

    Convention:
      - free columns correspond to basis vectors
      - for a free column f, the basis vector has x_f = 1, and for each pivot row
        the pivot variable is set according to the row's coefficient in column f.

    Complexity: O(rank * (#free)).
    """
    pivot_set = set(pivots)
    free_cols = [c for c in range(ncols) if c not in pivot_set]
    basis: List[int] = []
    # Each pivot row corresponds to one pivot column
    for f in free_cols:
        v = 1 << f
        bit_f = 1 << f
        for row, pcol in zip(rref_rows, pivots):
            if row & bit_f:
                v |= 1 << pcol
        basis.append(v)
    return basis


def min_distance_exact_gray(
    basis: Sequence[int],
    ncols: int,
    *,
    stop_at: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Exact minimum distance by enumerating all nonzero codewords using Gray codes.

    Returns (d, codewords_checked).

    `stop_at` enables early exit: if we find a codeword of weight <= stop_at,
    we stop and return that value (still exact if stop_at is None).
    """
    k = len(basis)
    if k == 0:
        # Only {0}; treat as "infinite"
        return ncols + 1, 0

    best = ncols + 1
    cw = 0
    prev_gray = 0
    checked = 0

    # Enumerate 1..2^k-1
    limit = 1 << k
    for t in range(1, limit):
        gray = t ^ (t >> 1)
        diff = gray ^ prev_gray
        # index of toggled bit
        idx = (diff & -diff).bit_length() - 1
        cw ^= basis[idx]
        prev_gray = gray

        checked += 1
        w = cw.bit_count()
        if w and w < best:
            best = w
            if stop_at is not None and best <= stop_at:
                return best, checked

    return best, checked


def min_distance_sample(
    basis: Sequence[int],
    ncols: int,
    *,
    samples: int = 20000,
    seed: Optional[int] = None,
    stop_at: Optional[int] = None,
    max_subset_size: int = 8,
) -> Tuple[int, int]:
    """
    Heuristic (non-exact) distance search for larger k via sampling small random subsets.

    Strategy:
      - check all single basis vectors (always cheap)
      - then sample random XORs of 2..max_subset_size basis vectors

    Returns (best_weight_found, codewords_checked).
    """
    k = len(basis)
    if k == 0:
        return ncols + 1, 0

    rng = random.Random(seed)
    best = ncols + 1
    checked = 0

    # Singletons
    for v in basis:
        checked += 1
        w = v.bit_count()
        if w and w < best:
            best = w
            if stop_at is not None and best <= stop_at:
                return best, checked

    if k == 1:
        return best, checked

    # Random small subsets
    max_s = max(2, min(max_subset_size, k))
    for _ in range(max(0, samples)):
        s = rng.randint(2, max_s)
        # sample s distinct indices
        idxs = rng.sample(range(k), s)
        cw = 0
        for i in idxs:
            cw ^= basis[i]
        checked += 1
        w = cw.bit_count()
        if w and w < best:
            best = w
            if stop_at is not None and best <= stop_at:
                return best, checked

    return best, checked


def analyze_parity_check_bitrows(
    rows: Sequence[int],
    ncols: int,
    *,
    stop_at: Optional[int] = None,
    max_k_exact: int = 16,
    samples: int = 20000,
    seed: Optional[int] = None,
) -> ClassicalCodeAnalysis:
    """
    Analyze a binary linear code given by parity-check rows over GF(2).

    Computes:
      rank(H), k = n - rank(H), and minimum distance of ker(H).

    If k <= max_k_exact, enumerate all 2^k-1 nonzero codewords exactly.
    Otherwise sample to find a (usually good) upper bound.

    `stop_at` allows early exit: if we find distance <= stop_at we stop (useful
    for filtering).
    """
    m = len(rows)
    if ncols < 0:
        raise ValueError("ncols must be nonnegative")

    rref_rows, pivots = rref_bitrows(rows, ncols)
    rank = len(pivots)
    k = ncols - rank
    basis = nullspace_basis_from_rref(rref_rows, pivots, ncols)

    if k == 0:
        # Only zero codeword; treat as "infinite" for threshold comparisons.
        return ClassicalCodeAnalysis(
            m=m,
            n=ncols,
            rank=rank,
            k=0,
            d=ncols + 1,
            exact=True,
            codewords_checked=0,
            method="trivial",
        )

    if k <= max_k_exact:
        d, checked = min_distance_exact_gray(basis, ncols, stop_at=stop_at)
        return ClassicalCodeAnalysis(
            m=m,
            n=ncols,
            rank=rank,
            k=k,
            d=d,
            exact=True if stop_at is None else (checked == (1 << k) - 1),
            codewords_checked=checked,
            method="exact_gray",
        )

    d, checked = min_distance_sample(
        basis,
        ncols,
        samples=samples,
        seed=seed,
        stop_at=stop_at,
    )
    return ClassicalCodeAnalysis(
        m=m,
        n=ncols,
        rank=rank,
        k=k,
        d=d,
        exact=False,
        codewords_checked=checked,
        method="sample",
    )


def analyze_mtx_parity_check(
    path: str | Path,
    *,
    stop_at: Optional[int] = None,
    max_k_exact: int = 16,
    samples: int = 20000,
    seed: Optional[int] = None,
) -> ClassicalCodeAnalysis:
    """Read a MatrixMarket .mtx and run `analyze_parity_check_bitrows`."""
    m, n, rows = read_mtx_coordinate_binary(path)
    return analyze_parity_check_bitrows(
        rows,
        n,
        stop_at=stop_at,
        max_k_exact=max_k_exact,
        samples=samples,
        seed=seed,
    )


def analyze_mtx_parity_check_many(
    paths: Sequence[str | Path],
    *,
    stop_at: Optional[int] = None,
    max_k_exact: int = 16,
    samples: int = 20000,
    seed: Optional[int] = None,
) -> List[ClassicalCodeAnalysis]:
    """Analyze multiple .mtx files."""
    return [
        analyze_mtx_parity_check(
            p,
            stop_at=stop_at,
            max_k_exact=max_k_exact,
            samples=samples,
            seed=seed,
        )
        for p in paths
    ]


def analyze_mtx_parity_check_dir(
    dir_path: str | Path,
    *,
    glob: str = "slice*.mtx",
    stop_at: Optional[int] = None,
    max_k_exact: int = 16,
    samples: int = 20000,
    seed: Optional[int] = None,
) -> List[Tuple[Path, ClassicalCodeAnalysis]]:
    """
    Analyze all matching .mtx files in a directory, returning (path, analysis) pairs.

    By default, it matches "slice*.mtx".
    """
    d = Path(dir_path)
    paths = sorted(d.glob(glob))
    return [
        (
            p,
            analyze_mtx_parity_check(
                p,
                stop_at=stop_at,
                max_k_exact=max_k_exact,
                samples=samples,
                seed=seed,
            ),
        )
        for p in paths
    ]
