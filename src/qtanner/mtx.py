"""MatrixMarket writers for sparse binary matrices."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def write_mtx(path: str, n_rows: int, n_cols: int, ones: List[Tuple[int, int]]) -> None:
    """Write a MatrixMarket coordinate integer general matrix with 1s."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"{n_rows} {n_cols} {len(ones)}\n")
        for r, c in ones:
            f.write(f"{r + 1} {c + 1} 1\n")


def write_mtx_from_rows(
    path: str, n_rows: int, n_cols: int, row_ones: List[List[int]]
) -> None:
    """Write MatrixMarket data given list of 1-indices per row."""
    ones = []
    for r, cols in enumerate(row_ones):
        for c in cols:
            ones.append((r, c))
    write_mtx(path, n_rows, n_cols, ones)


def write_mtx_from_bitrows(path: str, rows: List[int], n_cols: int) -> None:
    """Write MatrixMarket data from int bitset rows."""
    n_rows = len(rows)
    ones: List[Tuple[int, int]] = []
    for r, row in enumerate(rows):
        x = row
        while x:
            lsb = x & -x
            c = lsb.bit_length() - 1
            ones.append((r, c))
            x -= lsb
    write_mtx(path, n_rows, n_cols, ones)


__all__ = ["write_mtx", "write_mtx_from_rows", "write_mtx_from_bitrows"]
