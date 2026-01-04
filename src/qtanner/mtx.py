"""MatrixMarket writers for sparse binary matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

MATRIX_MARKET_HEADER = "%%MatrixMarket matrix coordinate integer general"


def write_mtx(path: str, n_rows: int, n_cols: int, ones: List[Tuple[int, int]]) -> None:
    """Write a MatrixMarket coordinate integer general matrix with 1s."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{MATRIX_MARKET_HEADER}\n")
        sorted_ones = sorted(ones)
        f.write(f"{n_rows} {n_cols} {len(sorted_ones)}\n")
        # MatrixMarket uses 1-based coordinates; inputs are 0-based.
        for r, c in sorted_ones:
            f.write(f"{r + 1} {c + 1} 1\n")


def write_mtx_from_rows(
    path: str, n_rows: int, n_cols: int, row_ones: List[List[int]]
) -> None:
    """Write MatrixMarket data given list of 0-based column indices per row."""
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


def validate_mtx_for_qdistrnd(path: Path) -> dict:
    """Validate MTX files for QDistRnd and return basic stats."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError("Invalid MTX at line 1: <EOF>")

    header = lines[0].rstrip("\n")
    if header.endswith("\r"):
        header = header.rstrip("\r")
    if header != MATRIX_MARKET_HEADER:
        raise RuntimeError(f"Invalid MTX at line 1: {lines[0].rstrip()}")

    idx = 1
    dims_line_no = None
    dims_line_text = None
    rows = cols = nnz_declared = None
    while idx < len(lines):
        raw = lines[idx].strip()
        if raw == "" or raw.startswith("%"):
            idx += 1
            continue
        dims_line_no = idx + 1
        dims_line_text = lines[idx].rstrip("\n")
        parts = raw.split()
        if len(parts) != 3:
            raise RuntimeError(
                f"Invalid MTX at line {dims_line_no}: {dims_line_text}"
            )
        try:
            rows, cols, nnz_declared = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid MTX at line {dims_line_no}: {dims_line_text}"
            ) from exc
        if rows <= 0 or cols <= 0 or nnz_declared < 0:
            raise RuntimeError(
                f"Invalid MTX at line {dims_line_no}: {dims_line_text}"
            )
        idx += 1
        break
    if rows is None or cols is None or nnz_declared is None:
        raise RuntimeError(f"Invalid MTX at line {len(lines) + 1}: <EOF>")

    nnz_actual = 0
    min_i = max_i = min_j = max_j = None
    for line_idx in range(idx, len(lines)):
        raw = lines[line_idx].strip()
        if raw == "" or raw.startswith("%"):
            continue
        line_no = line_idx + 1
        line_text = lines[line_idx].rstrip("\n")
        parts = raw.split()
        if len(parts) != 3:
            raise RuntimeError(f"Invalid MTX at line {line_no}: {line_text}")
        try:
            i, j, _v = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise RuntimeError(f"Invalid MTX at line {line_no}: {line_text}") from exc
        if i < 1 or i > rows or j < 1 or j > cols:
            raise RuntimeError(f"Invalid MTX at line {line_no}: {line_text}")
        nnz_actual += 1
        if min_i is None or i < min_i:
            min_i = i
        if max_i is None or i > max_i:
            max_i = i
        if min_j is None or j < min_j:
            min_j = j
        if max_j is None or j > max_j:
            max_j = j

    if nnz_actual != nnz_declared:
        raise RuntimeError(
            "Invalid MTX at line "
            f"{dims_line_no}: {dims_line_text}"
        )

    return {
        "rows": rows,
        "cols": cols,
        "nnz_declared": nnz_declared,
        "nnz_actual": nnz_actual,
        "min_i": min_i,
        "max_i": max_i,
        "min_j": min_j,
        "max_j": max_j,
    }


__all__ = [
    "MATRIX_MARKET_HEADER",
    "write_mtx",
    "write_mtx_from_rows",
    "write_mtx_from_bitrows",
    "validate_mtx_for_qdistrnd",
]
