"""MatrixMarket header tests for MTX writers."""

from __future__ import annotations

from qtanner.mtx import (
    validate_mtx_for_qdistrnd,
    write_mtx,
    write_mtx_from_bitrows,
    write_mtx_from_rows,
)


def _read_first_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().rstrip("\n")


def test_mtx_writers_header(tmp_path):
    expected = "%%MatrixMarket matrix coordinate integer general"

    direct_path = tmp_path / "direct.mtx"
    write_mtx(str(direct_path), 2, 2, [(0, 1)])
    assert _read_first_line(str(direct_path)) == expected

    rows_path = tmp_path / "rows.mtx"
    write_mtx_from_rows(str(rows_path), 2, 2, [[1], []])
    assert _read_first_line(str(rows_path)) == expected

    bitrows_path = tmp_path / "bitrows.mtx"
    write_mtx_from_bitrows(str(bitrows_path), [0b10, 0b0], 2)
    assert _read_first_line(str(bitrows_path)) == expected


def test_mtx_writers_validation(tmp_path):
    path = tmp_path / "tiny.mtx"
    write_mtx(str(path), 2, 3, [(0, 0), (1, 2)])
    stats = validate_mtx_for_qdistrnd(path)
    assert stats["rows"] == 2
    assert stats["cols"] == 3
    assert stats["nnz_declared"] == 2
    assert stats["nnz_actual"] == 2
    assert stats["min_i"] >= 1
    assert stats["min_j"] >= 1
    assert stats["max_i"] <= stats["rows"]
    assert stats["max_j"] <= stats["cols"]
