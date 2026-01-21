import subprocess

import pytest

import qtanner.gap_backend as gap_backend


def test_gap_backend_parses_group_data(monkeypatch) -> None:
    gap_backend.gap_group_data.cache_clear()
    output = "\n".join(
        [
            "ORDER 2",
            "ISABELIAN true",
            "ELTS_BEGIN",
            "<identity>",
            "a",
            "ELTS_END",
            "MUL_BEGIN",
            "0,1",
            "1,0",
            "MUL_END",
            "INV_BEGIN",
            "0,1",
            "INV_END",
            "AUT_BEGIN",
            "0,1",
            "1,0",
            "AUT_END",
            "",
        ]
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(gap_backend.subprocess, "run", fake_run)

    data = gap_backend.gap_group_data("C2")
    assert data.order == 2
    assert data.elements == ["<identity>", "a"]
    assert data.mul_table == [[0, 1], [1, 0]]
    assert data.inv_table == [0, 1]
    assert data.automorphisms == [[0, 1], [1, 0]]
    assert data.is_abelian is True


def test_gap_backend_missing_gap(monkeypatch) -> None:
    gap_backend.gap_group_data.cache_clear()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("gap not found")

    monkeypatch.setattr(gap_backend.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        gap_backend.gap_group_data("C2")
    msg = str(excinfo.value)
    assert "brew install gap" in msg
    assert "--gap-cmd" in msg
