from pathlib import Path

from qtanner.best_codes_updater import CodeRecord, select_best_by_nk


def _rec(code_id: str, n: int, k: int, d: int, trials: int) -> CodeRecord:
    return CodeRecord(
        code_id=code_id,
        n=n,
        k=k,
        d=d,
        trials=trials,
        meta_path=Path("meta.json"),
        hx_path=Path("Hx.mtx"),
        hz_path=Path("Hz.mtx"),
    )


def test_select_best_prefers_max_trials() -> None:
    r1 = _rec("B", 100, 10, 11, 100)
    r2 = _rec("A", 100, 10, 20, 50)
    selected = select_best_by_nk([r1, r2])
    assert selected[(100, 10)].code_id == "B"


def test_select_best_prefers_max_distance_then_code_id() -> None:
    r1 = _rec("b", 100, 10, 15, 100)
    r2 = _rec("c", 100, 10, 20, 100)
    r3 = _rec("a", 100, 10, 20, 100)
    selected = select_best_by_nk([r1, r2, r3])
    assert selected[(100, 10)].code_id == "a"


def test_distance_floor_except_n36() -> None:
    low = _rec("low", 144, 20, 4, 1000)
    high = _rec("high", 144, 20, 6, 10)
    sel = select_best_by_nk([low, high])
    assert sel[(144, 20)].code_id == "high"

    low36 = _rec("low36", 36, 6, 3, 10)
    sel36 = select_best_by_nk([low36])
    assert sel36[(36, 6)].code_id == "low36"
