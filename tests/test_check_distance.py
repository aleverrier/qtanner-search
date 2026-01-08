from pathlib import Path

from qtanner.check_distance import _find_code_dir


def test_find_code_dir_prefers_best_codes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code_id = "C4_c00001"
    best_dir = run_dir / "best_codes" / "C4" / "n144" / "k12" / code_id
    best_dir.mkdir(parents=True)
    found = _find_code_dir(run_dir, code_id)
    assert found == best_dir


def test_find_code_dir_falls_back_to_promising(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code_id = "C4_c00002"
    prom_dir = run_dir / "promising" / code_id
    prom_dir.mkdir(parents=True)
    found = _find_code_dir(run_dir, code_id)
    assert found == prom_dir
