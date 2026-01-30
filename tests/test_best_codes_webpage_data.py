from pathlib import Path

from qtanner.best_codes_updater import CodeRecord, update_best_codes_webpage_data


def test_webpage_data_uses_meta_distance(tmp_path: Path) -> None:
    best_dir = tmp_path / "best_codes"
    meta_dir = best_dir / "meta"
    meta_dir.mkdir(parents=True)

    code_id = "Example_k2_d50"
    meta_path = meta_dir / f"{code_id}.json"
    meta_path.write_text(
        (
            "{\n"
            '  "code_id": "Example_k2_d50",\n'
            '  "group": "Example",\n'
            '  "n": 10,\n'
            '  "k": 2,\n'
            '  "d_ub": 33,\n'
            '  "distance": {"d_ub": 33, "steps_used_total": 1000}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    rec = CodeRecord(code_id=code_id, n=10, k=2, d=50, trials=1000)
    update_best_codes_webpage_data({(10, 2): rec}, tmp_path, dry_run=False, verbose=False)

    data = (best_dir / "data.json").read_text(encoding="utf-8")
    assert '"d_ub": 33' in data

    index = (best_dir / "index.tsv").read_text(encoding="utf-8")
    assert "\t33\t" in index
