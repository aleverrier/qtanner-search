from qtanner.search import _best_by_k_table


def test_best_by_k_table_formatting() -> None:
    entry = {
        "candidate_id": "C4_c00001",
        "group": {"spec": "C4"},
        "n": 144,
        "k": 12,
        "qdistrnd": {
            "d_ub": 11,
            "dx_ub": 11,
            "dz_ub": 12,
            "steps": 500,
        },
    }
    table = _best_by_k_table({("C4", 144, 12): entry})
    assert "[best_by_k]" in table
    assert "C4" in table
    assert "11" in table
