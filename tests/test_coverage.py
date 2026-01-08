from qtanner.search import _coverage_ratios, _coverage_summary


def test_coverage_ratios_and_summary() -> None:
    entry = {
        "group": "C1",
        "A_candidates_selected": 2,
        "B_candidates_selected": 3,
        "Q_possible_ge_sqrt": 10,
        "Q_possible_selected": 6,
        "Q_meas_run": 3,
    }
    ratios = _coverage_ratios(entry)
    assert ratios["Q_meas_over_possible_ge_sqrt"] == 0.3
    assert ratios["Q_meas_over_possible_selected"] == 0.5
    summary = _coverage_summary(entry)
    assert "[coverage]" in summary
    assert "C1" in summary
