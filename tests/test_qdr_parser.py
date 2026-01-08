from qtanner.search import _parse_qdr_line


def test_parse_qdr_line_missing_mx_mz() -> None:
    idx, parsed = _parse_qdr_line("QDR|7|dx=3|dz=4|rx=10|rz=11|vx=2|vz=1")
    assert idx == 7
    assert parsed["mx"] == []
    assert parsed["mz"] == []
    assert parsed["vx"] == 2
    assert parsed["vz"] == 1


def test_parse_qdr_line_blank_vz_defaults() -> None:
    idx, parsed = _parse_qdr_line(
        "QDR|8|dx=3|dz=4|rx=10|rz=11|vx=2|vz=|mx=[1,2]|mz=[]"
    )
    assert idx == 8
    assert parsed["vz"] == 0
    assert parsed["mx"] == [1, 2]
    assert parsed["mz"] == []


def test_parse_qdr_line_missing_vx_vz_defaults() -> None:
    idx, parsed = _parse_qdr_line("QDR|9|dx=3|dz=4|rx=10|rz=11|mx=[1]|mz=[2]")
    assert idx == 9
    assert parsed["vx"] == 0
    assert parsed["vz"] == 0
