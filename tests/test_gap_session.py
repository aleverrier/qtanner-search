from qtanner.gap_session import _build_gap_server_script


def test_gap_session_script_uses_io_and_no_flush() -> None:
    script = _build_gap_server_script()
    assert "IO_ReadLine" in script
    assert "IO_WriteLine" in script
    assert "FlushOutput" not in script
