import pytest

import qtanner.dist_m4ri as dist_m4ri


class _Result:
    def __init__(self, stdout: str, stderr: str) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


def test_run_dist_m4ri_css_rw_builds_command(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, text, capture_output, check):
        calls.append(cmd)
        return _Result("ok d=8 done", "")

    monkeypatch.setattr(dist_m4ri.subprocess, "run", fake_run)

    d = dist_m4ri.run_dist_m4ri_css_rw(
        [0b1],
        [0b1],
        1,
        steps=10,
        wmin=3,
        seed=5,
        dist_m4ri_cmd="dist_m4ri",
    )

    assert d == 8
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "dist_m4ri"
    assert cmd[1] == "debug=0"
    assert cmd[2] == "method=1"
    assert cmd[3] == "steps=10"
    assert cmd[4] == "wmin=3"
    assert cmd[5] == "seed=5"
    assert cmd[6].startswith("fin=")


def test_run_dist_m4ri_css_rw_parses_negative(monkeypatch) -> None:
    def fake_run(cmd, text, capture_output, check):
        return _Result("d=-7", "")

    monkeypatch.setattr(dist_m4ri.subprocess, "run", fake_run)

    d = dist_m4ri.run_dist_m4ri_css_rw([0b1], [0b1], 1, steps=1, wmin=0)
    assert d == -7


def test_run_dist_m4ri_classical_rw_builds_command(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, text, capture_output, check):
        calls.append(cmd)
        return _Result("ok d=6 done", "")

    monkeypatch.setattr(dist_m4ri.subprocess, "run", fake_run)

    d = dist_m4ri.run_dist_m4ri_classical_rw(
        [0b1],
        1,
        steps=12,
        wmin=4,
        seed=7,
        dist_m4ri_cmd="dist_m4ri",
    )

    assert d == 6
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "dist_m4ri"
    assert cmd[1] == "debug=0"
    assert cmd[2] == "method=1"
    assert cmd[3] == "steps=12"
    assert cmd[4] == "wmin=4"
    assert cmd[5] == "seed=7"
    assert cmd[6].startswith("finH=")


def test_run_dist_m4ri_classical_rw_parses_bare(monkeypatch) -> None:
    def fake_run(cmd, text, capture_output, check):
        return _Result("-6", "")

    monkeypatch.setattr(dist_m4ri.subprocess, "run", fake_run)

    d = dist_m4ri.run_dist_m4ri_classical_rw([0b1], 1, steps=1, wmin=0)
    assert d == -6


def test_run_dist_m4ri_css_rw_missing_d(monkeypatch) -> None:
    def fake_run(cmd, text, capture_output, check):
        return _Result("no distance here", "")

    monkeypatch.setattr(dist_m4ri.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        dist_m4ri.run_dist_m4ri_css_rw([0b1], [0b1], 1, steps=1, wmin=0)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("-6\n", -6),
        (" \n-6\n", -6),
    ],
)
def test_parse_last_distance_bare_integer(output: str, expected: int) -> None:
    assert dist_m4ri._parse_last_distance(output) == expected


def test_parse_last_distance_multiple_integer_lines() -> None:
    output = "1\n-2\n  5 \n"
    assert dist_m4ri._parse_last_distance(output) == 5
