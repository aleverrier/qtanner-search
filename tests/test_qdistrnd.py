"""Optional tests for the GAP/QDistRnd wrapper."""

import os

import pytest

from qtanner.group import FiniteGroup
from qtanner.lift_matrices import build_hx_hz
from qtanner.local_codes import repetition_2
from qtanner.mtx import write_mtx, write_mtx_from_bitrows
from qtanner.qdistrnd import (
    _build_gap_script,
    _build_gap_script_dz,
    dist_rand_css_mtx,
    dist_rand_dz_mtx,
    gap_is_available,
    qd_stats_d_ub,
    qdistrnd_is_available,
)


def test_gap_script_omits_one_mul():
    script = _build_gap_script(
        "Hx.mtx",
        "Hz.mtx",
        num=1,
        mindist=0,
        debug=0,
        maxav=None,
        seed=None,
    )
    assert "One(F) * hxinfo[3]" not in script
    assert "One(F) * hzinfo[3]" not in script
    assert "ReadMMGF2" in script
    assert "HX := ReadMMGF2(" in script
    assert "HZ := ReadMMGF2(" in script
    assert "Hx.mtx" in script
    assert "Hz.mtx" in script
    assert "FlushOutput" not in script


def test_gap_script_dz_uses_single_zero_row():
    script = _build_gap_script_dz(
        "Hx.mtx",
        num=1,
        mindist=0,
        debug=0,
        seed=None,
    )
    assert "NullMat(1," in script
    assert "QDISTRESULT_Z" in script
    assert "FlushOutput" not in script


def test_dist_rand_dz_missing_gap_cmd_logs(tmp_path):
    hx_path = tmp_path / "Hx.mtx"
    write_mtx(str(hx_path), 1, 1, [])
    log_path = tmp_path / "qdistrnd.log"

    with pytest.raises(RuntimeError) as excinfo:
        dist_rand_dz_mtx(
            str(hx_path),
            num=1,
            mindist=0,
            debug=0,
            gap_cmd="gap-does-not-exist",
            timeout_sec=1,
            log_path=str(log_path),
        )

    assert "GAP command not found" in str(excinfo.value)
    assert "UnboundLocalError" not in str(excinfo.value)
    assert log_path.exists()
    assert "[stdout]" in log_path.read_text(encoding="utf-8")


def test_qdistrnd_optional(tmp_path):
    if os.environ.get("QTANNER_RUN_GAP_TESTS") != "1":
        pytest.skip("Set QTANNER_RUN_GAP_TESTS=1 to run GAP/QDistRnd tests.")
    if not gap_is_available():
        pytest.skip("GAP is not available on PATH.")
    if not qdistrnd_is_available():
        pytest.skip("GAP QDistRnd package is not available.")

    group = FiniteGroup.cyclic(3)
    A = [0, 1]
    B = [0, 1]
    C0 = repetition_2()
    C1 = repetition_2()
    C0p = repetition_2()
    C1p = repetition_2()
    hx_rows, hz_rows, n_cols = build_hx_hz(group, A, B, C0, C1, C0p, C1p)

    hx_path = tmp_path / "Hx.mtx"
    hz_path = tmp_path / "Hz.mtx"
    write_mtx_from_bitrows(str(hx_path), hx_rows, n_cols)
    write_mtx_from_bitrows(str(hz_path), hz_rows, n_cols)

    summary = dist_rand_css_mtx(
        str(hx_path),
        str(hz_path),
        num=200,
        mindist=0,
        debug=2,
        timeout_sec=60,
    )
    for key in ["dX_ub", "dZ_ub", "d_ub", "runtime_sec", "gap_cmd"]:
        assert key in summary
    assert summary["d_ub"] >= 1


def test_qd_stats_d_ub_none():
    assert qd_stats_d_ub(None) is None
