"""dist-m4ri CLI wrapper for CSS distance estimation (RW method)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from .mtx import write_mtx_from_bitrows

_DOCS_HINT = "See README.md#dist-m4ri for setup instructions."


def write_mtx_gf2(path: str | Path, rows: Sequence[int], n_cols: int) -> None:
    """Write a sparse GF(2) matrix in MatrixMarket coordinate format."""
    write_mtx_from_bitrows(str(path), list(rows), n_cols)


def dist_m4ri_is_available(dist_m4ri_cmd: str = "dist_m4ri") -> bool:
    """Return True if the dist_m4ri binary is available on PATH."""
    return shutil.which(dist_m4ri_cmd) is not None


def _parse_last_distance(output: str) -> int:
    matches = list(re.finditer(r"(?<![A-Za-z0-9_])d=(-?\d+)", output))
    if matches:
        return int(matches[-1].group(1))

    line_matches = list(re.finditer(r"(?m)^[ \t\r]*(-?\d+)[ \t\r]*$", output))
    if line_matches:
        return int(line_matches[-1].group(1))

    raise RuntimeError(
        "dist_m4ri output did not include a parsable distance. "
        f"Output:\n{output}"
    )


def run_dist_m4ri_css_rw(
    hx_rows: Sequence[int],
    hz_rows: Sequence[int],
    n_cols: int,
    steps: int,
    wmin: int,
    *,
    seed: int = 0,
    dist_m4ri_cmd: str = "dist_m4ri",
) -> int:
    """Run dist-m4ri RW (method=1) on CSS code defined by Hx/Hz bitrows."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if wmin < 0:
        raise ValueError("wmin must be nonnegative.")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "code"
        hx_path = f"{base}X.mtx"
        hz_path = f"{base}Z.mtx"
        write_mtx_from_bitrows(hx_path, list(hx_rows), n_cols)
        write_mtx_from_bitrows(hz_path, list(hz_rows), n_cols)
        cmd = [
            dist_m4ri_cmd,
            "debug=0",
            "method=1",
            f"steps={int(steps)}",
            f"wmin={int(wmin)}",
            f"seed={int(seed)}",
            f"fin={base}",
        ]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"dist_m4ri not found on PATH (cmd='{dist_m4ri_cmd}'). "
                "Install dist-m4ri and ensure the dist_m4ri binary is available. "
                f"{_DOCS_HINT}"
            ) from exc
        output = (result.stdout or "") + (result.stderr or "")
        try:
            return _parse_last_distance(output)
        except RuntimeError as exc:
            raise RuntimeError(
                "dist_m4ri output did not contain a parsable distance. "
                f"Output:\n{output}"
            ) from exc


def run_dist_m4ri_classical_rw(
    h_rows: Sequence[int],
    n_cols: int,
    steps: int,
    wmin: int,
    *,
    seed: int = 0,
    dist_m4ri_cmd: str = "dist_m4ri",
) -> int:
    """Run dist-m4ri RW (method=1) for a classical code given parity-check rows."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if wmin < 0:
        raise ValueError("wmin must be nonnegative.")
    with tempfile.TemporaryDirectory() as tmpdir:
        h_path = Path(tmpdir) / "codeH.mtx"
        write_mtx_from_bitrows(str(h_path), list(h_rows), n_cols)
        cmd = [
            dist_m4ri_cmd,
            "debug=0",
            "method=1",
            f"steps={int(steps)}",
            f"wmin={int(wmin)}",
            f"seed={int(seed)}",
            f"finH={h_path}",
        ]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"dist_m4ri not found on PATH (cmd='{dist_m4ri_cmd}'). "
                "Install dist-m4ri and ensure the dist_m4ri binary is available. "
                f"{_DOCS_HINT}"
            ) from exc
        output = (result.stdout or "") + (result.stderr or "")
        try:
            return _parse_last_distance(output)
        except RuntimeError as exc:
            raise RuntimeError(
                "dist_m4ri output did not contain a parsable distance. "
                f"Output:\n{output}"
            ) from exc


__all__ = [
    "dist_m4ri_is_available",
    "run_dist_m4ri_css_rw",
    "run_dist_m4ri_classical_rw",
    "write_mtx_gf2",
]
