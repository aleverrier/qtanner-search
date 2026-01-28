"""Bootstrap helper to re-exec with a Python that supports required features."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Iterable, List


_REQUIRED_FEATURES = ("bit_count",)


def _has_required_features() -> bool:
    return hasattr(int, "bit_count")


def _iter_candidates() -> Iterable[str]:
    # Allow explicit override first.
    env_python = os.environ.get("QTANNER_PYTHON")
    if env_python:
        yield env_python
    # Prefer common Homebrew/local installs before PATH resolution.
    for path in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3"):
        if os.path.exists(path):
            yield path
    # Prefer newer explicit versions, then generic python3.
    for name in (
        "python3.14",
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
    ):
        found = shutil.which(name)
        if found:
            yield found


def _is_usable(candidate: str) -> bool:
    try:
        result = subprocess.run(
            [candidate, "-c", "import sys; sys.exit(0 if hasattr(int,'bit_count') else 1)"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _dedupe_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        real = os.path.realpath(p)
        if real in seen:
            continue
        seen.add(real)
        out.append(real)
    return out


def ensure_minimum_python() -> None:
    """Re-exec with a Python that supports required features, if needed."""
    if _has_required_features():
        return
    if os.environ.get("QTANNER_PYTHON_REEXEC") == "1":
        missing = ", ".join(_REQUIRED_FEATURES)
        raise SystemExit(
            "Required Python features missing ({}). Set QTANNER_PYTHON to a newer "
            "interpreter (e.g., Homebrew python3).".format(missing)
        )
    current = os.path.realpath(sys.executable)
    for candidate in _dedupe_paths(_iter_candidates()):
        if candidate == current:
            continue
        if _is_usable(candidate):
            env = os.environ.copy()
            env["QTANNER_PYTHON_REEXEC"] = "1"
            os.execve(candidate, [candidate, *sys.argv], env)
    missing = ", ".join(_REQUIRED_FEATURES)
    raise SystemExit(
        "Required Python features missing ({}). Install a newer Python or set "
        "QTANNER_PYTHON to its path.".format(missing)
    )


__all__ = ["ensure_minimum_python"]
