"""Trial schedule helpers."""

from __future__ import annotations

from typing import List, Tuple


def parse_trials_schedule(s: str) -> List[Tuple[int, int]]:
    """Parse schedule strings like '150:1000,300:2000' into sorted tuples."""
    if not s or not s.strip():
        return []
    schedule: List[Tuple[int, int]] = []
    for raw_part in s.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Empty schedule entry; expected 'n:trials'.")
        if ":" not in part:
            raise ValueError(f"Invalid schedule entry '{part}'; expected 'n:trials'.")
        key_str, val_str = part.split(":", 1)
        key_str = key_str.strip()
        val_str = val_str.strip()
        if not key_str or not val_str:
            raise ValueError(f"Invalid schedule entry '{part}'; expected 'n:trials'.")
        try:
            n_max = int(key_str)
            trials = int(val_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid schedule entry '{part}'; expected integers."
            ) from exc
        if n_max <= 0 or trials <= 0:
            raise ValueError(
                f"Invalid schedule entry '{part}'; values must be positive."
            )
        if schedule and n_max <= schedule[-1][0]:
            raise ValueError("Schedule thresholds must be strictly increasing.")
        schedule.append((n_max, trials))
    return schedule


def trials_for_n(n: int, schedule: List[Tuple[int, int]], default: int) -> int:
    """Return trial count for n according to schedule or default if none match."""
    for n_max, trials in schedule:
        if n <= n_max:
            return trials
    return default


__all__ = ["parse_trials_schedule", "trials_for_n"]
