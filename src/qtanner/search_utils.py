"""Helpers for search scripts."""

from __future__ import annotations

from typing import List, Optional


def parse_qd_batches(value: str) -> List[int]:
    if not value or not value.strip():
        raise ValueError("Empty batch list; expected comma-separated integers.")
    batches: List[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Empty batch entry; expected integers.")
        try:
            batch = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid batch entry '{part}'; expected integers."
            ) from exc
        if batch <= 0:
            raise ValueError(f"Invalid batch entry '{part}'; must be positive.")
        batches.append(batch)
    return batches


def threshold_to_beat(target: int, best_d_obs: Optional[int]) -> int:
    if best_d_obs is None:
        return target
    return max(target, int(best_d_obs))


def should_abort_after_batch(
    *, d_obs: int, target: int, best_d_obs: Optional[int]
) -> bool:
    if d_obs < target:
        return True
    if best_d_obs is not None and d_obs < best_d_obs:
        return True
    return False


__all__ = ["parse_qd_batches", "threshold_to_beat", "should_abort_after_batch"]
