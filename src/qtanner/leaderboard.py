"""Leaderboard helpers for tracking best codes by (n,k) or n."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_best_by_nk(path: str) -> Dict[str, Dict[str, object]]:
    best_path = Path(path)
    if not best_path.exists():
        return {}
    with best_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def save_best_by_nk(path: str, data: Dict[str, Dict[str, object]]) -> None:
    best_path = Path(path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def key_nk(n: int, k: int) -> str:
    return f"{n},{k}"


def maybe_update_best(data: Dict[str, Dict[str, object]], record: Dict[str, object]) -> bool:
    n = record.get("n")
    k = record.get("k")
    d_obs = record.get("d_obs")
    trials_used = record.get("trials_used")
    if n is None or k is None or d_obs is None or trials_used is None:
        raise ValueError("Record missing required fields n, k, d_obs, trials_used.")
    key = key_nk(int(n), int(k))
    current = data.get(key)
    if current is None:
        data[key] = record
        return True
    current_d = current.get("d_obs")
    current_trials = current.get("trials_used")
    if current_d is None or current_trials is None:
        data[key] = record
        return True
    if int(d_obs) > int(current_d):
        data[key] = record
        return True
    if int(d_obs) == int(current_d) and int(trials_used) < int(current_trials):
        data[key] = record
        return True
    return False


__all__ = ["load_best_by_nk", "save_best_by_nk", "key_nk", "maybe_update_best"]
