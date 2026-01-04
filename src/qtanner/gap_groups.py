"""Helpers for importing GAP SmallGroup data into FiniteGroup."""

from __future__ import annotations

import ast
import json
import os
import subprocess
from typing import List, Tuple

from .group import FiniteGroup


def _parse_gap_lists(output: str) -> Tuple[List[List[int]], List[int]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("GAP output missing expected multiplication table and inverse list.")
    mul_table = ast.literal_eval(lines[0])
    inv = ast.literal_eval(lines[1])
    return mul_table, inv


def smallgroup(order: int, gid: int, cache_dir: str = ".cache/gap_smallgroups") -> FiniteGroup:
    """Load a GAP SmallGroup as a FiniteGroup, with JSON caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"smallgroup_{order}_{gid}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return FiniteGroup(
            name=f"SmallGroup({order},{gid})",
            order=payload["order"],
            mul_table=payload["mul_table"],
            inv=payload["inv"],
        )

    script = "\n".join(
        [
            "G := SmallGroup(%d,%d);" % (order, gid),
            "elts := Elements(G);",
            "dict := NewDictionary(elts, true);",
            "for i in [1..Length(elts)] do AddDictionary(dict, elts[i], i-1); od;",
            "mul := List([1..Length(elts)], i -> List([1..Length(elts)], j -> LookupDictionary(dict, elts[i]*elts[j])));",
            "inv := List([1..Length(elts)], i -> LookupDictionary(dict, elts[i]^-1));",
            "Print(mul, \"\\n\", inv, \"\\n\");",
        ]
    )
    try:
        proc = subprocess.run(
            ["gap", "-q"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GAP executable not found. Install GAP and ensure `gap` is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GAP failed: {exc.stderr.strip()}") from exc

    mul_table, inv = _parse_gap_lists(proc.stdout)
    payload = {
        "order": order,
        "gid": gid,
        "mul_table": mul_table,
        "inv": inv,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return FiniteGroup(
        name=f"SmallGroup({order},{gid})",
        order=order,
        mul_table=mul_table,
        inv=inv,
    )


__all__ = ["smallgroup"]
