"""
qtanner.group_ops

A tiny interface for finite groups given by their multiplication table (0-based).

Key entry point:
    ensure_group_ops(group)

Accepted inputs:
  - GroupOps (returned as-is)
  - dict with:
      {"order": o, "gid": g}  (GAP will be used to build tables)
    or
      {"order": o, "gid": g?, "mul_table": ..., "inv": ...}  (no GAP)
  - objects with attributes:
      .order and (.mul_table or .mul) and .inv   (no GAP)
    Optionally .gid and/or .name like "SmallGroup(o,gid)".
  - strings: "o,gid" or "SmallGroup(o,gid)"
  - (o, gid) tuples/lists

If only (order, gid) are available, GAP is used to build the multiplication
and inverse tables (cached).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence


_GROUP_TABLES_BEGIN = "###QTANNER_GROUP_TABLES_BEGIN###"
_GROUP_TABLES_END = "###QTANNER_GROUP_TABLES_END###"


def _tail(s: str, n: int = 2000) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[-n:]


def _coerce_int(x: Any, *, name: str) -> int:
    if x is None:
        raise TypeError(f"Missing required {name}.")
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Could not coerce {name}={x!r} to int.") from e


def _parse_smallgroup_name(name: str) -> tuple[Optional[int], Optional[int]]:
    """
    Try to parse "SmallGroup(o,gid)" out of a name string.
    Returns (order, gid) or (None, None) if no match.
    """
    m = re.search(r"SmallGroup\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", name)
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


@dataclass(frozen=True)
class GroupOps:
    """
    Finite group represented by a multiplication table (0-based indices).

    Attributes:
        order: group order
        gid: SmallGroup id if known, else None
        mul_table: order x order table; mul_table[a][b] is product a*b
        inv: length-order table; inv[a] is a^{-1}
        name: optional display name
    """

    order: int
    gid: Optional[int]
    mul_table: tuple[tuple[int, ...], ...]
    inv: tuple[int, ...]
    name: Optional[str] = None

    # Preferred API
    def mul_of(self, a: int, b: int) -> int:
        return self.mul_table[a][b]

    def inv_of(self, a: int) -> int:
        return self.inv[a]

    # Backwards-compatible alias (older code calls group.mul(...))
    def mul(self, a: int, b: int) -> int:
        return self.mul_of(a, b)

    # NOTE: we cannot define a method named `inv` because `inv` is a table attribute.


def _normalize_table(
    order: int,
    mul_table: Sequence[Sequence[Any]],
    inv: Sequence[Any],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate/normalize tables to tuples of ints (0-based)."""
    o = _coerce_int(order, name="order")
    if o <= 0:
        raise ValueError(f"order must be positive, got {o}.")

    if len(mul_table) != o:
        raise ValueError(f"mul_table has {len(mul_table)} rows but order={o}.")
    norm_mul: list[tuple[int, ...]] = []
    for i, row in enumerate(mul_table):
        if len(row) != o:
            raise ValueError(f"mul_table row {i} has length {len(row)} but order={o}.")
        norm_row = tuple(_coerce_int(x, name=f"mul_table[{i}][*]") for x in row)
        for x in norm_row:
            if x < 0 or x >= o:
                raise ValueError(f"mul_table contains value {x} outside [0,{o-1}].")
        norm_mul.append(norm_row)

    if len(inv) != o:
        raise ValueError(f"inv has length {len(inv)} but order={o}.")
    norm_inv = tuple(_coerce_int(x, name="inv[*]") for x in inv)
    for x in norm_inv:
        if x < 0 or x >= o:
            raise ValueError(f"inv contains value {x} outside [0,{o-1}].")

    return (tuple(norm_mul), norm_inv)


def _extract_between_markers(stdout: str, *, begin: str, end: str) -> str:
    """
    Extract substring between the *last* BEGIN and the first END after it.
    Robust to GAP printing extra stuff before our markers.
    """
    if not stdout:
        raise RuntimeError("Empty stdout; cannot extract group tables markers.")

    begins = [m.start() for m in re.finditer(re.escape(begin), stdout)]
    if not begins:
        raise RuntimeError(f"Could not find BEGIN marker {begin!r} in stdout.")
    for i0 in reversed(begins):
        j0 = i0 + len(begin)
        i1 = stdout.find(end, j0)
        if i1 != -1:
            return stdout[j0:i1]
    raise RuntimeError(f"Could not find END marker {end!r} after any BEGIN marker.")


def _parse_tables_payload(payload: str) -> tuple[int, int, list[list[int]], list[int]]:
    """
    Parse the plain-text payload emitted by the GAP helper:

        <order>
        <gid>
        row0 CSV
        ...
        row{o-1} CSV
        INV
        inv CSV
    """
    lines = [ln.strip() for ln in payload.strip().splitlines() if ln.strip() != ""]
    if len(lines) < 4:
        raise RuntimeError(f"Too few lines in payload to parse tables: {lines!r}")

    o = _coerce_int(lines[0], name="order(line0)")
    gid = _coerce_int(lines[1], name="gid(line1)")

    if len(lines) < 2 + o + 2:
        raise RuntimeError(
            f"Payload too short for order={o}: got {len(lines)} non-empty lines."
        )

    mul_lines = lines[2 : 2 + o]
    inv_tag = lines[2 + o]
    if inv_tag != "INV":
        raise RuntimeError(f"Expected 'INV' line after mul_table, got {inv_tag!r}")
    inv_line = lines[2 + o + 1]

    mul: list[list[int]] = []
    for r, ln in enumerate(mul_lines):
        parts = [p for p in ln.split(",") if p != ""]
        if len(parts) != o:
            raise RuntimeError(f"mul row {r} has {len(parts)} entries, expected {o}.")
        mul.append([_coerce_int(p, name=f"mul[{r}][*]") for p in parts])

    inv_parts = [p for p in inv_line.split(",") if p != ""]
    if len(inv_parts) != o:
        raise RuntimeError(f"inv line has {len(inv_parts)} entries, expected {o}.")
    inv = [_coerce_int(p, name="inv[*]") for p in inv_parts]

    return o, gid, mul, inv


def _run_gap_script(gap_cmd: str, script_text: str, *, timeout_s: int) -> tuple[str, str, int]:
    """
    Run a GAP script (written to a temp file) and return (stdout, stderr, returncode).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".g", delete=False) as tf:
        tf.write(script_text)
        script_path = tf.name

    try:
        cmd = [
            gap_cmd,
            "-q",
            "-b",
            "--quitonbreak",
            "--norepl",
            "--nobanner",
            script_path,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass


def _gap_smallgroup_tables(
    order: int,
    gid: int,
    *,
    gap_cmd: str,
    timeout_s: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Ask GAP for (mul_table, inv) for SmallGroup(order, gid), using a plain-text,
    marker-delimited protocol (no JSON, no quote-escaping).
    """
    o = int(order)
    g = int(gid)

    gap_script = f"""
LoadPackage("smallgrp");;
o := {o};;
gid := {g};;
grp := SmallGroup(o, gid);;
els := Elements(grp);;

mul := List([1..o], i -> List([1..o], j -> Position(els, els[i]*els[j]) - 1));;
inv := List([1..o], i -> Position(els, Inverse(els[i])) - 1);;

Print("{_GROUP_TABLES_BEGIN}\\n");;
Print(o, "\\n");;
Print(gid, "\\n");;

for i in [1..o] do
  for j in [1..o] do
    Print(mul[i][j]);;
    if j < o then
      Print(",");;
    fi;;
  od;;
  Print("\\n");;
od;;

Print("INV\\n");;

for i in [1..o] do
  Print(inv[i]);;
  if i < o then
    Print(",");;
  fi;;
od;;
Print("\\n");;

Print("{_GROUP_TABLES_END}\\n");;
QUIT;;
"""
    stdout, stderr, rc = _run_gap_script(gap_cmd, gap_script, timeout_s=timeout_s)

    # GAP sometimes exits with rc=0 even if it printed a syntax error to stderr.
    if rc != 0:
        raise RuntimeError(
            "GAP exited with non-zero status while building SmallGroup tables.\n"
            f"cmd={gap_cmd!r} order={o} gid={g}\n"
            f"stdout_tail:\n{_tail(stdout)}\n\n"
            f"stderr_tail:\n{_tail(stderr)}"
        )

    try:
        payload = _extract_between_markers(stdout, begin=_GROUP_TABLES_BEGIN, end=_GROUP_TABLES_END)
    except Exception as e:
        raise RuntimeError(
            "Failed to locate group-table markers in GAP stdout.\n"
            f"cmd={gap_cmd!r} order={o} gid={g}\n"
            f"stdout_tail:\n{_tail(stdout)}\n\n"
            f"stderr_tail:\n{_tail(stderr)}"
        ) from e

    try:
        o2, gid2, mul, inv = _parse_tables_payload(payload)
    except Exception as e:
        raise RuntimeError(
            "Failed to parse group-table payload from GAP.\n"
            f"cmd={gap_cmd!r} order={o} gid={g}\n"
            f"payload_tail:\n{_tail(payload)}\n\n"
            f"stdout_tail:\n{_tail(stdout)}\n\n"
            f"stderr_tail:\n{_tail(stderr)}"
        ) from e

    if o2 != o or gid2 != g:
        raise RuntimeError(
            f"GAP returned tables for order={o2},gid={gid2} but expected order={o},gid={g}."
        )

    return mul, inv


@lru_cache(maxsize=None)
def _get_group_ops(order: int, gid: int, gap_cmd: str) -> GroupOps:
    mul, inv = _gap_smallgroup_tables(order, gid, gap_cmd=gap_cmd, timeout_s=60)
    norm_mul, norm_inv = _normalize_table(order, mul, inv)
    name = f"SmallGroup({int(order)},{int(gid)})"
    return GroupOps(order=int(order), gid=int(gid), mul_table=norm_mul, inv=norm_inv, name=name)


def ensure_group_ops(group: Any, *, gap_cmd: Optional[str] = None) -> GroupOps:
    """
    Convert `group` into a GroupOps instance.

    If `group` already has mul_table+inv tables, GAP will NOT be called.
    GAP is only used when we only have (order, gid).
    """
    if isinstance(group, GroupOps):
        return group

    gap_cmd_eff = gap_cmd or os.environ.get("QTANNER_GAP_CMD") or "gap"

    # 1) dict input
    if isinstance(group, dict):
        name = group.get("name")
        order = group.get("order")
        gid = group.get("gid")

        if gid is None and isinstance(name, str):
            o2, g2 = _parse_smallgroup_name(name)
            if g2 is not None:
                gid = g2
            if order is None and o2 is not None:
                order = o2

        mul_table = group.get("mul_table", group.get("mul"))
        inv = group.get("inv")

        if mul_table is not None and inv is not None and order is not None:
            o = _coerce_int(order, name="order")
            g = None if gid is None else _coerce_int(gid, name="gid")
            norm_mul, norm_inv = _normalize_table(o, mul_table, inv)
            return GroupOps(order=o, gid=g, mul_table=norm_mul, inv=norm_inv, name=name)

        if order is None or gid is None:
            raise TypeError(
                f"Dict group must provide (order,gid) or (order,mul_table,inv). Got: {group!r}"
            )
        return _get_group_ops(_coerce_int(order, name="order"), _coerce_int(gid, name="gid"), gap_cmd_eff)

    # 2) string input
    if isinstance(group, str):
        s = group.strip()
        m = re.fullmatch(r"(\d+)\s*,\s*(\d+)", s)
        if m:
            return _get_group_ops(int(m.group(1)), int(m.group(2)), gap_cmd_eff)
        o2, g2 = _parse_smallgroup_name(s)
        if o2 is not None and g2 is not None:
            return _get_group_ops(o2, g2, gap_cmd_eff)
        raise TypeError(f"Unrecognized group string {group!r}; expected 'o,gid' or 'SmallGroup(o,gid)'.")

    # 3) tuple/list (o, gid)
    if isinstance(group, (tuple, list)) and len(group) == 2:
        return _get_group_ops(_coerce_int(group[0], name="order"), _coerce_int(group[1], name="gid"), gap_cmd_eff)

    # 4) object with tables (mul_table / mul, inv)
    name = getattr(group, "name", None)
    order = getattr(group, "order", None)
    gid = getattr(group, "gid", None)

    if gid is None and isinstance(name, str):
        o2, g2 = _parse_smallgroup_name(name)
        if g2 is not None:
            gid = g2
        if order is None and o2 is not None:
            order = o2

    mul_table = getattr(group, "mul_table", None)
    if mul_table is None:
        mul_table = getattr(group, "mul", None)
    inv = getattr(group, "inv", None)

    if mul_table is not None and inv is not None and order is not None:
        o = _coerce_int(order, name="order")
        g = None if gid is None else _coerce_int(gid, name="gid")
        norm_mul, norm_inv = _normalize_table(o, mul_table, inv)
        return GroupOps(order=o, gid=g, mul_table=norm_mul, inv=norm_inv, name=name)

    # 5) final fallback: if we at least have order+gid, use GAP
    if order is not None and gid is not None:
        return _get_group_ops(_coerce_int(order, name="order"), _coerce_int(gid, name="gid"), gap_cmd_eff)

    raise TypeError(
        f"Could not interpret group={group!r}. Need a dict/object with mul_table+inv, or (order,gid)."
    )


def smallgroup(order: int, gid: int, *, gap_cmd: Optional[str] = None) -> GroupOps:
    """Convenience wrapper for SmallGroup(order,gid), backed by the GAP cache."""
    gap_cmd_eff = gap_cmd or os.environ.get("QTANNER_GAP_CMD") or "gap"
    return _get_group_ops(int(order), int(gid), gap_cmd_eff)
