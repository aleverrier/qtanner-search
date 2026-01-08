"""Finite group helpers with 0-based index conventions."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence


class FiniteGroup:
    """Small interface for finite groups with elements 0..|G|-1."""

    name: str
    order: int

    def mul(self, a: int, b: int) -> int:
        raise NotImplementedError

    def inv(self, a: int) -> int:
        raise NotImplementedError

    def id(self) -> int:
        return 0

    def elements(self) -> range:
        return range(self.order)

    def repr(self, a: int) -> str:
        return str(a)

    # Backwards-compatible aliases.
    def mul_of(self, a: int, b: int) -> int:
        return self.mul(a, b)

    def inv_of(self, a: int) -> int:
        return self.inv(a)

    @classmethod
    def cyclic(cls, n: int) -> "FiniteGroup":
        return CyclicGroup(n)


class CyclicGroup(FiniteGroup):
    """Cyclic group C_n with additive notation modulo n."""

    def __init__(self, n: int, *, name: Optional[str] = None) -> None:
        n_int = int(n)
        if n_int <= 0:
            raise ValueError(f"CyclicGroup order must be positive, got {n}.")
        self.order = n_int
        self.name = name or f"C{n_int}"

    def mul(self, a: int, b: int) -> int:
        return (int(a) + int(b)) % self.order

    def inv(self, a: int) -> int:
        return (-int(a)) % self.order

    def repr(self, a: int) -> str:
        return str(int(a))


class DirectProductGroup(FiniteGroup):
    """Direct product of two finite groups with element IDs packed as (a,b)."""

    def __init__(
        self,
        g1: FiniteGroup,
        g2: FiniteGroup,
        *,
        name: Optional[str] = None,
    ) -> None:
        self.g1 = g1
        self.g2 = g2
        self.order = g1.order * g2.order
        self.name = name or f"{g1.name}x{g2.name}"

    def _decode(self, x: int) -> tuple[int, int]:
        q, r = divmod(int(x), self.g2.order)
        return q, r

    def _encode(self, a: int, b: int) -> int:
        return int(a) * self.g2.order + int(b)

    def mul(self, a: int, b: int) -> int:
        a1, b1 = self._decode(a)
        a2, b2 = self._decode(b)
        return self._encode(self.g1.mul(a1, a2), self.g2.mul(b1, b2))

    def inv(self, a: int) -> int:
        a1, b1 = self._decode(a)
        return self._encode(self.g1.inv(a1), self.g2.inv(b1))

    def repr(self, a: int) -> str:
        a1, b1 = self._decode(a)
        return f"({self.g1.repr(a1)},{self.g2.repr(b1)})"


class TableGroup(FiniteGroup):
    """Finite group defined by 0-based multiplication and inverse tables."""

    def __init__(
        self,
        name: str,
        mul_table: Sequence[Sequence[int]],
        inv_table: Sequence[int],
        *,
        element_repr: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = str(name)
        self.order = len(mul_table)
        norm_mul, norm_inv = _normalize_table(self.order, mul_table, inv_table)
        self.mul_table = norm_mul
        self.inv_table = norm_inv
        if element_repr is None:
            self._repr_table = None
        else:
            if len(element_repr) != self.order:
                raise ValueError("element_repr length does not match group order.")
            self._repr_table = [str(x) for x in element_repr]

    def mul(self, a: int, b: int) -> int:
        return self.mul_table[int(a)][int(b)]

    def inv(self, a: int) -> int:
        return self.inv_table[int(a)]

    def repr(self, a: int) -> str:
        idx = int(a)
        if self._repr_table is None:
            return str(idx)
        return self._repr_table[idx]


def _normalize_table(
    order: int,
    mul_table: Sequence[Sequence[int]],
    inv_table: Sequence[int],
) -> tuple[List[List[int]], List[int]]:
    n = int(order)
    if n <= 0:
        raise ValueError(f"Group order must be positive, got {n}.")
    if len(mul_table) != n:
        raise ValueError(f"mul_table has {len(mul_table)} rows but order={n}.")
    norm: List[List[int]] = []
    for i, row in enumerate(mul_table):
        if len(row) != n:
            raise ValueError(f"mul_table row {i} length {len(row)} but order={n}.")
        norm_row = [int(x) for x in row]
        for x in norm_row:
            if x < 0 or x >= n:
                raise ValueError(f"mul_table entry {x} out of range [0,{n-1}].")
        norm.append(norm_row)
    inv = [int(x) for x in inv_table]
    if len(inv) != n:
        raise ValueError(f"inv_table length {len(inv)} but order={n}.")
    for x in inv:
        if x < 0 or x >= n:
            raise ValueError(f"inv_table entry {x} out of range [0,{n-1}].")
    return norm, inv


_SMALLGROUP_RE = re.compile(r"SmallGroup\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CYCLIC_RE = re.compile(r"([CZ])(\d+)$", re.IGNORECASE)
_DIRECT_RE = re.compile(r"([CZ])(\d+)x([CZ])(\d+)$", re.IGNORECASE)


def canonical_group_spec(spec: str) -> str:
    if spec is None:
        raise ValueError("Group spec cannot be None.")
    s = re.sub(r"\s+", "", spec)
    if not s:
        raise ValueError("Group spec cannot be empty.")
    if s.lower() == "v4":
        return "C2xC2"
    m = _SMALLGROUP_RE.fullmatch(s)
    if m:
        return f"SmallGroup({int(m.group(1))},{int(m.group(2))})"
    m = _DIRECT_RE.fullmatch(s)
    if m:
        return f"C{int(m.group(2))}xC{int(m.group(4))}"
    m = _CYCLIC_RE.fullmatch(s)
    if m:
        return f"C{int(m.group(2))}"
    raise ValueError(f"Unrecognized group spec {spec!r}.")


def group_from_spec(
    spec: str,
    *,
    gap_cmd: str = "gap",
    cache_dir: Optional[Path] = None,
) -> FiniteGroup:
    spec_norm = canonical_group_spec(spec)
    if spec_norm.startswith("SmallGroup"):
        return _load_smallgroup(spec_norm, gap_cmd=gap_cmd, cache_dir=cache_dir)
    if "x" in spec_norm:
        left, right = spec_norm.split("x", 1)
        m = int(left[1:])
        n = int(right[1:])
        return DirectProductGroup(CyclicGroup(m), CyclicGroup(n), name=spec_norm)
    return CyclicGroup(int(spec_norm[1:]), name=spec_norm)


def _load_smallgroup(
    spec_norm: str,
    *,
    gap_cmd: str,
    cache_dir: Optional[Path],
) -> FiniteGroup:
    m = _SMALLGROUP_RE.fullmatch(spec_norm)
    if not m:
        raise ValueError(f"Invalid SmallGroup spec {spec_norm!r}.")
    order = int(m.group(1))
    gid = int(m.group(2))
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{spec_norm}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            mul = payload.get("mul_table")
            inv = payload.get("inv_table", payload.get("inv"))
            if isinstance(mul, list) and isinstance(inv, list):
                return TableGroup(spec_norm, mul, inv)
    mul, inv = _gap_smallgroup_tables(order, gid, gap_cmd=gap_cmd)
    if cache_path is not None:
        payload = {
            "format_version": 1,
            "spec": spec_norm,
            "order": order,
            "gid": gid,
            "mul_table": mul,
            "inv_table": inv,
        }
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return TableGroup(spec_norm, mul, inv)


def _gap_smallgroup_tables(
    order: int,
    gid: int,
    *,
    gap_cmd: str,
    timeout_s: int = 60,
) -> tuple[List[List[int]], List[int]]:
    script = "\n".join(
        [
            'if LoadPackage("smallgrp") = fail then',
            '  Print("QTANNER_GAP_ERROR smallgrp_missing\\n");',
            "  QuitGap(2);",
            "fi;",
            f"G := SmallGroup({int(order)},{int(gid)});;",
            "elts := ShallowCopy(Elements(G));;",
            "id := One(G);;",
            "pos := Position(elts, id);;",
            "if pos <> 1 then",
            "  tmp := elts[1];; elts[1] := elts[pos];; elts[pos] := tmp;;",
            "fi;",
            "n := Length(elts);;",
            'Print("MUL_BEGIN\\n");',
            "for i in [1..n] do",
            "  for j in [1..n] do",
            "    Print(Position(elts, elts[i]*elts[j]) - 1);",
            "    if j < n then",
            '      Print(",");',
            "    fi;",
            "  od;",
            '  Print("\\n");',
            "od;",
            'Print("MUL_END\\n");',
            'Print("INV_BEGIN\\n");',
            "for i in [1..n] do",
            "  Print(Position(elts, Inverse(elts[i])) - 1);",
            "  if i < n then",
            '    Print(",");',
            "  fi;",
            "od;",
            'Print("\\nINV_END\\n");',
            "QuitGap(0);",
        ]
    )
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".g", delete=False) as tmp:
            tmp.write(script)
            script_path = tmp.name
        result = subprocess.run(
            [gap_cmd, "-q", "-b", "--quitonbreak", script_path],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
    except FileNotFoundError as exc:
        raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise RuntimeError(
            f"GAP timed out after {timeout_s} seconds while loading SmallGroup({order},{gid})."
        ) from exc
    finally:
        if "script_path" in locals():
            try:
                os.remove(script_path)
            except OSError:
                pass
    if returncode != 0:
        raise RuntimeError(
            "GAP exited with non-zero status while building SmallGroup tables.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        )
    try:
        mul, inv = _parse_gap_marked_output(stdout, order)
    except Exception as exc:
        raise RuntimeError(
            "Failed to parse GAP output for SmallGroup tables.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        ) from exc
    return mul, inv


def _tail(text: str, n: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[-n:]


def _parse_gap_marked_output(output: str, n: int) -> tuple[List[List[int]], List[int]]:
    lines = output.splitlines()
    mul_begin = None
    mul_end = None
    inv_begin = None
    inv_end = None
    for idx, line in enumerate(lines):
        marker = line.strip()
        if marker == "MUL_BEGIN":
            mul_begin = idx
        elif marker == "MUL_END":
            mul_end = idx
        elif marker == "INV_BEGIN":
            inv_begin = idx
        elif marker == "INV_END":
            inv_end = idx
    if mul_begin is None or mul_end is None or mul_begin >= mul_end:
        raise RuntimeError("GAP output missing MUL_BEGIN/MUL_END markers.")
    if inv_begin is None or inv_end is None or inv_begin >= inv_end:
        raise RuntimeError("GAP output missing INV_BEGIN/INV_END markers.")

    mul_lines: List[str] = []
    for line in lines[mul_begin + 1 : mul_end]:
        stripped = line.strip()
        if stripped:
            mul_lines.append(stripped)
    if len(mul_lines) != n:
        raise RuntimeError(f"GAP output has {len(mul_lines)} mul rows; expected {n}.")

    mul_table: List[List[int]] = []
    for row_idx, line in enumerate(mul_lines):
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) != n:
            raise RuntimeError(
                f"GAP output row {row_idx} has {len(parts)} entries; expected {n}."
            )
        mul_table.append([int(part) for part in parts])

    inv_tokens: List[str] = []
    for line in lines[inv_begin + 1 : inv_end]:
        inv_tokens.extend([p for p in line.strip().split(",") if p != ""])
    if len(inv_tokens) != n:
        raise RuntimeError(f"GAP output has {len(inv_tokens)} inverses; expected {n}.")
    inv = [int(token) for token in inv_tokens]
    return mul_table, inv


__all__ = [
    "FiniteGroup",
    "CyclicGroup",
    "DirectProductGroup",
    "TableGroup",
    "canonical_group_spec",
    "group_from_spec",
]
