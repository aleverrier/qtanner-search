"""GAP-backed group data helpers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

_SMALLGROUP_RE = re.compile(r"(?:SmallGroup|SG)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_CYCLIC_RE = re.compile(r"([CZ])(\d+)$", re.IGNORECASE)

_ORDER_PREFIX = "ORDER "
_ISABELIAN_PREFIX = "ISABELIAN "
_ELTS_BEGIN = "ELTS_BEGIN"
_ELTS_END = "ELTS_END"
_MUL_BEGIN = "MUL_BEGIN"
_MUL_END = "MUL_END"
_INV_BEGIN = "INV_BEGIN"
_INV_END = "INV_END"
_AUT_BEGIN = "AUT_BEGIN"
_AUT_END = "AUT_END"


@dataclass(frozen=True)
class GapGroupData:
    """Group data parsed from GAP output (treat as read-only)."""

    spec: str
    order: int
    elements: List[str]
    mul_table: List[List[int]]
    inv_table: List[int]
    automorphisms: List[List[int]]
    is_abelian: bool


def _normalize_spec(spec: str) -> str:
    if spec is None:
        raise ValueError("Group spec cannot be None.")
    s = re.sub(r"\s+", "", str(spec))
    if not s:
        raise ValueError("Group spec cannot be empty.")
    if s.lower() == "v4":
        return "C2xC2"
    return s.replace("X", "x")


def _spec_to_gap_expr(spec: str) -> Tuple[str, bool]:
    s = _normalize_spec(spec)
    m = _SMALLGROUP_RE.fullmatch(s)
    if m:
        order = int(m.group(1))
        gid = int(m.group(2))
        return f"SmallGroup({order},{gid})", True
    if "x" in s:
        parts = s.split("x")
        orders: List[int] = []
        for part in parts:
            m = _CYCLIC_RE.fullmatch(part)
            if not m:
                raise ValueError(f"Unsupported group spec {spec!r}.")
            orders.append(int(m.group(2)))
        if len(orders) == 1:
            return f"CyclicGroup({orders[0]})", False
        groups = ",".join(f"CyclicGroup({order})" for order in orders)
        return f"DirectProduct({groups})", False
    m = _CYCLIC_RE.fullmatch(s)
    if m:
        return f"CyclicGroup({int(m.group(2))})", False
    raise ValueError(f"Unsupported group spec {spec!r}.")


def _gap_missing_error(gap_cmd: str, context: str) -> RuntimeError:
    return RuntimeError(
        f"GAP is required for {context}, but '{gap_cmd}' was not found on PATH. "
        "Install it with `brew install gap` and re-run with `--gap-cmd gap` "
        "or pass the full path to the GAP binary."
    )


def _tail(text: str, n: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[-n:]


def _run_gap_script(script: str, *, gap_cmd: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [gap_cmd, "-q", "--quitonbreak"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _gap_missing_error(gap_cmd, "GAP-backed group data") from exc


def _build_group_script(gap_expr: str, *, needs_smallgrp: bool) -> str:
    lines = []
    if needs_smallgrp:
        lines.extend(
            [
                'if LoadPackage("smallgrp") = fail then',
                '  Print("QTANNER_GAP_ERROR smallgrp_missing\\n");',
                "  QuitGap(2);",
                "fi;",
            ]
        )
    lines.extend(
        [
            f"G := {gap_expr};;",
            "elts := ShallowCopy(Elements(G));;",
            "id := One(G);;",
            "pos := Position(elts, id);;",
            "if pos <> 1 then",
            "  tmp := elts[1];; elts[1] := elts[pos];; elts[pos] := tmp;;",
            "fi;",
            "n := Length(elts);;",
            f'Print("{_ORDER_PREFIX}", n, "\\n");',
            f'Print("{_ISABELIAN_PREFIX}", IsAbelian(G), "\\n");',
            f'Print("{_ELTS_BEGIN}\\n");',
            "for i in [1..n] do",
            "  Print(String(elts[i]), \"\\n\");",
            "od;",
            f'Print("{_ELTS_END}\\n");',
            f'Print("{_MUL_BEGIN}\\n");',
            "for i in [1..n] do",
            "  for j in [1..n] do",
            "    Print(Position(elts, elts[i]*elts[j]) - 1);",
            "    if j < n then",
            '      Print(",");',
            "    fi;",
            "  od;",
            '  Print("\\n");',
            "od;",
            f'Print("{_MUL_END}\\n");',
            f'Print("{_INV_BEGIN}\\n");',
            "for i in [1..n] do",
            "  Print(Position(elts, Inverse(elts[i])) - 1);",
            "  if i < n then",
            '    Print(",");',
            "  fi;",
            "od;",
            f'Print("\\n{_INV_END}\\n");',
            "auts := Elements(AutomorphismGroup(G));;",
            f'Print("{_AUT_BEGIN}\\n");',
            "for phi in auts do",
            "  for i in [1..n] do",
            "    Print(Position(elts, Image(phi, elts[i])) - 1);",
            "    if i < n then",
            '      Print(",");',
            "    fi;",
            "  od;",
            '  Print("\\n");',
            "od;",
            f'Print("{_AUT_END}\\n");',
            "QuitGap(0);",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_gap_group_output(output: str, spec: str) -> GapGroupData:
    lines = output.splitlines()
    order = None
    is_abelian: Optional[bool] = None
    for line in lines:
        if line.startswith(_ORDER_PREFIX):
            try:
                order = int(line[len(_ORDER_PREFIX) :].strip())
            except ValueError as exc:
                raise RuntimeError(f"Invalid GAP order line: {line!r}") from exc
            break
    for line in lines:
        if line.startswith(_ISABELIAN_PREFIX):
            token = line[len(_ISABELIAN_PREFIX) :].strip().lower()
            if token in ("true", "false"):
                is_abelian = token == "true"
            elif token in ("1", "0"):
                is_abelian = token == "1"
            else:
                raise RuntimeError(f"Invalid GAP IsAbelian line: {line!r}")
            break
    if order is None:
        raise RuntimeError("GAP output missing ORDER line.")

    def _find_block(begin: str, end: str) -> List[str]:
        begin_idx = None
        end_idx = None
        for idx, line in enumerate(lines):
            marker = line.strip()
            if marker == begin:
                begin_idx = idx
            elif marker == end:
                end_idx = idx
        if begin_idx is None or end_idx is None or begin_idx >= end_idx:
            raise RuntimeError(f"GAP output missing {begin}/{end} markers.")
        return lines[begin_idx + 1 : end_idx]

    element_lines = _find_block(_ELTS_BEGIN, _ELTS_END)
    elements = [line.rstrip("\n") for line in element_lines if line.strip() != ""]
    if len(elements) != order:
        raise RuntimeError(
            f"GAP output has {len(elements)} elements; expected {order}."
        )

    mul_lines = [line.strip() for line in _find_block(_MUL_BEGIN, _MUL_END) if line.strip()]
    if len(mul_lines) != order:
        raise RuntimeError(f"GAP output has {len(mul_lines)} mul rows; expected {order}.")
    mul_table: List[List[int]] = []
    for row_idx, line in enumerate(mul_lines):
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) != order:
            raise RuntimeError(
                f"GAP output row {row_idx} has {len(parts)} entries; expected {order}."
            )
        mul_table.append([int(part) for part in parts])

    inv_tokens: List[str] = []
    for line in _find_block(_INV_BEGIN, _INV_END):
        inv_tokens.extend([p for p in line.strip().split(",") if p != ""])
    if len(inv_tokens) != order:
        raise RuntimeError(
            f"GAP output has {len(inv_tokens)} inverses; expected {order}."
        )
    inv_table = [int(token) for token in inv_tokens]

    aut_lines = [line.strip() for line in _find_block(_AUT_BEGIN, _AUT_END) if line.strip()]
    automorphisms: List[List[int]] = []
    for line in aut_lines:
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) != order:
            raise RuntimeError(
                f"GAP output permutation has {len(parts)} entries; expected {order}."
            )
        automorphisms.append([int(part) for part in parts])
    if not automorphisms:
        automorphisms = [list(range(order))]

    if is_abelian is None:
        is_abelian = all(
            mul_table[i][j] == mul_table[j][i]
            for i in range(order)
            for j in range(order)
        )

    return GapGroupData(
        spec=spec,
        order=order,
        elements=elements,
        mul_table=mul_table,
        inv_table=inv_table,
        automorphisms=automorphisms,
        is_abelian=bool(is_abelian),
    )


@lru_cache(maxsize=None)
def gap_group_data(
    spec: str,
    *,
    gap_cmd: str = "gap",
    timeout_s: int = 60,
) -> GapGroupData:
    """Fetch order/Elements/multiplication table/automorphisms for a group spec."""
    gap_expr, needs_smallgrp = _spec_to_gap_expr(spec)
    script = _build_group_script(gap_expr, needs_smallgrp=needs_smallgrp)
    result = _run_gap_script(script, gap_cmd=gap_cmd, timeout_s=timeout_s)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        raise RuntimeError(
            "GAP exited with non-zero status while building group data.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        )
    try:
        return _parse_gap_group_output(stdout, spec)
    except Exception as exc:
        raise RuntimeError(
            "Failed to parse GAP output for group data.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        ) from exc


@lru_cache(maxsize=None)
def gap_list_small_groups(
    max_order: int,
    *,
    gap_cmd: str = "gap",
    timeout_s: int = 60,
) -> Tuple[Tuple[int, int, str], ...]:
    """Return (order, id, structure_description) tuples for SmallGroup <= max_order."""
    nmax = int(max_order)
    if nmax <= 0:
        raise ValueError("max_order must be positive.")
    script_lines = [
        'if LoadPackage("smallgrp") = fail then',
        '  Print("QTANNER_GAP_ERROR smallgrp_missing\\n");',
        "  QuitGap(2);",
        "fi;",
        f"for n in [1..{nmax}] do",
        "  count := NrSmallGroups(n);",
        "  for i in [1..count] do",
        "    desc := StructureDescription(SmallGroup(n, i));",
        '    Print(n, "|", i, "|", desc, "\\n");',
        "  od;",
        "od;",
        "QuitGap(0);",
    ]
    script = "\n".join(script_lines) + "\n"
    result = _run_gap_script(script, gap_cmd=gap_cmd, timeout_s=timeout_s)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        raise RuntimeError(
            "GAP exited with non-zero status while listing small groups.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        )
    records: List[Tuple[int, int, str]] = []
    for line in stdout.splitlines():
        if "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        order_s, gid_s, desc = parts
        try:
            order = int(order_s.strip())
            gid = int(gid_s.strip())
        except ValueError:
            continue
        records.append((order, gid, desc.strip()))
    if not records:
        raise RuntimeError("GAP did not return any SmallGroup records.")
    return tuple(records)


__all__ = ["GapGroupData", "gap_group_data", "gap_list_small_groups"]
