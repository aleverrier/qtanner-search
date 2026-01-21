"""Finite group helpers with 0-based index conventions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .gap_backend import gap_group_data, gap_list_small_groups


class FiniteGroup:
    """Small interface for finite groups with elements 0..|G|-1."""

    name: str
    order: int
    _is_abelian: Optional[bool] = None

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

    def _compute_is_abelian(self) -> bool:
        for a in self.elements():
            for b in self.elements():
                if self.mul(a, b) != self.mul(b, a):
                    return False
        return True

    @property
    def is_abelian(self) -> bool:
        cached = getattr(self, "_is_abelian", None)
        if cached is None:
            cached = self._compute_is_abelian()
            self._is_abelian = cached
        return bool(cached)

    def automorphisms(
        self,
        *,
        gap_cmd: str = "gap",
        cache_dir: Optional[Path] = None,
    ) -> List[List[int]]:
        """Return automorphisms as permutations of element IDs."""
        if hasattr(self, "_automorphisms") and self._automorphisms is not None:
            return list(self._automorphisms)
        perms = _load_automorphisms(self, gap_cmd=gap_cmd, cache_dir=cache_dir)
        self._automorphisms = perms
        return list(perms)

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
        self._is_abelian = True

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
        self._is_abelian = g1.is_abelian and g2.is_abelian

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
        is_abelian: Optional[bool] = None,
    ) -> None:
        self.name = str(name)
        self.order = len(mul_table)
        norm_mul, norm_inv = _normalize_table(self.order, mul_table, inv_table)
        self.mul_table = norm_mul
        self.inv_table = norm_inv
        if is_abelian is not None:
            self._is_abelian = bool(is_abelian)
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


@dataclass(frozen=True)
class SmallGroupRecord:
    """Summary record for SmallGroup identifiers."""

    order: int
    gid: int
    spec: str
    structure_description: str


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


_SMALLGROUP_RE = re.compile(r"(?:SmallGroup|SG)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_CYCLIC_RE = re.compile(r"([CZ])(\d+)$", re.IGNORECASE)
_AUT_BEGIN = "AUT_BEGIN"
_AUT_END = "AUT_END"
_AUT_CACHE_VERSION = 1


def canonical_group_spec(spec: str) -> str:
    if spec is None:
        raise ValueError("Group spec cannot be None.")
    s = re.sub(r"\s+", "", spec).replace("X", "x")
    if not s:
        raise ValueError("Group spec cannot be empty.")
    if s.lower() == "v4":
        return "C2xC2"
    m = _SMALLGROUP_RE.fullmatch(s)
    if m:
        return f"SmallGroup({int(m.group(1))},{int(m.group(2))})"
    if "x" in s:
        parts = s.split("x")
        norm_parts = []
        for part in parts:
            m = _CYCLIC_RE.fullmatch(part)
            if not m:
                raise ValueError(f"Unrecognized group spec {spec!r}.")
            norm_parts.append(f"C{int(m.group(2))}")
        return "x".join(norm_parts)
    m = _CYCLIC_RE.fullmatch(s)
    if m:
        return f"C{int(m.group(2))}"
    raise ValueError(f"Unrecognized group spec {spec!r}.")


def _cyclic_factors_from_spec(spec_norm: str) -> Optional[List[int]]:
    """Return cyclic factor orders for specs like C2xC2xC2, else None."""
    if "x" in spec_norm:
        parts = spec_norm.split("x")
    else:
        parts = [spec_norm]
    orders: List[int] = []
    for part in parts:
        m = _CYCLIC_RE.fullmatch(part)
        if not m:
            return None
        orders.append(int(m.group(2)))
    return orders


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
        parts = spec_norm.split("x")
        groups = [CyclicGroup(int(part[1:]), name=part) for part in parts]
        group = groups[0]
        for idx, rhs in enumerate(groups[1:], start=1):
            name = spec_norm if idx == len(groups) - 1 else None
            group = DirectProductGroup(group, rhs, name=name)
        return group
    return CyclicGroup(int(spec_norm[1:]), name=spec_norm)


def list_small_groups(
    max_order: int,
    *,
    gap_cmd: str = "gap",
) -> List[SmallGroupRecord]:
    """List GAP SmallGroup identifiers up to a maximum order."""
    records: List[SmallGroupRecord] = []
    for order, gid, desc in gap_list_small_groups(max_order, gap_cmd=gap_cmd):
        records.append(
            SmallGroupRecord(
                order=order,
                gid=gid,
                spec=f"SG({order},{gid})",
                structure_description=desc,
            )
        )
    return records


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
                is_abelian = payload.get("is_abelian")
                return TableGroup(
                    spec_norm,
                    mul,
                    inv,
                    is_abelian=is_abelian if isinstance(is_abelian, bool) else None,
                )
    data = gap_group_data(spec_norm, gap_cmd=gap_cmd)
    mul = data.mul_table
    inv = data.inv_table
    if cache_path is not None:
        payload = {
            "format_version": 1,
            "spec": spec_norm,
            "order": order,
            "gid": gid,
            "mul_table": mul,
            "inv_table": inv,
            "is_abelian": bool(data.is_abelian),
        }
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return TableGroup(spec_norm, mul, inv, is_abelian=bool(data.is_abelian))


def _tail(text: str, n: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[-n:]


def _parse_gap_aut_output(output: str, n: int) -> List[List[int]]:
    lines = output.splitlines()
    begin = None
    end = None
    for idx, line in enumerate(lines):
        marker = line.strip()
        if marker == _AUT_BEGIN:
            begin = idx
        elif marker == _AUT_END:
            end = idx
    if begin is None or end is None or begin >= end:
        raise RuntimeError("GAP output missing AUT_BEGIN/AUT_END markers.")
    perm_lines = [line.strip() for line in lines[begin + 1 : end] if line.strip()]
    perms: List[List[int]] = []
    for line in perm_lines:
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) != n:
            raise RuntimeError(
                f"GAP output permutation has {len(parts)} entries; expected {n}."
            )
        perm = [int(part) for part in parts]
        perms.append(perm)
    if not perms:
        perms = [list(range(n))]
    return perms


def _normalize_aut_perms(perms: Sequence[Sequence[int]], n: int) -> List[List[int]]:
    seen: Dict[Tuple[int, ...], None] = {}
    for perm in perms:
        if len(perm) != n:
            raise ValueError(f"Automorphism length {len(perm)} but order={n}.")
        norm = tuple(int(x) for x in perm)
        if any(x < 0 or x >= n for x in norm):
            raise ValueError("Automorphism entries out of range.")
        if len(set(norm)) != n:
            raise ValueError("Automorphism is not a permutation.")
        seen[norm] = None
    if tuple(range(n)) not in seen:
        seen[tuple(range(n))] = None
    return [list(perm) for perm in sorted(seen)]


def _aut_cache_path(cache_dir: Optional[Path], spec_norm: str) -> Optional[Path]:
    if cache_dir is None:
        cache_dir = Path(".cache") / "qtanner" / "aut"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{spec_norm}.json"


def _gap_missing_error(gap_cmd: str, context: str) -> RuntimeError:
    return RuntimeError(
        f"GAP is required for {context}, but '{gap_cmd}' was not found on PATH. "
        "Install it with `brew install gap` and re-run with `--gap-cmd gap` "
        "or pass the full path to the GAP binary."
    )


def _gap_cyclic_product_expr(orders: Sequence[int]) -> str:
    if len(orders) == 1:
        return f"CyclicGroup({orders[0]})"
    groups = ",".join(f"CyclicGroup({order})" for order in orders)
    return f"DirectProduct({groups})"


def _gap_cyclic_product_element_lines(orders: Sequence[int]) -> List[str]:
    """Build GAP loops that match the Python direct product element ordering."""
    lines: List[str] = ["elts := [];;"]
    for idx, order in enumerate(orders, start=1):
        indent = "  " * (idx - 1)
        lines.append(f"{indent}for a{idx} in [0..{order - 1}] do")
    product = " * ".join(f"gens[{idx}]^a{idx}" for idx in range(1, len(orders) + 1))
    lines.append(f"{'  ' * len(orders)}Add(elts, {product});;")
    for idx in range(len(orders) - 1, -1, -1):
        lines.append(f"{'  ' * idx}od;")
    return lines


def _gap_automorphisms_for_spec(
    spec_norm: str,
    *,
    gap_cmd: str,
    timeout_s: int = 60,
) -> List[List[int]]:
    m = _SMALLGROUP_RE.fullmatch(spec_norm)
    if m:
        data = gap_group_data(spec_norm, gap_cmd=gap_cmd, timeout_s=timeout_s)
        return list(data.automorphisms)
    orders = _cyclic_factors_from_spec(spec_norm)
    if not orders:
        raise ValueError(f"Unsupported group spec for automorphisms: {spec_norm!r}")
    order = 1
    for value in orders:
        order *= value
    gap_expr = _gap_cyclic_product_expr(orders)
    gens_list = ", ".join(f"G.{idx}" for idx in range(1, len(orders) + 1))
    script_lines = [
        f"G := {gap_expr};;",
        f"gens := [{gens_list}];;",
        *_gap_cyclic_product_element_lines(orders),
        "auts := Elements(AutomorphismGroup(G));;",
        "n := Length(elts);;",
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
    script = "\n".join(script_lines) + "\n"
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        result = subprocess.run(
            [gap_cmd, "-q", "--quitonbreak"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
    except FileNotFoundError as exc:
        raise _gap_missing_error(gap_cmd, "group automorphisms") from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise RuntimeError(
            f"GAP timed out after {timeout_s} seconds while building automorphisms for {spec_norm}."
        ) from exc
    if returncode != 0:
        raise RuntimeError(
            "GAP exited with non-zero status while building automorphisms.\n"
            f"stdout_tail:\n{_tail(stdout)}\n\nstderr_tail:\n{_tail(stderr)}"
        )
    perms = _parse_gap_aut_output(stdout, order)
    return perms


def _load_automorphisms(
    group: FiniteGroup,
    *,
    gap_cmd: str,
    cache_dir: Optional[Path],
) -> List[List[int]]:
    spec_norm = None
    try:
        spec_norm = canonical_group_spec(group.name)
    except ValueError:
        spec_norm = group.name
    cache_path = _aut_cache_path(cache_dir, spec_norm)
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        perms = payload.get("perms") or payload.get("automorphisms")
        if (
            payload.get("format_version") == _AUT_CACHE_VERSION
            and int(payload.get("order", 0)) == group.order
            and isinstance(perms, list)
        ):
            return _normalize_aut_perms(perms, group.order)
    perms = _gap_automorphisms_for_spec(spec_norm, gap_cmd=gap_cmd)
    perms = _normalize_aut_perms(perms, group.order)
    if cache_path is not None:
        payload = {
            "format_version": _AUT_CACHE_VERSION,
            "spec": spec_norm,
            "order": group.order,
            "perms": perms,
        }
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return perms


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="qtanner group utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-small", help="List GAP SmallGroup identifiers."
    )
    list_parser.add_argument(
        "--max-order",
        type=int,
        default=20,
        help="Maximum group order to list.",
    )
    list_parser.add_argument(
        "--gap-cmd",
        type=str,
        default="gap",
        help="GAP command.",
    )

    args = parser.parse_args(argv)
    if args.command == "list-small":
        for record in list_small_groups(args.max_order, gap_cmd=args.gap_cmd):
            print(f"{record.order}  {record.spec}  {record.structure_description}")
        return 0
    return 1


__all__ = [
    "FiniteGroup",
    "CyclicGroup",
    "DirectProductGroup",
    "TableGroup",
    "SmallGroupRecord",
    "canonical_group_spec",
    "group_from_spec",
    "list_small_groups",
]


if __name__ == "__main__":
    raise SystemExit(_main())
