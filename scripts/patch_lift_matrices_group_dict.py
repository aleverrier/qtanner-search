#!/usr/bin/env python3
"""
Patch src/qtanner/lift_matrices.py so that build_hx_hz accepts `group` passed as a dict
(e.g. {"order": 4, "gid": 2}) by constructing a minimal group-ops object that provides:

  - order, gid
  - inv_of(a)
  - mul_of(a,b)
  - (helpers: perm_of, left_perm_of, right_perm_of)

We build multiplication/inverse tables using GAP's SmallGroup(order,gid) once per group,
cached via functools.lru_cache.

Usage:
  # restore clean version
  ./scripts/py scripts/patch_lift_matrices_group_dict.py --git-restore

  # apply patch
  ./scripts/py scripts/patch_lift_matrices_group_dict.py
"""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


HELPER_BLOCK = r'''# BEGIN PATCH: smallgroup ops fallback for dict-like group
import ast as _ast
import functools as _functools
import subprocess as _subprocess
from typing import Any as _Any, List as _List, Tuple as _Tuple


class _SmallGroupOps:
    __slots__ = ("order", "gid", "_inv", "_mul")

    def __init__(self, order: int, gid: int, inv: _List[int], mul: _List[_List[int]]) -> None:
        self.order = int(order)
        self.gid = int(gid)
        self._inv = inv
        self._mul = mul

    def inv_of(self, a: int) -> int:
        return self._inv[int(a)]

    def mul_of(self, a: int, b: int) -> int:
        return self._mul[int(a)][int(b)]

    def perm_of(self, a: int) -> _List[int]:
        # left multiplication permutation: b -> a*b
        return self._mul[int(a)]

    def left_perm_of(self, a: int) -> _List[int]:
        return self.perm_of(a)

    def right_perm_of(self, a: int) -> _List[int]:
        aa = int(a)
        return [row[aa] for row in self._mul]


@_functools.lru_cache(maxsize=None)
def _smallgroup_tables(order: int, gid: int, gap_cmd: str = "gap") -> _Tuple[_List[int], _List[_List[int]]]:
    """
    Return (inv, mul) tables for GAP SmallGroup(order,gid) using the element ordering of Elements(G).

    inv[a]      = index of inverse of element a
    mul[a][b]   = index of product a*b
    """
    order_i = int(order)
    gid_i = int(gid)

    script = f"""
G := SmallGroup({order_i}, {gid_i});
elts := Elements(G);
inv := List(elts, x -> Position(elts, x^-1) - 1);
mul := List(elts, a -> List(elts, b -> Position(elts, a*b) - 1));
Print([inv, mul]);
QuitGap(0);
"""
    proc = _subprocess.run([gap_cmd, "-q"], input=script, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("GAP failed while building group tables. stderr tail:\n" + (proc.stderr or "")[-500:])

    out = (proc.stdout or "").strip()
    try:
        inv, mul = _ast.literal_eval(out)
    except Exception as e:
        raise RuntimeError("Could not parse GAP output for group tables. stdout tail:\n" + out[-500:]) from e

    if not isinstance(inv, list) or not isinstance(mul, list):
        raise RuntimeError("Unexpected GAP output type for group tables.")

    return inv, mul


@_functools.lru_cache(maxsize=None)
def _smallgroup_ops(order: int, gid: int, gap_cmd: str = "gap") -> _SmallGroupOps:
    inv, mul = _smallgroup_tables(int(order), int(gid), gap_cmd=gap_cmd)
    return _SmallGroupOps(int(order), int(gid), inv, mul)


def _ensure_group_ops(group: _Any, gap_cmd: str = "gap") -> _Any:
    """
    If `group` already provides inv_of(), return it.
    If it's dict-like / namespace-like with (order,gid), return a cached _SmallGroupOps.
    """
    if hasattr(group, "inv_of") and callable(getattr(group, "inv_of")):
        return group

    order = None
    gid = None
    if isinstance(group, dict):
        order = group.get("order")
        gid = group.get("gid")
    else:
        order = getattr(group, "order", None)
        gid = getattr(group, "gid", None)

    if order is None or gid is None:
        raise TypeError(f"Unsupported group object (missing inv_of and order/gid): {type(group)}")

    return _smallgroup_ops(int(order), int(gid), gap_cmd=gap_cmd)
# END PATCH: smallgroup ops fallback for dict-like group
'''

ENSURE_LINE_MARK = "group = _ensure_group_ops(group)  # PATCH: ensure_group_ops\n"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def strip_patch_blocks(lines: List[str]) -> List[str]:
    """Remove known patch blocks so the patch is re-applicable."""
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# BEGIN PATCH: accept group passed as dict"):
            i += 1
            while i < len(lines) and not lines[i].startswith("# END PATCH"):
                i += 1
            if i < len(lines):
                i += 1
            continue

        if line.startswith("# BEGIN PATCH: smallgroup ops fallback for dict-like group"):
            i += 1
            while i < len(lines) and not lines[i].startswith("# END PATCH: smallgroup ops fallback for dict-like group"):
                i += 1
            if i < len(lines):
                i += 1
            continue

        if "PATCH: ensure_group_ops" in line:
            i += 1
            continue

        out.append(line)
        i += 1
    return out


def find_def_build_hx_hz(lines: List[str]) -> int:
    for idx, line in enumerate(lines):
        if line.startswith("def build_hx_hz("):
            return idx
    raise RuntimeError("Could not find 'def build_hx_hz(' in lift_matrices.py")


def insert_helper_block(lines: List[str], def_idx: int) -> List[str]:
    helper_lines = [l + "\n" if not l.endswith("\n") else l for l in HELPER_BLOCK.splitlines()]
    if helper_lines and helper_lines[-1] != "\n":
        helper_lines.append("\n")
    return lines[:def_idx] + helper_lines + ["\n"] + lines[def_idx:]


def _is_docstring_line(s: str) -> bool:
    st = s.lstrip()
    return st.startswith('"""') or st.startswith("'''")


def insert_ensure_line(lines: List[str], def_idx: int) -> List[str]:
    """
    Insert `group = _ensure_group_ops(group)` as the first executable statement inside build_hx_hz,
    but after any docstring.
    """
    i = def_idx + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        raise RuntimeError("build_hx_hz appears to be empty; cannot patch.")

    body_indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]

    j = i
    if _is_docstring_line(lines[i]):
        quote = lines[i].lstrip()[:3]
        if lines[i].lstrip().count(quote) >= 2:
            j = i + 1
        else:
            j = i + 1
            while j < len(lines):
                if quote in lines[j]:
                    j += 1
                    break
                j += 1

    ensure_line = body_indent + ENSURE_LINE_MARK
    return lines[:j] + [ensure_line] + lines[j:]


def restore_from_git(path: Path) -> None:
    root = repo_root()
    subprocess.run(["git", "restore", str(path)], cwd=str(root), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="src/qtanner/lift_matrices.py", help="Path to lift_matrices.py")
    ap.add_argument("--git-restore", action="store_true", help="Restore target file from git and exit.")
    args = ap.parse_args()

    root = repo_root()
    target = (root / args.path).resolve()
    if not target.exists():
        print(f"ERROR: file not found: {target}", file=sys.stderr)
        return 2

    if args.git_restore:
        restore_from_git(target)
        print(f"Restored from git: {target}")
        return 0

    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    lines = strip_patch_blocks(lines)
    def_idx = find_def_build_hx_hz(lines)

    lines = insert_helper_block(lines, def_idx)
    def_idx = find_def_build_hx_hz(lines)

    lines = insert_ensure_line(lines, def_idx)

    target.write_text("".join(lines), encoding="utf-8")
    print(f"Patched: {target} (inserted smallgroup ops + ensure_group_ops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
