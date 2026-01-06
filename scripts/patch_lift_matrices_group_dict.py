#!/usr/bin/env python3
"""patch_lift_matrices_group_dict.py

Patches src/qtanner/lift_matrices.py so build_hx_hz() accepts `group` passed as a
plain dict (e.g. when a group record was serialized/deserialized).

Idempotent: it will not apply the patch twice.

Typical usage (from repo root):
  python scripts/patch_lift_matrices_group_dict.py --git-restore
  python scripts/patch_lift_matrices_group_dict.py
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

_MARKER = "qtanner-search compat: accept dict group"


def _repo_root() -> Path:
    # scripts/patch_*.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _norm_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _run(cmd: list[str]) -> int:
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"ERROR: command not found: {cmd[0]}", file=sys.stderr)
        return 127
    return proc.returncode


def _git_restore(target: Path) -> int:
    # Prefer `git restore`; fall back to `git checkout --` for older git.
    rc = _run(["git", "restore", str(target)])
    if rc == 0:
        return 0
    return _run(["git", "checkout", "--", str(target)])


def _find_fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Could not find function {name}() in AST")


def _doc_expr(fn: ast.FunctionDef) -> Optional[ast.Expr]:
    if not fn.body:
        return None
    first = fn.body[0]
    if not isinstance(first, ast.Expr):
        return None
    val = first.value
    if isinstance(val, ast.Constant) and isinstance(val.value, str):
        return first
    return None


def _insertion_lineno(fn: ast.FunctionDef) -> int:
    """1-indexed line number where we should insert the compat snippet."""

    if not fn.body:
        raise RuntimeError("build_hx_hz() appears to have an empty body")

    doc = _doc_expr(fn)
    if doc is not None:
        end = getattr(doc, "end_lineno", None) or doc.lineno
        return int(end) + 1

    # No docstring: insert before the first statement.
    return int(fn.body[0].lineno)


def _patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if _MARKER in text:
        print(f"Already patched: {path}")
        return

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(
            "ERROR: target file does not parse as Python.\n"
            "Run with --git-restore to restore it from git, then re-run this patch.\n"
            f"SyntaxError: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    fn = _find_fn(tree, "build_hx_hz")
    ins_line = _insertion_lineno(fn)  # 1-indexed

    lines = text.splitlines(keepends=True)
    if not (1 <= ins_line <= len(lines) + 1):
        raise RuntimeError(
            f"Computed insertion line {ins_line} out of range for file with {len(lines)} lines"
        )

    # Determine indentation from the next line (or last line if inserting at EOF).
    if ins_line <= len(lines):
        probe = lines[ins_line - 1]
    else:
        probe = "    "

    indent = re.match(r"^(\s*)", probe).group(1)

    snippet = (
        f"{indent}# {_MARKER}\n"
        f"{indent}if isinstance(group, dict):\n"
        f"{indent}    from types import SimpleNamespace\n"
        f"{indent}    group = SimpleNamespace(**group)\n\n"
    )

    lines.insert(ins_line - 1, snippet)
    new_text = "".join(lines)

    # Verify we didn't break parsing.
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        print(
            "ERROR: patch would result in a SyntaxError; refusing to write.\n"
            f"SyntaxError: {e}",
            file=sys.stderr,
        )
        raise SystemExit(3)

    path.write_text(new_text, encoding="utf-8")
    print(f"Patched: {path} (inserted at line {ins_line})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="src/qtanner/lift_matrices.py",
        help="Path to lift_matrices.py (default: src/qtanner/lift_matrices.py)",
    )
    ap.add_argument(
        "--git-restore",
        action="store_true",
        help="Restore the target file from git (useful if it was corrupted)",
    )
    args = ap.parse_args()

    target = _norm_path(args.path)
    if not target.exists():
        print(f"ERROR: target file not found: {target}", file=sys.stderr)
        raise SystemExit(1)

    if args.git_restore:
        rc = _git_restore(target)
        if rc != 0:
            print(
                "ERROR: git restore failed. Are you in the repo root?\n"
                f"Tried to restore: {target}",
                file=sys.stderr,
            )
            raise SystemExit(rc)
        print(f"Restored from git: {target}")
        return

    _patch_file(target)


if __name__ == "__main__":
    main()
