"""Helpers for importing GAP SmallGroup data into FiniteGroup."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import List, Tuple

from .group import TableGroup


def _parse_gap_marked_output(output: str, n: int) -> Tuple[List[List[int]], List[int]]:
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

    mul_lines = []
    for line in lines[mul_begin + 1 : mul_end]:
        stripped = line.strip()
        if stripped:
            mul_lines.append(stripped)
    if len(mul_lines) != n:
        raise RuntimeError(f"GAP output has {len(mul_lines)} mul rows; expected {n}.")

    mul_table: List[List[int]] = []
    for row_idx, line in enumerate(mul_lines):
        parts = line.split()
        if len(parts) != n:
            raise RuntimeError(
                f"GAP output row {row_idx} has {len(parts)} entries; expected {n}."
            )
        try:
            row = [int(part) - 1 for part in parts]
        except ValueError as exc:
            raise RuntimeError(f"Invalid GAP mul entry in row {row_idx}: {line}") from exc
        mul_table.append(row)

    inv_tokens: List[str] = []
    for line in lines[inv_begin + 1 : inv_end]:
        inv_tokens.extend(line.split())
    if len(inv_tokens) != n:
        raise RuntimeError(f"GAP output has {len(inv_tokens)} inverses; expected {n}.")
    try:
        inv = [int(token) - 1 for token in inv_tokens]
    except ValueError as exc:
        raise RuntimeError("Invalid GAP inverse entry.") from exc

    return mul_table, inv


def _gap_error_preview(stdout: str, stderr: str) -> str:
    stdout_lines = stdout.splitlines()[:20]
    stderr_lines = stderr.splitlines()[:20]
    parts = []
    if stdout_lines:
        parts.append(f"stdout (first {len(stdout_lines)} lines):\n" + "\n".join(stdout_lines))
    if stderr_lines:
        parts.append(f"stderr (first {len(stderr_lines)} lines):\n" + "\n".join(stderr_lines))
    if not parts:
        parts.append("stdout/stderr empty.")
    return "\n".join(parts)


def _run_gap_script(script: str) -> subprocess.CompletedProcess[str]:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".g", delete=False) as tmp_file:
            tmp_file.write(script)
            tmp_path = tmp_file.name
        proc = subprocess.run(
            ["gap", "-q", "-b", "--quitonbreak", tmp_path],
            text=True,
            capture_output=True,
            check=True,
        )
        return proc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GAP executable not found. Install GAP and ensure `gap` is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        preview = _gap_error_preview(exc.stdout or "", exc.stderr or "")
        raise RuntimeError(f"GAP failed.\n{preview}") from exc
    finally:
        if "tmp_path" in locals():
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def smallgroup(order: int, gid: int, cache_dir: str = ".cache/gap_smallgroups") -> TableGroup:
    """Load a GAP SmallGroup as a FiniteGroup, with JSON caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"smallgroup_{order}_{gid}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format_version") == 2:
            return TableGroup(
                name=f"SmallGroup({order},{gid})",
                mul_table=payload["mul_table"],
                inv_table=payload["inv"],
            )

    script = "\n".join(
        [
            "G := SmallGroup(%d,%d);;" % (order, gid),
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
            "    Print(Position(elts, elts[i]*elts[j]), \" \");",
            "  od;",
            '  Print("\\n");',
            "od;",
            'Print("MUL_END\\n");',
            'Print("INV_BEGIN\\n");',
            "for i in [1..n] do",
            "  Print(Position(elts, Inverse(elts[i])), \" \");",
            "od;",
            'Print("\\nINV_END\\n");',
            "QuitGap(0);",
        ]
    )
    proc = _run_gap_script(script)

    mul_table, inv = _parse_gap_marked_output(proc.stdout, order)
    payload = {
        "format_version": 2,
        "order": order,
        "gid": gid,
        "mul_table": mul_table,
        "inv": inv,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return TableGroup(
        name=f"SmallGroup({order},{gid})",
        mul_table=mul_table,
        inv_table=inv,
    )


def nr_smallgroups(order: int, gap_cmd: str = "gap") -> int:
    """Return NrSmallGroups(order), using a JSON cache."""
    cache_dir = ".cache/gap_smallgroups"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "nr_smallgroups.json")
    payload = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    key = str(order)
    if key in payload:
        return int(payload[key])

    script = f'Print(NrSmallGroups({order}), "\\n");'
    try:
        proc = subprocess.run(
            [gap_cmd, "-q"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GAP failed: {exc.stderr.strip()}") from exc

    output = proc.stdout.strip().split()
    if not output:
        raise RuntimeError("GAP did not return NrSmallGroups output.")
    try:
        count = int(output[0])
    except ValueError as exc:
        raise RuntimeError(f"Invalid NrSmallGroups output: {proc.stdout.strip()}") from exc

    payload[key] = count
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return count


__all__ = ["smallgroup", "nr_smallgroups"]
