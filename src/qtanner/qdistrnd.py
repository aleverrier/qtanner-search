"""GAP/QDistRnd wrappers for randomized CSS distance estimates."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .gap_session import GapSession

from qtanner.mtx import validate_mtx_for_qdistrnd

_LAST_RUN_INFO: Optional[Dict[str, str]] = None
_TAIL_LINES = 30


def qd_stats_d_ub(qd_stats: Optional[Dict[str, object]]) -> Optional[int]:
    if not qd_stats:
        return None
    d_ub = qd_stats.get("d_ub")
    if d_ub is None:
        return None
    return int(d_ub)


def _gap_string_literal(text: str) -> str:
    """Escape a Python string for use as a GAP double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _gap_read_mmgf2_lines() -> list[str]:
    return [
        "ReadMMGF2 := function(path)",
        "  local stream, line, parts, rows, cols, nnz, idx, M, i, j, val;",
        "  stream := InputTextFile(path);",
        "  if stream = fail then",
        '    Print("QDISTERROR ReadMMGF2OpenFailed ", path, "\\n");',
        "    QuitGap(3);",
        "  fi;",
        "  line := ReadLine(stream);",
        "  while true do",
        "    line := ReadLine(stream);",
        "    if line = fail then",
        "      CloseStream(stream);",
        '      Print("QDISTERROR ReadMMGF2MissingSize ", path, "\\n");',
        "      QuitGap(3);",
        "    fi;",
        '    line := Filtered(line, c -> not c in "\\r\\n");',
        "    idx := PositionProperty(line, c -> not (c = ' ' or c = '\\t'));",
        "    if idx = fail then",
        "      continue;",
        "    fi;",
        "    if line[idx] = '%' then",
        "      continue;",
        "    fi;",
        "    break;",
        "  od;",
        '  parts := Filtered(SplitString(line, " \\t"), x -> x <> "");',
        "  if Length(parts) < 3 then",
        "    CloseStream(stream);",
        '    Print("QDISTERROR ReadMMGF2BadSize ", path, "\\n");',
        "    QuitGap(3);",
        "  fi;",
        "  rows := Int(parts[1]);",
        "  cols := Int(parts[2]);",
        "  nnz := Int(parts[3]);",
        "  M := NullMat(rows, cols, GF(2));",
        "  while true do",
        "    line := ReadLine(stream);",
        "    if line = fail then",
        "      break;",
        "    fi;",
        '    line := Filtered(line, c -> not c in "\\r\\n");',
        "    idx := PositionProperty(line, c -> not (c = ' ' or c = '\\t'));",
        "    if idx = fail then",
        "      continue;",
        "    fi;",
        "    if line[idx] = '%' then",
        "      continue;",
        "    fi;",
        '    parts := Filtered(SplitString(line, " \\t"), x -> x <> "");',
        "    if Length(parts) < 2 then",
        "      continue;",
        "    fi;",
        "    i := Int(parts[1]);",
        "    j := Int(parts[2]);",
        "    if Length(parts) >= 3 then",
        "      val := Int(parts[3]);",
        "    else",
        "      val := 1;",
        "    fi;",
        "    if (val mod 2) = 1 then",
        "      M[i][j] := M[i][j] + One(GF(2));",
        "    fi;",
        "  od;",
        "  CloseStream(stream);",
        "  return M;",
        "end;",
    ]


def _build_gap_script(
    hx_path: str,
    hz_path: str,
    num: int,
    mindist: int,
    debug: int,
    maxav: float | None,
    seed: int | None,
) -> str:
    hx_abs = os.path.abspath(hx_path)
    hz_abs = os.path.abspath(hz_path)
    hx_literal = _gap_string_literal(hx_abs)
    hz_literal = _gap_string_literal(hz_abs)
    base_args = f"HX, HZ, {num}, {mindist}, {debug} : field := GF(2)"
    base_args_swapped = f"HZ, HX, {num}, {mindist}, {debug} : field := GF(2)"
    if maxav is not None:
        base_args += f", maxav:={maxav}"
        base_args_swapped += f", maxav:={maxav}"
    lines = [
        "OnBreak := function()",
        '  Print("QDISTERROR GAPBreak\\n");',
        "  QuitGap(3);",
        "end;",
        'if LoadPackage("QDistRnd") = fail then',
        '  Print("QDISTERROR QDistRndNotLoaded\\n");',
        "  QuitGap(2);",
        "fi;",
        *_gap_read_mmgf2_lines(),
        f'HX := ReadMMGF2("{hx_literal}");;',
        f'HZ := ReadMMGF2("{hz_literal}");;',
    ]
    if seed is not None:
        lines.append(f"Reset(GlobalMersenneTwister, {seed});")
    lines.extend(
        [
            f"dZ_raw := DistRandCSS({base_args});;",
            f"dX_raw := DistRandCSS({base_args_swapped});;",
            'Print("QDISTRESULT ", dX_raw, " ", dZ_raw, "\\n");',
            "QuitGap(0);",
        ]
    )
    return "\n".join(lines) + "\n"


def _tail_excerpt(text: str, *, max_lines: int = _TAIL_LINES) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n".join(tail)


def _append_tail_section(
    *,
    stdout: str,
    stderr: str,
    max_lines: int = _TAIL_LINES,
) -> str:
    stdout_tail = _tail_excerpt(stdout, max_lines=max_lines)
    stderr_tail = _tail_excerpt(stderr, max_lines=max_lines)
    return "\n".join(
        [
            "[tail]",
            "[stdout_tail]",
            stdout_tail,
            "[stderr_tail]",
            stderr_tail,
        ]
    ).rstrip("\n") + "\n"


def _validate_mtx_header(path: str) -> None:
    """Ensure MatrixMarket header matches QDistRnd expectations."""
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip("\n")
    if line.endswith("\r"):
        line = line.rstrip("\r")
    if not line.startswith("%%MatrixMarket matrix coordinate") or "integer general" not in line:
        raise RuntimeError(
            f"Invalid MatrixMarket header in {path}: "
            "QDistRnd requires %%MatrixMarket matrix coordinate type general."
        )


def _parse_qdist_result(stdout: str, *, log_path: str, stderr: str) -> tuple[int, int]:
    error_lines = []
    combined_lines = []
    for line in stdout.splitlines():
        combined_lines.append(line)
        if "QDISTERROR" in line or "QTANNER_GAP_ERROR" in line:
            error_lines.append(line.strip())
    for line in stderr.splitlines():
        combined_lines.append(line)
        if (
            "QDISTERROR" in line
            or "QTANNER_GAP_ERROR" in line
            or line.startswith("Error,")
            or line.startswith("Syntax error")
        ):
            error_lines.append(line.strip())
    if error_lines:
        preview = "\n".join(error_lines[:20])
        raise RuntimeError(
            "GAP/QDistRnd reported errors. "
            f"See log at {log_path}.\n"
            f"errors (first {len(error_lines[:20])} lines):\n{preview}"
        )
    dX_raw: Optional[int] = None
    dZ_raw: Optional[int] = None
    for line in combined_lines:
        if "QDISTRESULT" in line:
            match = re.search(r"QDISTRESULT\s+(-?\d+)\s+(-?\d+)", line)
            if match:
                dX_raw = int(match.group(1))
                dZ_raw = int(match.group(2))
    if dX_raw is None or dZ_raw is None:
        stderr_lines = stderr.splitlines()[:20]
        stderr_preview = "\n".join(stderr_lines)
        raise RuntimeError(
            "Missing QDISTRESULT line in GAP output. "
            f"See log at {log_path}.\n"
            f"stderr (first {len(stderr_lines)} lines):\n{stderr_preview}"
        )
    return dX_raw, dZ_raw


def _build_gap_script_dz(
    hx_path: str,
    num: int,
    mindist: int,
    debug: int,
    seed: int | None,
) -> str:
    hx_abs = os.path.abspath(hx_path)
    hx_literal = _gap_string_literal(hx_abs)
    base_args = f"HX, HZ, {num}, {mindist}, {debug} : field := GF(2)"
    lines = [
        "OnBreak := function()",
        '  Print("QDISTERROR GAPBreak\\n");',
        "  QuitGap(3);",
        "end;",
        'if LoadPackage("QDistRnd") = fail then',
        '  Print("QDISTERROR QDistRndNotLoaded\\n");',
        "  QuitGap(2);",
        "fi;",
        *_gap_read_mmgf2_lines(),
        f'HX := ReadMMGF2("{hx_literal}");;',
        "ncols := NumberColumns(HX);;",
        "HZ := NullMat(1, ncols, GF(2));",
    ]
    if seed is not None:
        lines.append(f"Reset(GlobalMersenneTwister, {seed});")
    lines.extend(
        [
            f"dZ_raw := DistRandCSS({base_args});;",
            'Print("QDISTRESULT_Z ", dZ_raw, "\\n");',
            "QuitGap(0);",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_qdist_result_z(stdout: str, *, log_path: str, stderr: str) -> int:
    error_lines = []
    combined_lines = []
    for line in stdout.splitlines():
        combined_lines.append(line)
        if "QDISTERROR" in line or "QTANNER_GAP_ERROR" in line:
            error_lines.append(line.strip())
    for line in stderr.splitlines():
        combined_lines.append(line)
        if (
            "QDISTERROR" in line
            or "QTANNER_GAP_ERROR" in line
            or line.startswith("Error,")
            or line.startswith("Syntax error")
        ):
            error_lines.append(line.strip())
    if error_lines:
        preview = "\n".join(error_lines[:20])
        raise RuntimeError(
            "GAP/QDistRnd reported errors. "
            f"See log at {log_path}.\n"
            f"errors (first {len(error_lines[:20])} lines):\n{preview}"
        )
    dZ_raw: Optional[int] = None
    for line in combined_lines:
        if "QDISTRESULT_Z" in line:
            match = re.search(r"QDISTRESULT_Z\s+(-?\d+)", line)
            if match:
                dZ_raw = int(match.group(1))
    if dZ_raw is None:
        stderr_lines = stderr.splitlines()[:20]
        stderr_preview = "\n".join(stderr_lines)
        raise RuntimeError(
            "Missing QDISTRESULT_Z line in GAP output. "
            f"See log at {log_path}.\n"
            f"stderr (first {len(stderr_lines)} lines):\n{stderr_preview}"
        )
    return dZ_raw


def _run_gap_qdistrnd(
    hx_path: str,
    hz_path: str,
    *,
    num: int,
    mindist: int,
    debug: int,
    maxav: float | None,
    seed: int | None,
    gap_cmd: str,
    timeout_sec: float | None,
    log_path: str,
) -> Tuple[list[str], str, str, str, int, float, str]:
    script = _build_gap_script(hx_path, hz_path, num, mindist, debug, maxav, seed)
    script_path = str(Path(log_path).with_suffix(".g"))
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    cmd = [gap_cmd, "-q", "-b", "--quitonbreak", script_path]
    start = time.monotonic()
    result = None
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        runtime_sec = time.monotonic() - start
        write_qdistrnd_log(
            log_path,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
    except subprocess.TimeoutExpired as exc:
        runtime_sec = time.monotonic() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        write_qdistrnd_log(
            log_path,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(
            f"GAP timed out after {timeout_sec} seconds (runtime {runtime_sec:.2f}s)."
        ) from exc
    runtime_sec = time.monotonic() - start
    if result is not None:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode
    return cmd, script, stdout, stderr, returncode, runtime_sec, script_path


def write_qdistrnd_log(
    log_path: str,
    *,
    cmd: list[str],
    script: str,
    script_path: str,
    stdout: str,
    stderr: str,
    returncode: int,
    runtime_sec: float,
    hx_stats: dict | None = None,
    hz_stats: dict | None = None,
    include_tail: bool = False,
) -> None:
    """Write a QDistRnd run log with command, script, and outputs."""
    with open(log_path, "w", encoding="utf-8") as f:
        if hx_stats is not None or hz_stats is not None:
            f.write("[validation]\n")
            if hx_stats is not None:
                f.write(f"HX {hx_stats}\n")
            if hz_stats is not None:
                f.write(f"HZ {hz_stats}\n")
            f.write("\n")
        f.write("[command]\n")
        f.write(" ".join(shlex.quote(part) for part in cmd))
        f.write("\n\n[script_path]\n")
        f.write(script_path)
        f.write("\n\n[script]\n")
        f.write(script)
        if not script.endswith("\n"):
            f.write("\n")
        f.write("\n[stdout]\n")
        f.write(stdout)
        if stdout and not stdout.endswith("\n"):
            f.write("\n")
        f.write("\n[stderr]\n")
        f.write(stderr)
        if stderr and not stderr.endswith("\n"):
            f.write("\n")
        f.write("\n[returncode/runtime]\n")
        f.write(f"{returncode} {runtime_sec:.2f}\n")
        if include_tail:
            f.write("\n")
            f.write(_append_tail_section(stdout=stdout, stderr=stderr))


def _default_log_path(hx_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(hx_path)), "qdistrnd.log")


def _write_validation_log(
    log_path: str,
    *,
    hx_stats: dict | None = None,
    hz_stats: dict | None = None,
    error: Exception | None = None,
) -> None:
    """Write validation details to the QDistRnd log."""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[validation]\n")
        if hx_stats is not None:
            f.write(f"HX {hx_stats}\n")
        if hz_stats is not None:
            f.write(f"HZ {hz_stats}\n")
        if error is not None:
            f.write(f"error {error}\n")


def _run_info_from_details(
    *,
    cmd: list[str],
    script: str,
    script_path: str,
    stdout: str,
    stderr: str,
    returncode: int,
    runtime_sec: float,
) -> Dict[str, str]:
    return {
        "cmdline": " ".join(shlex.quote(part) for part in cmd),
        "script": script,
        "script_path": script_path,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": str(returncode),
        "runtime_sec": f"{runtime_sec:.2f}",
    }


def get_last_qdistrnd_run() -> Dict[str, str]:
    """Return details from the most recent QDistRnd run."""
    if _LAST_RUN_INFO is None:
        raise RuntimeError("No QDistRnd run info available yet.")
    return dict(_LAST_RUN_INFO)


def dist_rand_css_mtx(
    hx_path: str,
    hz_path: str,
    *,
    num: int = 50000,
    mindist: int = 0,
    debug: int = 2,
    maxav: float | None = None,
    seed: int | None = None,
    gap_cmd: str = "gap",
    timeout_sec: float | None = None,
    log_path: str | None = None,
    session: GapSession | None = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Estimate CSS distance upper bounds via GAP/QDistRnd using .mtx files."""
    log_path_final = log_path or _default_log_path(hx_path)
    returncode = -1
    try:
        hx_stats = validate_mtx_for_qdistrnd(Path(hx_path))
        hz_stats = validate_mtx_for_qdistrnd(Path(hz_path))
    except RuntimeError as exc:
        _write_validation_log(log_path_final, error=exc)
        raise
    _write_validation_log(log_path_final, hx_stats=hx_stats, hz_stats=hz_stats)
    if session is None:
        (
            cmd,
            script,
            stdout,
            stderr,
            returncode,
            runtime_sec,
            script_path,
        ) = _run_gap_qdistrnd(
            hx_path,
            hz_path,
            num=num,
            mindist=mindist,
            debug=debug,
            maxav=maxav,
            seed=seed,
            gap_cmd=gap_cmd,
            timeout_sec=timeout_sec,
            log_path=log_path_final,
        )
        write_qdistrnd_log(
            log_path_final,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            runtime_sec=runtime_sec,
            hx_stats=hx_stats,
            hz_stats=hz_stats,
            include_tail=returncode != 0,
        )
        if returncode != 0:
            message = f"GAP exited with code {returncode}. See log at {log_path_final}."
            if verbose:
                message = (
                    f"{message}\n"
                    "[stdout_tail]\n"
                    f"{_tail_excerpt(stdout)}\n"
                    "[stderr_tail]\n"
                    f"{_tail_excerpt(stderr)}"
                )
            raise RuntimeError(message)
        try:
            dX_raw, dZ_raw = _parse_qdist_result(
                stdout, log_path=log_path_final, stderr=stderr
            )
        except RuntimeError as exc:
            write_qdistrnd_log(
                log_path_final,
                cmd=cmd,
                script=script,
                script_path=script_path,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                runtime_sec=runtime_sec,
                hx_stats=hx_stats,
                hz_stats=hz_stats,
                include_tail=True,
            )
            message = str(exc)
            if verbose:
                message = (
                    f"{message}\n"
                    "[stdout_tail]\n"
                    f"{_tail_excerpt(stdout)}\n"
                    "[stderr_tail]\n"
                    f"{_tail_excerpt(stderr)}"
                )
            raise RuntimeError(message) from exc
    else:
        dX_raw, dZ_raw, lines, runtime_sec = session.run_css(
            hx_path,
            hz_path,
            num=num,
            mindist=mindist,
            debug=debug,
            timeout_sec=timeout_sec,
        )
        stdout = "\n".join(lines)
        stderr = ""
        script = session.server_script or ""
        script_path = str(Path(log_path_final).with_suffix(".g"))
        if script:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
        cmd = [gap_cmd, "-q", "-b", "--quitonbreak", script_path]
        returncode = 0
        write_qdistrnd_log(
            log_path_final,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            runtime_sec=runtime_sec,
            hx_stats=hx_stats,
            hz_stats=hz_stats,
        )
    summary = {
        "dX_ub": abs(dX_raw),
        "dZ_ub": abs(dZ_raw),
        "d_ub": min(abs(dX_raw), abs(dZ_raw)),
        "dX_raw": dX_raw,
        "dZ_raw": dZ_raw,
        "terminated_early_X": dX_raw < 0,
        "terminated_early_Z": dZ_raw < 0,
        "num": num,
        "mindist": mindist,
        "debug": debug,
        "maxav": maxav,
        "seed": seed,
        "runtime_sec": runtime_sec,
        "gap_cmd": gap_cmd,
        "session": session is not None,
    }
    run_info = _run_info_from_details(
        cmd=cmd,
        script=script,
        script_path=script_path,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        runtime_sec=runtime_sec,
    )
    global _LAST_RUN_INFO
    _LAST_RUN_INFO = run_info
    return summary


def dist_rand_dz_mtx(
    hx_path: str,
    *,
    num: int,
    mindist: int,
    debug: int = 2,
    seed: int | None = None,
    gap_cmd: str = "gap",
    timeout_sec: float | None = None,
    log_path: str | None = None,
    session: GapSession | None = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Estimate Z-distance upper bounds for a single HX using GAP/QDistRnd."""
    log_path_final = log_path or _default_log_path(hx_path)
    result = None
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        hx_stats = validate_mtx_for_qdistrnd(Path(hx_path))
    except RuntimeError as exc:
        _write_validation_log(log_path_final, error=exc)
        raise
    _write_validation_log(log_path_final, hx_stats=hx_stats)
    if session is None:
        script = _build_gap_script_dz(hx_path, num, mindist, debug, seed)
        script_path = str(Path(log_path_final).with_suffix(".g"))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        cmd = [gap_cmd, "-q", "-b", "--quitonbreak", script_path]
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except FileNotFoundError as exc:
            runtime_sec = time.monotonic() - start
            write_qdistrnd_log(
                log_path_final,
                cmd=cmd,
                script=script,
                script_path=script_path,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                runtime_sec=runtime_sec,
                hx_stats=hx_stats,
                include_tail=True,
            )
            raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
        except subprocess.TimeoutExpired as exc:
            runtime_sec = time.monotonic() - start
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            write_qdistrnd_log(
                log_path_final,
                cmd=cmd,
                script=script,
                script_path=script_path,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                runtime_sec=runtime_sec,
                hx_stats=hx_stats,
                include_tail=True,
            )
            raise RuntimeError(
                f"GAP timed out after {timeout_sec} seconds (runtime {runtime_sec:.2f}s)."
            ) from exc
        runtime_sec = time.monotonic() - start
        if result is not None:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            returncode = result.returncode
        write_qdistrnd_log(
            log_path_final,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            runtime_sec=runtime_sec,
            hx_stats=hx_stats,
            include_tail=returncode != 0,
        )
        if returncode != 0:
            returncode_text = returncode if result is not None else "unknown"
            message = (
                f"GAP exited with code {returncode_text}. See log at {log_path_final}."
            )
            if verbose:
                message = (
                    f"{message}\n"
                    "[stdout_tail]\n"
                    f"{_tail_excerpt(stdout)}\n"
                    "[stderr_tail]\n"
                    f"{_tail_excerpt(stderr)}"
                )
            raise RuntimeError(message)
        try:
            dZ_raw = _parse_qdist_result_z(
                stdout, log_path=log_path_final, stderr=stderr
            )
        except RuntimeError as exc:
            write_qdistrnd_log(
                log_path_final,
                cmd=cmd,
                script=script,
                script_path=script_path,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                runtime_sec=runtime_sec,
                hx_stats=hx_stats,
                include_tail=True,
            )
            message = str(exc)
            if verbose:
                message = (
                    f"{message}\n"
                    "[stdout_tail]\n"
                    f"{_tail_excerpt(stdout)}\n"
                    "[stderr_tail]\n"
                    f"{_tail_excerpt(stderr)}"
                )
            raise RuntimeError(message) from exc
    else:
        dZ_raw, lines, runtime_sec = session.run_dz(
            hx_path,
            num=num,
            mindist=mindist,
            debug=debug,
            timeout_sec=timeout_sec,
        )
        stdout = "\n".join(lines)
        stderr = ""
        script = session.server_script or ""
        script_path = str(Path(log_path_final).with_suffix(".g"))
        if script:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
        cmd = [gap_cmd, "-q", "-b", "--quitonbreak", script_path]
        write_qdistrnd_log(
            log_path_final,
            cmd=cmd,
            script=script,
            script_path=script_path,
            stdout=stdout,
            stderr=stderr,
            returncode=0,
            runtime_sec=runtime_sec,
            hx_stats=hx_stats,
        )
        returncode = 0
    summary = {
        "dZ_raw": dZ_raw,
        "dZ_ub": abs(dZ_raw),
        "terminated_early": dZ_raw < 0,
        "num": num,
        "mindist": mindist,
        "debug": debug,
        "seed": seed,
        "runtime_sec": runtime_sec,
        "gap_cmd": gap_cmd,
        "session": session is not None,
    }
    run_info = _run_info_from_details(
        cmd=cmd,
        script=script,
        script_path=script_path,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        runtime_sec=runtime_sec,
    )
    global _LAST_RUN_INFO
    _LAST_RUN_INFO = run_info
    return summary


def gap_is_available(gap_cmd: str = "gap") -> bool:
    """Return True if GAP is runnable."""
    try:
        result = subprocess.run(
            [gap_cmd, "-q", "-b", "--quitonbreak", "--norepl"],
            input="QuitGap(0);\n",
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def qdistrnd_is_available(gap_cmd: str = "gap") -> bool:
    """Return True if GAP can load the QDistRnd package."""
    script = '\n'.join(
        [
            'if LoadPackage("QDistRnd") = fail then',
            "  QuitGap(1);",
            "fi;",
            "QuitGap(0);",
        ]
    )
    try:
        result = subprocess.run(
            [gap_cmd, "-q", "-b", "--quitonbreak", "--norepl"],
            input=script,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


__all__ = [
    "dist_rand_css_mtx",
    "dist_rand_dz_mtx",
    "gap_is_available",
    "qdistrnd_is_available",
    "qd_stats_d_ub",
    "get_last_qdistrnd_run",
    "write_qdistrnd_log",
]
