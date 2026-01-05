"""Persistent GAP session helper for QDistRnd."""

from __future__ import annotations

import contextlib
import os
import re
import select
import subprocess
import tempfile
import time
from pathlib import Path


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


class GapSession:
    """Manage a persistent GAP process for QDistRnd calls."""

    def __init__(
        self,
        *,
        gap_cmd: str = "gap",
        timeout_sec: float | None = None,
        work_dir: str | None = None,
    ) -> None:
        self.gap_cmd = gap_cmd
        self.timeout_sec = timeout_sec
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self._process: subprocess.Popen[str] | None = None
        self._script_path: str | None = None
        self._server_script: str | None = None
        self._start_process()

    @property
    def script_path(self) -> str | None:
        return self._script_path

    @property
    def server_script(self) -> str | None:
        return self._server_script

    def _build_server_script(self) -> str:
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
            "stream := InputTextFile(\"*stdin*\");",
            "if stream = fail then",
            '  Print("QDISTERROR StdinOpenFailed\\n");',
            "  QuitGap(3);",
            "fi;",
            "while true do",
            "  line := ReadLine(stream);",
            "  if line = fail then",
            "    QuitGap(0);",
            "  fi;",
            '  line := Filtered(line, c -> not c in "\\r\\n");',
            "  if line = \"\" then",
            "    continue;",
            "  fi;",
            '  parts := SplitString(line, "|");',
            "  cmd := parts[1];",
            "  if cmd = \"QUIT\" then",
            "    QuitGap(0);",
            "  fi;",
            "  if cmd = \"CSS\" then",
            "    if Length(parts) < 6 then",
            '      Print("QDISTERROR BadCommand ", line, "\\n");',
            "      FlushOutput();",
            "      continue;",
            "    fi;",
            "    HX := ReadMMGF2(parts[2]);",
            "    HZ := ReadMMGF2(parts[3]);",
            "    num := Int(parts[4]);",
            "    mindist := Int(parts[5]);",
            "    debug := Int(parts[6]);",
            "    dZ_raw := DistRandCSS(HX, HZ, num, mindist, debug : field := GF(2));",
            "    dX_raw := DistRandCSS(HZ, HX, num, mindist, debug : field := GF(2));",
            '    Print("QDISTRESULT ", dX_raw, " ", dZ_raw, "\\n");',
            "    FlushOutput();",
            "    continue;",
            "  fi;",
            "  if cmd = \"DZ\" then",
            "    if Length(parts) < 5 then",
            '      Print("QDISTERROR BadCommand ", line, "\\n");',
            "      FlushOutput();",
            "      continue;",
            "    fi;",
            "    HX := ReadMMGF2(parts[2]);",
            "    num := Int(parts[3]);",
            "    mindist := Int(parts[4]);",
            "    debug := Int(parts[5]);",
            "    ncols := NumberColumns(HX);",
            "    HZ := NullMat(1, ncols, GF(2));",
            "    dZ_raw := DistRandCSS(HX, HZ, num, mindist, debug : field := GF(2));",
            '    Print("QDISTRESULT_Z ", dZ_raw, "\\n");',
            "    FlushOutput();",
            "    continue;",
            "  fi;",
            '  Print("QDISTERROR UnknownCommand ", line, "\\n");',
            "  FlushOutput();",
            "od;",
        ]
        return "\n".join(lines) + "\n"

    def _start_process(self) -> None:
        self._server_script = self._build_server_script()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".g", dir=self.work_dir, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(self._server_script)
            self._script_path = tmp.name
        cmd = [self.gap_cmd, "-q", "-b", "--quitonbreak", self._script_path]
        try:
            self._process = subprocess.Popen(
                cmd,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"GAP command not found: {self.gap_cmd}") from exc

    def _ensure_alive(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._restart()

    def _restart(self) -> None:
        self.close()
        self._start_process()

    def close(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            with contextlib.suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            with contextlib.suppress(Exception):
                if proc.stdout:
                    proc.stdout.close()
            self._process = None
            if self._script_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(self._script_path)
                self._script_path = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _format_failure(self, message: str, lines: list[str]) -> str:
        tail = lines[-20:]
        snippet = "\n".join(tail) if tail else "<no output captured>"
        return f"{message}\nLast output:\n{snippet}"

    def _read_until(
        self, *, prefix: str, timeout_sec: float | None
    ) -> tuple[str, list[str]]:
        if self._process is None or self._process.stdout is None:
            raise GapSessionError("GAP session is not running.", kind="not_running")
        stdout = self._process.stdout
        deadline = None
        if timeout_sec is not None:
            deadline = time.monotonic() + timeout_sec
        lines: list[str] = []
        while True:
            timeout = None
            if deadline is not None:
                timeout = max(deadline - time.monotonic(), 0)
                if timeout == 0:
                    raise GapSessionError(
                        "GAP session timed out waiting for response.",
                        lines=lines,
                        kind="timeout",
                    )
            ready, _, _ = select.select([stdout], [], [], timeout)
            if not ready:
                raise GapSessionError(
                    "GAP session timed out waiting for response.",
                    lines=lines,
                    kind="timeout",
                )
            line = stdout.readline()
            if line == "":
                raise GapSessionError(
                    "GAP session exited unexpectedly.",
                    lines=lines,
                    kind="eof",
                )
            line = line.rstrip("\n")
            lines.append(line)
            if any(
                marker in line for marker in ("QDISTERROR", "Error,", "Syntax error")
            ):
                raise GapSessionError(
                    "GAP/QDistRnd reported errors.", lines=lines, kind="error"
                )
            if line.startswith(prefix):
                return line, lines

    def _run_command(
        self, command: str, *, expect_prefix: str, timeout_sec: float | None
    ) -> tuple[str, list[str]]:
        if self._process is None or self._process.stdin is None:
            raise GapSessionError("GAP session is not running.", kind="not_running")
        self._process.stdin.write(command)
        self._process.stdin.flush()
        return self._read_until(prefix=expect_prefix, timeout_sec=timeout_sec)

    def run_css(
        self,
        hx_path: str,
        hz_path: str,
        *,
        num: int,
        mindist: int,
        debug: int = 2,
        timeout_sec: float | None = None,
    ) -> tuple[int, int, list[str], float]:
        hx_abs = os.path.abspath(hx_path)
        hz_abs = os.path.abspath(hz_path)
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        last_lines: list[str] = []
        last_error = "GAP session failed."
        for attempt in range(2):
            self._ensure_alive()
            try:
                start = time.monotonic()
                line, lines = self._run_command(
                    f"CSS|{hx_abs}|{hz_abs}|{num}|{mindist}|{debug}\n",
                    expect_prefix="QDISTRESULT",
                    timeout_sec=timeout,
                )
                runtime_sec = time.monotonic() - start
                match = re.search(r"QDISTRESULT\s+(-?\d+)\s+(-?\d+)", line)
                if not match:
                    raise RuntimeError(
                        self._format_failure(
                            "Missing QDISTRESULT line in GAP session output.", lines
                        )
                    )
                return int(match.group(1)), int(match.group(2)), lines, runtime_sec
            except BrokenPipeError as exc:
                last_error = "GAP session died while sending command."
                last_lines = []
                self._restart()
                if attempt == 0:
                    continue
                raise RuntimeError(self._format_failure(last_error, last_lines)) from exc
            except GapSessionError as exc:
                last_lines = exc.lines
                last_error = str(exc)
                if exc.kind in ("timeout", "eof", "not_running"):
                    self._restart()
                    if attempt == 0:
                        continue
                raise RuntimeError(self._format_failure(last_error, last_lines)) from exc
        raise RuntimeError(self._format_failure(last_error, last_lines))

    def run_dz(
        self,
        hx_path: str,
        *,
        num: int,
        mindist: int,
        debug: int = 2,
        timeout_sec: float | None = None,
    ) -> tuple[int, list[str], float]:
        hx_abs = os.path.abspath(hx_path)
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        last_lines: list[str] = []
        last_error = "GAP session failed."
        for attempt in range(2):
            self._ensure_alive()
            try:
                start = time.monotonic()
                line, lines = self._run_command(
                    f"DZ|{hx_abs}|{num}|{mindist}|{debug}\n",
                    expect_prefix="QDISTRESULT_Z",
                    timeout_sec=timeout,
                )
                runtime_sec = time.monotonic() - start
                match = re.search(r"QDISTRESULT_Z\s+(-?\d+)", line)
                if not match:
                    raise RuntimeError(
                        self._format_failure(
                            "Missing QDISTRESULT_Z line in GAP session output.", lines
                        )
                    )
                return int(match.group(1)), lines, runtime_sec
            except BrokenPipeError as exc:
                last_error = "GAP session died while sending command."
                last_lines = []
                self._restart()
                if attempt == 0:
                    continue
                raise RuntimeError(self._format_failure(last_error, last_lines)) from exc
            except GapSessionError as exc:
                last_lines = exc.lines
                last_error = str(exc)
                if exc.kind in ("timeout", "eof", "not_running"):
                    self._restart()
                    if attempt == 0:
                        continue
                raise RuntimeError(self._format_failure(last_error, last_lines)) from exc
        raise RuntimeError(self._format_failure(last_error, last_lines))


class GapSessionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        lines: list[str] | None = None,
        kind: str = "error",
    ) -> None:
        super().__init__(message)
        self.lines = lines or []
        self.kind = kind


__all__ = ["GapSession"]
