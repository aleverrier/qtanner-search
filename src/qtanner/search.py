"""Pilot search for cyclic-group quantum Tanner codes ([6,3,3] local codes only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .classical_distance import ClassicalCodeAnalysis, analyze_parity_check_bitrows
from .gf2 import gf2_rank
from .group import FiniteGroup
from .lift_matrices import build_hx_hz, css_commutes
from .local_codes import (
    LocalCode,
    apply_col_perm_to_rows,
    hamming_6_3_3_shortened,
    variants_6_3_3,
)
from .mtx import write_mtx_from_bitrows
from .qdistrnd import gap_is_available, qdistrnd_is_available, write_qdistrnd_log


@dataclass(frozen=True)
class SliceScore:
    score: int
    h: ClassicalCodeAnalysis
    g: ClassicalCodeAnalysis

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "h": self.h.as_dict(),
            "g": self.g.as_dict(),
        }


@dataclass(frozen=True)
class SliceCandidate:
    elements: List[int]
    score: SliceScore

    def as_dict(self) -> dict:
        return {"elements": list(self.elements), "score": self.score.as_dict()}


def _parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _ceil_sqrt(n: int) -> int:
    root = math.isqrt(n)
    return root if root * root == n else root + 1


def _slice_dist_threshold(spec: str, *, n: int) -> int:
    if spec is None:
        raise ValueError("--min-slice-dist must be provided.")
    value = spec.strip()
    if not value:
        raise ValueError("--min-slice-dist must be non-empty.")
    if value.lower() == "sqrtn":
        return _ceil_sqrt(n)
    try:
        dist = int(value)
    except ValueError as exc:
        raise ValueError(
            f"--min-slice-dist must be an integer or 'sqrtN' (got '{spec}')."
        ) from exc
    if dist < 0:
        raise ValueError("--min-slice-dist must be nonnegative.")
    return dist


def _timestamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _apply_variant(code: LocalCode, variant_idx: int) -> LocalCode:
    variants = variants_6_3_3()
    if variant_idx < 0 or variant_idx >= len(variants):
        raise ValueError(f"Variant index {variant_idx} out of range for 6_3_3.")
    perm = variants[variant_idx]
    H_rows = apply_col_perm_to_rows(code.H_rows, perm, code.n)
    G_rows = apply_col_perm_to_rows(code.G_rows, perm, code.n)
    return LocalCode(
        name=f"{code.name}_var{variant_idx}",
        n=code.n,
        k=code.k,
        H_rows=H_rows,
        G_rows=G_rows,
    )


def _choose_permutations(total: int, count: int, rng: random.Random) -> List[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    pool = list(range(total))
    rng.shuffle(pool)
    if count == 1:
        return [0]
    selected = [0]
    for idx in pool:
        if idx == 0:
            continue
        selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def _comb_count(m: int) -> int:
    if m < 6:
        return 0
    return math.comb(m - 1, 5)


def _multiset_count(m: int) -> int:
    if m <= 0:
        return 0
    return math.comb(m + 4, 5)


def _sample_combinations(
    *,
    m: int,
    count: int,
    rng: random.Random,
) -> List[List[int]]:
    if count <= 0:
        return []
    if m < 6:
        return []
    pool = list(range(1, m))
    chosen: set[Tuple[int, ...]] = set()
    max_attempts = max(50, count * 50)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        comb = tuple(sorted(rng.sample(pool, 5)))
        chosen.add(comb)
        attempts += 1
    if len(chosen) < count:
        # fallback: deterministic prefix if sampling couldn't hit enough
        combs = list(combinations(pool, 5))[:count]
        chosen.update(combs)
    ordered = sorted(chosen)
    if len(ordered) > count:
        ordered = ordered[:count]
    return [[0, *comb] for comb in ordered]


def _sample_multisets(
    *,
    m: int,
    count: int,
    rng: random.Random,
) -> List[List[int]]:
    if count <= 0 or m <= 0:
        return []
    chosen: set[Tuple[int, ...]] = set()
    max_attempts = max(50, count * 50)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        sample = [rng.randrange(m) for _ in range(6)]
        if 0 not in sample:
            attempts += 1
            continue
        chosen.add(tuple(sorted(sample)))
        attempts += 1
    if len(chosen) < count:
        for comb in combinations_with_replacement(range(m), 6):
            if comb[0] != 0:
                continue
            chosen.add(comb)
            if len(chosen) >= count:
                break
    ordered = list(chosen)
    rng.shuffle(ordered)
    if len(ordered) > count:
        ordered = ordered[:count]
    return [list(comb) for comb in ordered]


def _enumerate_sets(
    *,
    m: int,
    max_sets: Optional[int],
    rng: random.Random,
    feasible_limit: int,
    allow_repeats: bool = False,
) -> List[List[int]]:
    if allow_repeats:
        total = _multiset_count(m)
        if total == 0:
            return []
        if total <= feasible_limit:
            combos = [
                list(comb)
                for comb in combinations_with_replacement(range(m), 6)
                if comb[0] == 0
            ]
            if max_sets is not None and len(combos) > max_sets:
                rng.shuffle(combos)
                combos = combos[:max_sets]
            return combos
        target = max_sets if max_sets is not None else min(feasible_limit, total)
        return _sample_multisets(m=m, count=min(target, total), rng=rng)
    total = _comb_count(m)
    if total == 0:
        return []
    if max_sets is None and total <= feasible_limit:
        return [[0, *comb] for comb in combinations(range(1, m), 5)]
    target = max_sets if max_sets is not None else min(feasible_limit, total)
    return _sample_combinations(m=m, count=target, rng=rng)


def _analyze_slice(
    rows: Sequence[int],
    ncols: int,
    *,
    rng: random.Random,
) -> ClassicalCodeAnalysis:
    return analyze_parity_check_bitrows(
        rows,
        ncols,
        max_k_exact=10,
        samples=1 << 10,
        seed=rng.randrange(1 << 31),
    )


def _score_a_slice(
    group: FiniteGroup,
    A: List[int],
    C0: LocalCode,
    C1: LocalCode,
    *,
    rng: random.Random,
) -> SliceScore:
    from .slice_codes import build_a_slice_checks_G, build_a_slice_checks_H

    h_rows, n_cols = build_a_slice_checks_H(group, A, C0, C1)
    g_rows, _ = build_a_slice_checks_G(group, A, C0, C1)
    h_stats = _analyze_slice(h_rows, n_cols, rng=rng)
    g_stats = _analyze_slice(g_rows, n_cols, rng=rng)
    return SliceScore(score=min(h_stats.d, g_stats.d), h=h_stats, g=g_stats)


def _score_b_slice(
    group: FiniteGroup,
    B: List[int],
    C0p: LocalCode,
    C1p: LocalCode,
    *,
    rng: random.Random,
) -> SliceScore:
    from .slice_codes import build_b_slice_checks_Gp, build_b_slice_checks_Hp

    h_rows, n_cols = build_b_slice_checks_Hp(group, B, C0p, C1p)
    g_rows, _ = build_b_slice_checks_Gp(group, B, C0p, C1p)
    h_stats = _analyze_slice(h_rows, n_cols, rng=rng)
    g_stats = _analyze_slice(g_rows, n_cols, rng=rng)
    return SliceScore(score=min(h_stats.d, g_stats.d), h=h_stats, g=g_stats)


def _select_candidates(
    scored: List[SliceCandidate],
    *,
    top_n: int,
    explore_n: int,
    rng: random.Random,
) -> List[SliceCandidate]:
    if not scored:
        return []
    ordered = sorted(scored, key=lambda item: item.score.score, reverse=True)
    top = ordered[: max(0, top_n)]
    remainder = ordered[len(top) :]
    explore = []
    if explore_n > 0 and remainder:
        explore = rng.sample(remainder, min(explore_n, len(remainder)))
    seen = {tuple(item.elements) for item in top}
    for item in explore:
        key = tuple(item.elements)
        if key not in seen:
            top.append(item)
            seen.add(key)
    return top


def _filter_slice_candidates(
    scored: List[SliceCandidate],
    *,
    threshold: int,
    top_n: int,
    explore_n: int,
    rng: random.Random,
    label: str,
) -> List[SliceCandidate]:
    if not scored:
        return []
    filtered = [item for item in scored if item.score.score >= threshold]
    if filtered:
        return _select_candidates(
            filtered, top_n=top_n, explore_n=explore_n, rng=rng
        )
    keep_n = min(top_n, len(scored))
    ordered = sorted(scored, key=lambda item: item.score.score, reverse=True)
    kept = ordered[:keep_n]
    print(
        f"[pilot] WARNING: no {label} slices met min_slice_dist={threshold}; "
        f"keeping best {keep_n} of {len(scored)} by score."
    )
    return kept


def _mult_stats(mult: Sequence[int]) -> Tuple[int, Optional[float]]:
    uniq = len(mult)
    if uniq == 0:
        return 0, None
    return uniq, sum(mult) / uniq


def _qd_side_stats(
    *,
    d_signed: int,
    rounds_done: int,
    vec_count_total: int,
    mult: List[int],
) -> Dict[str, object]:
    uniq, avg = _mult_stats(mult)
    return {
        "d_signed": d_signed,
        "rounds_done": rounds_done,
        "vec_count_total": vec_count_total,
        "mult": mult,
        "uniq": uniq,
        "avg": avg,
        "early_stop": d_signed < 0,
    }


def _format_avg(avg: Optional[float]) -> str:
    if avg is None:
        return "n/a"
    return f"{avg:.3f}"


def _record_best_so_far(
    best_by_nk: Dict[Tuple[int, int], Dict[str, object]],
    *,
    entry: Dict[str, object],
    dx: int,
    dz: int,
    d: int,
    trials_requested: int,
    qd_x: Dict[str, object],
    qd_z: Dict[str, object],
) -> None:
    key = (int(entry["n"]), int(entry["k"]))
    current = best_by_nk.get(key)
    if current is not None and int(current["d"]) >= d:
        return
    best_by_nk[key] = {
        "n": key[0],
        "k": key[1],
        "d": d,
        "dx": dx,
        "dz": dz,
        "trials_requested": trials_requested,
        "qd_x": qd_x,
        "qd_z": qd_z,
        "candidate_id": entry["candidate_id"],
    }
    print("[pilot] best so far by (n,k):")
    for n_key, k_key in sorted(best_by_nk):
        rec = best_by_nk[(n_key, k_key)]
        qd_x = rec["qd_x"]
        qd_z = rec["qd_z"]
        print(
            f"  n={rec['n']} k={rec['k']} d={rec['d']} "
            f"dx={rec['dx']} dz={rec['dz']} "
            f"trials={rec['trials_requested']} "
            f"dx: done={qd_x['rounds_done']} avg={_format_avg(qd_x['avg'])} "
            f"uniq={qd_x['uniq']} "
            f"dz: done={qd_z['rounds_done']} avg={_format_avg(qd_z['avg'])} "
            f"uniq={qd_z['uniq']} id={rec['candidate_id']}"
        )


def _gap_string_literal(text: str) -> str:
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


def _gap_dist_rand_css_stats_lines() -> list[str]:
    return [
        "DistRandCSS_stats := function(GX, GZ, num, mindist)",
        "  local DistBound, i, j, dimsWZ, rowsWZ, colsWZ, WZ, WX,",
        "        TempVec, TempWeight, per, WZ1, WZ2, VecCount,",
        "        CodeWords, mult, pos, early_stop, rounds_done,",
        "        d_signed, uniq, avg, s1, s2, x2;",
        "  WZ := NullspaceMat(TransposedMatMutable(GX));",
        "  WX := NullspaceMat(TransposedMatMutable(GZ));",
        "  dimsWZ := DimensionsMat(WZ);",
        "  rowsWZ := dimsWZ[1];",
        "  colsWZ := dimsWZ[2];",
        "  DistBound := colsWZ + 1;",
        "  VecCount := 0;",
        "  CodeWords := [];",
        "  mult := [];",
        "  early_stop := false;",
        "  rounds_done := 0;",
        "  for i in [1..num] do",
        "    rounds_done := i;",
        "    if colsWZ > 1 then",
        "      per := Random(SymmetricGroup(colsWZ));",
        "    else",
        "      per := ();",
        "    fi;",
        "    WZ1 := PermutedCols(WZ, per);",
        "    WZ2 := TriangulizedMat(WZ1);",
        "    WZ2 := PermutedCols(WZ2, Inverse(per));",
        "    for j in [1..rowsWZ] do",
        "      TempVec := WZ2[j];",
        "      TempWeight := WeightVecFFE(TempVec);",
        "      if (TempWeight > 0) and (TempWeight <= DistBound) then",
        "        if WeightVecFFE(WX * TempVec) > 0 then",
        "          if TempWeight < DistBound then",
        "            DistBound := TempWeight;",
        "            VecCount := 1;",
        "            CodeWords := [TempVec];",
        "            mult := [1];",
        "          elif TempWeight = DistBound then",
        "            VecCount := VecCount + 1;",
        "            pos := Position(CodeWords, TempVec);",
        "            if (pos = fail) and (Length(mult) < 100) then",
        "              Add(CodeWords, TempVec);",
        "              Add(mult, 1);",
        "            elif (pos <> fail) then",
        "              mult[pos] := mult[pos] + 1;",
        "            fi;",
        "          fi;",
        "        fi;",
        "      fi;",
        "      if DistBound <= mindist then",
        "        early_stop := true;",
        "        break;",
        "      fi;",
        "    od;",
        "    if early_stop then",
        "      break;",
        "    fi;",
        "  od;",
        "  if early_stop then",
        "    d_signed := -DistBound;",
        "  else",
        "    d_signed := DistBound;",
        "  fi;",
        "  uniq := Length(mult);",
        "  if uniq > 0 then",
        "    avg := QDR_AverageCalc(mult);",
        "  else",
        "    avg := fail;",
        "  fi;",
        "  if uniq > 1 then",
        "    s1 := Sum(mult);",
        "    s2 := Sum(mult, x -> x^2);",
        "    x2 := Float(s2 * uniq - s1^2) / s1;",
        "  else",
        "    x2 := fail;",
        "  fi;",
        "  return rec(",
        "    d_signed := d_signed,",
        "    rounds_done := rounds_done,",
        "    vec_count_total := VecCount,",
        "    mult := mult,",
        "    uniq := uniq,",
        "    avg := avg,",
        "    x2 := x2",
        "  );",
        "end;",
        "QDR_ListString := function(list)",
        "  if Length(list) = 0 then",
        '    return "[]";',
        "  fi;",
        '  return Concatenation("[", JoinStringsWithSeparator(List(list, String), ","), "]");',
        "end;",
    ]


def _build_gap_batch_script(
    batch: Sequence[Tuple[int, str, str]],
    *,
    num: int,
    mindist: int,
    debug: int,
    seed: Optional[int],
) -> str:
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
        *_gap_dist_rand_css_stats_lines(),
    ]
    if seed is not None:
        lines.append(f"Reset(GlobalMersenneTwister, {seed});")
    for idx, hx_path, hz_path in batch:
        hx_literal = _gap_string_literal(os.path.abspath(hx_path))
        hz_literal = _gap_string_literal(os.path.abspath(hz_path))
        lines.extend(
            [
                f'HX := ReadMMGF2("{hx_literal}");;',
                f'HZ := ReadMMGF2("{hz_literal}");;',
                f"zz := DistRandCSS_stats(HX, HZ, {num}, {mindist});;",
                f"zx := DistRandCSS_stats(HZ, HX, {num}, {mindist});;",
                f'Print("QDR|{idx}|dx=", zx.d_signed, "|dz=", zz.d_signed,',
                '      "|rx=", zx.rounds_done, "|rz=", zz.rounds_done,',
                '      "|vx=", zx.vec_count_total, "|vz=", zz.vec_count_total,',
                '      "|mx=", QDR_ListString(zx.mult), "|mz=", QDR_ListString(zz.mult), "\\n");',
            ]
        )
    lines.append("QuitGap(0);")
    return "\n".join(lines) + "\n"


def _parse_mult_list(text: str) -> List[int]:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError(f"Invalid mult list: {text}")
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    parts = [part.strip() for part in inner.split(",") if part.strip()]
    return [int(part) for part in parts]


def _parse_qdr_line(line: str) -> Tuple[int, Dict[str, object]]:
    parts = [part for part in line.strip().split("|") if part != ""]
    if len(parts) < 3 or parts[0] != "QDR":
        raise ValueError(f"Invalid QDR line: {line}")
    try:
        idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid QDR index in line: {line}") from exc
    fields: Dict[str, str] = {}
    for part in parts[2:]:
        key, sep, value = part.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid QDR field '{part}' in line: {line}")
        fields[key] = value
    required = ("dx", "dz", "rx", "rz", "vx", "vz", "mx", "mz")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing QDR fields {missing} in line: {line}")
    return idx, {
        "dx_signed": int(fields["dx"]),
        "dz_signed": int(fields["dz"]),
        "rx": int(fields["rx"]),
        "rz": int(fields["rz"]),
        "vx": int(fields["vx"]),
        "vz": int(fields["vz"]),
        "mx": _parse_mult_list(fields["mx"]),
        "mz": _parse_mult_list(fields["mz"]),
    }


def _parse_batch_output(
    *,
    stdout: str,
    stderr: str,
    expected: Iterable[int],
    log_path: Path,
) -> Dict[int, Dict[str, object]]:
    errors = []
    for line in stdout.splitlines():
        if "QDISTERROR" in line or "QTANNER_GAP_ERROR" in line:
            errors.append(line.strip())
    for line in stderr.splitlines():
        if (
            "QDISTERROR" in line
            or "QTANNER_GAP_ERROR" in line
            or line.startswith("Error,")
            or line.startswith("Syntax error")
        ):
            errors.append(line.strip())
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(
            "GAP/QDistRnd reported errors. "
            f"See log at {log_path}.\n"
            f"errors (first {len(errors[:20])} lines):\n{preview}"
        )
    results: Dict[int, Dict[str, object]] = {}
    for line in stdout.splitlines():
        if not line.startswith("QDR|"):
            continue
        idx, parsed = _parse_qdr_line(line)
        results[idx] = parsed
    missing = [idx for idx in expected if idx not in results]
    if missing:
        raise RuntimeError(
            "Missing QDR lines for some candidates. "
            f"See log at {log_path}. Missing indices: {missing}"
        )
    return results


def _run_gap_batch(
    *,
    batch: Sequence[Tuple[int, str, str]],
    num: int,
    mindist: int,
    debug: int,
    seed: Optional[int],
    outdir: Path,
    batch_id: int,
    gap_cmd: str = "gap",
    timeout_sec: Optional[float] = None,
) -> Tuple[Dict[int, Dict[str, object]], float]:
    script = _build_gap_batch_script(batch, num=num, mindist=mindist, debug=debug, seed=seed)
    script_path = outdir / f"qdistrnd_batch_{batch_id:04d}.g"
    log_path = outdir / f"qdistrnd_batch_{batch_id:04d}.log"
    script_path.write_text(script, encoding="utf-8")
    cmd = [gap_cmd, "-q", "-b", "--quitonbreak", str(script_path)]
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
            str(log_path),
            cmd=cmd,
            script=script,
            script_path=str(script_path),
            stdout="",
            stderr="",
            returncode=-1,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(f"GAP command not found: {gap_cmd}") from exc
    except subprocess.TimeoutExpired as exc:
        runtime_sec = time.monotonic() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        write_qdistrnd_log(
            str(log_path),
            cmd=cmd,
            script=script,
            script_path=str(script_path),
            stdout=stdout,
            stderr=stderr,
            returncode=-1,
            runtime_sec=runtime_sec,
            include_tail=True,
        )
        raise RuntimeError(
            f"GAP timed out after {timeout_sec} seconds (runtime {runtime_sec:.2f}s)."
        ) from exc
    runtime_sec = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    write_qdistrnd_log(
        str(log_path),
        cmd=cmd,
        script=script,
        script_path=str(script_path),
        stdout=stdout,
        stderr=stderr,
        returncode=result.returncode,
        runtime_sec=runtime_sec,
        include_tail=result.returncode != 0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"GAP exited with code {result.returncode}. See log at {log_path}."
        )
    expected = [idx for idx, _, _ in batch]
    results = _parse_batch_output(
        stdout=stdout, stderr=stderr, expected=expected, log_path=log_path
    )
    return results, runtime_sec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot search for cyclic-group Tanner codes ([6,3,3] local codes)."
    )
    parser.add_argument("--m-list", required=True, help="Comma list of group orders.")
    parser.add_argument(
        "--max-n",
        type=int,
        default=200,
        help="Skip m with n=36*m above this value.",
    )
    parser.add_argument(
        "--min-slice-dist",
        type=str,
        default="sqrtN",
        help="Minimum slice score (int or 'sqrtN').",
    )
    parser.add_argument(
        "--allow-repeats",
        action="store_true",
        help="Allow repeated elements in A/B multisets.",
    )
    parser.add_argument("--maxA", type=int, default=None, help="Limit A multisets.")
    parser.add_argument("--maxB", type=int, default=None, help="Limit B multisets.")
    parser.add_argument("--permH1", type=int, default=5, help="Permutations for H1.")
    parser.add_argument("--permH1p", type=int, default=5, help="Permutations for H1p.")
    parser.add_argument("--topA", type=int, default=30, help="Keep top A by slice score.")
    parser.add_argument("--topB", type=int, default=30, help="Keep top B by slice score.")
    parser.add_argument(
        "--exploreA", type=int, default=0, help="Random tail A candidates."
    )
    parser.add_argument(
        "--exploreB", type=int, default=0, help="Random tail B candidates."
    )
    parser.add_argument("--trials", type=int, default=20, help="QDistRnd trials.")
    parser.add_argument("--mindist", type=int, default=10, help="QDistRnd mindist.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Candidates per GAP/QDistRnd batch.",
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for the run."
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--gap-cmd", type=str, default="gap", help="GAP command.")
    parser.add_argument(
        "--qd-timeout", type=float, default=None, help="Timeout per batch (sec)."
    )
    parser.add_argument("--qd-debug", type=int, default=2, help="QDistRnd debug.")
    args = parser.parse_args()

    m_list = _parse_int_list(args.m_list)
    if not m_list:
        raise ValueError("--m-list must contain at least one integer.")
    if args.max_n <= 0:
        raise ValueError("--max-n must be positive.")
    if args.min_slice_dist is None or not args.min_slice_dist.strip():
        raise ValueError("--min-slice-dist must be non-empty.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.mindist < 0:
        raise ValueError("--mindist must be nonnegative.")
    for name in ("maxA", "maxB", "topA", "topB", "exploreA", "exploreB"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be nonnegative.")
    if not gap_is_available(args.gap_cmd):
        raise RuntimeError(f"GAP is not available on PATH as '{args.gap_cmd}'.")
    if not qdistrnd_is_available(args.gap_cmd):
        raise RuntimeError("GAP QDistRnd package is not available.")

    seed = args.seed
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 31)
    rng = random.Random(seed)

    outdir = Path(args.outdir) if args.outdir else Path("runs") / f"pilot_{_timestamp_utc()}"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root = outdir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    promising_root = outdir / "promising"
    promising_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "m_list": m_list,
        "max_n": args.max_n,
        "min_slice_dist": args.min_slice_dist,
        "allow_repeats": args.allow_repeats,
        "maxA": args.maxA,
        "maxB": args.maxB,
        "permH1": args.permH1,
        "permH1p": args.permH1p,
        "topA": args.topA,
        "topB": args.topB,
        "exploreA": args.exploreA,
        "exploreB": args.exploreB,
        "trials": args.trials,
        "mindist": args.mindist,
        "batch_size": args.batch_size,
        "seed": seed,
        "gap_cmd": args.gap_cmd,
        "qd_timeout": args.qd_timeout,
        "qd_debug": args.qd_debug,
        "slice_scoring": {"max_k_exact": 10, "samples": 1 << 10},
        "note": "slice scores use canonical C1/C1p (variant 0)",
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    base_code = hamming_6_3_3_shortened()
    perm_total = len(variants_6_3_3())
    a_perm_indices = _choose_permutations(perm_total, args.permH1, rng)
    b_perm_indices = _choose_permutations(perm_total, args.permH1p, rng)

    results_path = outdir / "candidates.jsonl"
    summary_records: List[Dict[str, object]] = []
    best_by_nk: Dict[Tuple[int, int], Dict[str, object]] = {}
    candidate_counter = 0
    batch_id = 0

    with results_path.open("w", encoding="utf-8") as results_file:
        for m in m_list:
            n_est = 36 * m
            if n_est > args.max_n:
                print(
                    f"[pilot] skipping m={m} (n=36*m={n_est} exceeds --max-n={args.max_n})."
                )
                continue
            if not args.allow_repeats and m < 6:
                print(
                    f"[pilot] skipping m={m} (need m>=6 for 5 distinct nonzero; "
                    "use --allow-repeats to permit repetitions)."
                )
                continue
            min_slice_dist = _slice_dist_threshold(args.min_slice_dist, n=n_est)
            group = FiniteGroup.cyclic(m)
            feasible_limit = 2000
            A_sets = _enumerate_sets(
                m=m,
                max_sets=args.maxA,
                rng=rng,
                feasible_limit=feasible_limit,
                allow_repeats=args.allow_repeats,
            )
            B_sets = _enumerate_sets(
                m=m,
                max_sets=args.maxB,
                rng=rng,
                feasible_limit=feasible_limit,
                allow_repeats=args.allow_repeats,
            )
            if not A_sets or not B_sets:
                print(f"[pilot] no A/B sets for m={m}; skipping.")
                continue

            a_scored: List[SliceCandidate] = []
            for A in A_sets:
                score = _score_a_slice(group, A, base_code, base_code, rng=rng)
                a_scored.append(SliceCandidate(elements=A, score=score))
            b_scored: List[SliceCandidate] = []
            for B in B_sets:
                score = _score_b_slice(group, B, base_code, base_code, rng=rng)
                b_scored.append(SliceCandidate(elements=B, score=score))

            A_keep = _filter_slice_candidates(
                a_scored,
                threshold=min_slice_dist,
                top_n=args.topA,
                explore_n=args.exploreA,
                rng=rng,
                label="A",
            )
            B_keep = _filter_slice_candidates(
                b_scored,
                threshold=min_slice_dist,
                top_n=args.topB,
                explore_n=args.exploreB,
                rng=rng,
                label="B",
            )

            print(
                f"[pilot] m={m} n={n_est} slice_min={min_slice_dist} "
                f"A_sets={len(A_sets)} -> keep {len(A_keep)}; "
                f"B_sets={len(B_sets)} -> keep {len(B_keep)}"
            )

            batch_items: List[Tuple[int, str, str, dict]] = []
            for A_info in A_keep:
                for B_info in B_keep:
                    for a1v in a_perm_indices:
                        for b1v in b_perm_indices:
                            candidate_id = f"m{m}_c{candidate_counter:05d}"
                            candidate_counter += 1
                            C1 = _apply_variant(base_code, a1v)
                            C1p = _apply_variant(base_code, b1v)
                            hx_rows, hz_rows, n_cols = build_hx_hz(
                                group,
                                A_info.elements,
                                B_info.elements,
                                base_code,
                                C1,
                                base_code,
                                C1p,
                            )
                            if not css_commutes(hx_rows, hz_rows):
                                raise RuntimeError(
                                    f"HX/HZ do not commute for {candidate_id}."
                                )
                            rank_hx = gf2_rank(hx_rows[:], n_cols)
                            rank_hz = gf2_rank(hz_rows[:], n_cols)
                            k = n_cols - rank_hx - rank_hz
                            entry = {
                                "candidate_id": candidate_id,
                                "group": {"type": "cyclic", "order": m},
                                "A": list(A_info.elements),
                                "B": list(B_info.elements),
                                "local_codes": {
                                    "C0": base_code.name,
                                    "C1": C1.name,
                                    "C0p": base_code.name,
                                    "C1p": C1p.name,
                                    "a1v": a1v,
                                    "b1v": b1v,
                                },
                                "slice_scores": {
                                    "A": A_info.score.as_dict(),
                                    "B": B_info.score.as_dict(),
                                },
                                "n": n_cols,
                                "k": k,
                                "seed": seed,
                                "column_order": "col = ((i*nB + j)*|G| + g), g fastest",
                            }
                            if k == 0:
                                entry["skipped_reason"] = "k=0"
                                results_file.write(
                                    json.dumps(entry, sort_keys=True) + "\n"
                                )
                                results_file.flush()
                                continue

                            cand_dir = tmp_root / candidate_id
                            cand_dir.mkdir(parents=True, exist_ok=True)
                            hx_path = cand_dir / "Hx.mtx"
                            hz_path = cand_dir / "Hz.mtx"
                            write_mtx_from_bitrows(str(hx_path), hx_rows, n_cols)
                            write_mtx_from_bitrows(str(hz_path), hz_rows, n_cols)

                            idx = len(batch_items)
                            batch_items.append((idx, str(hx_path), str(hz_path), entry))

                            if len(batch_items) >= args.batch_size:
                                batch_seed = (seed + batch_id) % (1 << 31)
                                qd_results, runtime_sec = _run_gap_batch(
                                    batch=[(i, hx, hz) for i, hx, hz, _ in batch_items],
                                    num=args.trials,
                                    mindist=args.mindist,
                                    debug=args.qd_debug,
                                    seed=batch_seed,
                                    outdir=outdir,
                                    batch_id=batch_id,
                                    gap_cmd=args.gap_cmd,
                                    timeout_sec=args.qd_timeout,
                                )
                                for idx_key, _, _, entry in batch_items:
                                    qd_stats = qd_results[idx_key]
                                    dx_signed = int(qd_stats["dx_signed"])
                                    dz_signed = int(qd_stats["dz_signed"])
                                    dx = abs(dx_signed)
                                    dz = abs(dz_signed)
                                    d_ub = min(dx, dz)
                                    qd_x = _qd_side_stats(
                                        d_signed=dx_signed,
                                        rounds_done=int(qd_stats["rx"]),
                                        vec_count_total=int(qd_stats["vx"]),
                                        mult=list(qd_stats["mx"]),
                                    )
                                    qd_z = _qd_side_stats(
                                        d_signed=dz_signed,
                                        rounds_done=int(qd_stats["rz"]),
                                        vec_count_total=int(qd_stats["vz"]),
                                        mult=list(qd_stats["mz"]),
                                    )
                                    _record_best_so_far(
                                        best_by_nk,
                                        entry=entry,
                                        dx=dx,
                                        dz=dz,
                                        d=d_ub,
                                        trials_requested=args.trials,
                                        qd_x=qd_x,
                                        qd_z=qd_z,
                                    )
                                    target = math.isqrt(entry["n"])
                                    promising = d_ub >= target and entry["k"] * d_ub >= entry["n"]
                                    qd = {
                                        "trials_requested": args.trials,
                                        "mindist": args.mindist,
                                        "dx": dx,
                                        "dz": dz,
                                        "d": d_ub,
                                        "d_ub": d_ub,
                                        "qd_x": qd_x,
                                        "qd_z": qd_z,
                                        "seed": batch_seed,
                                        "runtime_sec": runtime_sec,
                                        "gap_cmd": args.gap_cmd,
                                    }
                                    entry["qdistrnd"] = qd
                                    entry["promising"] = promising
                                    entry["saved_path"] = None

                                    cand_dir = tmp_root / entry["candidate_id"]
                                    hx_path = cand_dir / "Hx.mtx"
                                    hz_path = cand_dir / "Hz.mtx"
                                    meta = {
                                        "group": entry["group"],
                                        "A": entry["A"],
                                        "B": entry["B"],
                                        "local_codes": entry["local_codes"],
                                        "n": entry["n"],
                                        "k": entry["k"],
                                        "distance_estimate": {
                                            "method": "QDistRnd",
                                            "trials_requested": args.trials,
                                            "mindist": args.mindist,
                                            "rng_seed": batch_seed,
                                            "dx": dx,
                                            "dz": dz,
                                            "d": d_ub,
                                            "d_ub": d_ub,
                                            "qd_x": qd_x,
                                            "qd_z": qd_z,
                                        },
                                        "slice_scores": entry["slice_scores"],
                                    }

                                    if promising:
                                        out_path = promising_root / entry["candidate_id"]
                                        out_path.mkdir(parents=True, exist_ok=True)
                                        hx_path.replace(out_path / "Hx.mtx")
                                        hz_path.replace(out_path / "Hz.mtx")
                                        (out_path / "meta.json").write_text(
                                            json.dumps(meta, indent=2, sort_keys=True),
                                            encoding="utf-8",
                                        )
                                        entry["saved_path"] = str(out_path)
                                    else:
                                        if hx_path.exists():
                                            hx_path.unlink()
                                        if hz_path.exists():
                                            hz_path.unlink()
                                    if cand_dir.exists():
                                        try:
                                            cand_dir.rmdir()
                                        except OSError:
                                            pass

                                    results_file.write(
                                        json.dumps(entry, sort_keys=True) + "\n"
                                    )
                                    results_file.flush()
                                    summary_records.append(entry)

                                batch_items = []
                                batch_id += 1

            if batch_items:
                batch_seed = (seed + batch_id) % (1 << 31)
                qd_results, runtime_sec = _run_gap_batch(
                    batch=[(i, hx, hz) for i, hx, hz, _ in batch_items],
                    num=args.trials,
                    mindist=args.mindist,
                    debug=args.qd_debug,
                    seed=batch_seed,
                    outdir=outdir,
                    batch_id=batch_id,
                    gap_cmd=args.gap_cmd,
                    timeout_sec=args.qd_timeout,
                )
                for idx_key, _, _, entry in batch_items:
                    qd_stats = qd_results[idx_key]
                    dx_signed = int(qd_stats["dx_signed"])
                    dz_signed = int(qd_stats["dz_signed"])
                    dx = abs(dx_signed)
                    dz = abs(dz_signed)
                    d_ub = min(dx, dz)
                    qd_x = _qd_side_stats(
                        d_signed=dx_signed,
                        rounds_done=int(qd_stats["rx"]),
                        vec_count_total=int(qd_stats["vx"]),
                        mult=list(qd_stats["mx"]),
                    )
                    qd_z = _qd_side_stats(
                        d_signed=dz_signed,
                        rounds_done=int(qd_stats["rz"]),
                        vec_count_total=int(qd_stats["vz"]),
                        mult=list(qd_stats["mz"]),
                    )
                    _record_best_so_far(
                        best_by_nk,
                        entry=entry,
                        dx=dx,
                        dz=dz,
                        d=d_ub,
                        trials_requested=args.trials,
                        qd_x=qd_x,
                        qd_z=qd_z,
                    )
                    target = math.isqrt(entry["n"])
                    promising = d_ub >= target and entry["k"] * d_ub >= entry["n"]
                    qd = {
                        "trials_requested": args.trials,
                        "mindist": args.mindist,
                        "dx": dx,
                        "dz": dz,
                        "d": d_ub,
                        "d_ub": d_ub,
                        "qd_x": qd_x,
                        "qd_z": qd_z,
                        "seed": batch_seed,
                        "runtime_sec": runtime_sec,
                        "gap_cmd": args.gap_cmd,
                    }
                    entry["qdistrnd"] = qd
                    entry["promising"] = promising
                    entry["saved_path"] = None

                    cand_dir = tmp_root / entry["candidate_id"]
                    hx_path = cand_dir / "Hx.mtx"
                    hz_path = cand_dir / "Hz.mtx"
                    meta = {
                        "group": entry["group"],
                        "A": entry["A"],
                        "B": entry["B"],
                        "local_codes": entry["local_codes"],
                        "n": entry["n"],
                        "k": entry["k"],
                        "distance_estimate": {
                            "method": "QDistRnd",
                            "trials_requested": args.trials,
                            "mindist": args.mindist,
                            "rng_seed": batch_seed,
                            "dx": dx,
                            "dz": dz,
                            "d": d_ub,
                            "d_ub": d_ub,
                            "qd_x": qd_x,
                            "qd_z": qd_z,
                        },
                        "slice_scores": entry["slice_scores"],
                    }

                    if promising:
                        out_path = promising_root / entry["candidate_id"]
                        out_path.mkdir(parents=True, exist_ok=True)
                        hx_path.replace(out_path / "Hx.mtx")
                        hz_path.replace(out_path / "Hz.mtx")
                        (out_path / "meta.json").write_text(
                            json.dumps(meta, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        entry["saved_path"] = str(out_path)
                    else:
                        if hx_path.exists():
                            hx_path.unlink()
                        if hz_path.exists():
                            hz_path.unlink()
                    if cand_dir.exists():
                        try:
                            cand_dir.rmdir()
                        except OSError:
                            pass

                    results_file.write(json.dumps(entry, sort_keys=True) + "\n")
                    results_file.flush()
                    summary_records.append(entry)

                batch_id += 1

    if tmp_root.exists():
        try:
            tmp_root.rmdir()
        except OSError:
            pass

    ranked = [
        rec
        for rec in summary_records
        if rec.get("qdistrnd") and rec["qdistrnd"].get("d_ub") is not None
    ]
    ranked.sort(
        key=lambda rec: (
            rec["qdistrnd"]["d_ub"],
            rec["k"] * rec["qdistrnd"]["d_ub"],
        ),
        reverse=True,
    )
    print("[pilot] top results:")
    for idx, rec in enumerate(ranked[:10], start=1):
        qd = rec["qdistrnd"]
        print(
            f"  {idx:02d} m={rec['group']['order']} n={rec['n']} k={rec['k']} "
            f"d_ub={qd['d_ub']} a1v={rec['local_codes']['a1v']} "
            f"b1v={rec['local_codes']['b1v']} saved={bool(rec['saved_path'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
