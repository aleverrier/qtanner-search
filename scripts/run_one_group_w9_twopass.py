#!/usr/bin/env python3
"""
Two-pass search runner for w=9 small-group experiments.

Pass 1: run a coarse search (fast QDistRnd) and rely on run_search_w9_smallgroups.py to write
        candidate directories under data/tmp/search_w9_smallgroups (symlinked to --tmpdir)
        and a "best_by_nk_group_{group}_*.jsonl" under data/results (symlinked to --outdir).

Pass 2: refine the best candidates using more QDistRnd trials via refine_qdistrnd_dir.py.

This wrapper writes enriched JSONL lines containing n,k,d_ub,kd_over_n,d_over_sqrt_n so you
can rank refined results directly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_RE_CAND = re.compile(r"o(?P<order>\d+)_g(?P<gid>\d+)_p(?P<pair>\d+)_a(?P<a1v>\d+)_b(?P<b1v>\d+)$")


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_pythonpath_env(env: Dict[str, str], repo_root: Path) -> Dict[str, str]:
    """
    Ensure subprocesses can import qtanner even if PYTHONPATH wasn't set in the shell.
    """
    env2 = dict(env)
    src = str((repo_root / "src").resolve())
    old = env2.get("PYTHONPATH", "")
    parts = [p for p in old.split(os.pathsep) if p]
    if src not in parts:
        parts.insert(0, src)
    env2["PYTHONPATH"] = os.pathsep.join(parts)
    return env2


def _parse_group(s: str) -> str:
    s = s.strip()
    if not s:
        raise argparse.ArgumentTypeError("group must be non-empty, e.g. '4,2'")
    if "," not in s:
        raise argparse.ArgumentTypeError("group must be 'order,id' e.g. '4,2'")
    a, b = s.split(",", 1)
    if not a.isdigit() or not b.isdigit():
        raise argparse.ArgumentTypeError("group must be 'order,id' with integers e.g. '4,2'")
    return f"{int(a)},{int(b)}"


def _parse_pairs(s: str) -> Tuple[int, int]:
    """
    Accept:
      - "1-200"  -> (1,200)
      - "200"    -> (1,200)
      - "1:200"  -> (1,200)
    """
    s = s.strip()
    if not s:
        raise argparse.ArgumentTypeError("pairs must be like '1-200' or '200'")
    for sep in ("-", ":"):
        if sep in s:
            lo_s, hi_s = s.split(sep, 1)
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if not lo_s.isdigit() or not hi_s.isdigit():
                raise argparse.ArgumentTypeError("pairs must be integers like '1-200'")
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo:
                raise argparse.ArgumentTypeError("pairs must satisfy 1 <= lo <= hi")
            return lo, hi
    if not s.isdigit():
        raise argparse.ArgumentTypeError("pairs must be like '1-200' or '200'")
    hi = int(s)
    if hi < 1:
        raise argparse.ArgumentTypeError("pairs must be >= 1")
    return 1, hi


def _tail(s: str, max_chars: int = 4000) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool = False) -> int:
    print("[twopass] exec:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _find_latest_best_by_nk(outdir: Path, group: str) -> Optional[Path]:
    pat = str(outdir / f"best_by_nk_group_{group.replace(',', '_')}_*.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    files.sort(key=lambda p: (os.path.getmtime(p), p))
    return Path(files[-1])


def _scan_tmp_for_cands(tmpdir: Path) -> List[Path]:
    pat = str(tmpdir / "o*_g*_p*_a*_b*")
    dirs = [Path(p) for p in glob.glob(pat) if os.path.isdir(p)]
    dirs.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return dirs


def _load_best_records(best_path: Path) -> List[dict]:
    recs: List[dict] = []
    with open(best_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _score_record(rec: dict) -> float:
    n = rec.get("n")
    k = rec.get("k")
    d = rec.get("d_obs")
    if d is None:
        d = rec.get("d_ub")
    if d is None:
        d = rec.get("d")
    if isinstance(n, int) and isinstance(k, int) and isinstance(d, int) and n > 0:
        return (k * d) / n
    kd = rec.get("kd_over_n")
    if isinstance(kd, (int, float)):
        return float(kd)
    return 0.0


def _select_cands_for_refine(
    *,
    group: str,
    outdir: Path,
    tmpdir: Path,
    refine_from: str,
    refine_max: int,
) -> Tuple[List[Path], Dict[str, dict], str]:
    best_meta: Dict[str, dict] = {}

    if refine_from == "best_by_nk":
        best_path = _find_latest_best_by_nk(outdir, group)
        if best_path is not None:
            recs = _load_best_records(best_path)
            recs.sort(key=lambda r: (_score_record(r), str(r.get("cand_dir", ""))), reverse=True)
            dirs: List[Path] = []
            seen = set()
            for r in recs:
                cand_dir = r.get("cand_dir")
                if not isinstance(cand_dir, str) or not cand_dir:
                    continue
                p = Path(cand_dir)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if not p.is_dir():
                    continue
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                dirs.append(p)
                best_meta[key] = r
                if len(dirs) >= refine_max:
                    break
            if dirs:
                return dirs, best_meta, f"best_by_nk:{best_path.name}"

    dirs = _scan_tmp_for_cands(tmpdir)
    if refine_max > 0:
        dirs = dirs[-refine_max:]
    return dirs, best_meta, "scan_tmp"


def _parse_qdist_line(line: str) -> Optional[dict]:
    if not line.startswith("QDISTRESULT"):
        return None
    parts = line.split()
    out: Dict[str, object] = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k in {"dX", "dZ", "d", "qd_num", "mindist"}:
            try:
                out[k] = int(v)
            except ValueError:
                pass
        else:
            out[k] = v
    if "d" not in out:
        return None
    return out  # type: ignore[return-value]


def _read_mtx_bitrows(path: Path) -> Tuple[int, int, List[int]]:
    raw = path.read_text().splitlines()
    filtered: List[str] = []
    for ln in raw:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("%"):
            continue
        filtered.append(ln)

    if not filtered:
        return 0, 0, []

    head = filtered[0].split()
    idx = 1
    if len(head) == 3:
        m, n, _nnz = map(int, head)
    elif len(head) == 2:
        m, n = map(int, head)
        if idx < len(filtered) and len(filtered[idx].split()) == 1:
            idx += 1
    else:
        raise ValueError(f"Unrecognized MTX header in {path}: {filtered[0]}")

    rowbits = [0] * m
    for ln in filtered[idx:]:
        ps = ln.split()
        if len(ps) < 2:
            continue
        i = int(ps[0]) - 1
        j = int(ps[1]) - 1
        val = 1
        if len(ps) >= 3:
            try:
                val = int(float(ps[2]))
            except ValueError:
                val = 1
        if val & 1:
            if 0 <= i < m and 0 <= j < n:
                rowbits[i] ^= (1 << j)

    return m, n, rowbits


def _rank_gf2_from_bitrows(rowbits: List[int]) -> int:
    basis: Dict[int, int] = {}
    rank = 0
    for r in rowbits:
        x = r
        while x:
            p = x.bit_length() - 1
            b = basis.get(p)
            if b is None:
                basis[p] = x
                rank += 1
                break
            x ^= b
    return rank


def _rank_mtx_gf2(path: Path) -> Tuple[int, int, int]:
    m, n, rowbits = _read_mtx_bitrows(path)
    return _rank_gf2_from_bitrows(rowbits), m, n


@dataclass
class RefineResult:
    ok: bool
    rec: dict


def _refine_one(
    *,
    cand_dir: Path,
    refine_script: Path,
    qd_num: int,
    qd_timeout: int,
    repo_root: Path,
    group: str,
    best_meta: Optional[dict],
    dry_run: bool,
) -> RefineResult:
    env = _ensure_pythonpath_env(os.environ, repo_root)
    cmd = [
        sys.executable,
        str(refine_script),
        str(cand_dir),
        "--qd-num",
        str(qd_num),
        "--timeout",
        str(qd_timeout),
    ]

    if dry_run:
        print("[twopass] (dry-run) refine:", " ".join(cmd))
        rc = 0
        out = ""
        err = ""
        qd_line = None
    else:
        p = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = p.communicate()
        rc = p.returncode or 0
        qd_line = None
        if rc == 0:
            for ln in out.splitlines():
                if ln.startswith("QDISTRESULT"):
                    qd_line = ln.strip()
                    break

    cand_name = cand_dir.name
    m = _RE_CAND.match(cand_name)
    meta_from_name = {}
    if m:
        meta_from_name = {k: int(v) for k, v in m.groupdict().items()}

    rec: dict = {
        "timestamp": _now_utc_iso(),
        "group": group,
        "cand_dir": str(cand_dir),
        "cand_name": cand_name,
        **meta_from_name,
        "cmd": cmd,
        "returncode": rc,
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    if best_meta is not None:
        rec["pass1"] = best_meta

    qd = None
    if qd_line:
        qd = _parse_qdist_line(qd_line)
        rec["qd_line"] = qd_line
        if qd is not None:
            rec["qd"] = qd

    try:
        hx_path = cand_dir / "Hx.mtx"
        hz_path = cand_dir / "Hz.mtx"
        if hx_path.is_file() and hz_path.is_file():
            rx, mx, nx = _rank_mtx_gf2(hx_path)
            rz, mz, nz = _rank_mtx_gf2(hz_path)
            if nx != nz:
                rec["n_warning"] = f"Hx cols {nx} != Hz cols {nz}"
            n = nx
            k = n - rx - rz
            rec.update(
                {
                    "n": n,
                    "rank_hx": rx,
                    "rank_hz": rz,
                    "rows_hx": mx,
                    "rows_hz": mz,
                    "k": k,
                }
            )
            if isinstance(qd, dict) and isinstance(qd.get("d"), int) and n > 0:
                d = int(qd["d"])
                rec["d_ub"] = d
                rec["dX_ub"] = int(qd.get("dX", d)) if isinstance(qd.get("dX"), int) else None
                rec["dZ_ub"] = int(qd.get("dZ", d)) if isinstance(qd.get("dZ"), int) else None
                rec["kd_over_n"] = (k * d) / n
                rec["d_over_sqrt_n"] = d / math.sqrt(n)
    except Exception as e:
        rec["nk_error"] = f"{type(e).__name__}: {e}"

    ok = (rc == 0) and (qd is not None)
    return RefineResult(ok=ok, rec=rec)


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    p = argparse.ArgumentParser(description="Two-pass search for w=9 small groups (fast + refine).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--group", type=_parse_group, help="Group as 'order,id' (e.g. 4,2)")
    g.add_argument("--only-group", type=_parse_group, help="Alias for --group (compat)")

    p.add_argument("--pairs", type=_parse_pairs, default=(1, 200), help="Pair range like '1-200' (used as max-pairs)")
    p.add_argument("--qd-fast", type=int, default=100, help="QDistRnd trials for pass 1 (coarse)")
    p.add_argument("--qd-refine", type=int, default=2000, help="QDistRnd trials for refinement (pass 2)")
    p.add_argument("--qd-timeout", type=int, default=240, help="Timeout (seconds) for QDistRnd in pass 1")
    p.add_argument("--qd-refine-timeout", type=int, default=240, help="Timeout (seconds) for QDistRnd in pass 2")
    p.add_argument("--refine-max", type=int, default=50, help="Maximum number of candidate dirs to refine")
    p.add_argument("--top-per-nk", type=int, default=20, help="(Passed to pass1 wrapper; may be ignored)")
    p.add_argument("--tmpdir", type=str, default="data/tmp/search_w9_smallgroups", help="TMP root for candidates")
    p.add_argument("--outdir", type=str, default="data/results", help="Results directory root")
    p.add_argument("--refine-from", choices=["best_by_nk", "scan_tmp"], default="best_by_nk", help="How to select candidates")
    p.add_argument("--no-pass1", action="store_true", help="Skip pass 1 and only refine existing candidates")
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    args, extra = p.parse_known_args(argv)

    group = args.group or args.only_group
    assert group is not None
    lo, hi = args.pairs

    tmpdir = Path(args.tmpdir).resolve()
    outdir = Path(args.outdir).resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    run_one_group = scripts_dir / "run_one_group_w9.py"
    refine_script = scripts_dir / "refine_qdistrnd_dir.py"

    if not run_one_group.is_file():
        print(f"[twopass] ERROR: missing {run_one_group}", file=sys.stderr)
        return 2
    if not refine_script.is_file():
        print(f"[twopass] ERROR: missing {refine_script}", file=sys.stderr)
        return 2

    env = _ensure_pythonpath_env(os.environ, repo_root)

    if not args.no_pass1:
        pass1_cmd = [
            sys.executable,
            str(run_one_group),
            "--group",
            group,
            "--pairs",
            f"{lo}-{hi}",
            "--tmpdir",
            str(tmpdir),
            "--outdir",
            str(outdir),
            "--qd-num",
            str(args.qd_fast),
            "--qd-batches",
            "1,1",
            "--qd-timeout",
            str(args.qd_timeout),
            "--top-per-nk",
            str(args.top_per_nk),
        ] + extra

        print("[twopass] Pass 1 cmd:", " ".join(pass1_cmd), flush=True)
        rc1 = _run(pass1_cmd, cwd=repo_root, env=env, dry_run=args.dry_run)
        if rc1 != 0:
            print(f"[twopass] Pass 1 exited with code {rc1}. Continuing to refinement anyway.", flush=True)

    cand_dirs, best_meta_by_dir, source = _select_cands_for_refine(
        group=group,
        outdir=outdir,
        tmpdir=tmpdir,
        refine_from=args.refine_from,
        refine_max=args.refine_max,
    )
    print(f"[twopass] Selected {len(cand_dirs)} dirs for refinement via {source}.", flush=True)

    stamp = _now_utc_iso().replace(":", "").replace("-", "")
    out_path = outdir / f"refine_w9_twopass_{group.replace(',', '_')}_{stamp}.jsonl"

    ok = 0
    rows_for_print: List[dict] = []
    with open(out_path, "w") as out_f:
        for cand_dir in cand_dirs:
            bm = best_meta_by_dir.get(str(cand_dir))
            rr = _refine_one(
                cand_dir=cand_dir,
                refine_script=refine_script,
                qd_num=args.qd_refine,
                qd_timeout=args.qd_refine_timeout,
                repo_root=repo_root,
                group=group,
                best_meta=bm,
                dry_run=args.dry_run,
            )
            out_f.write(json.dumps(rr.rec, sort_keys=True) + "\n")
            out_f.flush()
            if rr.ok:
                ok += 1
            rows_for_print.append(rr.rec)

    print(f"[twopass] refine ok={ok}/{len(cand_dirs)}", flush=True)
    print(f"[twopass] wrote {out_path}", flush=True)

    scored = []
    for r in rows_for_print:
        kd = r.get("kd_over_n")
        if isinstance(kd, (int, float)):
            scored.append((float(kd), r))
    scored.sort(key=lambda t: t[0], reverse=True)

    if scored:
        print("\n[twopass] Top 20 refined by kd/n:", flush=True)
        for kd, r in scored[:20]:
            n = r.get("n")
            k = r.get("k")
            d = r.get("d_ub")
            ds = r.get("d_over_sqrt_n")
            cand = r.get("cand_name")
            a1v = r.get("a1v")
            b1v = r.get("b1v")
            pair = r.get("pair")
            print(
                f"  kd/n={kd:.3f}  d/sqrt(n)={ds:.3f}  n={n} k={k} d_ub={d}  pair={pair} a1v={a1v} b1v={b1v}  {cand}",
                flush=True,
            )
    else:
        print("[twopass] No refined records contained kd_over_n (maybe QDistRnd failed or matrices missing).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
