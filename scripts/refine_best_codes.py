#!/usr/bin/env python3
"""
Refine (re-estimate) distances for all codes in the "best codes" folder.

What it does:
- Finds pairs of MTX files describing CSS codes:
    * ...X.mtx + ...Z.mtx   OR
    * ...Hx.mtx + ...Hz.mtx
- For each code, runs GAP+QDistRnd DistRandCSS with a large number of trials.
- Appends results to: results/refined_distances.csv
- Generates a website-friendly Markdown table: docs/refined_distances.md

Notes:
- DistRandCSS returns an upper bound (a randomized estimate). More trials can
  keep the same value or decrease it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, NamedTuple


class CodePair(NamedTuple):
    code_id: str
    hx: Path
    hz: Path


def repo_root() -> Path:
    # scripts/ is one level below repo root
    return Path(__file__).resolve().parents[1]


def pick_best_dir(root: Path, best_dir_arg: Optional[str]) -> Path:
    if best_dir_arg:
        p = Path(best_dir_arg)
        return (root / p).resolve() if not p.is_absolute() else p

    # common folder names
    for name in ["best_codes", "best-codes", "best codes"]:
        p = root / name
        if p.exists() and p.is_dir():
            return p

    # default
    return root / "best_codes"


def _candidate_with_suffix(dirpath: Path, base: str, suffixes: list[str]) -> Optional[Path]:
    # Try several naming variants (case differences are common)
    for suf in suffixes:
        p = dirpath / f"{base}{suf}.mtx"
        if p.exists():
            return p
    return None


def find_code_pairs(best_dir: Path) -> list[CodePair]:
    """
    Find code pairs by looking for *X.mtx (or *Hx.mtx) and matching *Z.mtx (or *Hz.mtx).
    We only start from the X/Hx file to avoid duplicates.
    """
    pairs: dict[str, CodePair] = {}

    for x_path in best_dir.rglob("*.mtx"):
        if not x_path.is_file():
            continue

        stem = x_path.stem
        low = stem.lower()

        base = None
        hz_path = None

        if low.endswith("hx"):
            base = stem[:-2]
            hz_path = _candidate_with_suffix(
                x_path.parent,
                base,
                suffixes=["Hz", "hz", "HZ"],
            )
        elif low.endswith("x"):
            base = stem[:-1]
            hz_path = _candidate_with_suffix(
                x_path.parent,
                base,
                suffixes=["Z", "z"],
            )
        else:
            continue

        if base is None or hz_path is None:
            continue

        rel = x_path.relative_to(best_dir)
        # code_id is a stable relative identifier (folder + base name)
        # Skip entries that lead to an empty base name (e.g. hidden files like .git)
        if not base:
            continue
        code_id = rel.with_name(base).as_posix()

        pairs[code_id] = CodePair(code_id=code_id, hx=x_path, hz=hz_path)

    return sorted(pairs.values(), key=lambda c: c.code_id)


def gap_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def run_gap_qdistrnd(
    hx: Path,
    hz: Path,
    steps: int,
    mindist: int,
    gap_cmd: str,
    timeout: Optional[int],
) -> dict:
    hx_s = gap_escape(str(hx))
    hz_s = gap_escape(str(hz))

    # We print exactly one machine-readable line starting with "__JSON__ "
    gap_script = f"""
if LoadPackage("QDistRnd") = fail then
  Print("__JSON__ {{\\"error\\":\\"Could not load QDistRnd\\"}}\\n");
  QUIT;
fi;

Hx := ReadMTXE("{hx_s}")[3];
Hz := ReadMTXE("{hz_s}")[3];

dims := DimensionsMat(Hx);
n := dims[2];

dZ := DistRandCSS(Hx, Hz, {steps}, {mindist}, 0);
dX := DistRandCSS(Hz, Hx, {steps}, {mindist}, 0);
dMin := Minimum(dX, dZ);

Print("__JSON__ {{\\"n\\":", n, ",\\"dX\\":", dX, ",\\"dZ\\":", dZ, ",\\"d\\":", dMin, "}}\\n");
QUIT;
"""

    proc = subprocess.run(
        [gap_cmd, "-q"],
        input=gap_script,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    result: dict = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    json_line = None
    for ln in proc.stdout.splitlines():
        if ln.startswith("__JSON__ "):
            json_line = ln[len("__JSON__ "):].strip()
            break

    if json_line is None:
        result["error"] = "No __JSON__ line found in GAP output."
        return result

    try:
        result.update(json.loads(json_line))
    except json.JSONDecodeError:
        result["error"] = f"Could not parse JSON: {json_line}"

    return result


def read_existing_done(csv_path: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not csv_path.exists():
        return done

    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("status") != "ok":
                continue
            try:
                done.add((row["code_id"], int(row["steps"]), int(row["mindist"])))
            except Exception:
                continue
    return done


def append_csv_row(csv_path: Path, fieldnames: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            wr.writeheader()
        wr.writerow(row)


def generate_markdown(csv_path: Path, md_path: Path, steps: int, mindist: int) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("status") != "ok":
                    continue
                if int(r.get("steps", -1)) != steps:
                    continue
                if int(r.get("mindist", -1)) != mindist:
                    continue
                rows.append(r)

    # keep latest per code_id (by timestamp string)
    latest: dict[str, dict] = {}
    for r in rows:
        code_id = r["code_id"]
        if (code_id not in latest) or (r["timestamp_utc"] > latest[code_id]["timestamp_utc"]):
            latest[code_id] = r

    table = list(latest.values())
    table.sort(key=lambda r: (-int(r["d"]), r["code_id"]))

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    with md_path.open("w") as f:
        f.write("# Refined distance estimates\n\n")
        f.write("This page is **auto-generated** by `scripts/refine_best_codes.py`.\n\n")
        f.write(f"- Generated: **{now}**\n")
        f.write(f"- Trials (steps): **{steps}**\n")
        f.write(f"- mindist: **{mindist}** (0 means “run normally”, no early-stop)\n\n")
        f.write("**Interpretation:** these are *randomized upper bounds* (estimates). With more trials, values can stay the same or decrease.\n\n")
        f.write("| code_id | n | dX_ub | dZ_ub | d_ub=min | last_run_utc |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in table:
            f.write(f"| `{r['code_id']}` | {r['n']} | {r['dX']} | {r['dZ']} | **{r['d']}** | {r['timestamp_utc']} |\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default=None, help="Folder containing best codes (default: best_codes/ or best-codes/)")
    ap.add_argument("--steps", type=int, default=500_000, help="Number of trials/steps for DistRandCSS (default: 500000)")
    ap.add_argument("--mindist", type=int, default=0, help="Early-stop threshold (default: 0 means no early stop)")
    ap.add_argument("--gap", default="gap", help="GAP executable (default: gap)")
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) per code")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N codes")
    ap.add_argument("--pattern", default=None, help="Only process code_id containing this substring")
    ap.add_argument("--dry-run", action="store_true", help="Only list discovered codes, do not run anything")
    ap.add_argument("--force", action="store_true", help="Re-run even if an identical (code_id,steps,mindist) entry exists")
    ap.add_argument("--out-csv", default="results/refined_distances.csv", help="CSV log output path")
    ap.add_argument("--out-md", default="docs/refined_distances.md", help="Markdown output path (website)")

    args = ap.parse_args()

    root = repo_root()
    best_dir = pick_best_dir(root, args.best_dir)

    if not best_dir.exists():
        print(f"[error] best dir not found: {best_dir}", file=sys.stderr)
        print("        Use --best-dir to point to your folder (e.g. --best-dir 'best_codes')", file=sys.stderr)
        return 2

    codes = find_code_pairs(best_dir)
    if args.pattern:
        codes = [c for c in codes if args.pattern in c.code_id]
    if args.limit is not None:
        codes = codes[: args.limit]

    print(f"[info] best_dir = {best_dir}")
    print(f"[info] discovered {len(codes)} code(s)")

    if args.dry_run:
        for c in codes:
            print(f"  - {c.code_id}")
            print(f"      Hx: {c.hx}")
            print(f"      Hz: {c.hz}")
        return 0

    out_csv = (root / args.out_csv).resolve()
    out_md = (root / args.out_md).resolve()

    done = set()
    if not args.force:
        done = read_existing_done(out_csv)

    fieldnames = [
        "timestamp_utc",
        "code_id",
        "hx_path",
        "hz_path",
        "n",
        "steps",
        "mindist",
        "dX",
        "dZ",
        "d",
        "status",
        "error",
    ]

    for idx, c in enumerate(codes, start=1):
        key = (c.code_id, args.steps, args.mindist)
        if (not args.force) and (key in done):
            print(f"[skip] ({idx}/{len(codes)}) {c.code_id} already has steps={args.steps} mindist={args.mindist}")
            continue

        print(f"[run ] ({idx}/{len(codes)}) {c.code_id}  steps={args.steps} mindist={args.mindist}")

        timestamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        res = {}
        status = "ok"
        err = ""

        try:
            res = run_gap_qdistrnd(c.hx, c.hz, args.steps, args.mindist, args.gap, args.timeout)
            if "error" in res or res.get("returncode", 0) != 0:
                status = "error"
                err = res.get("error", f"gap returned code {res.get('returncode')}")
        except FileNotFoundError:
            status = "error"
            err = f"GAP executable not found: {args.gap}"
        except subprocess.TimeoutExpired:
            status = "error"
            err = "timeout"
        except Exception as e:
            status = "error"
            err = f"exception: {e}"

        row = {
            "timestamp_utc": timestamp,
            "code_id": c.code_id,
            "hx_path": str(c.hx.relative_to(root)) if c.hx.is_relative_to(root) else str(c.hx),
            "hz_path": str(c.hz.relative_to(root)) if c.hz.is_relative_to(root) else str(c.hz),
            "n": res.get("n", ""),
            "steps": args.steps,
            "mindist": args.mindist,
            "dX": res.get("dX", ""),
            "dZ": res.get("dZ", ""),
            "d": res.get("d", ""),
            "status": status,
            "error": err,
        }

        append_csv_row(out_csv, fieldnames, row)

        if status == "ok":
            print(f"      -> n={row['n']} dX_ub={row['dX']} dZ_ub={row['dZ']} d_ub={row['d']}")
        else:
            print(f"      -> ERROR: {err}")

    generate_markdown(out_csv, out_md, steps=args.steps, mindist=args.mindist)
    print(f"[info] wrote CSV: {out_csv}")
    print(f"[info] wrote MD : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
