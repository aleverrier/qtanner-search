from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_CODE_ID_D_RE = re.compile(r"_d(\d+)(?:_|$)")
_CODE_ID_K_RE = re.compile(r"_k(\d+)(?:_|$)")

_SKIP_DIR_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules"}

_SUMMARY_JSON_NAMES = {
    "best_by_nk.json",
    "best_by_n.json",
    "best_current.json",
}

_SUMMARY_JSONL_PATTERNS = (
    "milestones.jsonl",
    "recheck",
    "best",
    "report",
)

_MAX_JSONL_BYTES = 5_000_000

_HISTORY_STAGING_ROOT = Path(tempfile.gettempdir()) / "qtanner_best_history"


@dataclass
class CodeRecord:
    code_id: str
    n: Optional[int]
    k: Optional[int]
    d: Optional[int]
    trials: Optional[int]
    group: Optional[str] = None
    source_kind: str = ""
    source_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    code_dir: Optional[Path] = None
    hx_path: Optional[Path] = None
    hz_path: Optional[Path] = None
    meta: Optional[Dict[str, Any]] = None
    history_meta_commit: Optional[str] = None
    history_meta_relpath: Optional[str] = None
    history_hx_commit: Optional[str] = None
    history_hx_relpath: Optional[str] = None
    history_hz_commit: Optional[str] = None
    history_hz_relpath: Optional[str] = None
    issues: List[str] = field(default_factory=list)

    def has_artifacts(self) -> bool:
        return self.meta_path is not None and self.hx_path is not None and self.hz_path is not None


class GitNonFastForwardError(RuntimeError):
    pass


@dataclass
class BestCodesUpdateResult:
    records: List[CodeRecord]
    selected: Dict[Tuple[int, int], CodeRecord]
    attempts: int = 1
    committed: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _default_commit_message(context: Optional[str] = None) -> str:
    ts = _utc_now_iso()
    if context:
        return f"best_codes: refresh best-by-nk after {context} ({ts})"
    return f"best_codes: refresh best-by-nk ({ts})"


def _to_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s and re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except ValueError:
                return None
    return None


def _get_nested(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _get_first_int(obj: Any, paths: Iterable[str]) -> Optional[int]:
    for path in paths:
        val = _get_nested(obj, path)
        iv = _to_int(val)
        if iv is not None:
            return iv
    return None


def _parse_code_id_suffix(code_id: str, key_re: re.Pattern[str]) -> Optional[int]:
    m = key_re.search(code_id or "")
    return int(m.group(1)) if m else None


def _group_from_code_id(code_id: str) -> Optional[str]:
    if not code_id:
        return None
    s = str(code_id)
    m = re.match(r"^(SmallGroup[_\(\s]*\d+[, _]+\d+\)?)(?:__|_)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if "_AA" in s:
        return s.split("_AA", 1)[0]
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _extract_group(meta: Optional[Dict[str, Any]], code_id: str) -> Optional[str]:
    if isinstance(meta, dict):
        g = meta.get("group") or meta.get("G")
        if isinstance(g, str):
            return g
        if isinstance(g, dict):
            for key in ("spec", "id", "name", "group"):
                v = g.get(key)
                if isinstance(v, str) and v:
                    return v
    return _group_from_code_id(code_id)


def _extract_distance(meta: Optional[Dict[str, Any]], code_id: str) -> Optional[int]:
    if not isinstance(meta, dict):
        return _parse_code_id_suffix(code_id, _CODE_ID_D_RE)

    d = _get_first_int(meta, ["d_ub", "d", "d_recorded", "d_in_id", "d_obs"])
    if d is not None:
        return d

    d = _get_first_int(meta, ["distance.d_ub", "distance.d", "distance_ub", "distance"])
    if d is not None:
        return d

    dx = _get_first_int(meta, ["distance.dX_best", "distance.dX_ub", "distance.dx_best", "distance.dx_ub", "distance.dX"])
    dz = _get_first_int(meta, ["distance.dZ_best", "distance.dZ_ub", "distance.dz_best", "distance.dz_ub", "distance.dZ"])
    if dx is not None or dz is not None:
        vals = [v for v in (dx, dz) if v is not None]
        return min(vals) if vals else None

    dx = _get_first_int(meta, ["distance.fast.dx.d_ub", "distance.fast.dx.d", "distance.fast.dx.signed"])
    dz = _get_first_int(meta, ["distance.fast.dz.d_ub", "distance.fast.dz.d", "distance.fast.dz.signed"])
    if dx is not None or dz is not None:
        vals = [v for v in (dx, dz) if v is not None]
        return min(vals) if vals else None

    return _parse_code_id_suffix(code_id, _CODE_ID_D_RE)


def _extract_trials(meta: Optional[Dict[str, Any]], run_meta: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if not isinstance(meta, dict):
        if isinstance(run_meta, dict):
            return _get_first_int(run_meta, ["quantum_steps_slow", "quantum_steps_fast", "steps", "trials"])
        return None

    total = _get_first_int(meta, ["distance.steps_used_total", "distance.steps_total", "distance.steps", "distance.trials", "distance.m4ri_steps"])
    sx = _get_first_int(meta, ["distance.steps_used_x", "distance.steps_x"])
    sz = _get_first_int(meta, ["distance.steps_used_z", "distance.steps_z"])
    if total is None and sx is not None and sz is not None:
        total = sx + sz

    if total is None:
        total = _get_first_int(
            meta,
            [
                "m4ri_trials",
                "m4ri_steps",
                "trials",
                "steps",
                "steps_used",
                "steps_used_total",
                "distance_trials",
                "distance_steps",
                "distance.steps_fast",
                "distance.steps_slow",
            ],
        )

    if total is None and isinstance(run_meta, dict):
        total = _get_first_int(run_meta, ["quantum_steps_slow", "quantum_steps_fast", "steps", "trials"])

    return total


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _resolve_artifact_path(repo_root: Path, raw: Any) -> Optional[Path]:
    if raw is None:
        return None
    if not isinstance(raw, (str, Path)):
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _git_run(repo_root: Path, args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo_root), text=True, capture_output=True, check=check)


def _git_log_path_mapping(repo_root: Path, pathspec: str, *, diff_filter: str = "AM", verbose: bool = False) -> Dict[str, str]:
    try:
        res = _git_run(
            repo_root,
            [
                "log",
                "--all",
                f"--diff-filter={diff_filter}",
                "--pretty=format:%H",
                "--name-only",
                "--",
                pathspec,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        _log(verbose, f"[history] git log failed for {pathspec}: {exc.stderr.strip()}")
        return {}

    commit = ""
    mapping: Dict[str, str] = {}
    for raw in res.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[0-9a-f]{40}", line):
            commit = line
            continue
        if commit:
            # First occurrence is the most recent commit where the path exists.
            mapping.setdefault(line, commit)
    return mapping


def _git_show_text(repo_root: Path, commit: str, relpath: str) -> Optional[str]:
    res = _git_run(repo_root, ["show", f"{commit}:{relpath}"], check=False)
    if res.returncode != 0:
        return None
    return res.stdout


def _matrix_kind(name: str) -> Optional[str]:
    lower = name.lower()
    if "hx" in lower or "h_x" in lower:
        return "hx"
    if "hz" in lower or "h_z" in lower:
        return "hz"
    return None


def _strip_matrix_suffix(stem: str) -> str:
    lower = stem.lower()
    suffixes = [
        "__hx",
        "__hz",
        "_hx",
        "_hz",
        "-hx",
        "-hz",
        "__h_x",
        "__h_z",
        "_h_x",
        "_h_z",
    ]
    for suf in suffixes:
        if lower.endswith(suf):
            return stem[: -len(suf)]
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def _code_id_from_matrix_filename(name: str) -> str:
    stem = Path(name).stem
    return _strip_matrix_suffix(stem)


def _build_history_matrix_index(repo_root: Path, *, verbose: bool = False) -> Dict[str, Dict[str, Tuple[str, str]]]:
    mapping = _git_log_path_mapping(repo_root, "best_codes/matrices", diff_filter="AM", verbose=verbose)
    index: Dict[str, Dict[str, Tuple[str, str, bool]]] = {}
    for relpath, commit in mapping.items():
        if not relpath.lower().endswith(".mtx"):
            continue
        name = Path(relpath).name
        kind = _matrix_kind(name)
        if kind is None:
            continue
        code_id = _code_id_from_matrix_filename(name)
        canonical = "__h" in name.lower()
        entry = index.setdefault(code_id, {})
        cur = entry.get(kind)
        if cur is None:
            entry[kind] = (relpath, commit, canonical)
            continue
        # Prefer canonical names when available.
        if canonical and not cur[2]:
            entry[kind] = (relpath, commit, canonical)

    out: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for code_id, kinds in index.items():
        out[code_id] = {}
        for kind, (relpath, commit, _canonical) in kinds.items():
            out[code_id][kind] = (relpath, commit)
    return out


def _is_ignored_path(path: Path) -> bool:
    for part in path.parts:
        if part in _SKIP_DIR_PARTS:
            return True
    return False


def _find_meta_file(code_dir: Path) -> Optional[Path]:
    p = code_dir / "meta.json"
    if p.exists():
        return p
    for cand in sorted(code_dir.glob("*meta*.json")):
        if cand.is_file():
            return cand
    return None


def _find_hx_hz_in_dir(code_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    hx = None
    hz = None
    for p in sorted(code_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".mtx":
            continue
        name = p.name.lower()
        if "hx" in name or "h_x" in name:
            hx = hx or p
        elif "hz" in name or "h_z" in name:
            hz = hz or p
    return hx, hz


def _find_hx_hz_in_matrices(matrices_dir: Path, code_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    if not matrices_dir.exists():
        return None, None
    hx = None
    hz = None
    for p in matrices_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".mtx":
            continue
        if not p.name.startswith(code_id):
            continue
        name = p.name.lower()
        if "hx" in name or "h_x" in name:
            hx = hx or p
        elif "hz" in name or "h_z" in name:
            hz = hz or p
    return hx, hz


def _record_from_code_dir(code_dir: Path, source_kind: str, *, run_meta: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Optional[CodeRecord]:
    meta_path = _find_meta_file(code_dir)
    meta = _safe_load_json(meta_path) if meta_path else None
    code_id = ""
    if isinstance(meta, dict):
        code_id = meta.get("code_id") or meta.get("id") or meta.get("name") or ""
    if not code_id:
        code_id = code_dir.name
    n = _get_first_int(meta, ["n"]) if meta else None
    k = _get_first_int(meta, ["k"]) if meta else None
    if k is None:
        k = _parse_code_id_suffix(code_id, _CODE_ID_K_RE)
    d = _extract_distance(meta, code_id)
    trials = _extract_trials(meta, run_meta)
    group = _extract_group(meta, code_id)

    hx, hz = _find_hx_hz_in_dir(code_dir)
    rec = CodeRecord(
        code_id=code_id,
        n=n,
        k=k,
        d=d,
        trials=trials,
        group=group,
        source_kind=source_kind,
        source_path=code_dir,
        meta_path=meta_path,
        code_dir=code_dir,
        hx_path=hx,
        hz_path=hz,
    )
    if meta_path is None:
        rec.issues.append("missing meta.json")
    if hx is None or hz is None:
        rec.issues.append("missing Hx/Hz .mtx")
    if n is None:
        rec.issues.append("missing n")
    if k is None:
        rec.issues.append("missing k")
    if d is None:
        rec.issues.append("missing d")
    if trials is None:
        rec.issues.append("missing trials")
    if rec.issues:
        _log(verbose, f"[scan] {code_dir}: " + "; ".join(rec.issues))
    return rec


def _record_from_meta_file(meta_path: Path, repo_root: Path, source_kind: str, *, verbose: bool = False) -> Optional[CodeRecord]:
    meta = _safe_load_json(meta_path)
    if meta is None:
        _log(verbose, f"[scan] {meta_path}: invalid JSON")
        return None
    code_id = meta.get("code_id") or meta.get("id") or meta.get("name") or meta_path.stem
    n = _get_first_int(meta, ["n"])
    k = _get_first_int(meta, ["k"])
    if k is None:
        k = _parse_code_id_suffix(code_id, _CODE_ID_K_RE)
    d = _extract_distance(meta, code_id)
    run_meta = meta.get("run_meta") if isinstance(meta.get("run_meta"), dict) else None
    trials = _extract_trials(meta, run_meta)
    group = _extract_group(meta, code_id)

    code_dir = None
    hx = None
    hz = None
    artifacts = meta.get("artifacts") if isinstance(meta.get("artifacts"), dict) else None
    if isinstance(artifacts, dict):
        code_dir = _resolve_artifact_path(repo_root, artifacts.get("code_dir"))
        hx = _resolve_artifact_path(repo_root, artifacts.get("hx_path"))
        hz = _resolve_artifact_path(repo_root, artifacts.get("hz_path"))
        if code_dir is not None and (hx is None or hz is None):
            hx2, hz2 = _find_hx_hz_in_dir(code_dir)
            hx = hx or hx2
            hz = hz or hz2

    collected_dir = meta.get("collected_dir") if isinstance(meta.get("collected_dir"), str) else None
    if code_dir is None and collected_dir:
        cand = repo_root / collected_dir
        if cand.exists():
            code_dir = cand
            hx, hz = _find_hx_hz_in_dir(cand)

    if hx is None or hz is None:
        matrices_dir = repo_root / "best_codes" / "matrices"
        hx2, hz2 = _find_hx_hz_in_matrices(matrices_dir, code_id)
        hx = hx or hx2
        hz = hz or hz2

    rec = CodeRecord(
        code_id=code_id,
        n=n,
        k=k,
        d=d,
        trials=trials,
        group=group,
        source_kind=source_kind,
        source_path=meta_path,
        meta_path=meta_path,
        code_dir=code_dir,
        hx_path=hx,
        hz_path=hz,
    )
    if hx is None or hz is None:
        rec.issues.append("missing Hx/Hz .mtx")
    if n is None:
        rec.issues.append("missing n")
    if k is None:
        rec.issues.append("missing k")
    if d is None:
        rec.issues.append("missing d")
    if trials is None:
        rec.issues.append("missing trials")
    if rec.issues:
        _log(verbose, f"[scan] {meta_path}: " + "; ".join(rec.issues))
    return rec


def _looks_like_code_record(obj: Dict[str, Any]) -> bool:
    if "code_id" in obj or "candidate_id" in obj:
        return True
    if "n" in obj and "k" in obj:
        return True
    return False


def _iter_records_from_json(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if _looks_like_code_record(obj):
            yield obj
        else:
            for v in obj.values():
                yield from _iter_records_from_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_records_from_json(v)


def _record_from_summary_dict(row: Dict[str, Any], source_path: Path, source_kind: str) -> CodeRecord:
    code_id = row.get("code_id") or row.get("candidate_id") or row.get("id") or row.get("name") or ""
    n = _get_first_int(row, ["n"])
    k = _get_first_int(row, ["k"])
    d = _get_first_int(row, ["d_ub", "d", "d_obs", "distance", "distance_ub"])
    trials = _get_first_int(row, ["trials", "trials_used", "steps", "eval", "evals"])
    group = None
    g = row.get("group")
    if isinstance(g, dict):
        group = g.get("spec") or g.get("name")
        if not group and "order" in g and "gid" in g:
            group = f"SmallGroup_{g['order']}_{g['gid']}"
    elif isinstance(g, str):
        group = g
    if not group:
        group = _group_from_code_id(code_id)
    return CodeRecord(
        code_id=code_id or f"{source_path.name}#row",
        n=n,
        k=k,
        d=d,
        trials=trials,
        group=group,
        source_kind=source_kind,
        source_path=source_path,
        meta_path=None,
        code_dir=None,
        hx_path=None,
        hz_path=None,
        issues=["summary-only"],
    )


def scan_all_codes(repo_root: Path, *, verbose: bool = False, include_git_history: bool = True) -> List[CodeRecord]:
    root = Path(repo_root)
    records: List[CodeRecord] = []

    # 1) Scan run-level best_codes directories (excluding repo_root/best_codes).
    for best_dir in sorted(root.rglob("best_codes")):
        if _is_ignored_path(best_dir):
            continue
        if best_dir == root / "best_codes":
            continue
        if not best_dir.is_dir():
            continue
        run_meta = _safe_load_json(best_dir.parent / "run_meta.json")
        for code_dir in sorted(best_dir.iterdir()):
            if not code_dir.is_dir():
                continue
            rec = _record_from_code_dir(code_dir, "run_best_codes", run_meta=run_meta, verbose=verbose)
            if rec:
                records.append(rec)

    # 2) Scan best_codes/collected
    collected_dir = root / "best_codes" / "collected"
    if collected_dir.exists():
        for code_dir in sorted(collected_dir.iterdir()):
            if not code_dir.is_dir():
                continue
            rec = _record_from_code_dir(code_dir, "best_codes_collected", verbose=verbose)
            if rec:
                records.append(rec)

    # 3) Scan best_codes/meta
    meta_dir = root / "best_codes" / "meta"
    if meta_dir.exists():
        for meta_path in sorted(meta_dir.glob("*.json")):
            rec = _record_from_meta_file(meta_path, root, "best_codes_meta", verbose=verbose)
            if rec:
                records.append(rec)

    # 3b) Scan new_best artifacts in codes/pending
    pending_dir = root / "codes" / "pending"
    if pending_dir.exists():
        for meta_path in sorted(pending_dir.rglob("*.json")):
            if not meta_path.is_file():
                continue
            rec = _record_from_meta_file(meta_path, root, "pending_json", verbose=verbose)
            if rec:
                records.append(rec)

    # 4) Scan summary JSON files
    for name in _SUMMARY_JSON_NAMES:
        p = root / "results" / name
        if not p.exists():
            continue
        data = _safe_load_json(p)
        if data is None:
            _log(verbose, f"[scan] {p}: invalid JSON")
            continue
        for row in _iter_records_from_json(data):
            records.append(_record_from_summary_dict(row, p, "summary_json"))

    # 5) Scan selected JSONL logs
    for path in sorted(root.rglob("*.jsonl")):
        if _is_ignored_path(path):
            continue
        name = path.name
        if path.stat().st_size > _MAX_JSONL_BYTES:
            continue
        if not any(tok in name for tok in _SUMMARY_JSONL_PATTERNS):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if not _looks_like_code_record(obj):
                        continue
                    records.append(_record_from_summary_dict(obj, path, "summary_jsonl"))
        except Exception as exc:
            _log(verbose, f"[scan] {path}: {exc}")

    # 6) Scan git history for best_codes/meta and best_codes/matrices.
    if include_git_history:
        try:
            (_HISTORY_STAGING_ROOT / "meta").mkdir(parents=True, exist_ok=True)
            (_HISTORY_STAGING_ROOT / "matrices").mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log(verbose, f"[history] staging root unavailable: {exc}")
            return records

        history_meta = _git_log_path_mapping(root, "best_codes/meta", diff_filter="AM", verbose=verbose)
        history_mats = _build_history_matrix_index(root, verbose=verbose)

        for relpath, commit in sorted(history_meta.items()):
            if not relpath.endswith(".json"):
                continue
            meta_text = _git_show_text(root, commit, relpath)
            if not meta_text:
                _log(verbose, f"[history] missing meta {relpath} @ {commit[:12]}")
                continue
            try:
                meta = json.loads(meta_text)
            except Exception:
                _log(verbose, f"[history] invalid meta {relpath} @ {commit[:12]}")
                continue
            if not isinstance(meta, dict):
                continue

            code_id = meta.get("code_id") or meta.get("id") or meta.get("name") or Path(relpath).stem
            n = _get_first_int(meta, ["n"])
            k = _get_first_int(meta, ["k"])
            if k is None:
                k = _parse_code_id_suffix(code_id, _CODE_ID_K_RE)
            d = _extract_distance(meta, code_id)
            trials = _extract_trials(meta, meta.get("run_meta") if isinstance(meta.get("run_meta"), dict) else None)
            group = _extract_group(meta, code_id)

            hx_local, hz_local = _find_hx_hz_in_matrices(root / "best_codes" / "matrices", code_id)
            hx_hist = history_mats.get(code_id, {}).get("hx")
            hz_hist = history_mats.get(code_id, {}).get("hz")

            hx_path: Optional[Path] = hx_local
            hz_path: Optional[Path] = hz_local
            hx_commit = None
            hx_relpath = None
            hz_commit = None
            hz_relpath = None

            if hx_path is None and hx_hist:
                hx_relpath, hx_commit = hx_hist
                hx_path = _HISTORY_STAGING_ROOT / "matrices" / f"{code_id}__Hx.mtx"
            if hz_path is None and hz_hist:
                hz_relpath, hz_commit = hz_hist
                hz_path = _HISTORY_STAGING_ROOT / "matrices" / f"{code_id}__Hz.mtx"

            if hx_path is None or hz_path is None:
                continue

            meta_path = root / relpath if (root / relpath).exists() else (_HISTORY_STAGING_ROOT / "meta" / f"{code_id}.json")

            rec = CodeRecord(
                code_id=code_id,
                n=n,
                k=k,
                d=d,
                trials=trials,
                group=group,
                source_kind="git_history",
                source_path=root / relpath,
                meta_path=meta_path,
                code_dir=None,
                hx_path=hx_path,
                hz_path=hz_path,
                meta=meta,
                history_meta_commit=commit,
                history_meta_relpath=relpath,
                history_hx_commit=hx_commit,
                history_hx_relpath=hx_relpath,
                history_hz_commit=hz_commit,
                history_hz_relpath=hz_relpath,
            )
            records.append(rec)

    return records


def _is_promising(n: int, k: int, d: int) -> bool:
    if n <= 0 or k <= 0 or d <= 0:
        return False
    if d < math.sqrt(n):
        return False
    if k * d < n:
        return False
    return True


def select_best_by_nk(records: List[CodeRecord]) -> Dict[Tuple[int, int], CodeRecord]:
    eligible: Dict[Tuple[int, int], List[CodeRecord]] = {}
    for rec in records:
        if not rec.code_id:
            continue
        if rec.n is None or rec.k is None or rec.d is None:
            continue
        # Drop very small distances everywhere except the tiny n=36 cases.
        if int(rec.n) != 36 and int(rec.d) < 5:
            continue
        if not rec.has_artifacts():
            continue
        key = (int(rec.n), int(rec.k))
        eligible.setdefault(key, []).append(rec)

    selected: Dict[Tuple[int, int], CodeRecord] = {}
    for key, group in eligible.items():
        # Prefer promising codes when any exist for this (n,k),
        # but fall back to the full set to avoid dropping legacy entries.
        promising = [r for r in group if _is_promising(int(r.n), int(r.k), int(r.d))]
        pool = promising if promising else list(group)

        trials_vals = [r.trials for r in pool if isinstance(r.trials, int)]
        max_trials = max(trials_vals) if trials_vals else None
        if max_trials is None:
            cand = list(pool)
        else:
            cand = [r for r in pool if r.trials == max_trials]
        if not cand:
            continue

        def sort_key(r: CodeRecord) -> Tuple[int, str, str]:
            d_val = r.d if isinstance(r.d, int) else -1
            code_id = r.code_id or ""
            src = str(r.source_path) if r.source_path else ""
            return (-d_val, code_id, src)

        cand.sort(key=sort_key)
        selected[key] = cand[0]

    return selected


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    _atomic_write_text(path, text + "\n")


def _ensure_history_meta(rec: CodeRecord, repo_root: Path, *, verbose: bool = False) -> bool:
    if rec.meta_path is None:
        return False
    if rec.meta_path.exists():
        return True
    if rec.meta is not None:
        try:
            _atomic_write_json(rec.meta_path, rec.meta)
            return True
        except Exception as exc:
            _log(verbose, f"[history] failed to write staged meta for {rec.code_id}: {exc}")
            return False
    if rec.history_meta_commit and rec.history_meta_relpath:
        text = _git_show_text(repo_root, rec.history_meta_commit, rec.history_meta_relpath)
        if not text:
            return False
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                _atomic_write_json(rec.meta_path, data)
            else:
                _atomic_write_text(rec.meta_path, text if text.endswith("\n") else text + "\n")
            return True
        except Exception:
            _atomic_write_text(rec.meta_path, text if text.endswith("\n") else text + "\n")
            return True
    return False


def _ensure_history_matrix(rec: CodeRecord, repo_root: Path, which: str, *, verbose: bool = False) -> bool:
    if which not in {"hx", "hz"}:
        raise ValueError(which)
    path = rec.hx_path if which == "hx" else rec.hz_path
    if path is None:
        return False
    if path.exists():
        return True
    commit = rec.history_hx_commit if which == "hx" else rec.history_hz_commit
    relpath = rec.history_hx_relpath if which == "hx" else rec.history_hz_relpath
    if not commit or not relpath:
        return False
    text = _git_show_text(repo_root, commit, relpath)
    if text is None:
        _log(verbose, f"[history] missing {which.upper()} for {rec.code_id} @ {commit[:12]}")
        return False
    try:
        _atomic_write_text(path, text if text.endswith("\n") else text + "\n")
        return True
    except Exception as exc:
        _log(verbose, f"[history] failed to write staged {which.upper()} for {rec.code_id}: {exc}")
        return False


def sync_best_codes_folder(
    selected: Dict[Tuple[int, int], CodeRecord],
    repo_root: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    root = Path(repo_root)
    best_dir = root / "best_codes"
    meta_dir = best_dir / "meta"
    collected_dir = best_dir / "collected"
    matrices_dir = best_dir / "matrices"

    best_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    collected_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = {rec.code_id for rec in selected.values()}

    report = {
        "removed_meta": 0,
        "removed_collected": 0,
        "removed_matrices": 0,
        "copied_meta": 0,
        "copied_collected": 0,
        "copied_matrices": 0,
    }

    # Remove stale meta
    for p in meta_dir.glob("*.json"):
        if p.stem not in selected_ids:
            report["removed_meta"] += 1
            _log(verbose, f"[sync] remove {p}")
            if not dry_run:
                p.unlink()

    # Remove stale collected directories
    for d in collected_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name not in selected_ids:
            report["removed_collected"] += 1
            _log(verbose, f"[sync] remove {d}")
            if not dry_run:
                shutil.rmtree(d)

    # Remove stale matrices
    for p in matrices_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".mtx":
            continue
        keep = False
        for cid in selected_ids:
            if p.name.startswith(f"{cid}__"):
                keep = True
                break
        if not keep:
            report["removed_matrices"] += 1
            _log(verbose, f"[sync] remove {p}")
            if not dry_run:
                p.unlink()

    # Copy selected artifacts
    for rec in sorted(selected.values(), key=lambda r: (r.n or -1, r.k or -1, r.code_id)):
        if not rec.meta_path or not rec.hx_path or not rec.hz_path:
            raise RuntimeError(f"Selected code {rec.code_id} missing artifacts.")

        if not dry_run:
            if not _ensure_history_meta(rec, root, verbose=verbose):
                raise RuntimeError(f"Could not materialize meta for {rec.code_id}.")
            if not _ensure_history_matrix(rec, root, "hx", verbose=verbose):
                raise RuntimeError(f"Could not materialize Hx for {rec.code_id}.")
            if not _ensure_history_matrix(rec, root, "hz", verbose=verbose):
                raise RuntimeError(f"Could not materialize Hz for {rec.code_id}.")

        meta_dst = meta_dir / f"{rec.code_id}.json"
        if meta_dst.resolve() != rec.meta_path.resolve():
            report["copied_meta"] += 1
            _log(verbose, f"[sync] copy {rec.meta_path} -> {meta_dst}")
            if not dry_run:
                shutil.copy2(rec.meta_path, meta_dst)

        # Copy collected directory if we have one
        if rec.code_dir and rec.code_dir.exists():
            dst_dir = collected_dir / rec.code_id
            if rec.code_dir.resolve() != dst_dir.resolve():
                if dst_dir.exists():
                    report["removed_collected"] += 1
                    if not dry_run:
                        shutil.rmtree(dst_dir)
                report["copied_collected"] += 1
                _log(verbose, f"[sync] copytree {rec.code_dir} -> {dst_dir}")
                if not dry_run:
                    shutil.copytree(rec.code_dir, dst_dir)

        # Copy matrices (canonical names)
        hx_dst = matrices_dir / f"{rec.code_id}__Hx.mtx"
        hz_dst = matrices_dir / f"{rec.code_id}__Hz.mtx"
        for src, dst in [(rec.hx_path, hx_dst), (rec.hz_path, hz_dst)]:
            if dst.resolve() == src.resolve():
                continue
            report["copied_matrices"] += 1
            _log(verbose, f"[sync] copy {src} -> {dst}")
            if not dry_run:
                shutil.copy2(src, dst)

    return report


def update_best_codes_webpage_data(
    selected: Dict[Tuple[int, int], CodeRecord],
    repo_root: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    root = Path(repo_root)
    best_dir = root / "best_codes"
    meta_dir = best_dir / "meta"

    codes: List[Dict[str, Any]] = []
    for rec in sorted(selected.values(), key=lambda r: (r.n or -1, r.k or -1, r.code_id)):
        meta_path = meta_dir / f"{rec.code_id}.json"
        meta = _safe_load_json(meta_path)
        if meta is None:
            _log(verbose, f"[web] missing meta for {rec.code_id}")
            continue

        group = _extract_group(meta, rec.code_id) or rec.group
        n = _get_first_int(meta, ["n"]) or rec.n
        k = _get_first_int(meta, ["k"]) or rec.k
        d = _extract_distance(meta, rec.code_id)
        trials = _extract_trials(meta, meta.get("run_meta") if isinstance(meta.get("run_meta"), dict) else None)

        dX = _get_first_int(meta, ["distance.dX_ub", "distance.dX_best", "distance.dx_ub", "distance.dx_best"])
        dZ = _get_first_int(meta, ["distance.dZ_ub", "distance.dZ_best", "distance.dz_ub", "distance.dz_best"])

        codes.append(
            {
                "code_id": rec.code_id,
                "group": group,
                "n": n,
                "k": k,
                "d_ub": d,
                "dX_ub": dX,
                "dZ_ub": dZ,
                "trials": trials,
                "m4ri_trials": trials,
                "steps_used_total": trials,
                "meta": meta,
            }
        )

    out = {
        "generated_at_utc": _utc_now_iso(),
        "total_codes": len(codes),
        "codes": codes,
    }

    report = {"codes": len(codes)}

    if dry_run:
        return report

    _atomic_write_json(best_dir / "data.json", out)

    # index.tsv
    idx_lines = ["group\tn\tk\td_ub\tm4ri_trials\tcode_id"]
    for r in codes:
        idx_lines.append(
            "\t".join(
                [
                    str(r.get("group") or ""),
                    str(r.get("n") or ""),
                    str(r.get("k") or ""),
                    str(r.get("d_ub") if r.get("d_ub") is not None else ""),
                    str(r.get("m4ri_trials") if r.get("m4ri_trials") is not None else ""),
                    r.get("code_id") or "",
                ]
            )
        )
    _atomic_write_text(best_dir / "index.tsv", "\n".join(idx_lines) + "\n")

    # best_by_group_k.tsv
    best_by_group: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in codes:
        g = str(r.get("group") or "")
        k = int(r.get("k")) if isinstance(r.get("k"), int) else None
        if k is None:
            continue
        key = (g, k)
        cur = best_by_group.get(key)
        if cur is None:
            best_by_group[key] = r
            continue
        d_new = r.get("d_ub") if isinstance(r.get("d_ub"), int) else -1
        t_new = r.get("m4ri_trials") if isinstance(r.get("m4ri_trials"), int) else -1
        d_old = cur.get("d_ub") if isinstance(cur.get("d_ub"), int) else -1
        t_old = cur.get("m4ri_trials") if isinstance(cur.get("m4ri_trials"), int) else -1
        if (d_new, t_new) > (d_old, t_old):
            best_by_group[key] = r
        elif (d_new, t_new) == (d_old, t_old):
            if str(r.get("code_id") or "") < str(cur.get("code_id") or ""):
                best_by_group[key] = r

    best_lines = ["group\tk\tn\td_ub\tm4ri_trials\tcode_id"]
    for (g, k), r in sorted(best_by_group.items(), key=lambda x: (x[0][0], x[0][1])):
        best_lines.append(
            "\t".join(
                [
                    g,
                    str(k),
                    str(r.get("n") or ""),
                    str(r.get("d_ub") if r.get("d_ub") is not None else ""),
                    str(r.get("m4ri_trials") if r.get("m4ri_trials") is not None else ""),
                    r.get("code_id") or "",
                ]
            )
        )
    _atomic_write_text(best_dir / "best_by_group_k.tsv", "\n".join(best_lines) + "\n")

    return report


def _run_git(cmd: List[str], repo_root: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, check=check)


def git_pull_rebase(repo_root: Path, *, verbose: bool = False) -> None:
    def _run(cmd: List[str]) -> subprocess.CompletedProcess:
        if verbose:
            print("[git] " + " ".join(cmd))
        return subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)

    branch_res = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch_res.returncode != 0:
        raise subprocess.CalledProcessError(
            branch_res.returncode,
            branch_res.args,
            branch_res.stdout,
            branch_res.stderr,
        )
    branch = (branch_res.stdout or "").strip() or "main"

    fetch_cmd = ["git", "fetch", "origin", branch]
    fetch_res = _run(fetch_cmd)
    if fetch_res.returncode != 0:
        raise subprocess.CalledProcessError(
            fetch_res.returncode,
            fetch_cmd,
            fetch_res.stdout,
            fetch_res.stderr,
        )

    rebase_cmd = ["git", "rebase", "--autostash", f"origin/{branch}"]
    rebase_res = _run(rebase_cmd)
    if rebase_res.returncode != 0:
        raise subprocess.CalledProcessError(
            rebase_res.returncode,
            rebase_cmd,
            rebase_res.stdout,
            rebase_res.stderr,
        )


def git_commit_and_push(repo_root: Path, commit_message: str, retry_on_nonfastforward: bool = True) -> bool:
    root = Path(repo_root)
    _run_git(["git", "add", "best_codes"], root, check=True)

    diff = _run_git(["git", "diff", "--cached", "--quiet"], root, check=False)
    if diff.returncode == 0:
        push = _run_git(["git", "push"], root, check=False)
        if push.returncode != 0:
            stderr = (push.stderr or "").lower()
            if retry_on_nonfastforward and ("non-fast-forward" in stderr or "fetch first" in stderr):
                raise GitNonFastForwardError(push.stderr.strip())
            raise RuntimeError(push.stderr.strip() or "git push failed")
        return False

    _run_git(["git", "commit", "-m", commit_message], root, check=True)

    push = _run_git(["git", "push"], root, check=False)
    if push.returncode != 0:
        stderr = (push.stderr or "").lower()
        if retry_on_nonfastforward and ("non-fast-forward" in stderr or "fetch first" in stderr):
            raise GitNonFastForwardError(push.stderr.strip())
        raise RuntimeError(push.stderr.strip() or "git push failed")
    return True


def run_best_codes_update(
    repo_root: Path,
    *,
    dry_run: bool = False,
    no_git: bool = False,
    no_publish: bool = False,
    verbose: bool = False,
    include_git_history: bool = True,
    max_attempts: int = 3,
    commit_message: Optional[str] = None,
) -> BestCodesUpdateResult:
    """Scan, select, sync, optionally publish, and optionally git-push best codes.

    Git retries reuse the same non-fast-forward handling used by the publish script.
    """
    root = Path(repo_root)
    attempts = 0
    attempts_cap = max(1, int(max_attempts))
    last_records: List[CodeRecord] = []
    last_selected: Dict[Tuple[int, int], CodeRecord] = {}

    while attempts < attempts_cap:
        attempts += 1
        last_records = scan_all_codes(root, verbose=verbose, include_git_history=include_git_history)
        last_selected = select_best_by_nk(last_records)

        if dry_run:
            return BestCodesUpdateResult(
                records=last_records,
                selected=last_selected,
                attempts=attempts,
                committed=False,
            )

        sync_best_codes_folder(last_selected, root, dry_run=False, verbose=verbose)
        if not no_publish:
            update_best_codes_webpage_data(last_selected, root, dry_run=False, verbose=verbose)

        if no_git:
            return BestCodesUpdateResult(
                records=last_records,
                selected=last_selected,
                attempts=attempts,
                committed=False,
            )

        git_pull_rebase(root, verbose=verbose)
        msg = commit_message or _default_commit_message()
        try:
            committed = git_commit_and_push(
                root,
                msg,
                retry_on_nonfastforward=True,
            )
            return BestCodesUpdateResult(
                records=last_records,
                selected=last_selected,
                attempts=attempts,
                committed=bool(committed),
            )
        except GitNonFastForwardError:
            if attempts >= attempts_cap:
                raise
            if verbose:
                print(
                    f"[git] non-fast-forward push; retrying "
                    f"({attempts}/{attempts_cap})"
                )
            git_pull_rebase(root, verbose=verbose)
            # Loop will rescan/reselect after the rebase.

    return BestCodesUpdateResult(
        records=last_records,
        selected=last_selected,
        attempts=attempts,
        committed=False,
    )
