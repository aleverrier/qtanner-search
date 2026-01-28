#!/usr/bin/env python3
"""
Generate best_codes/data.json from best_codes/meta/*.json.

This turns collected metadata into a static website data file.
"""
from __future__ import annotations
from _ensure_python import ensure_minimum_python
ensure_minimum_python()



import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(out)


def git_remote_origin() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    except Exception:
        return None


def git_current_branch() -> str:
    try:
        b = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return b if b else "main"
    except Exception:
        return "main"


def parse_github_owner_repo(remote: str) -> Optional[Tuple[str, str]]:
    remote = remote.strip()
    if remote.startswith("git@github.com:"):
        s = remote[len("git@github.com:"):]
    elif remote.startswith("https://github.com/"):
        s = remote[len("https://github.com/"):]
    elif remote.startswith("ssh://git@github.com/"):
        s = remote[len("ssh://git@github.com/"):]
    else:
        return None

    if s.endswith(".git"):
        s = s[:-4]
    parts = s.split("/")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def gh_blob_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"


def gh_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def classify_matrix_filename(name: str) -> str:
    s = name.lower()
    if "hx" in s or "h_x" in s:
        return "HX"
    if "hz" in s or "h_z" in s:
        return "HZ"
    if "classical" in s:
        return "CLASSICAL"
    return "MTX"


def safe_load_json(fp: Path) -> Optional[Dict[str, Any]]:
    try:
        x = json.loads(fp.read_text(errors="replace"))
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="best_codes/meta", help="Folder with meta JSON files")
    ap.add_argument("--site-dir", default="best_codes", help="Website folder to write data.json into")
    ap.add_argument("--branch", default="", help="Git branch for GitHub links (default: current branch)")
    args = ap.parse_args()

    root = repo_root()
    meta_dir = root / args.meta_dir
    site_dir = root / args.site_dir
    site_dir.mkdir(parents=True, exist_ok=True)

    if not meta_dir.exists():
        raise SystemExit(f"Missing {meta_dir}. Run collect_best_codes.py first.")

    remote = git_remote_origin() or ""
    parsed = parse_github_owner_repo(remote) if remote else None
    owner, repo = parsed if parsed else ("", "")

    branch = args.branch.strip() or git_current_branch()

    codes: List[Dict[str, Any]] = []
    meta_files = sorted(meta_dir.glob("*.json"))
    for mf in meta_files:
        meta = safe_load_json(mf)
        if not meta:
            continue

        code_id = meta.get("code_id") or mf.stem
        group = meta.get("group", "UNKNOWN")
        n = meta.get("n", None)
        k = meta.get("k", None)
        d = meta.get("d_recorded", None)

        run_dir = meta.get("run_dir", "")
        src_dir = meta.get("src_dir", "")
        collected_dir = meta.get("collected_dir", "")

        settings_path = meta.get("settings_path", "")

        matrices_flat = meta.get("matrices_flat", []) or []
        collected_files = meta.get("collected_files", []) or []

        distance_method = meta.get("distance_method", None)
        distance_trials = meta.get("distance_trials", None)
        distance_seed = meta.get("distance_seed", None)

        # GitHub links
        meta_path = f"best_codes/meta/{code_id}.json"
        meta_url_blob = gh_blob_url(owner, repo, branch, meta_path) if owner else ""
        meta_url_raw = gh_raw_url(owner, repo, branch, meta_path) if owner else ""

        src_dir_url_blob = gh_blob_url(owner, repo, branch, src_dir) if owner and src_dir else ""
        collected_dir_url_blob = gh_blob_url(owner, repo, branch, collected_dir) if owner and collected_dir else ""
        settings_url_blob = gh_blob_url(owner, repo, branch, settings_path) if owner and settings_path else ""

        matrices: List[Dict[str, Any]] = []
        for p in matrices_flat:
            filename = Path(p).name
            matrices.append({
                "path": p,
                "filename": filename,
                "kind": classify_matrix_filename(filename),
                "url_blob": gh_blob_url(owner, repo, branch, p) if owner else "",
                "url_raw": gh_raw_url(owner, repo, branch, p) if owner else "",
            })

        extra_files: List[Dict[str, Any]] = []
        for p in collected_files:
            if p.lower().endswith(".mtx"):
                continue
            extra_files.append({
                "path": p,
                "url_blob": gh_blob_url(owner, repo, branch, p) if owner else "",
            })

        codes.append({
            "code_id": code_id,
            "group": group,
            "n": n,
            "k": k,
            "d_recorded": d,
            "d_recorded_kind": meta.get("d_recorded_kind", "recorded"),
            "A_id": meta.get("A_id", ""),
            "B_id": meta.get("B_id", ""),
            "A_elems": meta.get("A_elems", []),
            "B_elems": meta.get("B_elems", []),

            "distance_method": distance_method,
            "distance_trials": distance_trials,
            "distance_seed": distance_seed,

            "run_dir": run_dir,
            "src_dir": src_dir,
            "collected_dir": collected_dir,

            "meta_path": meta_path,
            "meta_url_blob": meta_url_blob,
            "meta_url_raw": meta_url_raw,
            "src_dir_url_blob": src_dir_url_blob,
            "collected_dir_url_blob": collected_dir_url_blob,
            "settings_url_blob": settings_url_blob,

            "matrices": matrices,
            "extra_files": extra_files,

            # Embed the full original meta so the modal can show it
            "source_meta": meta.get("source_meta", None),
        })

    # Keep only entries with numeric n,k so the table can place them
    filtered = [c for c in codes if isinstance(c.get("n"), int) and isinstance(c.get("k"), int)]

    data = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": {
            "remote_origin": remote,
            "owner": owner,
            "repo": repo,
            "branch": branch,
        },
        "codes": filtered,
        "dropped_codes_missing_n_or_k": len(codes) - len(filtered),
    }

    out_path = site_dir / "data.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path.relative_to(root)} with {len(filtered)} codes (dropped {data['dropped_codes_missing_n_or_k']} missing n/k).")


if __name__ == "__main__":
    main()
