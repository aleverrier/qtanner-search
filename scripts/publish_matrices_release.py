#!/usr/bin/env python3
"""Publish matrix files as GitHub release assets for download counting."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import parse, request, error

DEFAULT_TAG = "best-codes-matrices"
TRACK_DIRS = {
    "633": "best_codes",
    "844": "best_codes_844",
    "212": "best_codes_212",
}
DIR_TO_PREFIX = {v: k for k, v in TRACK_DIRS.items()}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def infer_repo() -> tuple[str, str]:
    env = os.environ.get("GITHUB_REPOSITORY")
    if env and "/" in env:
        owner, repo = env.split("/", 1)
        return owner, repo

    url = run(["git", "remote", "get-url", "origin"]).strip()
    path = ""
    if url.startswith("git@"):
        # git@github.com:owner/repo.git
        path = url.split(":", 1)[1]
    elif "github.com/" in url:
        # https://github.com/owner/repo.git
        path = url.split("github.com/", 1)[1]
    else:
        raise RuntimeError(f"Unsupported git remote URL: {url}")

    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    owner, repo = path.split("/", 1)
    return owner, repo


def api_request_json(method: str, url: str, token: str | None, payload: dict | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "qtanner-matrix-release",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail}") from None


def api_request_binary(method: str, url: str, token: str | None, data: bytes):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "qtanner-matrix-release",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(data)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req) as resp:
            _ = resp.read()
            return resp.status
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail}") from None


def get_release(owner: str, repo: str, tag: str, token: str | None):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{parse.quote(tag)}"
    try:
        return api_request_json("GET", url, token)
    except RuntimeError as exc:
        if " -> 404" in str(exc):
            return None
        raise


def create_release(owner: str, repo: str, tag: str, token: str | None):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    payload = {
        "tag_name": tag,
        "name": tag,
        "draft": False,
        "prerelease": False,
    }
    return api_request_json("POST", url, token, payload)


def list_assets(owner: str, repo: str, release_id: int, token: str | None):
    assets = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/releases/"
            f"{release_id}/assets?per_page=100&page={page}"
        )
        batch = api_request_json("GET", url, token) or []
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return assets


def parse_tracks(value: str | None) -> list[str]:
    if not value:
        return list(TRACK_DIRS.keys())
    items = [v.strip() for v in value.split(",") if v.strip()]
    tracks: list[str] = []
    for item in items:
        if item in TRACK_DIRS:
            tracks.append(item)
            continue
        if item in DIR_TO_PREFIX:
            tracks.append(DIR_TO_PREFIX[item])
            continue
        raise ValueError(f"Unknown track '{item}'. Use one of: {', '.join(TRACK_DIRS.keys())}")
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag name")
    parser.add_argument("--owner", default="", help="GitHub owner (optional)")
    parser.add_argument("--repo", default="", help="GitHub repo (optional)")
    parser.add_argument("--tracks", default="", help="Comma list: 633,844,212")
    parser.add_argument("--limit", type=int, default=0, help="Max uploads (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "")
    args = parser.parse_args()

    owner = args.owner
    repo = args.repo
    if not owner or not repo:
        owner, repo = infer_repo()

    token = args.token.strip() or None
    if not token and not args.dry_run:
        raise SystemExit("Missing token: set GITHUB_TOKEN or GH_TOKEN, or pass --token.")

    tracks = parse_tracks(args.tracks)
    repo_root = Path(__file__).resolve().parents[1]

    release = get_release(owner, repo, args.tag, token)
    if release is None:
        if args.dry_run:
            print(f"[dry-run] create release {args.tag}")
            release = {"id": None, "upload_url": ""}
        else:
            release = create_release(owner, repo, args.tag, token)
            print(f"Created release {args.tag} ({release.get('html_url', '')})")

    release_id = release.get("id")
    upload_url = (release.get("upload_url") or "").split("{", 1)[0]
    if not upload_url and not args.dry_run:
        raise SystemExit("Release upload_url missing; cannot upload assets.")

    existing = set()
    if release_id and token:
        for asset in list_assets(owner, repo, int(release_id), token):
            if "name" in asset:
                existing.add(asset["name"])

    to_upload: list[tuple[str, Path]] = []
    for track in tracks:
        base = repo_root / TRACK_DIRS[track] / "matrices"
        if not base.exists():
            print(f"Skipping missing directory: {base}")
            continue
        for path in sorted(base.glob("*.mtx")):
            asset_name = f"{track}__{path.name}"
            if asset_name in existing:
                continue
            to_upload.append((asset_name, path))

    if args.limit and len(to_upload) > args.limit:
        to_upload = to_upload[: args.limit]

    print(f"Release tag: {args.tag}")
    print(f"Repo: {owner}/{repo}")
    print(f"Tracks: {', '.join(tracks)}")
    print(f"Missing assets: {len(to_upload)}")

    if args.dry_run:
        for name, path in to_upload[:10]:
            print(f"[dry-run] upload {name} ({path})")
        if len(to_upload) > 10:
            print(f"[dry-run] ... {len(to_upload) - 10} more")
        return 0

    uploaded = 0
    for name, path in to_upload:
        url = f"{upload_url}?name={parse.quote(name)}"
        data = path.read_bytes()
        api_request_binary("POST", url, token, data)
        uploaded += 1
        print(f"Uploaded {name} ({uploaded}/{len(to_upload)})")

    print(f"Done. Uploaded {uploaded} assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
