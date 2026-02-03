#!/usr/bin/env python3
"""Report GitHub release asset download counts for matrix files."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import parse, request, error

DEFAULT_TAG_TEMPLATE = "best-codes-matrices-{track}"
DEFAULT_TRACKS = ("633", "844", "212")
PREFIX_TO_TRACK = {
    "633": "6_3_3",
    "844": "8_4_4",
    "212": "2_1_2",
}


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
        path = url.split(":", 1)[1]
    elif "github.com/" in url:
        path = url.split("github.com/", 1)[1]
    else:
        raise RuntimeError(f"Unsupported git remote URL: {url}")

    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    owner, repo = path.split("/", 1)
    return owner, repo


def api_request_json(method: str, url: str, token: str | None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "qtanner-matrix-release",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = request.Request(url, method=method, headers=headers)
    try:
        with request.urlopen(req) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail}") from None


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


def parse_asset(name: str) -> tuple[str, str, str]:
    # Expected name: <prefix>__<code_id>__Hx.mtx
    track = ""
    code_id = ""
    matrix = ""
    if "__" in name:
        prefix, rest = name.split("__", 1)
        track = PREFIX_TO_TRACK.get(prefix, prefix)
        if rest.endswith("__Hx.mtx"):
            matrix = "Hx"
            code_id = rest[: -len("__Hx.mtx")]
        elif rest.endswith("__Hz.mtx"):
            matrix = "Hz"
            code_id = rest[: -len("__Hz.mtx")]
        else:
            code_id = rest
    return track, code_id, matrix


def parse_tracks(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_TRACKS)
    items = [v.strip() for v in value.split(",") if v.strip()]
    tracks: list[str] = []
    for item in items:
        if item in DEFAULT_TRACKS:
            tracks.append(item)
            continue
        raise ValueError(f"Unknown track '{item}'. Use one of: {', '.join(DEFAULT_TRACKS)}")
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Release tag name (single release)")
    parser.add_argument("--tags", default="", help="Comma list of release tags")
    parser.add_argument("--tag-template", default=DEFAULT_TAG_TEMPLATE, help="Release tag template per track")
    parser.add_argument("--tracks", default="", help="Comma list: 633,844,212")
    parser.add_argument("--owner", default="", help="GitHub owner (optional)")
    parser.add_argument("--repo", default="", help="GitHub repo (optional)")
    parser.add_argument("--out", default="", help="Write TSV output to file")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "")
    args = parser.parse_args()

    owner = args.owner
    repo = args.repo
    if not owner or not repo:
        owner, repo = infer_repo()

    token = args.token.strip() or None

    tracks = parse_tracks(args.tracks)

    if args.tag:
        tags = [args.tag]
    elif args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        template = args.tag_template or DEFAULT_TAG_TEMPLATE
        tags = [template.format(track=t) for t in tracks]

    assets = []
    for tag in tags:
        release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{parse.quote(tag)}"
        try:
            release = api_request_json("GET", release_url, token)
        except RuntimeError as exc:
            if " -> 404" in str(exc):
                print(f"[warn] Release tag not found: {tag}", file=sys.stderr)
                continue
            raise
        release_id = release.get("id")
        if not release_id:
            print(f"[warn] Release tag not found: {tag}", file=sys.stderr)
            continue
        for asset in list_assets(owner, repo, int(release_id), token):
            asset["__tag"] = tag
            assets.append(asset)

    out_fh = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out_fh, delimiter="\t")
    writer.writerow(["asset", "tag", "track", "code_id", "matrix", "downloads", "size_bytes", "updated_at"])

    totals_by_track: dict[str, int] = {}
    totals_by_code: dict[tuple[str, str], int] = {}

    for asset in assets:
        name = asset.get("name", "")
        downloads = int(asset.get("download_count") or 0)
        size = int(asset.get("size") or 0)
        updated = asset.get("updated_at") or ""
        track, code_id, matrix = parse_asset(name)
        tag = asset.get("__tag", "")

        writer.writerow([name, tag, track, code_id, matrix, downloads, size, updated])

        if track:
            totals_by_track[track] = totals_by_track.get(track, 0) + downloads
        if track and code_id:
            key = (track, code_id)
            totals_by_code[key] = totals_by_code.get(key, 0) + downloads

    if args.out:
        out_fh.close()

    # Summary to stderr to keep TSV clean
    if totals_by_track:
        print("\nDownload totals by track:", file=sys.stderr)
        for track in sorted(totals_by_track.keys()):
            print(f"  {track}: {totals_by_track[track]}", file=sys.stderr)

    if totals_by_code:
        top = sorted(totals_by_code.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop 10 codes by downloads:", file=sys.stderr)
        for (track, code_id), count in top:
            print(f"  {track} {code_id}: {count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
