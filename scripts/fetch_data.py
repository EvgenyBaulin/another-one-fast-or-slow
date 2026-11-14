#!/usr/bin/env python3
"""Sync competition data from Kaggle into a local folder (stdlib + curl only).

Kaggle path data/data/abc.onnx is stored as <out>/data/abc.onnx, CSV files as
<out>/train.csv (the layout scripts/features.py expects). Files whose size already
matches the listing are skipped, partial files are resumed, so the script is
safe to re-run and is used both for the first sync and for the files that
appear later.

Token: env KAGGLE_API_TOKEN, else ~/.kaggle/access_token. The token is passed
to curl through stdin config, never through argv.

Usage:
  python3 scripts/fetch_data.py --out data            # sync everything
  python3 scripts/fetch_data.py --out data --check    # only report what is missing
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

COMP = "another-one-fast-or-slow-predict-ai-model-runtime-again"
API = "https://www.kaggle.com/api/v1"


def token() -> str:
    t = os.environ.get("KAGGLE_API_TOKEN")
    if t:
        return t.strip()
    return (Path.home() / ".kaggle" / "access_token").read_text().strip()


def list_files(tok: str) -> list:
    files, page = [], None
    while True:
        url = f"{API}/competitions/data/list/{COMP}?pageSize=200"
        if page:
            url += "&pageToken=" + urllib.parse.quote(page)
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        files += d["files"]
        page = d.get("nextPageTokenNullable")
        if not page:
            return files


def local_path(out: Path, name: str) -> Path:
    return out / (name[len("data/") :] if name.startswith("data/data/") else name)


def download(tok: str, out: Path, name: str, size: int) -> str:
    dst = local_path(out, name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/competitions/data/download/{COMP}/" + urllib.parse.quote(name, safe="")
    cfg = f'header = "Authorization: Bearer {tok}"\nurl = "{url}"\n'
    err = ""
    for _ in range(20):
        if dst.exists() and dst.stat().st_size > size:
            dst.unlink()
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "--continue-at", "-", "--retry", "5",
             "--retry-all-errors", "--connect-timeout", "20",
             "--speed-limit", "1024", "--speed-time", "60",
             "-o", str(dst), "--config", "-"],
            input=cfg, capture_output=True, text=True,
        )  # fmt: skip
        if dst.exists() and dst.stat().st_size == size:
            return f"ok    {name} ({size / 1e6:.1f} MB)"
        err = r.stderr.strip()[:200]
        time.sleep(2)
    return f"FAIL  {name}: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    tok = token()
    files = list_files(tok)
    todo = []
    for f in files:
        p = local_path(out, f["name"])
        if not (p.exists() and p.stat().st_size == f["totalBytes"]):
            todo.append(f)
    todo.sort(key=lambda f: f["totalBytes"])
    print(
        f"listing: {len(files)} files, {sum(f['totalBytes'] for f in files) / 1e9:.2f} GB; "
        f"to fetch: {len(todo)} files, {sum(f['totalBytes'] for f in todo) / 1e9:.2f} GB -> {out}",
        flush=True,
    )
    if a.check:
        for f in todo[:20]:
            print("  missing", f["name"], f["totalBytes"])
        sys.exit(1 if todo else 0)

    fails = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(download, tok, out, f["name"], f["totalBytes"]) for f in todo]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            msg = fu.result()
            fails += msg.startswith("FAIL")
            print(f"[{i}/{len(futs)}] {time.time() - t0:.0f}s {msg}", flush=True)
    print("DONE ok" if not fails else f"DONE with {fails} failures, re-run", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
