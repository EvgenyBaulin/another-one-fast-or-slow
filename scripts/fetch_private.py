#!/usr/bin/env python3
"""Fetch the private test ONNX files from the organiser's Kaggle dataset (stdlib + curl).

The files that were missing from the competition data were published as a separate
dataset (niobeus/fast-or-slow-private, files test/<hash>.onnx). Every file referenced by
<out>/test.csv (or train.csv) that is not yet in <out>/data/ is downloaded there; files of
the right size are skipped, partial files are resumed, so re-running is safe.

Token: env KAGGLE_API_TOKEN, else ~/.kaggle/access_token; passed to curl via stdin config.

  python3 scripts/fetch_private.py --out data [--check] [--all]
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "niobeus/fast-or-slow-private"
PACE = float(os.environ.get("FOS_PACE", "1.0"))  # seconds between files of one worker
API = "https://www.kaggle.com/api/v1"


def token() -> str:
    t = os.environ.get("KAGGLE_API_TOKEN")
    return t.strip() if t else (Path.home() / ".kaggle" / "access_token").read_text().strip()


def list_files(tok: str) -> list:
    files, page = [], None
    while True:
        url = f"{API}/datasets/list/{DATASET}?pageSize=200"
        if page:
            url += "&pageToken=" + urllib.parse.quote(page)
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        files += d["datasetFiles"]
        page = d.get("nextPageTokenNullable")
        if not page:
            return files


def download(tok: str, out: Path, name: str, size: int) -> str:
    dst = out / "data" / name.split("/")[-1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/datasets/download/{DATASET}/" + urllib.parse.quote(name, safe="")
    cfg = f'header = "Authorization: Bearer {tok}"\nurl = "{url}"\n'
    err = ""
    for _ in range(20):
        if dst.exists() and dst.stat().st_size > size:
            dst.unlink()
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "--continue-at", "-",
             "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "30",
             "-o", str(dst), "--config", "-"],
            input=cfg, capture_output=True, text=True,
        )  # fmt: skip
        if dst.exists() and dst.stat().st_size == size:
            time.sleep(PACE)
            return f"ok    {name} ({size / 1e6:.1f} MB)"
        err = r.stderr.strip()[:200]
        time.sleep(25 if "429" in err else 3)  # Kaggle rate limit: Retry-After is 20 s
    return f"FAIL  {name}: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true", help="every dataset file, not only the ones test.csv / train.csv need")
    a = ap.parse_args()
    out = Path(a.out)
    tok = token()
    files = list_files(tok)
    want = set()
    for split in ("test", "train"):
        p = out / f"{split}.csv"
        if p.exists():
            want |= {r["file"] for r in csv.DictReader(open(p))}
    todo = []
    for f in files:
        base = f["name"].split("/")[-1]
        if not a.all and base not in want:
            continue
        dst = out / "data" / base
        if not (dst.exists() and dst.stat().st_size == f["totalBytes"]):
            todo.append(f)
    todo.sort(key=lambda f: f["totalBytes"])
    print(f"dataset: {len(files)} files, {sum(f['totalBytes'] for f in files) / 1e9:.2f} GB; "
          f"to fetch: {len(todo)} files, {sum(f['totalBytes'] for f in todo) / 1e9:.2f} GB -> {out / 'data'}", flush=True)  # fmt: skip
    if a.check:
        sys.exit(1 if todo else 0)
    fails, t0 = 0, time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(download, tok, out, f["name"], f["totalBytes"]) for f in todo]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            msg = fu.result()
            fails += msg.startswith("FAIL")
            if msg.startswith("FAIL") or i % 10 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {time.time() - t0:.0f}s {msg}", flush=True)
    print("DONE ok" if not fails else f"DONE with {fails} failures, re-run", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
