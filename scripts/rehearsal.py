#!/usr/bin/env python3
"""Rehearsal of the last hour on the server (stdlib only, nothing is copied or submitted).

Builds <out>/ (default data_rehearsal/, next to data/):
  - N single-row train files are hidden (seeded, same batch mix as the real test rows
    without a graph); their rows replace the real graph-less rows of test.csv at the same
    positions, so test.csv keeps 299 rows; true latencies go to truth.csv;
  - <out>/data/ holds symlinks to the real ONNX files;
  - rows of existing data/measure_<provider>.csv (+ .jsonl journal) are copied for the
    visible ids, so only the hidden rows have to be measured.
Phase A: hidden files absent  -> pipeline without measurement (rows get the batch prior).
Phase B: hidden files "arrive" (symlinks added) -> steps 2-4 through
         `scripts/final.sh __pipeline`, the very code the real run uses.
Then the hidden rows are scored: RMSLE overall and by batch, phase A vs phase B.

Usage (from ~/hacklab/fos):
  python scripts/rehearsal.py [--n 171] [--seed 0] [--measure-cap 600] [--no-measure]
                              [--measure-force] [--skip-phase-a] [--reuse-hidden-measure]
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMERIC = ("ms_median", "ms_min", "ms_first", "create_s", "n_runs", "measured_batch")


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def scale_mix(mix, n):
    """Batch mix scaled to n rows (largest remainder)."""
    total = sum(mix.values())
    if n == total:
        return dict(mix)
    raw = {b: c * n / total for b, c in mix.items()}
    out = {b: int(v) for b, v in raw.items()}
    for b in sorted(raw, key=lambda b: raw[b] - out[b], reverse=True)[: n - sum(out.values())]:
        out[b] += 1
    return out


def copy_measurements(src, dst, keep_ids):
    for prov in ("cuda", "cpu"):
        s, d = src / f"measure_{prov}.csv", dst / f"measure_{prov}.csv"
        for p in (d, Path(str(d) + ".jsonl")):
            p.unlink(missing_ok=True)
        if not s.exists():
            print(f"measure_{prov}.csv: no source file, nothing to reuse")
            continue
        fields, rows = read_csv(s)
        rows = [r for r in rows if r["id"] in keep_ids]
        write_csv(d, fields, rows)
        sj = Path(str(s) + ".jsonl")
        with open(str(d) + ".jsonl", "w") as out:
            if sj.exists():  # the journal is what measure.py resumes from
                n = 0
                for line in open(sj):
                    try:
                        if json.loads(line).get("id") in keep_ids:
                            out.write(line if line.endswith("\n") else line + "\n")
                            n += 1
                    except ValueError:
                        pass
            else:
                for r in rows:
                    rec = {k: (v if v != "" else None) for k, v in r.items()}
                    for k in NUMERIC:
                        if rec.get(k) is not None:
                            rec[k] = float(rec[k])
                    out.write(json.dumps(rec) + "\n")
                n = len(rows)
        print(f"measure_{prov}.csv: reused {len(rows)} rows ({n} journal lines)")


def run_pipeline(data, out, extra, env=None):
    """Run final.sh __pipeline, echo its output, return (rc, {step: seconds}, wall seconds)."""
    cmd = ["bash", "scripts/final.sh", "__pipeline", "--data", str(data), "--out", str(out)] + extra
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    steps = {}
    for line in p.stdout:
        print("   | " + line, end="", flush=True)
        m = re.match(r"\[final\] step (\d\w? \w+(?: \w+)?): (\d+)s", line)
        if m:
            steps[m.group(1)] = int(m.group(2))
    return p.wait(), steps, time.time() - t0


def score(sub_path, truth, sample_ids):
    """Validate the submission and score the hidden rows. Returns (overall, {batch: (n, rmsle)})."""
    _, sub = read_csv(sub_path)
    ids = [r["id"] for r in sub]
    assert ids == sample_ids, f"{sub_path}: ids differ from the sample order ({len(ids)} vs {len(sample_ids)} rows)"
    pred = {r["id"]: float(r["latency"]) for r in sub}
    assert all(math.isfinite(v) and v >= 0 for v in pred.values()), f"{sub_path}: non-finite or negative values"
    err = collections.defaultdict(list)
    for r in truth:
        e = (math.log1p(pred[r["id"]]) - math.log1p(float(r["latency"]))) ** 2
        err[float(r["batch"])].append(e)
    allv = [e for v in err.values() for e in v]
    rm = lambda v: math.sqrt(sum(v) / len(v))  # noqa: E731
    return rm(allv), {b: (len(v), rm(v)) for b, v in sorted(err.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_rehearsal")
    ap.add_argument("--n", type=int, help="rows to hide (default: number of real test rows without a graph)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-cap", type=int, default=600)
    ap.add_argument("--no-measure", action="store_true")
    ap.add_argument("--measure-force", action="store_true", help="measure even if train was never measured")
    ap.add_argument("--skip-phase-a", action="store_true")
    ap.add_argument("--reuse-hidden-measure", action="store_true", help="also copy measurements of the hidden rows")
    a = ap.parse_args()
    src, dst = (ROOT / a.data).resolve(), (ROOT / a.out).resolve()
    assert src != dst and src.is_dir()

    tfields, train = read_csv(src / "train.csv")
    sfields, test = read_csv(src / "test.csv")
    smp_fields, sample = read_csv(src / "sample_submission.csv")
    present = set(os.listdir(src / "data"))
    blind = [r for r in test if r["file"] not in present]
    if blind:
        mix = collections.Counter(float(r["batch"]) for r in blind)
    elif (dst / "truth.csv").exists():  # the real files have already arrived: repeat the last mix
        mix = collections.Counter(float(r["batch"]) for r in read_csv(dst / "truth.csv")[1])
        print("no graph-less test rows any more: batch mix taken from the previous truth.csv, rows appended to test")
    else:
        sys.exit("no test rows without a graph and no previous truth.csv: nothing to imitate")
    mix = scale_mix(mix, a.n or sum(mix.values()))

    # candidates: train files with exactly one row, graph present, not used by test rows
    per_file = collections.Counter(r["file"] for r in train)
    test_files = {r["file"] for r in test}
    by_batch = collections.defaultdict(list)
    for r in sorted(train, key=lambda r: r["id"]):
        if per_file[r["file"]] == 1 and r["file"] in present and r["file"] not in test_files:
            by_batch[float(r["batch"])].append(r)
    rng = random.Random(a.seed)
    hidden, spare, deficit = [], [], 0
    for b in sorted(mix):
        pool = by_batch.get(b, [])
        rng.shuffle(pool)
        hidden += pool[: mix[b]]
        spare += pool[mix[b]:]
        deficit += max(0, mix[b] - len(pool))
    if deficit:
        rng.shuffle(spare)
        hidden += spare[:deficit]
        print(f"WARNING: {deficit} rows could not be matched by batch, filled from other batches")
    hid_ids, hid_files = {r["id"] for r in hidden}, {r["file"] for r in hidden}
    got = collections.Counter(float(r["batch"]) for r in hidden)
    print(f"hidden rows: {len(hidden)} of {sum(len(v) for v in by_batch.values())} single-row candidates; "
          f"batch mix wanted {dict(sorted(mix.items()))} got {dict(sorted(got.items()))}")

    # rehearsal CSVs: hidden rows take the places of the real graph-less rows (batch-matched)
    queue = collections.defaultdict(list)
    for r in hidden:
        queue[float(r["batch"])].append(r)
    new_test, rest = [], []
    for r in test:
        if r["file"] in present:
            new_test.append(r)
        else:
            q = queue.get(float(r["batch"]))
            new_test.append(q.pop() if q else None)
    rest = [r for q in queue.values() for r in q]
    new_test = [r if r is not None else (rest.pop() if rest else None) for r in new_test]
    new_test = [r for r in new_test if r is not None] + rest
    const = sample[0]["latency"] if sample else "1.0"
    dst.mkdir(exist_ok=True)
    write_csv(dst / "train.csv", tfields, [r for r in train if r["id"] not in hid_ids])
    write_csv(dst / "test.csv", sfields, new_test)
    write_csv(dst / "sample_submission.csv", smp_fields, [{"id": r["id"], "latency": const} for r in new_test])
    write_csv(dst / "truth.csv", tfields, hidden)
    sample_ids = [r["id"] for r in new_test]

    # phase A layout: symlinks for everything except the hidden files
    shutil.rmtree(dst / "data", ignore_errors=True)
    shutil.rmtree(dst / "preds", ignore_errors=True)  # stacking inputs of a previous run
    (dst / "data").mkdir()
    needed = ({r["file"] for r in train} | {r["file"] for r in new_test}) & present
    for f in needed - hid_files:
        os.symlink(src / "data" / f, dst / "data" / f)
    for f in ("features.csv", "sub_phaseA.csv", "sub_phaseB.csv"):
        (dst / f).unlink(missing_ok=True)
    keep = ({r["id"] for r in train} | {r["id"] for r in new_test}) - (set() if a.reuse_hidden_measure else hid_ids)
    copy_measurements(src, dst, keep)
    print(f"{dst}: train {len(train) - len(hidden)} rows, test {len(new_test)} rows "
          f"({len(hidden)} hidden), {len(needed - hid_files)} symlinks")

    report = {"n_hidden": len(hidden), "seed": a.seed, "batch_mix": {str(k): v for k, v in sorted(got.items())}}
    if not a.skip_phase_a:
        print("\n=== phase A: hidden files absent (batch prior) ===", flush=True)
        rc, steps, wall = run_pipeline(dst, dst / "sub_phaseA.csv", ["--no-measure"])
        if rc != 0:
            sys.exit(f"phase A pipeline failed (rc={rc})")
        report["phase_a"] = {"steps_s": steps, "wall_s": round(wall, 1)}
        report["phase_a"]["rmsle"], report["phase_a"]["by_batch"] = score(dst / "sub_phaseA.csv", hidden, sample_ids)

    print("\n=== phase B: hidden files arrive ===", flush=True)
    for f in hid_files:
        os.symlink(src / "data" / f, dst / "data" / f)
    extra = ["--no-measure"] if a.no_measure else ["--measure-cap", str(a.measure_cap)]
    rc, steps, wall = run_pipeline(dst, dst / "sub_phaseB.csv", extra,
                                   {"MEASURE_FORCE": "1"} if a.measure_force else None)
    if rc != 0:
        sys.exit(f"phase B pipeline failed (rc={rc})")
    report["phase_b"] = {"steps_s": steps, "wall_s": round(wall, 1)}
    report["phase_b"]["rmsle"], report["phase_b"]["by_batch"] = score(dst / "sub_phaseB.csv", hidden, sample_ids)

    print("\n=== rehearsal result ===")
    print(f"hidden rows: {len(hidden)}, seed {a.seed}; both submissions valid ({len(sample_ids)} rows, sample order, finite, >= 0)")
    print("phase B step timings, s: " + ", ".join(f"{k} = {v}" for k, v in steps.items()) + f"; wall = {wall:.0f}")
    pa, pb = report.get("phase_a"), report["phase_b"]
    print(f"RMSLE on hidden rows: phase A (batch prior) = {pa['rmsle']:.4f}" if pa else "phase A skipped", end="")
    print(f" | phase B (graphs) = {pb['rmsle']:.4f}")
    print(f"{'batch':>6} {'n':>4} {'phase A':>8} {'phase B':>8}")
    for b, (n, v) in pb["by_batch"].items():
        av = f"{pa['by_batch'][b][1]:.4f}" if pa else "-"
        print(f"{b:>6g} {n:>4} {av:>8} {v:>8.4f}")
    for ph in ("phase_a", "phase_b"):
        if ph in report:
            report[ph]["by_batch"] = {str(b): {"n": n, "rmsle": v} for b, (n, v) in report[ph]["by_batch"].items()}
    with open(dst / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"report: {dst / 'report.json'}")


if __name__ == "__main__":
    main()
