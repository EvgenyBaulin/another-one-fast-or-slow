#!/usr/bin/env python3
"""Model identity keys per ONNX file, for G1 (scripts/g1.py).

The same network appears under several file names (static vs dynamic export,
re-export with other tensor names). Keys that survive such re-exports:

  wkey  - hash of the multiset of float weight tensors (dims + content digest),
          tensors with >= MIN_NUMEL elements, initializers and Constant nodes;
  skey  - hash of the multiset of their dims only (same architecture, any weights);
  okey  - hash of the multiset of "compute" op types (shape-plumbing ops ignored);
  wset  - content digests of the largest weight tensors (for partial matching);
  in_sig - non-batch dims of the graph inputs.

Files that are absent are skipped; rerun after new files arrive.

Usage:  python scripts/ident.py [--data data] [--out data/ident.csv] [--jobs 8]
"""

import argparse
import collections
import csv
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import onnx
from onnx import TensorProto, numpy_helper

MIN_NUMEL = 16
HEAD_BYTES = 1 << 16  # bytes of every tensor that go into its content digest
TOP_K = 12
FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}
PLUMBING = {
    "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Constant", "ConstantOfShape",
    "Reshape", "Cast", "Slice", "Identity", "Flatten", "Expand", "Range", "Equal", "Where",
    "Transpose", "Mul", "Add", "Sub", "Div", "Floor", "Ceil", "Dropout", "Pad", "Not", "Less",
    "Greater", "And", "Or", "If", "Tile", "Size", "ReduceProd", "Min", "Max", "Neg",
}  # fmt: skip


def all_nodes(graph):
    for n in graph.node:
        yield n
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from all_nodes(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    yield from all_nodes(g)


def numel(t) -> int:
    p = 1
    for d in t.dims:
        p *= d
    return p


def tensor_bytes(t) -> bytes:
    if t.raw_data:
        return t.raw_data[:HEAD_BYTES]
    try:
        return numpy_helper.to_array(t).tobytes()[:HEAD_BYTES]
    except Exception:
        return b""


def weight_tensors(model):
    for t in model.graph.initializer:
        yield t
    for n in all_nodes(model.graph):
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    yield a.t


def ident(path: str) -> dict:
    model = onnx.load(path, load_external_data=False)
    g = model.graph
    items = []  # (numel, dims, digest)
    for t in weight_tensors(model):
        n = numel(t)
        if t.data_type not in FLOAT_TYPES or n < MIN_NUMEL:
            continue
        dims = tuple(t.dims)
        d = hashlib.md5(str((dims, t.data_type)).encode() + tensor_bytes(t)).hexdigest()[:10]
        items.append((n, dims, d))
    hw, hs = hashlib.md5(), hashlib.md5()
    for d in sorted(i[2] for i in items):
        hw.update(d.encode())
    for dims in sorted(i[1] for i in items):
        hs.update(str(dims).encode())
    ops = collections.Counter(n.op_type for n in all_nodes(g) if n.op_type not in PLUMBING)
    ho = hashlib.md5(str(sorted(ops.items())).encode())
    inits = {i.name for i in g.initializer}
    sig = []
    for i in g.input:
        if i.name in inits:
            continue
        dims = [d.dim_value if d.HasField("dim_value") else "s" for d in i.type.tensor_type.shape.dim]
        sig.append("x".join(str(d) for d in dims[1:]))
    top = sorted(items, key=lambda i: (-i[0], i[2]))[:TOP_K]
    return {
        "file": os.path.basename(path),
        "wkey": hw.hexdigest()[:16] if items else "",
        "skey": hs.hexdigest()[:16] if items else "",
        "okey": ho.hexdigest()[:16],
        "n_w": len(items),
        "w_elems": sum(i[0] for i in items),
        "n_compute": sum(ops.values()),
        "in_sig": "|".join(sig),
        "wset": " ".join(i[2] for i in top),
    }


def safe_ident(path):
    try:
        return ident(path), None
    except Exception as e:
        return None, f"{os.path.basename(path)}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, "ident.csv")
    t0 = time.time()
    files = set()
    for split in ("train", "test"):
        for r in csv.DictReader(open(os.path.join(a.data, f"{split}.csv"))):
            files.add(r["file"])
    ddir = os.path.join(a.data, "data")
    present = sorted(
        (f for f in files if os.path.exists(os.path.join(ddir, f))),
        key=lambda f: -os.path.getsize(os.path.join(ddir, f)),
    )
    with Pool(a.jobs) as p:
        res = p.map(safe_ident, [os.path.join(ddir, f) for f in present], chunksize=4)
    rows = [r for r, e in res if r]
    errs = [e for r, e in res if e]
    rows.sort(key=lambda r: r["file"])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} files of {len(files)} listed, {time.time() - t0:.1f}s", flush=True)
    if errs:
        print(f"{len(errs)} files failed:", *errs[:20], sep="\n  ", file=sys.stderr)


if __name__ == "__main__":
    main()
