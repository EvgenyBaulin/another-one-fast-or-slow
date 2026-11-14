#!/usr/bin/env python3
"""Boundary features between "slow" and "fast" parts of a graph -> <data>/feat3.csv.

Idea (mobile-NPU latency studies): ops the accelerator cannot run natively fall back to a
slow path, and every switch between the two paths costs a copy. We already price the
elements of standalone ops; here we count the *switches*:
  slow node  = standalone (unfused) elementwise / normalisation, Resize, 5-D/6-D Transpose,
               recurrent layer
  fast node  = conv-like kernel (with whatever is fused into it), pooling
  neutral    = everything else (Concat, Slice, Gather, low-rank Transpose, ...); no-ops are
               looked through.
Columns (all independent of the batch):
  f3_slow_seg        number of connected groups of slow nodes
  f3_fast_seg        number of connected groups of fast nodes
  f3_cross           tensors passed between a slow and a fast node
  f3_cross_per_io    elements on those tensors / graph input+output elements
  f3_slow_share      output elements of slow nodes / output elements of all real nodes
  f3_in_slow         1 if a graph input is consumed by a slow node
  f3_out_slow        1 if a graph output is produced by a slow node
  f3_sandwich        fast groups that have slow nodes both before and after them

  python scripts/feat3.py --data data      (about 10 s on the server)
"""

import argparse
import collections
import csv
import os
import sys
import time
from multiprocessing import Pool

import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feat2 as F2  # noqa: E402
import features as F  # noqa: E402

COLS = ["f3_slow_seg", "f3_fast_seg", "f3_cross", "f3_cross_per_io", "f3_slow_share",
        "f3_in_slow", "f3_out_slow", "f3_sandwich"]  # fmt: skip
RNN = {"LSTM", "GRU", "RNN"}
FAST_EXTRA = F.POOL | {"GlobalAveragePool", "GlobalMaxPool"}


def graph_feat3(m, shp, inits) -> dict:
    g = m.graph
    nodes = [n for n in F.all_nodes(g) if n.op_type != "Constant"]

    def E(x):
        return F.prod(shp[x]) if x in shp else 0

    # class per node: 's' slow, 'f' fast, 'n' neutral, 'x' no-op (looked through)
    cls, fused_src = {}, {}
    for k, n in enumerate(nodes):
        op = n.op_type
        ins_fused = [fused_src.get(i, False) for i in n.input if i]
        c = "n"
        if op in F.NOP:
            c = "x"
            for o in n.output:  # a no-op passes the fused state through
                fused_src[o] = bool(ins_fused and ins_fused[0])
        elif op in F.CONV_LIKE:
            c = "f"
            for o in n.output:
                fused_src[o] = True
        elif op in F.NORM:
            fz = bool(ins_fused and ins_fused[0])
            c = "f" if fz else "s"
            for o in n.output:
                fused_src[o] = fz
        elif op in F.ELEMENTWISE:
            c = "f" if any(ins_fused) else "s"
        elif op in FAST_EXTRA:
            c = "f"
        elif op in RNN or op in ("Resize", "Upsample"):
            c = "s"
        elif op == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            rank = len(perm) if perm else len(shp.get(n.input[0], []))
            c = "s" if rank >= 5 else "n"
        cls[k] = c

    prod_of = {o: k for k, n in enumerate(nodes) for o in n.output if o}
    g_in = {i.name for i in g.input if i.name not in inits}
    g_out = {o.name for o in g.output}

    def real_src(t):
        """Index of the real node producing tensor t (no-ops looked through); -1 for a graph input, None otherwise."""
        for _ in range(40):
            if t in g_in:
                return -1
            k = prod_of.get(t)
            if k is None:
                return None
            if cls[k] != "x":
                return k
            ins = [i for i in nodes[k].input if i]
            if not ins:
                return None
            t = ins[0]
        return None

    # edges between real nodes
    adj = collections.defaultdict(set)
    cross = cross_el = 0
    in_slow = out_slow = 0
    for k, n in enumerate(nodes):
        if cls[k] == "x":
            continue
        for t in n.input:
            if not t:
                continue
            s = real_src(t)
            if s == -1:
                in_slow |= cls[k] == "s"
            elif s is not None:
                adj[s].add(k)
                adj[k].add(s)
                if {cls[s], cls[k]} == {"s", "f"}:
                    cross += 1
                    cross_el += E(t)
    for t in g_out:
        s = real_src(t)
        if s is not None and s >= 0 and cls[s] == "s":
            out_slow = 1

    def groups(kind):
        seen, comps = set(), []
        for k in cls:
            if cls[k] != kind or k in seen:
                continue
            comp, stack = [], [k]
            seen.add(k)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if cls[v] == kind and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        return comps

    slow_g, fast_g = groups("s"), groups("f")
    sandwich = 0
    for comp in fast_g:
        lo, hi = min(comp), max(comp)
        nb = {v for u in comp for v in adj[u] if cls[v] == "s"}
        if any(v < lo for v in nb) and any(v > hi for v in nb):
            sandwich += 1
    tot = sum(sum(E(o) for o in n.output if o) for k, n in enumerate(nodes) if cls[k] != "x")
    slow_el = sum(sum(E(o) for o in n.output if o) for k, n in enumerate(nodes) if cls[k] == "s")
    io = sum(E(t) for t in g_in) + sum(E(t) for t in g_out)
    return {
        "f3_slow_seg": len(slow_g), "f3_fast_seg": len(fast_g), "f3_cross": cross,
        "f3_cross_per_io": cross_el / io if io else float("nan"),
        "f3_slow_share": slow_el / tot if tot else 0.0,
        "f3_in_slow": int(bool(in_slow)), "f3_out_slow": out_slow, "f3_sandwich": sandwich,
    }  # fmt: skip


def process_file(args):
    data_dir, fname, rows = args
    try:
        model = onnx.load(os.path.join(data_dir, "data", fname), load_external_data=False)
    except Exception:
        return [{"id": r["id"]} for r in rows]
    cache, out = {}, []
    for r in rows:
        spec = r["shape"]
        if spec not in cache:
            try:
                cache[spec] = graph_feat3(*F2.reference(model, spec))
            except Exception:
                cache[spec] = {}
        out.append({"id": r["id"], **cache[spec]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    for split in ("train", "test"):
        rows += list(csv.DictReader(open(os.path.join(a.data, f"{split}.csv"))))
    present = set(os.listdir(os.path.join(a.data, "data")))
    by_file = collections.defaultdict(list)
    for r in rows:
        if r["file"] in present:
            by_file[r["file"]].append(r)
    files = sorted(by_file, key=lambda f: -os.path.getsize(os.path.join(a.data, "data", f)))
    res = {}
    with Pool(a.jobs, maxtasksperchild=20) as p:
        for out in p.imap_unordered(process_file, [(a.data, f, by_file[f]) for f in files]):
            for o in out:
                res[o["id"]] = o
    path = os.path.join(a.data, "feat3.csv")
    n_bad = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id"] + COLS)
        w.writeheader()
        for r in rows:
            if r["id"] in res:
                o = res[r["id"]]
                n_bad += len(o) == 1
                w.writerow({c: (o.get(c, "") if c == "id" else (round(o[c], 6) if c in o and o[c] == o[c] else "")) for c in ["id"] + COLS})
    print(f"wrote {path}: {len(res)} rows x {len(COLS)} features, {n_bad} rows without features; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
