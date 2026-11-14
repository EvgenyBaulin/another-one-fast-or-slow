#!/usr/bin/env python3
"""Second feature table: structure around the graph inputs / outputs, high-rank Transposes,
Resize and recurrent layers.

  python scripts/feat2.py --data data      -> data/feat2.csv  (id + f2_* columns)

One row per train/test row whose ONNX file exists. A graph that cannot be parsed gets NaN in
every column (the script never raises on a single file). Shape inference is the same as in
features.py (reference batch, onnxruntime symbolic inference with an onnx fallback).

Columns (all scale-free unless noted, so the batch does not matter):
  f2_rnn_nodes        LSTM / GRU / RNN nodes
  f2_rnn_steps        sum over them of  seq_len * directions
  f2_rnn_work         sum of  seq_len * directions * hidden_size   (not scaled by batch: the
                      device runs the steps one after another, 4-LSTM models cost 58 ms at
                      batch 1 and 73 ms at batch 32)
  f2_rnn_only         1 if every MAC of the graph sits in recurrent nodes
  f2_in_fan           max number of real (non no-op) consumers of one graph input
  f2_in_conv_frac     share of graph-input elements read only by conv-like nodes
  f2_in_move_frac     share of graph-input elements read by a data-movement node
                      (Slice, Split, Transpose, Concat, Resize, Pad, Gather, pooling)
  f2_out_conv_frac    share of graph-output elements written by a conv-like node or an
                      activation fused into it
  f2_out_ew2_frac     ... written by a binary elementwise node with two large operands
  f2_out_move_frac    ... written by a data-movement node
  f2_res_in_per_io    elements of binary elementwise nodes that take a graph input as an
                      operand, per input+output element (skip connection from the input)
  f2_ew2_per_io       elements of all binary elementwise nodes with two large operands, per io
  f2_tr_hi_per_io     elements moved by Transposes of rank >= 5, per io
  f2_tr_hi_in_per_io  the same, only Transposes that read a graph input (through no-ops)
  f2_tr_hi_conv_per_io  only Transposes whose result goes to a conv-like node
  f2_tr_hi_out_per_io only Transposes whose result reaches a graph output through no-ops or
                      one elementwise node
  f2_tr_hi_moved      max number of axes that change position in a rank >= 5 Transpose
  f2_tr_hi_ch_stays   number of rank >= 5 Transposes that keep axis 1 (channels) in place
  f2_tr_multi_per_io  elements moved by Transposes (any rank) that move more than one axis
  f2_rs_out_per_io    Resize output elements per io
  f2_rs_max_scale     largest spatial scale factor of a Resize
  f2_rs_linear        number of non-nearest Resize nodes
  f2_rs_conv_per_io   Resize output elements that go to a conv-like node, per io
  f2_max_kernel       largest Conv kernel area
  f2_peak_per_io      largest intermediate tensor per io
"""

import argparse
import collections
import csv
import os
import sys
import time
from multiprocessing import Pool

import onnx
from onnx import shape_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

COLS = [
    "f2_rnn_nodes", "f2_rnn_steps", "f2_rnn_work", "f2_rnn_only",
    "f2_in_fan", "f2_in_conv_frac", "f2_in_move_frac",
    "f2_out_conv_frac", "f2_out_ew2_frac", "f2_out_move_frac",
    "f2_res_in_per_io", "f2_ew2_per_io",
    "f2_tr_hi_per_io", "f2_tr_hi_in_per_io", "f2_tr_hi_conv_per_io", "f2_tr_hi_out_per_io",
    "f2_tr_hi_moved", "f2_tr_hi_ch_stays", "f2_tr_multi_per_io",
    "f2_rs_out_per_io", "f2_rs_max_scale", "f2_rs_linear", "f2_rs_conv_per_io",
    "f2_max_kernel", "f2_peak_per_io",
]  # fmt: skip
RNN = {"LSTM", "GRU", "RNN"}
MOVE = {"Slice", "Split", "Transpose", "Concat", "Resize", "Upsample", "Pad", "Gather", "MaxPool", "AveragePool",
        "GlobalAveragePool", "GlobalMaxPool", "DepthToSpace", "SpaceToDepth", "Expand", "Tile"}  # fmt: skip
EW2 = {"Add", "Mul", "Sub", "Div"}
UNARY = {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Clip", "HardSigmoid", "HardSwish", "PRelu", "Elu", "Selu", "Softplus"}


def infer(model, batch, spec):
    """Shape map of the model at `batch` (same engine order as features.graph_features)."""
    m, moved = F.strip_for_inference(model)
    inits = {t.name for t in m.graph.initializer} | moved
    F.set_inputs(m, batch, F.parse_shape_spec(spec), inits)
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        ssi = SymbolicShapeInference(2**31 - 1, True, False, 0)
        ssi._preprocess(m)
        while ssi.run_:
            ssi._infer_impl()
        ssi._update_output_from_vi()
        m = ssi.out_mp_
    except Exception:
        try:
            m = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
        except Exception:
            pass
    return m, F.shape_map(m), inits


def reference(model, spec):
    """(m, shp, inits) at the reference batch with the best shape coverage."""
    nb = F.native_batch(model) if spec == "default" else None
    cands = ([None] if nb else []) + [b for b in F.REF_BATCHES if b != nb]
    best = None
    for b in cands:
        m, shp, inits = infer(model, b, spec)
        outs = [o for n in F.all_nodes(m.graph) if n.op_type != "Constant" for o in n.output if o]
        frac = sum(o in shp for o in outs) / len(outs) if outs else 1.0
        if best is None or frac > best[0] + 1e-9:
            best = (frac, m, shp, inits)
        if frac >= 0.999:
            break
    return best[1:]


def graph_feat2(m, shp, inits) -> dict:
    g = m.graph
    nodes = [n for n in F.all_nodes(g) if n.op_type != "Constant"]
    prod_of = {o: n for n in nodes for o in n.output if o}
    cons = collections.defaultdict(list)
    for n in nodes:
        for i in n.input:
            if i:
                cons[i].append(n)
    g_in = [i.name for i in g.input if i.name not in inits]
    g_in_set = set(g_in)
    g_out = [o.name for o in g.output]
    g_out_set = set(g_out)

    def E(x):
        return F.prod(shp[x]) if x in shp else 0

    def src(t):
        """Real producer of tensor t, looking back through no-ops: ('IN' | op | 'W', tensor)."""
        for _ in range(30):
            if t in prod_of and prod_of[t].op_type in F.NOP and prod_of[t].input:
                t = prod_of[t].input[0]
            else:
                break
        return ("IN" if t in g_in_set else prod_of[t].op_type if t in prod_of else "W"), t

    def real_cons(t):
        """Real consumers of tensor t, looking forward through no-ops; 'OUT' for a graph output."""
        out, stack, seen = [], [t], 0
        while stack and seen < 200:
            x = stack.pop()
            seen += 1
            if x in g_out_set:
                out.append("OUT")
            for c in cons.get(x, []):
                if c.op_type in F.NOP:
                    stack += [o for o in c.output if o]
                else:
                    out.append(c.op_type)
        return out

    io = sum(E(x) for x in g_in) + sum(E(x) for x in g_out)
    f = dict.fromkeys(COLS, 0.0)
    den = max(io, 1)

    # recurrent layers
    macs_rnn = macs_all = 0
    for n in nodes:
        mc = F.node_macs(n, shp)
        if mc > 0:
            macs_all += mc
            if n.op_type in RNN:
                macs_rnn += mc
        if n.op_type in RNN:
            at = {a.name: a for a in n.attribute}
            hid = at["hidden_size"].i if "hidden_size" in at else 0
            dirs = 2 if "direction" in at and at["direction"].s == b"bidirectional" else 1
            layout = at["layout"].i if "layout" in at else 0
            x = shp.get(n.input[0])
            if not hid and len(n.input) > 2 and n.input[2] in shp:
                hid = shp[n.input[2]][-1]
            t = x[1 if layout else 0] if x and len(x) == 3 else 0
            f["f2_rnn_nodes"] += 1
            f["f2_rnn_steps"] += t * dirs
            f["f2_rnn_work"] += t * dirs * hid
    f["f2_rnn_only"] = float(f["f2_rnn_nodes"] > 0 and macs_all == macs_rnn)

    # graph inputs
    tot_in = sum(E(x) for x in g_in)
    for x in g_in:
        rc = [c for c in real_cons(x) if c != "OUT"]
        f["f2_in_fan"] = max(f["f2_in_fan"], len(rc))
        if rc and all(c in F.CONV_LIKE for c in rc):
            f["f2_in_conv_frac"] += E(x) / max(tot_in, 1)
        if any(c in MOVE for c in rc):
            f["f2_in_move_frac"] += E(x) / max(tot_in, 1)

    # graph outputs
    tot_out = sum(E(x) for x in g_out)
    for o in g_out:
        op, t = src(o)
        w = E(o) / max(tot_out, 1)
        if op in UNARY and t in prod_of and prod_of[t].input:
            op2, _ = src(prod_of[t].input[0])
            if op2 in F.CONV_LIKE:
                op = op2
        if op in F.CONV_LIKE:
            f["f2_out_conv_frac"] += w
        elif op in EW2 and t in prod_of and sum(E(i) > 1 for i in prod_of[t].input if i) >= 2:
            f["f2_out_ew2_frac"] += w
        elif op in MOVE:
            f["f2_out_move_frac"] += w

    peak = 0
    for n in nodes:
        op = n.op_type
        oe = sum(E(o) for o in n.output if o)
        peak = max(peak, oe)
        if op in EW2:
            big = [i for i in n.input if i and E(i) > 1]
            if len(big) >= 2:
                f["f2_ew2_per_io"] += oe / den
                if any(src(i)[0] == "IN" for i in big):
                    f["f2_res_in_per_io"] += oe / den
        elif op == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            rank = len(perm) if perm else len(shp.get(n.input[0], []))
            if perm is None:
                perm = list(range(rank))[::-1]
            moved = sum(p != i for i, p in enumerate(perm))
            # a single axis moved to another place leaves the other axes in order
            rest = [p for p in perm]
            single = False
            for a in range(rank):
                if [p for p in rest if p != a] == sorted(p for p in rest if p != a):
                    single = True
                    break
            if moved and not single:
                f["f2_tr_multi_per_io"] += oe / den
            if rank >= 5:
                f["f2_tr_hi_per_io"] += oe / den
                f["f2_tr_hi_moved"] = max(f["f2_tr_hi_moved"], moved)
                f["f2_tr_hi_ch_stays"] += float(len(perm) > 1 and perm[1] == 1)
                if src(n.input[0])[0] == "IN":
                    f["f2_tr_hi_in_per_io"] += oe / den
                rc = real_cons(n.output[0])
                if any(c in F.CONV_LIKE for c in rc):
                    f["f2_tr_hi_conv_per_io"] += oe / den
                to_out = "OUT" in rc
                if not to_out:
                    for c in cons.get(n.output[0], []):
                        for c2 in ([c] if c.op_type not in F.NOP else [x for o in c.output for x in cons.get(o, [])]):
                            if c2.op_type in EW2 and any(o in g_out_set for o in c2.output):
                                to_out = True
                if to_out:
                    f["f2_tr_hi_out_per_io"] += oe / den
        elif op in ("Resize", "Upsample"):
            f["f2_rs_out_per_io"] += oe / den
            mode = next((a.s for a in n.attribute if a.name == "mode"), b"nearest")
            f["f2_rs_linear"] += float(mode != b"nearest")
            a, b = shp.get(n.input[0]), shp.get(n.output[0])
            if a and b and len(a) == len(b) and len(a) >= 3 and a[-1] > 0:
                f["f2_rs_max_scale"] = max(f["f2_rs_max_scale"], b[-1] / a[-1])
            if any(c in F.CONV_LIKE for c in real_cons(n.output[0])):
                f["f2_rs_conv_per_io"] += oe / den
        elif op == "Conv":
            ks = next((list(a.ints) for a in n.attribute if a.name == "kernel_shape"), None)
            if ks is None and len(n.input) > 1 and n.input[1] in shp:
                ks = shp[n.input[1]][2:]
            if ks:
                f["f2_max_kernel"] = max(f["f2_max_kernel"], F.prod(ks))
    f["f2_peak_per_io"] = peak / den
    if io <= 0:  # shapes of the graph inputs / outputs unknown: the ratios would be fake
        for c in COLS:
            if c.endswith("_per_io"):
                f[c] = float("nan")
    return f


def process_file(args):
    data_dir, fname, rows = args
    out = []
    try:
        model = onnx.load(os.path.join(data_dir, "data", fname), load_external_data=False)
    except Exception:
        return [{"id": r["id"]} for r in rows]
    cache = {}
    for r in rows:
        spec = r["shape"]
        if spec not in cache:
            try:
                cache[spec] = graph_feat2(*reference(model, spec))
            except Exception:
                cache[spec] = {}
        out.append({"id": r["id"], **cache[spec]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", type=int, default=6)
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
    path = os.path.join(a.data, "feat2.csv")
    n_bad = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id"] + COLS)
        w.writeheader()
        for r in rows:
            if r["id"] in res:
                o = res[r["id"]]
                n_bad += len(o) == 1
                w.writerow({c: o.get(c, "") if c == "id" else (round(o[c], 6) if c in o else "") for c in ["id"] + COLS})
    print(f"wrote {path}: {len(res)} rows x {len(COLS)} features, {n_bad} rows without features (parse failure); {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
