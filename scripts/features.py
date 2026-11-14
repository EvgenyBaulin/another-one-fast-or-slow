#!/usr/bin/env python3
"""Extract per-row features from ONNX graphs for train.csv + test.csv.

For every (file, batch, shape) row the graph inputs are fixed to concrete
shapes (dim 0 = batch, `shape` column overrides the named input), ONNX shape
inference is run on a weight-stripped copy of the graph, and per-node costs
(MACs, activation elements by op type) are summed.

Rows whose ONNX file is absent get only batch/shape features (has_graph=0).

Usage:  .venv/bin/python scripts/features.py [--data data] [--out data/features.csv] [--jobs 6]
"""

import argparse
import collections
import csv
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import onnx
from onnx import TensorProto, helper, shape_inference
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

logging.disable(logging.WARNING)

# Unknown non-batch symbolic input dims (height, sequence_length, ...) get this value.
DEFAULT_SYM_DIM = 128
# Initializers larger than this (elements) are turned into typed graph inputs for
# shape inference, so inference does not copy hundreds of MB of weights.
STRIP_NUMEL = 64

ELEMENTWISE = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Exp",
    "Log",
    "Neg",
    "Abs",
    "Erf",
    "Tanh",
    "Sigmoid",
    "Relu",
    "LeakyRelu",
    "PRelu",
    "Elu",
    "Selu",
    "Softplus",
    "HardSigmoid",
    "HardSwish",
    "Clip",
    "Gelu",
    "Mish",
    "Cos",
    "Sin",
    "Reciprocal",
    "Floor",
    "Ceil",
    "Round",
    "Min",
    "Max",
    "Where",
    "Equal",
    "Less",
    "Greater",
    "Not",
    "And",
    "Or",
    "Mod",
    "Sign",
    "Cast",
    "ThresholdedRelu",
    "Celu",
    "Softsign",
    "IsNaN",
    "IsInf",
    "LessOrEqual",
    "GreaterOrEqual",
    "BitShift",
}
MOVEMENT = {
    "Reshape",
    "Transpose",
    "Concat",
    "Slice",
    "Gather",
    "GatherElements",
    "GatherND",
    "Split",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Expand",
    "Tile",
    "Pad",
    "Resize",
    "Identity",
    "ScatterND",
    "ScatterElements",
    "DepthToSpace",
    "SpaceToDepth",
    "Upsample",
    "Compress",
    "NonZero",
    "TopK",
    "OneHot",
}
NORM = {
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "LpNormalization",
    "MeanVarianceNormalization",
}
REDUCE = {
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL2",
    "ReduceL1",
    "ReduceSumSquare",
    "ReduceLogSumExp",
    "ArgMax",
    "ArgMin",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Softmax",
    "LogSoftmax",
    "CumSum",
}
POOL = {"MaxPool", "AveragePool", "LpPool"}
NOP = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Dropout", "Shape", "Cast"}
CONV_LIKE = {"Conv", "ConvTranspose", "Gemm", "MatMul", "ConvInteger", "MatMulInteger"}


def parse_shape_spec(spec: str) -> dict:
    """'images:1x3x800x800' -> {'images': [1, 3, 800, 800]}; 'default' -> {}."""
    out = {}
    if spec == "default":
        return out
    for part in re.split(r"[;,]", spec):
        if ":" in part:
            name, dims = part.rsplit(":", 1)
            out[name] = [int(x) for x in dims.split("x")]
    return out


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def tensor_numel(t: TensorProto) -> int:
    return prod(t.dims)


def all_nodes(graph):
    """Nodes of the graph including If/Loop/Scan subgraphs."""
    for n in graph.node:
        yield n
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from all_nodes(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    yield from all_nodes(g)


def fingerprint(model) -> str:
    """Structure hash: op sequence with attributes + initializer shapes + non-batch input dims.
    Ignores tensor names and weight values, so re-exports of the same net collide."""
    g = model.graph
    h = hashlib.md5()
    for n in all_nodes(g):
        h.update(n.op_type.encode())
        for a in sorted(n.attribute, key=lambda a: a.name):
            if a.type in (
                onnx.AttributeProto.GRAPH,
                onnx.AttributeProto.GRAPHS,
                onnx.AttributeProto.TENSOR,
            ):
                continue
            h.update(a.name.encode())
            h.update(str(helper.get_attribute_value(a)).encode())
    for t in sorted(tuple(i.dims) for i in g.initializer):
        h.update(str(t).encode())
    inits = {i.name for i in g.initializer}
    for i in g.input:
        if i.name in inits:
            continue
        dims = [
            d.dim_value if d.HasField("dim_value") else "s"
            for d in i.type.tensor_type.shape.dim
        ]
        h.update(str(dims[1:]).encode())
    return h.hexdigest()[:16]


def strip_for_inference(model):
    """Copy of the model with large initializers replaced by typed graph inputs."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    keep, moved = [], []
    for t in g.initializer:
        if tensor_numel(t) > STRIP_NUMEL:
            moved.append(
                helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
            )
        else:
            keep.append(t)
    del g.initializer[:]
    g.initializer.extend(keep)
    g.input.extend(moved)
    del g.value_info[:]
    for o in g.output:  # drop stale symbolic output shapes, keep dtype
        o.type.tensor_type.ClearField("shape")
    return m, {v.name for v in moved}


def set_inputs(model, batch: int, override: dict, inits: set):
    """Fix graph input dims in place. Returns number of symbolic dims we had to guess."""
    guessed = 0
    for i in model.graph.input:
        if i.name in inits or not i.type.HasField("tensor_type"):
            continue
        shp = i.type.tensor_type.shape
        if i.name in override:
            dims = list(override[i.name])
            if dims:
                dims[0] = batch
        else:
            dims = []
            for j, d in enumerate(shp.dim):
                if (
                    j == 0
                    and batch is None
                    and d.HasField("dim_value")
                    and d.dim_value > 0
                ):
                    dims.append(d.dim_value)  # native batch of a static graph
                elif j == 0:
                    dims.append(batch if batch is not None else REF_BATCHES[0])
                elif d.HasField("dim_value") and d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(DEFAULT_SYM_DIM)
                    guessed += 1
        if not shp.dim and not dims:
            continue  # scalar input
        shp.ClearField("dim")
        for v in dims:
            shp.dim.add().dim_value = int(v)
    return guessed


def shape_map(model) -> dict:
    out = {}
    g = model.graph

    def add(vi):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        dims = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                return  # not fully known
        out[vi.name] = dims

    def walk(gr):
        for vi in list(gr.input) + list(gr.output) + list(gr.value_info):
            add(vi)
        for t in gr.initializer:
            out[t.name] = list(t.dims)
        for n in gr.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        walk(sg)

    walk(g)
    for n in all_nodes(g):  # Constant node outputs
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    out.setdefault(n.output[0], list(a.t.dims))
    return out


def node_macs(n, shp) -> int:
    op = n.op_type
    try:
        if op == "Conv" and len(n.input) > 1:
            return prod(shp[n.output[0]]) * prod(shp[n.input[1]][1:])
        if op == "ConvTranspose" and len(n.input) > 1:
            return prod(shp[n.input[0]]) * prod(shp[n.input[1]][1:])
        if op in ("MatMul", "MatMulInteger"):
            return prod(shp[n.output[0]]) * shp[n.input[0]][-1]
        if op == "Gemm":
            a = shp[n.input[0]]
            trans_a = next((x.i for x in n.attribute if x.name == "transA"), 0)
            return prod(shp[n.output[0]]) * (a[0] if trans_a else a[-1])
        if op == "Einsum":
            return (
                prod(shp[n.output[0]])
                * max(prod(shp[x]) for x in n.input)
                // max(1, prod(shp[n.output[0]]))
            )
        if op in ("LSTM", "GRU", "RNN"):
            x, w, r = shp[n.input[0]], shp[n.input[1]], shp[n.input[2]]
            return x[0] * x[1] * (prod(w[1:]) + prod(r[1:])) * w[0]
    except (KeyError, IndexError, ValueError):
        return -1
    return 0


def infer_static(model, batch: int, spec: str):
    """Static shape inference at one batch -> (inferred model, shape_map, engine, guessed, inits)."""
    m, moved = strip_for_inference(model)
    inits = {t.name for t in m.graph.initializer} | moved
    guessed = set_inputs(m, batch, parse_shape_spec(spec), inits)
    # onnxruntime's symbolic inference propagates shape values (Shape->Gather->
    # Concat->Reshape) far better than onnx's; fall back to onnx when it asserts.
    engine = 1
    try:
        # Same as SymbolicShapeInference.infer_shapes(), but keeps the partial
        # result instead of raising (and dumping a temp file into cwd) when
        # some shapes stay unknown.
        ssi = SymbolicShapeInference(2**31 - 1, True, False, 0)
        ssi._preprocess(m)
        complete = False
        while ssi.run_:
            complete = ssi._infer_impl()
        ssi._update_output_from_vi()
        m = ssi.out_mp_
        if not complete:
            engine = 2
    except Exception:
        engine = 0
        try:
            m = shape_inference.infer_shapes(
                m, check_type=False, strict_mode=False, data_prop=True
            )
        except Exception:
            engine = -1
    return m, shape_map(m), engine, guessed, inits


def graph_features(model, batch: int, spec: str, common_ops: list) -> dict:
    return cost_features(*infer_static(model, batch, spec), common_ops)


def cost_features(m, shp, engine, guessed, inits, common_ops: list) -> dict:
    f = collections.defaultdict(float)
    f["infer_engine"] = engine
    f["guessed_dims"] = guessed
    f["shape_src"] = 0  # 0 static only, 1 completed by a real run, -1 run failed
    n_out = n_known = macs_unknown = 0
    fused_src = {}  # tensor name -> produced by a conv-like node (or a norm fused into one)
    for k in ("elems_norm_fused", "elems_norm_unfused", "elems_ew_fused", "elems_ew_unfused", "n_norm_unfused", "n_ew_unfused",
              "n_nop", "n_kernels", "elems_traffic", "tr_max_rank", "n_tr_hi", "elems_tr_hi", "n_tr6"):
        f[k] = 0.0
    for n in all_nodes(m.graph):
        op = n.op_type
        if op == "Constant":
            continue
        macs = node_macs(n, shp)
        if macs < 0:
            macs_unknown += 1
        else:
            f["macs"] += macs
            if macs:
                f[f"macs_{op}"] += macs
        oe = 0
        for o in n.output:
            if not o:
                continue
            n_out += 1
            if o in shp:
                n_known += 1
                oe += prod(shp[o])
        ie = sum(prod(shp[i]) for i in n.input if i and i in shp)
        f["out_elems"] += oe
        f["in_elems"] += ie
        # Fused vs standalone: on the accelerator an activation / normalisation right
        # after a conv-like node is free, the same op standing alone costs ~2x the
        # transfer of its tensors (seen on one-node graphs).
        ins_fused = [fused_src.get(i, False) for i in n.input if i]
        fz = False
        if op in CONV_LIKE:
            for o in n.output:
                fused_src[o] = True
        elif op in NORM:
            fz = bool(ins_fused and ins_fused[0])
            f["elems_norm_fused" if fz else "elems_norm_unfused"] += oe
            f["n_norm_unfused"] += 0 if fz else 1
            for o in n.output:
                fused_src[o] = fz
        elif op in ELEMENTWISE:
            fz = any(ins_fused)
            f["elems_ew_fused" if fz else "elems_ew_unfused"] += oe
            f["n_ew_unfused"] += 0 if fz else 1
        # axis permutations of 5-D / 6-D tensors (window partition, space-to-depth,
        # channel shuffle): graphs with them run 2-15x slower per element than the io law
        if op == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            rank = len(perm) if perm else len(shp.get(n.input[0], []))
            f["tr_max_rank"] = max(f["tr_max_rank"], rank)
            if rank >= 5:
                f["n_tr_hi"] += 1
                f["elems_tr_hi"] += oe
            if rank >= 6:
                f["n_tr6"] += 1
        # what an inference compiler leaves after fusion: real kernels and the
        # tensor traffic (inputs + outputs) they move; no-ops are dropped
        if op in NOP:
            f["n_nop"] += 1
        elif not fz:
            f["n_kernels"] += 1
            f["elems_traffic"] += ie + oe
        cat = (
            "elementwise"
            if op in ELEMENTWISE
            else "movement"
            if op in MOVEMENT
            else "norm"
            if op in NORM
            else "reduce"
            if op in REDUCE
            else "pool"
            if op in POOL
            else "matmul"
            if macs
            else "other"
        )
        f[f"elems_{cat}"] += oe
        f[f"n_{cat}"] += 1
        if op in common_ops:
            f[f"elems_{op}"] += oe
        f["max_tensor_elems"] = max(f["max_tensor_elems"], oe)
    f["shape_known_frac"] = n_known / n_out if n_out else 1.0
    f["macs_unknown_nodes"] = macs_unknown
    g_in = [i for i in m.graph.input if i.name not in inits]
    f["input_elems"] = sum(prod(shp.get(i.name, [0])) for i in g_in)
    f["output_elems"] = sum(prod(shp.get(o.name, [0])) for o in m.graph.output)
    return f


# Shape inference is run once per (file, shape) at a reference batch and the
# activation/MAC features are scaled linearly to the row's batch. Batch 1 is a
# bad reference: broadcasting [1,...] against an unknown dim loses the shape.
REF_BATCHES = (7, 1)
BATCH_SCALED = (
    "macs",
    "out_elems",
    "in_elems",
    "elems_",
    "max_tensor_elems",
    "input_elems",
    "output_elems",
)


def native_batch(model):
    """dim 0 of the first graph input if every graph input has a concrete dim 0, else None.
    Static graphs often bake the batch into Split/Reshape constants, so they are
    only inferable at their own batch."""
    g = model.graph
    inits = {t.name for t in g.initializer}
    d0 = []
    for i in g.input:
        if i.name in inits:
            continue
        dims = i.type.tensor_type.shape.dim
        if not dims:
            continue
        if not dims[0].HasField("dim_value") or dims[0].dim_value <= 0:
            return None
        d0.append(dims[0].dim_value)
    return d0[0] if d0 else None


def reference_features(model, spec: str, common_ops: list, path=None, exact=None):
    """Pick the reference batch with the best shape coverage (ties -> first).
    With `exact` (dict of limits, see main) and `path`, graphs whose static coverage is
    incomplete get the missing shapes from one real onnxruntime run."""
    nb = native_batch(model) if spec == "default" else None
    cands = ([(None, nb)] if nb else []) + [(b, b) for b in REF_BATCHES if b != nb]
    best = best_arg = None
    for arg, b in cands:
        f = dict(graph_features(model, arg, spec, common_ops))
        if best is None or f["shape_known_frac"] > best[1]["shape_known_frac"] + 1e-9:
            best, best_arg = (b, f), arg
        if f["shape_known_frac"] >= 0.999:
            break
    if exact and path and best[1]["shape_known_frac"] < 0.999:
        try:
            done = complete_by_run(model, spec, common_ops, path, best, best_arg, exact)
        except Exception:
            done = None
        if done is None:
            best[1]["shape_src"] = -1
        else:
            best = done
    return best


# ---- exact shapes: one real onnxruntime run for graphs static inference cannot finish ----

SEQ_OPS = ("SplitToSequence", "ConcatFromSequence")  # + everything named Sequence*/Optional*


def complete_by_run(model, spec, common_ops, path, best, best_arg, exact):
    """(ref_batch, features) with the unknown shapes filled from a run, or None.
    Static shapes stay authoritative; the result is used only if coverage grows."""
    if os.path.getsize(path) > exact["max_bytes"]:
        return None
    ref_b, f0 = best
    tries = [(best_arg, ref_b)]
    if ref_b != 1 and best_arg is not None:  # not for static graphs with a baked-in batch
        tries.append((1, 1))  # reference batch failed / does not fit -> batch 1 and scale
    if f0.get("max_tensor_elems", 0) * 4 * 6 > exact["mem_bytes"]:
        tries = tries[1:]
    for arg, b in tries:
        m, shp, engine, guessed, inits = infer_static(model, arg, spec)
        want = [
            o
            for n in m.graph.node
            if n.op_type != "Constant"
            and n.op_type not in SEQ_OPS
            and not n.op_type.startswith(("Sequence", "Optional"))
            for o in n.output
            if o and o not in shp
        ]
        got = run_shapes(path, arg, spec, want, exact) if want else None
        if not got:
            continue
        merged = {k: [int(x) for x in v] for k, v in got.items()}
        merged.update(shp)
        f = dict(cost_features(m, merged, engine, guessed, inits, common_ops))
        if f["shape_known_frac"] > f0["shape_known_frac"] + 1e-9:
            f["shape_src"] = 1
            return b, f
    return None


def run_shapes(path, batch, spec, want, exact):
    """Shapes of the tensors `want` from one CPU run in a child process, or None."""
    cpath = None
    if exact.get("cache"):  # successful runs are kept, so repeated runs are fast and identical
        key = json.dumps([os.path.getsize(path), batch, spec, want]).encode()
        cpath = os.path.join(
            exact["cache"], f"{os.path.basename(path)}.{hashlib.md5(key).hexdigest()[:16]}.json"
        )
        try:
            with open(cpath) as fh:
                return json.load(fh)
        except Exception:
            pass
    budget = min(exact["timeout"], exact["deadline"] - time.time())
    if budget < 3:
        return None
    with tempfile.TemporaryDirectory() as td:
        req = os.path.join(td, "req.json")
        with open(req, "w") as fh:
            json.dump(
                {
                    "path": path,
                    "batch": batch,
                    "spec": spec,
                    "want": want,
                    "mem_bytes": exact["mem_bytes"],
                    "threads": exact["threads"],
                },
                fh,
            )
        try:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--shape-run-child", req],
                timeout=budget,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=None if exact.get("debug") else subprocess.DEVNULL,
            )
            with open(req + ".out") as fh:
                got = json.load(fh)
        except Exception:
            return None
    if cpath and got:
        try:
            tmp = f"{cpath}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(got, fh)
            os.replace(tmp, cpath)
        except Exception:
            pass
    return got


def shape_run_child(req_path):
    """Child process: feed the graph like set_inputs / measure.make_feeds do, run once on
    CPU and write the real shapes of the requested tensors to <req>.out. A Shape node is
    attached to every requested tensor and only those are fetched, so memory stays at the
    level of a normal inference."""
    import numpy as np
    import onnxruntime as ort

    with open(req_path) as fh:
        req = json.load(fh)
    try:
        import resource

        lim = int(req["mem_bytes"])
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        pass
    model = onnx.load(req["path"])
    g = model.graph
    inits = {t.name for t in g.initializer} | {t.values.name for t in g.sparse_initializer}
    set_inputs(model, req["batch"], parse_shape_spec(req["spec"]), inits)
    rng = np.random.default_rng(0)
    feeds = {}
    for i in g.input:
        if i.name in inits:
            continue
        tt = i.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tt.elem_type))
        if np.issubdtype(dtype, np.floating):
            data = rng.standard_normal(dims).astype(dtype) if dims else np.asarray(0.5, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            data = np.zeros(dims, dtype=dtype) if dims else np.asarray(100, dtype=dtype)
        elif dtype == np.dtype(bool):
            data = np.ones(dims, dtype=dtype)
        else:
            raise ValueError(f"unsupported input dtype {dtype}")
        feeds[i.name] = data
    del g.value_info[:]
    produced = {o for n in g.node for o in n.output}
    want = [t for t in req["want"] if t in produced]
    del g.output[:]
    names = []
    for k, t in enumerate(want):
        names.append(f"__exact_shape_{k}")
        g.node.append(helper.make_node("Shape", [t], [names[-1]]))
        g.output.append(helper.make_tensor_value_info(names[-1], TensorProto.INT64, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    so.intra_op_num_threads = int(req["threads"])
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    del model
    res = sess.run(names, feeds)
    out = {t: [int(x) for x in r] for t, r in zip(want, res)}
    tmp = req_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, req_path + ".out")


def static_features(model, path: str, common_ops: list) -> dict:
    g = model.graph
    f = collections.defaultdict(float)
    f["file_bytes"] = os.path.getsize(path)
    f["opset"] = max(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        default=0,
    )
    nodes = list(all_nodes(g))
    f["n_nodes"] = len(nodes)
    # op signature of tiny graphs (<= 3 real nodes), used by the family rule in predict.py
    real = sorted(n.op_type for n in nodes if n.op_type not in ("Constant", "Identity"))
    f["sig1"] = "noop" if not real else ("+".join(real) if len(real) <= 3 else "")
    f["n_nodes_nonconst"] = sum(n.op_type != "Constant" for n in nodes)
    f["n_subgraph_nodes"] = len(nodes) - len(g.node)
    f["n_op_types"] = len({n.op_type for n in nodes})
    params = sum(tensor_numel(t) for t in g.initializer)
    params += sum(
        prod(a.t.dims)
        for n in nodes
        if n.op_type == "Constant"
        for a in n.attribute
        if a.type == onnx.AttributeProto.TENSOR
    )
    f["params"] = params
    f["n_initializers"] = len(g.initializer)
    inits = {t.name for t in g.initializer}
    ins = [i for i in g.input if i.name not in inits]
    f["n_inputs"] = len(ins)
    f["n_outputs"] = len(g.output)
    f["input_rank"] = max((len(i.type.tensor_type.shape.dim) for i in ins), default=0)
    f["dynamic_batch"] = int(
        any(
            i.type.tensor_type.shape.dim
            and not i.type.tensor_type.shape.dim[0].HasField("dim_value")
            for i in ins
        )
    )
    f["fp16"] = int(any(t.data_type == TensorProto.FLOAT16 for t in g.initializer))
    cnt = collections.Counter(n.op_type for n in nodes)
    for op, c in cnt.items():
        f[f"cnt_{op}" if op in common_ops else "cnt_other"] += c
    return f


def process_file(args):
    data_dir, fname, rows, common_ops, exact = args
    path = os.path.join(data_dir, "data", fname)
    out = []
    try:
        model = onnx.load(path, load_external_data=False)
        base = dict(static_features(model, path, common_ops))
        base["fp"] = fingerprint(model)
        cache = {}
        for r in rows:
            spec = r["shape"]
            if spec not in cache:
                cache[spec] = reference_features(model, spec, common_ops, path, exact)
            ref_b, gf = cache[spec]
            k = int(float(r["batch"])) / ref_b
            scaled = {
                c: (v * k if c.startswith(BATCH_SCALED) else v) for c, v in gf.items()
            }
            out.append({**r, "has_graph": 1, **base, **scaled, "ref_batch": ref_b})
        err = None
    except Exception as e:  # keep going; report at the end
        err = f"{fname}: {type(e).__name__}: {e}"
        out = [{**r, "has_graph": 0} for r in rows]
    return out, err


def scan_ops(path):
    m = onnx.load(path, load_external_data=False)
    return {n.op_type for n in all_nodes(m.graph)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--min-files-per-op", type=int, default=5)
    ap.add_argument("--exact-shapes", action=argparse.BooleanOptionalAction, default=True,
                    help="complete unknown shapes by one real onnxruntime run (graphs with coverage < 0.999)")
    ap.add_argument("--exact-timeout", type=float, default=60, help="seconds per run")
    ap.add_argument("--exact-budget", type=float, default=50,
                    help="wall-clock seconds after which no run may still be going")
    ap.add_argument("--exact-max-mb", type=float, default=700, help="skip larger model files")
    ap.add_argument("--exact-mem-gb", type=float, default=12, help="address-space cap of a run")
    ap.add_argument("--exact-threads", type=int, default=4)
    ap.add_argument("--exact-cache", default=None,
                    help="directory with shapes of earlier successful runs (default <data>/exact_shapes, '' = off)")
    ap.add_argument("--exact-debug", action="store_true", help="show stderr of the runs")
    ap.add_argument("--shape-run-child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.shape_run_child:
        shape_run_child(a.shape_run_child)
        return

    rows = []
    for split in ("train", "test"):
        for r in csv.DictReader(open(os.path.join(a.data, f"{split}.csv"))):
            r["split"] = split
            r.setdefault("latency", "")
            rows.append(r)
    present = set(os.listdir(os.path.join(a.data, "data")))
    by_file = collections.defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    files = sorted(
        (f for f in by_file if f in present),
        key=lambda f: -os.path.getsize(os.path.join(a.data, "data", f)),
    )
    with Pool(a.jobs) as p:
        op_sets = p.map(
            scan_ops, [os.path.join(a.data, "data", f) for f in files], chunksize=8
        )
    op_files = collections.Counter(op for s in op_sets for op in s)
    common_ops = sorted(op for op, c in op_files.items() if c >= a.min_files_per_op)
    print(f"{len(files)} graphs, {len(common_ops)} common op types", flush=True)

    results, errors = [], []
    for f in by_file:
        if f not in present:
            results += [{**r, "has_graph": 0} for r in by_file[f]]
    exact = None
    if a.exact_shapes:
        exact = {
            "deadline": time.time() + a.exact_budget,
            "timeout": a.exact_timeout,
            "max_bytes": a.exact_max_mb * 1e6,
            "mem_bytes": int(a.exact_mem_gb * 2**30),
            "threads": a.exact_threads,
            "debug": a.exact_debug,
            "cache": os.path.join(a.data, "exact_shapes") if a.exact_cache is None else a.exact_cache,
        }
        if exact["cache"]:
            os.makedirs(exact["cache"], exist_ok=True)
    with Pool(a.jobs, maxtasksperchild=20) as p:
        jobs = [(a.data, f, by_file[f], common_ops, exact) for f in files]
        for i, (out, err) in enumerate(p.imap_unordered(process_file, jobs), 1):
            results += out
            if err:
                errors.append(err)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)

    for r in results:
        b = float(r["batch"])
        r["batch"] = b
        r["log_batch"] = math.log2(b)
        r["shape_override"] = int(r["shape"] != "default")

    order = {r["id"]: i for i, r in enumerate(rows)}
    results.sort(key=lambda r: order[r["id"]])
    head = ["id", "split", "file", "batch", "shape", "latency"]
    cols = head + sorted({k for r in results for k in r} - set(head))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            # graph rows: absent counters mean 0; rows without graph: empty (NaN)
            fill = 0 if r["has_graph"] else ""
            w.writerow({c: r.get(c, fill) for c in cols})
    print(f"wrote {a.out}: {len(results)} rows x {len(cols)} cols", flush=True)
    src = collections.Counter(str(r.get("shape_src", "")) for r in results if r["has_graph"])
    print("shape_src rows:", dict(src), flush=True)
    if errors:
        print(
            f"{len(errors)} graphs failed:", *errors[:20], sep="\n  ", file=sys.stderr
        )


if __name__ == "__main__":
    main()
