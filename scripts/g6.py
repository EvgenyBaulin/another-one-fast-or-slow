#!/usr/bin/env python3
"""G6: additive graph neural network (optional ensemble member).

Every ONNX node gets symbolic quantities computed by formulas (output elements,
input elements, MACs, graph-input / graph-output elements it touches, kernel,
stride, groups, fused / standalone flag, op type and category). A small network
turns the node description into six non-negative RATES; the node cost is
    rate_out * out_elems + rate_mac * MACs + rate_in * graph_input_elems
    + rate_gout * graph_output_elems + rate_const + rate_overhead / nodes_in_graph
and the graph latency (ms) is softplus(const) + sum of node costs. With the
network output at zero this is the analytic base (0.25 ms + 2 ns per graph
input / output element); training on log1p error moves the rates per op, size
and context.

The default is an ensemble of two kinds of rate network with equal total weight:
  layers 0 - per-node model, the node sees only itself and graph-level inputs (14 seeds);
  layers 2 - two rounds of message passing (mean over predecessors, successors and
             the whole graph), so a node sees its neighbours (5 seeds).
Optional graph-level inputs (`--gfeat`, default none): `base2` gives every node the log1p
prediction of the kernel-level analytic base (scripts/base2.py; out-of-fold for train
rows, fitted on all train rows for test rows), `f2` the columns of feat2.csv. base2 makes
G6 alone better on single-row files (CV 0.254 -> 0.232..0.239) but the blend with the
stack does not change beyond noise, so it is off by default; f2 is worse on single-row
files. If such inputs fail for any reason the script says so and runs without them.
The networks are spread over the GPUs of `--gpus` (default 0,1; one worker process per
GPU; a failed worker's networks are recomputed in the main process). See HANDOFF.md.

Usage:
  python scripts/g6.py --data data               # about 4 min on two GPUs, cache warm
  python scripts/g6.py --data data --seeds 8,3   # about 2.5 min on two GPUs
  python scripts/g6.py --data data --gpus 1      # one GPU (or set CUDA_VISIBLE_DEVICES)

Writes (pred in log1p space, columns id,pred):
  <data>/preds/g6_gnn_oof.csv   every train row, folds from common.make_folds
  <data>/preds/g6_gnn_test.csv  test rows with has_graph == 1
Node tables are cached in <data>/g6_cache/ (one npz per file / shape / reference
batch; only missing ones are built, 748 tables take about 10 s).

Two interpreters are needed on the server: the cache step needs onnx + onnxruntime
(main venv), the training step needs torch (~/hacklab/fos/.venv-g5). The script
re-executes itself with the right one, like g5.py does.
"""

import argparse
import hashlib
import subprocess
import tempfile
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, HERE)
import common as C  # noqa: E402

warnings.filterwarnings("ignore")

CACHE_VERSION = 1
# raw per-node columns stored in the cache (values at the reference batch)
RAW = ("oe", "ie", "w", "macs", "gin", "gout", "kernel", "stride", "group", "fused", "unfused",
       "known", "rank", "n_act_in", "macs_unknown", "cat")  # fmt: skip
RI = {c: i for i, c in enumerate(RAW)}
CATS = ("elementwise", "movement", "norm", "reduce", "pool", "convlike", "other")
BATCH_SCALED = ("oe", "ie", "macs", "gin", "gout")


# ---------------------------------------------------------------- cache (needs onnx)
def cache_path(cache_dir: str, fname: str, spec: str, ref_b: int) -> str:
    h = hashlib.md5(spec.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{fname}__{h}__b{int(ref_b)}__v{CACHE_VERSION}.npz")


def build_one(args):
    """Node table of one (file, shape spec) at the reference batch -> npz. Returns error text or None."""
    data_dir, fname, spec, ref_b, out = args
    try:
        import onnx
        from onnx import shape_inference
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        import features as F

        model = onnx.load(os.path.join(data_dir, "data", fname), load_external_data=False)
        nb = F.native_batch(model) if spec == "default" else None
        arg = None if (nb and nb == int(ref_b)) else int(ref_b)  # same call as features.reference_features
        m, moved = F.strip_for_inference(model)
        del model
        inits = {t.name for t in m.graph.initializer} | moved
        F.set_inputs(m, arg, F.parse_shape_spec(spec), inits)
        try:  # same engine order as features.graph_features
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
        shp = F.shape_map(m)
        g = m.graph
        nodes = [n for n in F.all_nodes(g) if n.op_type != "Constant"]
        const = {o for n in F.all_nodes(g) if n.op_type == "Constant" for o in n.output} | inits
        prod = F.prod

        def numel(name):
            return float(prod(shp[name])) if name in shp else 0.0

        producer = {}
        for j, n in enumerate(nodes):
            for o in n.output:
                if o:
                    producer[o] = j
        consumers = {}
        for j, n in enumerate(nodes):
            for i in set(n.input):
                if i and i not in const:
                    consumers.setdefault(i, []).append(j)

        raw = np.zeros((len(nodes), len(RAW)), dtype=np.float64)
        ops, src, dst = [], [], []
        fused_src = {}
        n_out = n_known = 0
        for j, n in enumerate(nodes):
            op = n.op_type
            ops.append(op)
            r = raw[j]
            outs = [o for o in n.output if o]
            n_out += len(outs)
            n_known += sum(o in shp for o in outs)
            r[RI["oe"]] = sum(numel(o) for o in outs)
            r[RI["known"]] = float(all(o in shp for o in outs))
            r[RI["rank"]] = len(shp[outs[0]]) if outs and outs[0] in shp else 0
            act = [i for i in n.input if i and i not in const]
            r[RI["ie"]] = sum(numel(i) for i in act)
            r[RI["w"]] = sum(numel(i) for i in n.input if i and i in const)
            r[RI["n_act_in"]] = len(act)
            macs = F.node_macs(n, shp)
            r[RI["macs"]] = max(macs, 0)
            r[RI["macs_unknown"]] = float(macs < 0)
            at = {a.name: a for a in n.attribute}
            ks = list(at["kernel_shape"].ints) if "kernel_shape" in at else None
            if ks is None and op in ("Conv", "ConvTranspose") and len(n.input) > 1 and n.input[1] in shp:
                ks = shp[n.input[1]][2:]
            r[RI["kernel"]] = max(1, prod(ks)) if ks else 1
            r[RI["stride"]] = max(1, prod(list(at["strides"].ints))) if "strides" in at else 1
            r[RI["group"]] = max(1, at["group"].i) if "group" in at else 1
            # fused / standalone, same rule as features.graph_features
            ins_fused = [fused_src.get(i, False) for i in n.input if i]
            if op in F.CONV_LIKE:
                for o in n.output:
                    fused_src[o] = True
            elif op in F.NORM:
                fz = bool(ins_fused and ins_fused[0])
                r[RI["fused" if fz else "unfused"]] = 1
                for o in n.output:
                    fused_src[o] = fz
            elif op in F.ELEMENTWISE:
                r[RI["fused" if any(ins_fused) else "unfused"]] = 1
            cat = ("elementwise" if op in F.ELEMENTWISE else "movement" if op in F.MOVEMENT else "norm" if op in F.NORM
                   else "reduce" if op in F.REDUCE else "pool" if op in F.POOL
                   else "convlike" if (op in F.CONV_LIKE or macs > 0) else "other")  # fmt: skip
            r[RI["cat"]] = CATS.index(cat)
            for p in {producer[i] for i in act if i in producer}:
                if p != j:
                    src.append(p)
                    dst.append(j)

        # graph inputs / outputs: elements go to the nodes that touch them, the rest stays graph-level
        in_total = in_rest = out_total = out_rest = 0.0
        for i in g.input:
            if i.name in inits:
                continue
            e = numel(i.name)
            in_total += e
            cs = consumers.get(i.name)
            if cs:
                for c in cs:
                    raw[c, RI["gin"]] += e / len(cs)
            else:
                in_rest += e
        for o in g.output:
            e = numel(o.name)
            out_total += e
            if o.name in producer:
                raw[producer[o.name], RI["gout"]] += e
            else:
                out_rest += e
        meta = np.array([ref_b, in_total, out_total, in_rest, out_rest, n_known / n_out if n_out else 1.0], dtype=np.float64)
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, ops=np.array(ops, dtype=str), raw=raw, edges=np.array([src, dst], dtype=np.int32).reshape(2, -1), meta=meta)
        os.replace(tmp, out)
        return None
    except Exception as e:  # the row falls back to a one-node pseudo graph
        try:
            open(out + ".fail", "w").write(f"{type(e).__name__}: {e}"[:500])
        except OSError:
            pass
        return f"{fname} [{spec}]: {type(e).__name__}: {str(e)[:200]}"


def needed_tables(df: pd.DataFrame, cache_dir: str):
    """(file, spec, ref_batch, path) for every graph row; ref_batch comes from features.csv."""
    g = df[(df.has_graph == 1) & df.ref_batch.notna()]
    out = []
    for (f, s, b) in g[["file", "shape", "ref_batch"]].drop_duplicates().itertuples(index=False):
        out.append((f, s, int(b), cache_path(cache_dir, f, s, int(b))))
    return out


def build_cache(data_dir: str, todo: list, jobs: int) -> None:
    from multiprocessing import Pool

    t0 = time.time()
    todo = sorted(todo, key=lambda t: -os.path.getsize(os.path.join(data_dir, "data", t[0])))
    args = [(data_dir, f, s, b, p) for f, s, b, p in todo]
    errors = []
    with Pool(jobs, maxtasksperchild=20) as pool:
        for i, err in enumerate(pool.imap_unordered(build_one, args), 1):
            if err:
                errors.append(err)
            if i % 100 == 0 or i == len(args):
                print(f"  cache [{i}/{len(args)}] {time.time() - t0:.0f}s", flush=True)
    if errors:
        print(f"{len(errors)} node tables failed (pseudo graph is used):", *errors[:10], sep="\n  ", flush=True)


def other_python(venv: str):
    p = os.path.join(ROOT, venv, "bin", "python")
    return p if os.path.exists(p) else None


def reexec(py: str, hop: str):
    os.execve(py, [py] + sys.argv, dict(os.environ, G6_HOP=hop))


# ---------------------------------------------------------------- row -> arrays
def load_rows(df: pd.DataFrame, cache_dir: str):
    """Concatenated node arrays for all rows of df (one graph per row)."""
    tables = {}
    z = lambda r, c: float(r[c]) if c in r and np.isfinite(r[c]) else 0.0  # noqa: E731
    ops_all, raw_all, edges_all, gid_all, k_all = [], [], [], [], []
    G = len(df)
    gmeta = np.zeros((G, 5))  # batch, n_nodes, input_elems, output_elems (row batch), pseudo flag
    rest = np.zeros((G, 2))
    off = n_pseudo = 0
    for gi, (_, r) in enumerate(df.iterrows()):
        t = None
        if r.has_graph == 1 and np.isfinite(r.ref_batch):
            p = cache_path(cache_dir, r.file, r["shape"], int(r.ref_batch))
            if p not in tables:
                if os.path.exists(p):
                    d = np.load(p)
                    tables[p] = (d["ops"], d["raw"], d["edges"], d["meta"]) if len(d["ops"]) else None
                else:
                    tables[p] = None
            t = tables[p]
        if t is None:  # no node table: one unknown node carrying the row totals from features.csv
            n_pseudo += 1
            raw = np.zeros((1, len(RAW)))
            raw[0, RI["oe"]], raw[0, RI["ie"]], raw[0, RI["macs"]] = z(r, "out_elems"), z(r, "in_elems"), z(r, "macs")
            raw[0, RI["gin"]], raw[0, RI["gout"]] = z(r, "input_elems"), z(r, "output_elems")
            raw[0, RI["kernel"]] = raw[0, RI["stride"]] = raw[0, RI["group"]] = 1
            raw[0, RI["cat"]] = CATS.index("other")
            ops, edges, k = np.array(["<none>"]), np.zeros((2, 0), dtype=np.int32), 1.0
            gmeta[gi] = (r.batch, max(z(r, "n_nodes_nonconst"), 1), z(r, "input_elems"), z(r, "output_elems"), 1)
        else:
            ops, raw, edges, meta = t
            k = float(r.batch) / meta[0]
            gmeta[gi] = (r.batch, len(ops), meta[1] * k, meta[2] * k, 0)
            rest[gi] = (meta[3] * k, meta[4] * k)
        ops_all.append(ops)
        raw_all.append(raw)
        edges_all.append(edges.astype(np.int64) + off)
        gid_all.append(np.full(len(ops), gi, dtype=np.int64))
        k_all.append(np.full(len(ops), k))
        off += len(ops)
    return dict(
        ops=np.concatenate(ops_all), raw=np.concatenate(raw_all), edges=np.concatenate(edges_all, axis=1),
        gid=np.concatenate(gid_all), k=np.concatenate(k_all), gmeta=gmeta, rest=rest, n_pseudo=n_pseudo,
    )  # fmt: skip


def node_inputs(D: dict):
    """Model inputs: x (network features, all O(1)) and q (quantities the rates multiply)."""
    raw, k, gid = D["raw"].copy(), D["k"], D["gid"]
    for c in BATCH_SCALED:  # shape inference sometimes leaves negative dims: such counts are treated as unknown (0)
        raw[:, RI[c]] = np.clip(np.nan_to_num(raw[:, RI[c]] * k, nan=0.0, posinf=0.0), 0, None)
    raw[:, RI["w"]] = np.clip(raw[:, RI["w"]], 0, None)
    batch = D["gmeta"][gid, 0]
    per_sample_oe = raw[:, RI["oe"]] / np.maximum(batch, 1.0)
    col = lambda c: raw[:, RI[c]]  # noqa: E731
    cat = np.eye(len(CATS))[col("cat").astype(int)]
    gm = D["gmeta"][gid]
    # log sizes centred at e^12 elements, 4 log units per feature unit: the per-element cost changes
    # between ~4M and ~10M elements, the network has to resolve that range
    sz = lambda v: (np.log1p(v) - 12) / 4  # noqa: E731
    x = np.column_stack([
        sz(col("oe")), sz(col("ie")), sz(col("gin")), sz(col("gout")), sz(gm[:, 2] + gm[:, 3]), sz(per_sample_oe),  # SIZE_COLS
        sz(col("w")), (np.log1p(col("macs")) - 16) / 6, col("gin") > 0, col("gout") > 0,
        col("fused"), col("unfused"), np.log(col("kernel")) / 3, np.log(col("stride")) / 2, np.log(col("group")) / 6,
        col("known"), np.minimum(col("rank"), 6) / 5, np.minimum(col("n_act_in"), 4) / 4, col("macs_unknown"), cat,
        np.log2(np.maximum(gm[:, 0], 1)) / 7, (np.log1p(gm[:, 1]) - 2) / 3, gm[:, 1] == 1, gm[:, 4],
    ]).astype(np.float32)  # fmt: skip
    # quantities the rates multiply; scaled so that the initial rates are O(1):
    # out elems (1e7), MACs (1e9), graph-input elems (1e6), graph-output elems (1e6),
    # per-node constant (1 us), share of the per-graph overhead (1 / nodes in the graph, 0.1 ms)
    q = np.column_stack([col("oe") / 1e7, col("macs") / 1e9, col("gin") / 1e6, col("gout") / 1e6,
                         np.full(len(raw), 1e-3), 0.1 / np.maximum(gm[:, 1], 1)]).astype(np.float32)  # fmt: skip
    return x, q


N_SIZE_COLS = 6  # leading columns of x that get per-graph size noise during training


def graph_inputs(mode: str, data_dir: str, tr: pd.DataFrame, te: pd.DataFrame, y: np.ndarray):
    """Extra graph-level inputs, one row per (train + test) row, or None.
    f2:    columns f2_* of feat2.csv (already merged by common.load), signed log, z-scored on train;
    base2: log1p prediction of the kernel-level analytic base (scripts/base2.py): out-of-fold for
           train rows (base2.crossfit, grouped by fp as in predict.fit_predict_all), fitted on all
           train rows for test rows.
    Any failure returns None: the caller keeps the old inputs."""
    if mode == "none":
        return None
    try:
        rows = pd.concat([tr, te], ignore_index=True)
        parts = []
        if mode in ("f2", "both"):
            cols = [c for c in rows.columns if c.startswith("f2_") and pd.api.types.is_numeric_dtype(rows[c])]
            if not cols:
                raise RuntimeError("no f2_* columns (feat2.csv missing?)")
            V = rows[cols].astype(float).values
            V = np.sign(V) * np.log1p(np.abs(V))
            mu, sd = np.nanmean(V[: len(tr)], axis=0), np.nanstd(V[: len(tr)], axis=0)
            keep = np.isfinite(sd) & (sd > 1e-9)
            Z = np.clip((V[:, keep] - mu[keep]) / sd[keep], -5, 5)
            parts.append(np.nan_to_num(Z, nan=0.0))
        if mode in ("base2", "both"):
            import base2 as B2

            B2.DATA_DIR = data_dir
            grp = tr.fp.fillna(tr.file).astype(str).values
            b_tr, b_te, _ = B2.crossfit(tr, np.asarray(y, float), grp, te if len(te) else None)
            b = np.concatenate([b_tr, b_te if b_te is not None else np.zeros(0)])
            if len(b) != len(rows) or not np.isfinite(b).all():
                raise RuntimeError("base2 prediction is incomplete")
            parts.append(((b - 1.5) / 1.5)[:, None])
        G = np.hstack(parts).astype(np.float32)
        assert np.isfinite(G).all()
        return G
    except Exception as e:
        print(f"!!! g6: graph-level inputs '{mode}' failed ({type(e).__name__}: {str(e)[:200]}); running WITHOUT them (old behaviour)", flush=True)
        return None


# ---------------------------------------------------------------- model (needs torch)
def train_stage(a, df, cache_dir, t0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[(df.split == "test") & (df.has_graph == 1)].reset_index(drop=True)
    rows = pd.concat([tr, te], ignore_index=True)
    ntr = len(tr)
    y = tr.y.values.astype(np.float32)
    folds = C.make_folds(tr)
    single = (tr.groupby("file").file.transform("size") == 1).values

    D = load_rows(rows, cache_dir)
    x_np, q_np = node_inputs(D)
    # extra graph-level inputs (workers get the matrix from the parent)
    Gf = None
    if a.gfeat_file:
        Gf = np.load(a.gfeat_file)
    elif a.gfeat != "none":
        Gf = graph_inputs(a.gfeat, a.data, tr, te, y)
    if Gf is not None and len(Gf) == len(rows):
        x_np = np.hstack([x_np, Gf[D["gid"]]]).astype(np.float32)
        if not a.worker:
            print(f"graph-level inputs '{a.gfeat}': {Gf.shape[1]} columns, {time.time() - t0:.0f}s", flush=True)
    # op vocabulary: ops seen in at least 3 train files; everything else is id 0
    tr_node = D["gid"] < ntr
    files_of_node = rows.file.values[D["gid"]]
    opf = pd.DataFrame({"op": D["ops"][tr_node], "file": files_of_node[tr_node]}).drop_duplicates()
    cnt = opf.op.value_counts()
    vocab = {op: i + 1 for i, op in enumerate(sorted(cnt[cnt >= 3].index))}
    op_np = np.array([vocab.get(o, 0) for o in D["ops"]], dtype=np.int64)
    te_node = ~tr_node
    print(f"rows: train {ntr}, test {len(te)}; nodes {len(op_np)} (train {int(tr_node.sum())}), edges {D['edges'].shape[1]}, "
          f"op vocab {len(vocab)}, pseudo graphs {D['n_pseudo']}, unknown-op nodes: train {int((op_np[tr_node] == 0).sum())}, "
          f"test {int((op_np[te_node] == 0).sum())}; device {dev}; {time.time() - t0:.0f}s" if not a.worker else f"worker on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: {a.worker}", flush=True)  # fmt: skip

    T = lambda v, dt=None: torch.as_tensor(v, dtype=dt).to(dev)  # noqa: E731
    X, Q, OP = T(x_np), T(q_np), T(op_np)
    REST = T((D["rest"] / 1e6).astype(np.float32))
    Y = T(y)
    gid_np, e_np = D["gid"], D["edges"]

    class Batch:
        """Sub-batch of whole graphs (rows)."""

        def __init__(self, row_idx):
            row_idx = np.asarray(row_idx)
            gmap = np.full(len(rows), -1, dtype=np.int64)
            gmap[row_idx] = np.arange(len(row_idx))
            nmask = gmap[gid_np] >= 0
            nmap = np.cumsum(nmask) - 1
            emask = nmask[e_np[0]]
            ni = T(np.nonzero(nmask)[0])
            self.x, self.q, self.op = X[ni], Q[ni], OP[ni]
            self.gid = T(gmap[gid_np[nmask]])
            self.rest = REST[T(row_idx)]
            self.G, self.N = len(row_idx), int(nmask.sum())
            # row-normalised sparse operators: mean over predecessors, over successors, over the node's graph
            src, dst, gid = nmap[e_np[0][emask]], nmap[e_np[1][emask]], gmap[gid_np[nmask]]

            def mean_op(rows, cols, n_rows, n_cols):
                deg = np.maximum(np.bincount(rows, minlength=n_rows), 1)
                m = torch.sparse_coo_tensor(np.vstack([rows, cols]), (1.0 / deg[rows]).astype(np.float32), (n_rows, n_cols))
                return m.coalesce().to_sparse_csr().to(dev)

            self.A_in = mean_op(dst, src, self.N, self.N)
            self.A_out = mean_op(src, dst, self.N, self.N)
            self.P = mean_op(gid, np.arange(self.N), self.G, self.N)  # graph mean

    _batches = {}

    def get_batch(idx):  # the same folds are used by every member
        idx = np.asarray(idx)
        key = (len(idx), int(idx[0]), int(idx[-1]), int(idx.sum()))
        if key not in _batches:
            _batches[key] = Batch(idx)
        return _batches[key]

    def inv_softplus(v):
        return torch.log(torch.expm1(torch.tensor(v, dtype=torch.float32)))

    class Net(nn.Module):
        def __init__(self, n_ops, n_num, hid, emb, layers, drop, opdrop, sizenoise):
            super().__init__()
            self.emb = nn.Embedding(n_ops, emb)
            self.inp = nn.Sequential(nn.Linear(n_num + emb, hid), nn.SiLU(), nn.Linear(hid, hid))
            self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
            self.mp = nn.ModuleList([nn.Sequential(nn.Linear(4 * hid, hid), nn.SiLU(), nn.Linear(hid, hid)) for _ in range(layers)])
            self.out_norm = nn.LayerNorm(hid)
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 6))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            # start = analytic base: 0.25 ms (0.05 global + 0.2 per-graph overhead, which the network can
            # lower for Identity / pooling-only graphs) + 2 ns per graph input / output element; small rates elsewhere
            self.bias = nn.Parameter(inv_softplus([1.0, 1.0, 2.0, 2.0, 2.0, 2.0]))
            self.c0 = nn.Parameter(inv_softplus(0.05))
            self.rest = nn.Parameter(inv_softplus([2.0, 2.0]))
            self.drop, self.opdrop, self.sizenoise = nn.Dropout(drop), opdrop, sizenoise

        def forward(self, b):
            rate = Fn.softplus(self.bias + self.trunk(b))
            cost = (rate * b.q).sum(1)
            ms = Fn.softplus(self.c0) + torch.zeros(b.G, device=cost.device).index_add_(0, b.gid, cost)
            ms = ms + (Fn.softplus(self.rest) * b.rest).sum(1)
            return torch.log1p(ms)

        def trunk(self, b):
            op = b.op
            if self.training and self.opdrop > 0:  # an unseen op must fall back on category + sizes
                op = torch.where(torch.rand(op.shape, device=op.device) < self.opdrop, torch.zeros_like(op), op)
            x = b.x
            if self.training and self.sizenoise > 0:  # same shift for a whole graph: exact sizes cannot be memorised
                eps = (torch.randn(b.G, 1, device=x.device) * (self.sizenoise / 4)).index_select(0, b.gid)
                x = torch.cat([x[:, :N_SIZE_COLS] + eps, x[:, N_SIZE_COLS:]], dim=1)
            h = self.inp(torch.cat([x, self.emb(op)], dim=1))
            for norm, mp in zip(self.norms, self.mp):
                hn = norm(h)
                # means over predecessors / successors / the whole graph as sparse products
                m_in, m_out = torch.sparse.mm(b.A_in, hn), torch.sparse.mm(b.A_out, hn)
                g = torch.sparse.mm(b.P, hn).index_select(0, b.gid)
                h = h + self.drop(mp(torch.cat([hn, m_in, m_out, g], dim=1)))
            return self.head(self.drop(self.out_norm(h)))

    def fit_predict(train_idx, pred_idx, seed, layers):
        torch.manual_seed(seed)
        np.random.seed(seed)
        bt, bp = get_batch(train_idx), get_batch(pred_idx)
        yt = Y[T(np.asarray(train_idx))]
        wt = T(np.where(single[np.asarray(train_idx)], 1.0, a.multiw).astype(np.float32))  # row weights
        net = Net(len(vocab) + 1, X.shape[1], a.hid, a.emb, layers, a.drop, a.opdrop, a.sizenoise).to(dev)
        decay = [p for n, p in net.named_parameters() if p.ndim > 1 and not n.startswith("emb")]
        rest = [p for n, p in net.named_parameters() if not (p.ndim > 1 and not n.startswith("emb"))]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}], lr=a.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs, pct_start=0.1)
        for _ in range(a.epochs):
            net.train()
            err = net(bt) - yt
            loss_row = Fn.huber_loss(err, torch.zeros_like(err), delta=a.huber, reduction="none") if a.huber > 0 else err**2
            keep = (torch.rand_like(err) < a.rowkeep).float() * wt  # random subset of graphs carries the loss each step
            loss = (loss_row * keep).sum() / keep.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            tr_fit = float(((net(bt) - yt) ** 2).mean().sqrt())
            return net(bp).float().cpu().numpy(), tr_fit

    def score(p):
        multi = C.rmse(p[~single], y[~single]) if (~single).any() else float("nan")
        return C.rmse(p, y), C.rmse(p[single], y[single]), multi

    # ensemble members: every --layers value (0 = per-node model, no message passing) x seeds;
    # the two kinds get equal total weight
    kinds = [int(v) for v in str(a.layers).split(",") if v != ""]
    n_seeds = [int(v) for v in str(a.seeds).split(",") if v != ""]
    n_seeds = (n_seeds * len(kinds))[: len(kinds)] if len(n_seeds) == 1 else n_seeds
    assert len(n_seeds) == len(kinds), "--seeds needs one value or one per --layers value"
    seeds = [(L, s) for L, n in zip(kinds, n_seeds) for s in range(n)]
    members = lambda L: [i for i, (l, _) in enumerate(seeds) if l == L]  # noqa: E731
    combine = lambda P: np.mean([P[members(L)].mean(0) for L in kinds], axis=0)  # noqa: E731
    if a.hidden_sim:  # last-hour imitation: hide the single-row files of each split, train on the rest
        d = pd.read_csv(a.hidden_sim, usecols=["seed", "id"])
        outs = []
        for hs in [int(v) for v in a.hidden_seeds.split(",") if v != ""]:
            h = tr.id.isin(set(d.id[d.seed == hs])).values
            known, q = np.nonzero(~h)[0], np.nonzero(h)[0]
            if not len(q):
                print(f"hidden split {hs}: no rows in {a.hidden_sim}", flush=True)
                continue
            _batches.clear()
            if Gf is not None:  # the base prediction must not see the hidden rows: refit it on the known ones
                Gs = graph_inputs(a.gfeat, a.data, tr.iloc[known].reset_index(drop=True), tr.iloc[q].reset_index(drop=True), y[known])
                assert Gs is not None and Gs.shape[1] == Gf.shape[1], "graph-level inputs differ between splits"
                Gh = np.zeros((len(rows), Gs.shape[1]), dtype=np.float32)
                Gh[known], Gh[q] = Gs[: len(known)], Gs[len(known) :]
                xh = x_np.copy()
                xh[:, -Gs.shape[1] :] = Gh[D["gid"]]
                X = T(xh)
            Pm = np.array([fit_predict(known, q, 5000 + 100 * hs + sd, L)[0] for L, sd in seeds])
            p = combine(Pm)
            outs.append(pd.DataFrame({"seed": hs, "id": tr.id.values[q], "g6": p}))
            print("hidden split %d: %d hidden rows, G6 alone %.4f | %.0fs" % (hs, len(q), C.rmse(p, y[q]), time.time() - t0), flush=True)
        pd.concat(outs, ignore_index=True).to_csv(a.hidden_out, index=False)
        print(f"wrote {a.hidden_out}", flush=True)
        return

    want_test = bool(len(te)) and not a.no_test
    all_tr, te_idx = np.arange(ntr), np.arange(ntr, len(rows))

    def run_member(L, seed):
        """Out-of-fold predictions of one network kind / seed and its test predictions."""
        oof_m, fit = np.zeros(ntr), []
        for kf in range(5):
            va = np.nonzero(folds == kf)[0]
            tn = np.nonzero(folds != kf)[0]
            p, tf = fit_predict(tn, va, seed * 100 + kf, L)
            oof_m[va] = p
            fit.append(tf)
        test_m = fit_predict(all_tr, te_idx, 1000 + seed, L)[0] if want_test else np.zeros(0)
        print("layers %d seed %d: all %.4f | single %.4f | multi %.4f | train fit %.3f | %.0fs" % ((L, seed) + score(oof_m) + (np.mean(fit), time.time() - t0)), flush=True)
        return oof_m, test_m

    if a.worker:  # sub-process of a multi-GPU run: compute the given members, store them, exit
        mem = [tuple(int(v) for v in m.split(":")) for m in a.worker.split(";") if m]
        res = [run_member(L, s) for L, s in mem]
        np.savez(a.worker_out, members=np.array(mem), oof=np.array([r[0] for r in res]), test=np.array([r[1] for r in res]))
        return

    done = {}
    gpu_ids = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip() != ""]
    if dev.type == "cuda" and len(gpu_ids) >= 2 and torch.cuda.device_count() >= 2 and len(seeds) > 1:
        # one worker process per GPU; message passing costs about five times the per-node kind
        load = {g: 0.0 for g in gpu_ids}
        parts = {g: [] for g in gpu_ids}
        for L, s in sorted(seeds, key=lambda m: -(5 if m[0] else 1)):
            g = min(load, key=load.get)
            parts[g].append((L, s))
            load[g] += 5 if L else 1
        tmp = tempfile.mkdtemp(prefix="g6_")
        extra = ["--gfeat", "none"]  # workers never rebuild the graph-level inputs themselves
        if Gf is not None:
            np.save(os.path.join(tmp, "gfeat.npy"), Gf)
            extra = ["--gfeat-file", os.path.join(tmp, "gfeat.npy")]
        procs = []
        for g, mem in parts.items():
            if not mem:
                continue
            out = os.path.join(tmp, f"w{g}.npz")
            cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + extra + ["--worker", ";".join(f"{L}:{s}" for L, s in mem), "--worker-out", out]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, G6_HOP="train", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            procs.append((g, out, subprocess.Popen(cmd, env=env)))
        print(f"{len(seeds)} networks on GPUs {','.join(parts)}: " + " | ".join(f"GPU {g}: {len(m)}" for g, m in parts.items()), flush=True)
        for g, out, pr in procs:
            rc = pr.wait()
            try:
                r = np.load(out)
                for (L, s), o, t in zip(r["members"], r["oof"], r["test"]):
                    if np.isfinite(o).all() and np.isfinite(t).all():
                        done[(int(L), int(s))] = (o, t)
            except Exception as e:
                print(f"!!! g6: worker on GPU {g} failed (exit code {rc}, {type(e).__name__}); its networks are computed in the main process", flush=True)
        try:
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)
        except OSError:
            pass
    for L, s in seeds:  # single GPU / CPU, or whatever the workers did not deliver
        if (L, s) not in done:
            done[(L, s)] = run_member(L, s)
    oof_seed = np.array([done[m][0] for m in seeds], dtype=np.float64)
    oof = combine(oof_seed)
    for L in kinds:
        if len(kinds) > 1:
            print("layers %d, %d seeds: all %.4f | single %.4f | multi %.4f" % ((L, len(members(L))) + score(oof_seed[members(L)].mean(0))), flush=True)
    if a.dump:  # per-seed OOF for error analysis
        d = pd.DataFrame({"id": tr.id.values, "y": y, "fold": folds, "single": single.astype(int)})
        for si, (L, sd) in enumerate(seeds):
            d[f"s{si}_L{L}"] = oof_seed[si]
        d.to_csv(a.dump, index=False)

    test_pred = np.zeros(0)
    if want_test:
        test_pred = combine(np.array([done[m][1] for m in seeds]))
        print(f"test predictions done, {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all(), "non-finite predictions"
    # safety net for exotic graphs: an additive model can extrapolate far beyond anything seen in train
    lo, hi = 0.0, float(y.max()) + 0.7
    n_clip = int(((oof < lo) | (oof > hi)).sum() + ((test_pred < lo) | (test_pred > hi)).sum())
    oof, test_pred = np.clip(oof, lo, hi), np.clip(test_pred, lo, hi)
    if n_clip:
        print(f"clipped {n_clip} predictions to [{lo:.2f}, {hi:.2f}]", flush=True)

    if not a.no_save:
        prefix = a.out_prefix or os.path.join(a.data, "preds", "g6_gnn")
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        pd.DataFrame({"id": tr.id.values, "pred": oof}).to_csv(prefix + "_oof.csv", index=False)
        if not a.no_test:
            pd.DataFrame({"id": te.id.values, "pred": test_pred}).to_csv(prefix + "_test.csv", index=False)
        print(f"wrote {prefix}_oof.csv ({ntr} rows)" + ("" if a.no_test else f", {prefix}_test.csv ({len(test_pred)} rows)"), flush=True)

    print("\nCV RMSLE (log1p space)          all     single   multi")
    print("%-28s %.4f  %.4f  %.4f" % ((f"g6_gnn ({len(seeds)} models)",) + score(oof)))
    sp = a.stack_oof or os.path.join(a.data, "preds", "main_STACK_oof.csv")
    if os.path.exists(sp):
        st = pd.read_csv(sp).set_index("id").pred.reindex(tr.id.values).values
        if np.isfinite(st).all():
            print("%-28s %.4f  %.4f  %.4f" % (("main_STACK",) + score(st)))
            for w in (0.5, 0.3, 0.15):
                print("%-28s %.4f  %.4f  %.4f" % ((f"{w:.2f} g6 + {1 - w:.2f} STACK",) + score(w * oof + (1 - w) * st)))
        else:
            print("main_STACK_oof.csv does not cover all train rows, blend skipped")
    print(f"total {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", type=int, default=6, help="processes for the node-table cache")
    ap.add_argument("--seeds", default="14,5", help="seeds per --layers value (one number or a comma list)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--layers", default="0,2", help="message-passing layers; a comma list makes an ensemble of kinds")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--opdrop", type=float, default=0.1)
    ap.add_argument("--rowkeep", type=float, default=0.7)
    ap.add_argument("--multiw", type=float, default=1.0, help="loss weight of rows from multi-row files")
    ap.add_argument("--sizenoise", type=float, default=0.3, help="std of per-graph noise on log sizes (training only)")
    ap.add_argument("--huber", type=float, default=0.0, help="Huber delta in log1p space (0 = squared error)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="CV table only, no files in preds/")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--dump", default="", help="write per-seed OOF predictions to this csv (analysis)")
    ap.add_argument("--gpus", default="0,1", help="GPUs to spread the networks over (ignored when CUDA_VISIBLE_DEVICES is set)")
    ap.add_argument("--gfeat", default="none", choices=["none", "f2", "base2", "both"], help="extra graph-level inputs: feat2 columns, base2 prediction")
    ap.add_argument("--stack-oof", default="", help="stack OOF file for the blend table (default <data>/preds/main_STACK_oof.csv)")
    ap.add_argument("--out-prefix", default="", help="write <prefix>_oof.csv / <prefix>_test.csv instead of <data>/preds/g6_gnn_* (experiments)")
    ap.add_argument("--hidden-sim", default="", help="csv with seed,id of hidden rows (scripts/exp_hidden.py): predict them from the rest of train")
    ap.add_argument("--hidden-seeds", default="0,1,2,3,4")
    ap.add_argument("--hidden-out", default="", help="where --hidden-sim writes seed,id,g6")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--gfeat-file", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    t0 = time.time()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus

    df = C.load(a.data, measure=["cuda"])
    cache_dir = os.path.join(a.data, "g6_cache")
    os.makedirs(cache_dir, exist_ok=True)
    hop = os.environ.get("G6_HOP", "")

    todo = [t for t in needed_tables(df, cache_dir) if not os.path.exists(t[3]) and not os.path.exists(t[3] + ".fail")]
    if todo:
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401

            have_onnx = True
        except Exception:
            have_onnx = False
        if have_onnx:
            print(f"building {len(todo)} node tables -> {cache_dir}", flush=True)
            build_cache(a.data, todo, a.jobs)
        elif not hop and other_python(".venv"):
            reexec(other_python(".venv"), "cache")
        else:
            print(f"WARNING: {len(todo)} node tables missing and onnx is not importable; pseudo graphs are used", flush=True)
    if not a.worker:
        print(f"node-table cache ready, {time.time() - t0:.0f}s", flush=True)
    if a.cache_only:
        return

    try:
        import torch  # noqa: F401
    except Exception:
        py = other_python(".venv-g5")
        if py and hop != "train":
            reexec(py, "train")
        raise
    train_stage(a, df, cache_dir, t0)


if __name__ == "__main__":
    main()
