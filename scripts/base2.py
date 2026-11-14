#!/usr/bin/env python3
"""Analytic latency model, version 2: the graph as an inference compiler leaves it.

Compile pass (per graph, shapes inferred at the reference batch of features.csv):
  - constant folding: a node whose inputs are all constants (or Shape) is dropped;
  - no-ops (Identity, Reshape, Flatten, Squeeze, Unsqueeze, Dropout, Cast) are dropped;
  - an activation / normalisation right after a conv-like node is fused into it (free);
  - what is left are kernels of a few physical classes (CLASSES); ops the lists do not
    know fall into "other", which is priced like the generic kernels.

Model (non-negative weights, fitted on the log1p error like scripts/base.py; TERMS):

    latency = c_host + [graph has kernels] * c_dev
            + transfer(graph input + output elements: host-only graphs / device, with the knee)
            + sum over kernels  rate[class] * (elements the kernel writes)
            + surcharge for standalone elementwise / normalisation kernels when the graph has
              no conv-like kernel at all (nothing to fuse them into)
            + per-kernel overhead + MACs

Tried and left out (see candidates()): one rate per class, per-kernel size regimes, count
terms per class, input + output elements instead of written elements. They do not help the
base alone beyond optimiser noise and make the full chain worse.

Drop-in for scripts/base.py:  AdditiveCost().fit(df, y_log1p), .predict(df), crossfit(...).
The per-kernel aggregates come from <data>/base2_feats.csv (joined by `id`):

    python scripts/base2.py --data data        # needs <data>/features.csv; ~6 s on the server

The file is rebuilt automatically (serially) when it is missing or older than features.csv.
Rows without base2 features (graph not parsed) get the prediction of scripts/base.py fitted
on the same rows; crossfit() switches to base.py for all rows when coverage is below 90%.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base as B1  # noqa: E402

CLASSES = ("conv", "mm", "ew", "norm", "pool", "reduce", "move", "tr_lo", "tr_hi", "resize", "rnn", "other")
CI = {c: i for i, c in enumerate(CLASSES)}
C_NOP, C_FUSED = -1, -2
IO_KNEE = 8e6  # graph input + output elements (32 MB of float32)
# elementwise ops that features.ELEMENTWISE does not list
EXTRA_EW = {"Xor", "Sum", "Mean", "Acos", "Asin", "Atan", "Acosh", "Asinh", "Atanh", "Cosh", "Sinh", "Tan", "HardTanh", "Swish", "Shrink"}
K_BIG = 1e6  # a kernel that moves at least this many elements is "large" (b2_big_* columns, unused by TERMS)
FEATS_FILE = "base2_feats.csv"
DATA_DIR = None  # set by the caller, else FOS_DATA, else `--data X` of the running script, else "data"


def data_dir() -> str:
    if DATA_DIR:
        return DATA_DIR
    if os.environ.get("FOS_DATA"):
        return os.environ["FOS_DATA"]
    if "--data" in sys.argv[:-1]:
        return sys.argv[sys.argv.index("--data") + 1]
    return "data"


# ------------------------------------------------------------------ extractor (needs onnx)
def node_table(path: str, spec: str, ref_b: int) -> dict:
    """Kernels of one graph at the reference batch: class, input / output elements, MACs."""
    import onnx
    from onnx import shape_inference
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    import features as F

    model = onnx.load(path, load_external_data=False)
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
    prod = F.prod

    def numel(name):
        return float(prod(shp[name])) if name in shp else 0.0

    const = set(inits)
    for n in F.all_nodes(g):
        if n.op_type == "Constant":
            const.update(n.output)
    cls, ie, oe, macs, ops = [], [], [], [], []
    fused_src = {}
    n_out = n_known = 0
    for n in F.all_nodes(g):
        op = n.op_type
        if op == "Constant":
            continue
        ins = [i for i in n.input if i]
        outs = [o for o in n.output if o]
        n_out += len(outs)
        n_known += sum(o in shp for o in outs)
        if op in ("Shape", "Size", "ConstantOfShape") or (ins and all(i in const for i in ins)):
            const.update(outs)  # folded at compile time
            c, mc = C_NOP, 0
        else:
            ins_fused = [fused_src.get(i, False) for i in ins]
            has_w = any(i in const for i in ins)
            mc = F.node_macs(n, shp)
            if op in F.NOP:
                c = C_NOP
                for o in outs:  # a no-op does not hide the conv behind it
                    fused_src[o] = bool(ins_fused and ins_fused[0])
            elif op in ("LSTM", "GRU", "RNN"):
                c = CI["rnn"]
            elif op in F.CONV_LIKE:
                c = CI["conv"] if has_w else CI["mm"]
                for o in outs:
                    fused_src[o] = True
            elif op in F.NORM:
                fz = bool(ins_fused and ins_fused[0])
                c = C_FUSED if fz else CI["norm"]
                for o in outs:
                    fused_src[o] = fz
            elif op in F.ELEMENTWISE or op in EXTRA_EW:
                c = C_FUSED if any(ins_fused) else CI["ew"]
            elif op in F.POOL:
                c = CI["pool"]
            elif op in F.REDUCE:
                c = CI["reduce"]
            elif op == "Transpose":
                perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
                rank = len(perm) if perm else len(shp.get(ins[0], [])) if ins else 0
                c = CI["tr_hi"] if rank >= 5 else CI["tr_lo"]
            elif op in ("Resize", "Upsample"):
                c = CI["resize"]
            elif op in F.MOVEMENT:
                c = CI["move"]
            else:
                c = CI["other"]
        macs.append(float(max(mc, 0)))
        cls.append(c)
        ops.append(op)
        ie.append(sum(numel(i) for i in ins if i not in const))
        oe.append(sum(numel(o) for o in outs))
    gin = sum(numel(i.name) for i in g.input if i.name not in inits)
    gout = sum(numel(o.name) for o in g.output)
    return {"cls": np.array(cls, dtype=np.int64), "ie": np.array(ie, dtype=float), "oe": np.array(oe, dtype=float),
            "macs": np.array(macs, dtype=float), "ops": ops, "gin": float(gin), "gout": float(gout),
            "known": n_known / n_out if n_out else 1.0}  # fmt: skip


def aggregate(t: dict, k: float) -> dict:
    """Terms of one row: node table scaled by k = batch / reference batch."""
    cls = t["cls"]
    mv = (t["ie"] + t["oe"]) * k
    oe = t["oe"] * k
    ker = cls >= 0
    nc = len(CLASSES)
    f = {"b2_ok": 1, "b2_known": t["known"], "b2_gin": t["gin"] * k, "b2_gout": t["gout"] * k,
         "b2_n_k": int(ker.sum()), "b2_n_nop": int((cls == C_NOP).sum()), "b2_n_fused": int((cls == C_FUSED).sum()),
         "b2_oe_fused": float(oe[cls == C_FUSED].sum()), "b2_macs": float(t["macs"].sum() * k),
         "b2_max_k": float(mv[ker].max()) if ker.any() else 0.0}  # fmt: skip
    ck = cls[ker]
    big = mv[ker] >= K_BIG
    n = np.bincount(ck, minlength=nc)
    el = np.bincount(ck, weights=mv[ker], minlength=nc)
    eb = np.bincount(ck, weights=mv[ker] * big, minlength=nc)
    eo = np.bincount(ck, weights=oe[ker], minlength=nc)
    for c, i in CI.items():
        f[f"b2_n_{c}"] = int(n[i])
        f[f"b2_el_{c}"] = float(el[i])
        f[f"b2_big_{c}"] = float(eb[i])
        f[f"b2_oe_{c}"] = float(eo[i])
    return f


def build_file(args):
    ddir, fname, items = args  # items: [(spec, ref_b, [(id, batch), ...]), ...]
    rows, tabs, err = [], {}, None
    for spec, ref_b, ids in items:
        try:
            t = node_table(os.path.join(ddir, "data", fname), spec, ref_b)
            tabs[(fname, spec)] = t
            for rid, b in ids:
                rows.append({"id": rid, **aggregate(t, float(b) / float(ref_b))})
        except Exception as e:  # NaN terms: the model falls back to base.py for these rows
            err = f"{fname} [{spec}]: {type(e).__name__}: {str(e)[:200]}"
            rows += [{"id": rid, "b2_ok": 0} for rid, _ in ids]
    return rows, tabs, err


def build(ddir: str, jobs: int, dump_nodes: bool = False) -> None:
    from multiprocessing import Pool

    t0 = time.time()
    d = pd.read_csv(os.path.join(ddir, "features.csv"), usecols=["id", "file", "batch", "shape", "has_graph", "ref_batch"])
    d = d[(d.has_graph == 1) & d.ref_batch.notna()]
    tasks = []
    for fname, gf in d.groupby("file", sort=False):
        items = [(spec, int(rb), list(zip(gs.id.values, gs.batch.values)))
                 for (spec, rb), gs in gf.groupby(["shape", "ref_batch"], sort=False)]  # fmt: skip
        tasks.append((ddir, fname, items))
    tasks.sort(key=lambda t: -os.path.getsize(os.path.join(ddir, "data", t[1])) if os.path.exists(os.path.join(ddir, "data", t[1])) else 0)
    rows, errors, tabs = [], [], {}
    pool = Pool(jobs, maxtasksperchild=40) if jobs > 1 else None  # jobs <= 1: in this process
    try:
        for r, tb, err in pool.imap_unordered(build_file, tasks, chunksize=4) if pool else map(build_file, tasks):
            rows += r
            if dump_nodes:
                tabs.update(tb)
            if err:
                errors.append(err)
    finally:
        if pool:
            pool.terminate()
    out = pd.DataFrame(rows).set_index("id").reindex(d.id.values).reset_index()
    tmp = os.path.join(ddir, FEATS_FILE + f".tmp{os.getpid()}")
    out.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(ddir, FEATS_FILE))
    if dump_nodes:
        pd.to_pickle(tabs, os.path.join(ddir, "base2_nodes.pkl"))
    print(f"wrote {os.path.join(ddir, FEATS_FILE)}: {len(out)} rows x {out.shape[1]} cols, {int((out.b2_ok != 1).sum())} rows without terms, {time.time() - t0:.0f}s", flush=True)
    if errors:
        print(f"{len(errors)} graphs failed (base.py is used for them):", *errors[:10], sep="\n  ", flush=True)


# ------------------------------------------------------------------ model
_FEATS = {}
AUTO_BUILD = True  # rebuild base2_feats.csv when it is missing or older than features.csv


def feats(ddir: str = None) -> pd.DataFrame:
    ddir = ddir or data_dir()
    p, src = os.path.join(ddir, FEATS_FILE), os.path.join(ddir, "features.csv")
    stale = os.path.exists(src) and (not os.path.exists(p) or os.path.getmtime(p) < os.path.getmtime(src))
    if stale and AUTO_BUILD:
        try:  # new graphs arrived and features.py was re-run: the terms must follow
            print(f"  base2: {p} is missing or older than features.csv, building it (serial, ~30 s)", flush=True)
            build(ddir, 1)  # no process pool here: safe inside any caller
        except Exception as e:  # rows without terms fall back to base.py
            print(f"  !!! base2: could not build {p}: {type(e).__name__}: {e}", flush=True)
    key = (p, os.path.getmtime(p) if os.path.exists(p) else None)
    if key not in _FEATS:
        _FEATS.clear()
        _FEATS[key] = pd.read_csv(p).drop_duplicates("id").set_index("id") if os.path.exists(p) else pd.DataFrame(columns=["b2_ok"])
    return _FEATS[key]


# Terms of the model (see candidates()); everything else in base2_feats.csv is unused.
# Rates are per element written by a kernel. Few merged classes on purpose: with one rate per
# class (21-26 terms) the base alone is as good, but the full chain is worse (hidden-set
# simulation, 10 seeds: 0.1758-0.1778 against 0.172-0.174 for the merged sets): rare classes
# buy themselves unstable rates.
TERMS = (
    "const",  # host overhead of any graph
    "dev",  # extra overhead when something is left for the device after compilation
    "io_host",  # graph input + output elements of a graph that compiles to nothing (Identity only)
    "io_dev",  # graph input + output elements, device transfer
    "io_dev_big",  # ... surcharge from IO_KNEE elements
    "oe_cv",  # conv-like kernels (Conv, ConvTranspose, Gemm, MatMul) with fused activations / norms
    "oe_gen",  # pooling, reductions, Concat / Slice / Gather / Pad, low-rank Transpose, Resize, unknown ops
    "oe_ew",  # standalone elementwise
    "oe_norm",  # standalone normalisation
    "oe_tr_hi",  # Transpose of 5-D / 6-D tensors
    "oe_rnn",  # LSTM / GRU / RNN
    "ncoe_ew",  # surcharge for standalone elementwise in a graph without any conv-like kernel
    "ncoe_norm",  # the same for standalone normalisation
    "n_k",  # per-kernel overhead
    "macs",
)
# Rows of multi-row files (big real networks, many batches) get this weight in the fit: the
# rates are meant for the small single-measurement graphs. With weight 1 the big networks
# pull the rates down (base alone on single-row files 0.31 instead of 0.27).
W_MULTI = 0.1
# experiments only (scripts/exp_base2.py): BASE2_TERMS="a,b" replaces the terms, BASE2_DROP="a,b" leaves terms
# out, BASE2_W_MULTI overrides the weight
if os.environ.get("BASE2_TERMS"):
    TERMS = tuple(os.environ["BASE2_TERMS"].split(","))
if os.environ.get("BASE2_DROP"):
    TERMS = tuple(t for t in TERMS if t not in os.environ["BASE2_DROP"].split(","))
if os.environ.get("BASE2_W_MULTI"):
    W_MULTI = float(os.environ["BASE2_W_MULTI"])


def candidates(F: pd.DataFrame) -> dict:
    """All additive terms that can be built from base2_feats.csv rows (raw units)."""
    z = lambda c: np.nan_to_num(F[c].values.astype(float)) if c in F.columns else np.zeros(len(F))  # noqa: E731
    io = z("b2_gin") + z("b2_gout")
    dev = (z("b2_n_k") > 0).astype(float)  # something is left for the device after compilation
    hc = ((z("b2_n_conv") + z("b2_n_mm") + z("b2_n_rnn")) > 0).astype(float)  # the graph has a conv-like kernel
    nc = dev * (1 - hc)
    d = {"const": np.ones(len(F)), "dev": dev, "nc": nc, "io_host": io * (1 - dev), "io_dev": io * dev,
         "io_dev_big": io * dev * (io >= IO_KNEE), "io_nc": io * nc, "n_k": z("b2_n_k"), "n_k_nc": z("b2_n_k") * nc,
         "macs": z("b2_macs"), "peak": z("b2_max_k"), "peak_big": z("b2_max_k") * (z("b2_max_k") >= IO_KNEE)}  # fmt: skip
    tot, otot = np.zeros(len(F)), np.zeros(len(F))
    for c in CLASSES:
        el = z(f"b2_el_{c}")
        tot += el
        d[f"el_{c}"] = el  # elements moved by kernels of the class
        d[f"big_{c}"] = z(f"b2_big_{c}")  # ... by its large kernels only
        d[f"n_{c}"] = z(f"b2_n_{c}")
        d[f"nc_{c}"] = el * nc  # ... in graphs without a conv-like kernel
        d[f"ncbig_{c}"] = z(f"b2_big_{c}") * nc
        d[f"hc_{c}"] = el * hc
        oe = z(f"b2_oe_{c}")
        otot += oe
        d[f"oe_{c}"] = oe  # elements written by kernels of the class
        d[f"ncoe_{c}"] = oe * nc
    d["oe_all"] = otot
    d["oe_cv"] = d["oe_conv"] + d["oe_mm"]  # merged classes (fewer, more stable rates)
    d["oe_gen"] = sum(d[f"oe_{c}"] for c in ("pool", "reduce", "move", "tr_lo", "resize", "other"))
    d["oe_mv"] = sum(d[f"oe_{c}"] for c in ("move", "tr_lo", "resize"))
    d["oe_pr"] = d["oe_pool"] + d["oe_reduce"]
    d["oe_gen_all"] = d["oe_gen"] + d["oe_tr_hi"]
    d["io_dev_big6"] = io * dev * (io >= 6e6)
    d["io_dev_ramp"] = io * dev * np.clip((io - 4e6) / 5e6, 0, 1)  # 1.45 ns up to ~4M elements, 2.4-2.5 ns from ~9M
    d["oe_int"] = np.clip(otot - z("b2_gout"), 0, None)  # tensors between kernels
    d["oe_int_hc"] = d["oe_int"] * hc
    d["oe_int_big"] = d["oe_int"] * ((z("b2_gin") + otot) >= IO_KNEE)
    d["el_all"] = tot
    d["el_int"] = np.clip(tot - io, 0, None)  # traffic between kernels (not graph input / output)
    d["el_int_hc"] = d["el_int"] * hc
    return d


def design(df: pd.DataFrame, cols=None):
    """Raw (non-log) additive terms. Returns (matrix, column names, mask of rows that have terms)."""
    F = feats().reindex(df.id.values)
    ok = (F.b2_ok.values == 1) & np.isfinite(F.get("b2_gin", pd.Series(np.nan, index=F.index)).values.astype(float))
    d = candidates(F)
    names = list(cols) if cols is not None else list(TERMS)
    A = np.column_stack([d[c] for c in names]).astype(float)
    return A, names, ok


class AdditiveCost:
    def __init__(self, l2: float = 1e-4):
        self.l2 = l2

    def fit(self, df: pd.DataFrame, y_log1p: np.ndarray):
        y_all = np.asarray(y_log1p, float)
        self.fallback = B1.AdditiveCost().fit(df, y_all)
        A, self.cols, ok = design(df)
        self.w = np.zeros(len(self.cols))
        self.scale = np.ones(len(self.cols))
        self.fitted = int(ok.sum()) >= 3 * len(self.cols)
        if not self.fitted:  # no base2 features for this data: behave exactly like base.py
            print(f"  !!! base2: only {int(ok.sum())} of {len(ok)} train rows have base2 features, base.py is used instead "
                  f"(run: python scripts/base2.py --data {data_dir()})", flush=True)  # fmt: skip
            return self
        sw = np.where(df.groupby("file").file.transform("size").values > 1, W_MULTI, 1.0)[ok]
        sw = sw / sw.mean()
        A, y = A[ok], y_all[ok]
        lat = np.expm1(y)
        live = A.mean(axis=0) > 0  # a term absent from train must stay at 0, not at a free weight
        self.scale = np.where(live, np.maximum(A.mean(axis=0), 1e-12), 1.0)
        As = A / self.scale
        wgt = 1.0 / (lat + 0.3)
        w0, _ = nnls(As * wgt[:, None], lat * wgt, maxiter=20000)

        def loss(w):
            p = As @ w
            r = np.log1p(p) - y
            g = (2 * sw * r / (1 + p)) @ As / len(y) + 2 * self.l2 * w
            return float(np.mean(sw * r**2) + self.l2 * np.sum(w**2)), g

        res = minimize(loss, np.where(live, np.maximum(w0, 1e-6), 0.0), jac=True, method="L-BFGS-B",
                       bounds=[(0, None) if o else (0, 0) for o in live], options={"maxiter": 3000})  # fmt: skip
        self.w = np.where(live, res.x, 0.0)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """log1p of the predicted latency."""
        out = self.fallback.predict(df)
        if not self.fitted:
            return out
        A, _, ok = design(df, self.cols)
        p = np.log1p(np.maximum((A / self.scale) @ self.w, 0.0))
        ok = ok & np.isfinite(p)
        miss = ~ok & ((df.has_graph.values == 1) if "has_graph" in df.columns else True)
        if miss.any():  # mixing two bases silently would spoil the residual models
            print(f"  !!! base2: {int(miss.sum())} of {len(ok)} graph rows have no base2 features -> base.py prediction "
                  f"(re-run: python scripts/base2.py --data {data_dir()})", flush=True)  # fmt: skip
        return np.where(ok, p, out)

    def weights(self) -> pd.Series:
        """Cost per unit of each term (ms per element / MAC / kernel)."""
        return pd.Series(self.w / self.scale, index=self.cols)


def crossfit(df_tr: pd.DataFrame, y: np.ndarray, groups: np.ndarray, df_te: pd.DataFrame = None, n_splits: int = 5):
    """Out-of-fold base predictions for train rows (group-wise) and, if given,
    predictions for other rows from a model fitted on all train rows.
    If base2 features cover less than 90% of the graph rows on either side (stale or missing
    base2_feats.csv), everything is delegated to base.py: residual models must never see
    one base on train rows and another on the rows they predict."""
    for d in (df_tr, df_te):
        if d is None or not len(d):
            continue
        g = (d.has_graph.values == 1) if "has_graph" in d.columns else np.ones(len(d), bool)
        ok = design(d)[2]
        if g.any() and ok[g].mean() < 0.9:
            print(f"  !!! base2: features cover {int(ok[g].sum())} of {int(g.sum())} graph rows -> base.py is used for all rows", flush=True)
            return B1.crossfit(df_tr, y, groups, df_te, n_splits)
    uniq = np.unique(groups)
    rng = np.random.RandomState(1)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    oof = np.zeros(len(df_tr))
    for k in range(n_splits):
        va = f == k
        oof[va] = AdditiveCost().fit(df_tr[~va], y[~va]).predict(df_tr[va])
    full = AdditiveCost().fit(df_tr, y)
    return oof, (full.predict(df_te) if df_te is not None else None), full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(default action, kept for explicit command lines)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--dump-nodes", action="store_true", help="also write <data>/base2_nodes.pkl (analysis only)")
    a = ap.parse_args()
    build(a.data, a.jobs, a.dump_nodes)
