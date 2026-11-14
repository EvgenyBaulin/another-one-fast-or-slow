#!/usr/bin/env python3
"""Train on train rows, predict test rows, write a submission.

Pipeline
  1. analytic base (scripts/base.py), cross-fitted so train residuals are honest;
  2. LightGBM-type models on three targets: direct `y`, residual `y - base`, direct with
     the base as a feature (plain, extra-trees, Huber); `--zoo full` adds CatBoost/XGBoost;
  3. stack: plain mean of lgbx_res, lgbx_bf, lgbh_res (`--stack nnls` fits weights on
     out-of-fold predictions of single-row files and can add models from <data>/preds);
  4. shrink to the base where the models disagree;
  5. family rule for tiny graphs without conv-like nodes;
  6. G1 (scripts/g1.py): exact lookups / batch interpolation blended by expected error;
  7. rows without an ONNX file: 0 (`--blind zero`) or the batch prior (`--blind prior`).

Usage:
  python scripts/predict.py --data data --out submissions/x.csv [--blind prior] [--measure cuda]
"""

import argparse
import glob
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base as B  # noqa: E402
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
# Threads per model. Measured on the 24-core server: 8 threads 59 s, 24 threads 73 s for the
# whole run (862 rows are too few for more threads to pay off). Override with FOS_JOBS.
N_JOBS = int(os.environ.get("FOS_JOBS", 8))


# ----------------------------------------------------------------- learners
def _lgb(Xa, ya, Xb, seed):
    m = lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=5,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed,
        verbose=-1, n_jobs=N_JOBS,
    )  # fmt: skip
    return m.fit(Xa, ya).predict(Xb)


# Extra-trees parameters of the three stack models, from the Optuna search of
# scripts/exp_optuna.py (650 trials on the hidden-set simulation, trial 556). Held-out
# splits: 0.1775 -> 0.1759, better on 9 of 10; below our usual 0.003 threshold, adopted
# at the user's request. Previous values: 1000 trees, lr 0.03, 15 leaves, min_child 5,
# subsample 0.8, colsample 0.7, reg_lambda 5; down-weight 0.3 beyond 1.5.
XT = dict(
    n_estimators=1373, learning_rate=0.02203, num_leaves=33, min_child_samples=7,
    subsample=0.7433, subsample_freq=1, colsample_bytree=0.8404, reg_lambda=0.8152,
    reg_alpha=0.009604, min_split_gain=7.014e-05, extra_trees=True, max_depth=-1,
)  # fmt: skip
XT_W, XT_THR = 0.3474, 1.528  # weight of anomalous rows in _lgbw and the threshold for them
if os.environ.get("FOS_XT_OLD"):  # the hand-set values, for A/B checks only
    XT = dict(n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, extra_trees=True, reg_lambda=5.0)
    XT_W, XT_THR = 0.3, 1.5


def _lgbx(Xa, ya, Xb, seed):
    """Extra-trees LightGBM (repeated CV: -0.011 on single-row files vs plain)."""
    m = lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=N_JOBS, **XT)
    return m.fit(Xa, ya).predict(Xb)


def _lgbh(Xa, ya, Xb, seed):
    """Huber-loss LightGBM: does not chase the few anomalous graphs (10x off the io law)."""
    m = lgb.LGBMRegressor(
        objective="huber", alpha=1.0, n_estimators=900, learning_rate=0.03, num_leaves=15,
        min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=N_JOBS,
    )  # fmt: skip
    return m.fit(Xa, ya).predict(Xb)


MONO_UP = ("base", "d_io", "input_elems", "output_elems", "out_elems", "in_elems", "batch", "log_batch", "elems_traffic")


def _lgbm(Xa, ya, Xb, seed):
    """Extra-trees LightGBM, latency constrained to grow with size features (direct targets only)."""
    mc = [1 if c in MONO_UP else 0 for c in Xa.columns]
    m = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=5,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, extra_trees=True, reg_lambda=5.0,
        monotone_constraints=mc, monotone_constraints_method="advanced",
        random_state=seed, verbose=-1, n_jobs=N_JOBS,
    )  # fmt: skip
    return m.fit(Xa, ya).predict(Xb)


def _lgbl(Xa, ya, Xb, seed):
    """Linear-leaf LightGBM: features and target are logs, so a linear leaf is a power law
    in size and can interpolate between sizes of one family."""
    m = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, linear_tree=True, linear_lambda=1.0,
        reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=N_JOBS,
    )  # fmt: skip
    Xa, Xb = Xa.fillna(0), Xb.fillna(0)
    return m.fit(Xa, ya).predict(Xb)


def _lgbw(Xa, ya, Xb, seed):
    """Extra-trees LightGBM that down-weights the anomalous rows: |target| > 1.5 for a
    residual target means 4.5x off the analytic law."""
    w = np.where(np.abs(ya - np.median(ya)) > XT_THR, XT_W, 1.0)
    m = lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=N_JOBS, **XT)
    return m.fit(Xa, ya, sample_weight=w).predict(Xb)


def _cat(Xa, ya, Xb, seed):
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_seed=seed,
        verbose=0, thread_count=N_JOBS,
    )  # fmt: skip
    return m.fit(Xa, ya).predict(Xb)


def _xgb(Xa, ya, Xb, seed):
    import xgboost as xgb

    m = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=2,
        subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=N_JOBS,
    )  # fmt: skip
    return m.fit(Xa, ya).predict(Xb)


LEARNERS = {"lgb": _lgb, "lgbx": _lgbx, "lgbh": _lgbh, "lgbm": _lgbm, "lgbl": _lgbl, "lgbw": _lgbw, "cat": _cat, "xgb": _xgb}
# (name, learner, target kind): dir = y on X; res = y - base on X+base; bf = y on X+base
ZOO = [
    ("lgb_dir", "lgb", "dir"), ("lgb_res", "lgb", "res"), ("lgb_bf", "lgb", "bf"), ("lgbw_res", "lgbw", "res"),
    ("lgbx_res", "lgbx", "res"), ("lgbx_bf", "lgbx", "bf"), ("lgbh_res", "lgbh", "res"),
    ("cat_res", "cat", "res"), ("xgb_res", "xgb", "res"),
]  # fmt: skip


# candidates under test (scripts/exp_hidden.py --zoo exp); not part of the default run
EXP_ZOO = [z for z in ZOO if z[1].startswith("lgb")] + [
    ("lgbm_bf", "lgbm", "bf"), ("lgbm_dir", "lgbm", "dir"),
    ("lgbl_res", "lgbl", "res"), ("lgbl_bf", "lgbl", "bf"),
]  # fmt: skip


BASES = [B]  # main() adds scripts/base2.py (kernel-level cost model) when it is available


def fit_predict_all(tr, Xtr, y, te, Xte, seeds=(0,), zoo=ZOO):
    """Fit every model on (tr, Xtr, y) and predict te. Returns {name: pred} incl. 'base'.
    With two analytic bases every tree model is fitted once per base and the two versions
    are averaged (hidden-set simulation, 20 splits: 0.1769 -> 0.1722); 'base' is the last
    base in BASES (base2: 0.267 on single-row files against 0.349 for base.py)."""
    grp = tr.fp.fillna(tr.file).astype(str).values
    outs = []
    for Bm in BASES:
        b_tr, b_te, _ = Bm.crossfit(tr, y, grp, te)
        Xa = Xtr.copy()
        Xa["base"] = b_tr
        Xb = Xte.copy()
        Xb["base"] = b_te
        out = {"base": b_te}
        for name, learner, kind in zoo:
            f = LEARNERS[learner]
            if kind == "dir" and outs:  # no base involved: identical to the first pass
                out[name] = outs[0][name]
                continue
            ps = []
            for s in seeds:
                if kind == "dir":
                    ps.append(f(Xtr, y, Xte, s))
                elif kind == "res":
                    ps.append(b_te + f(Xa, y - b_tr, Xb, s))
                else:
                    ps.append(f(Xa, y, Xb, s))
            out[name] = np.mean(ps, axis=0)
        outs.append(out)
    res = {n: np.mean([o[n] for o in outs], axis=0) for n in outs[0] if n != "base"}
    res["base"] = outs[-1]["base"]
    return res


def fit_predict_lgb(Xtr, ytr, Xte, seeds=(0, 1, 2)):
    """Plain LightGBM on the direct target (the v0 baseline; other scripts import it)."""
    return np.mean([_lgb(Xtr, ytr, Xte, s) for s in seeds], axis=0)


# ----------------------------------------------------------------- stacking
def stack_weights(P: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> pd.Series:
    """Non-negative weights (normalised to 1) minimising squared error on rows `mask`."""
    w, _ = nnls(P.values[mask], y[mask])
    if w.sum() <= 0:
        w = np.ones(P.shape[1])
    return pd.Series(w / w.sum(), index=P.columns)


EW_P_FAST = 0.17  # share of standalone elementwise ops in the fast regime: Selu and Softplus of 12 known ops
EW_FAST_NS = 0.4  # ns per element in the fast regime (Softplus 0.23, Selu 0.61)


def _io_log(d):
    return np.log((d.input_elems.fillna(0) + d.output_elems.fillna(0)).clip(lower=1).values)


def _ew1(d):
    """Graphs made only of standalone elementwise ops: one activation / binary op, or a
    chain such as Mish (Softplus, Tanh, Mul) or GELU (Div, Erf, Add, Mul, Mul)."""
    z = lambda c: d[c].fillna(0).values if c in d.columns else np.zeros(len(d))  # noqa: E731
    n = z("n_nodes_nonconst")
    return (d.has_graph.values == 1) & (n >= 1) & (n <= 8) & (z("macs") == 0) & (z("n_ew_unfused") == n)


def tiny_rules(known, query, min_known: int = 1):
    """Rules for tiny graphs without conv-like nodes, where the cost per element depends on
    the op itself and trees cannot isolate 1-8 examples. Returns (log1p prediction, kind);
    NaN / "" where no rule applies.

    kind "fam" / "noop": the op signature `sig1` has known graphs (other files): anchor on
      the nearest one in log(io), move along  latency = c0 + rate * io^p  (Identity-only
      graphs: 0.15 ns per element; pooling 1.5; standalone normalisations 3-6).
    kind "ewlaw": one standalone elementwise op that is absent from train. Such ops run in
      the slow regime (2.7-4.1 ns per element: Elu, Tanh, Sigmoid, LeakyRelu, PRelu, Add;
      only Selu and Softplus were fast). Robust law over all known standalone elementwise
      graphs: Theil-Sen slope, median intercept, so the two fast ones do not move it.
      Leave-one-op-out on train: trees underpredict the slow ops by 0.3-0.7 and are as
      wrong as the law on the fast ones.
    """
    out = np.full(len(query), np.nan)
    kind = np.array([""] * len(query), dtype=object)
    if "sig1" not in known.columns or "sig1" not in query.columns:
        return out, kind
    def ok(d):  # shapes must be known: with a broken shape inference io and MACs are fake zeros
        io = (d.input_elems.fillna(0) + d.output_elems.fillna(0)).values
        return (d.has_graph.values == 1) & (d.macs.fillna(-1).values == 0) & (io > 0) & (d.shape_known_frac.fillna(0).values >= 0.999)

    sg = lambda d: d.sig1.fillna("").astype(str).values  # noqa: E731
    k = known[ok(known)]
    kx, ks, kf = _io_log(k), sg(k), k.file.values
    klat = k.latency.values.astype(float)
    kew = _ew1(k)
    one = lambda d: (d.n_nodes_nonconst.fillna(0).values == 1)  # noqa: E731
    big = lambda d: (d.params.fillna(0).values > 1024)  # noqa: E731  binary op with a constant tensor operand
    kn1, kbig, qn1, qbig = one(k), big(k), one(query), big(query)
    qm, qew = ok(query), _ew1(query)
    qx, qs, qf = _io_log(query), sg(query), query.file.values
    for i in np.where(qm)[0]:
        m = (ks == qs[i]) & (kf != qf[i]) & (qs[i] != "")
        if m.sum() >= min_known:
            x, lat = kx[m], klat[m]
            c0 = min(0.2, 0.8 * lat.min())
            big = lat > 3 * c0
            p = 1.1
            if big.sum() >= 3 and np.ptp(x[big]) > 0.7:
                p = float(np.clip(np.polyfit(x[big], np.log(lat[big] - c0), 1)[0], 0.8, 1.4))
            j = np.abs(x - qx[i]).argmin()
            if abs(x[j] - qx[i]) > 2.5 and qs[i] != "noop":  # > ~12x away in size: leave it to the model
                continue
            out[i] = np.log1p(c0 + (lat[j] - c0) * np.exp(p * (qx[i] - x[j])))
            kind[i] = "noop" if qs[i] == "noop" else "fam"
        elif qew[i]:
            m = kew & (kf != qf[i]) & (klat > 0.4)
            if m.sum() < 5:
                continue
            x, lat = kx[m], klat[m]
            # drop the fast regime (Selu, Softplus: < 35% of the typical cost per element)
            rate = (lat - 0.2) / np.exp(x)
            slow = rate >= 0.35 * np.median(rate)
            # prefer known graphs of the same kind (one node vs chain, with / without a big
            # constant operand) when there are enough of them
            same = slow & (kn1[m] == qn1[i]) & (kbig[m] == qbig[i])
            use = same if same.sum() >= 3 else slow
            xs, ys = x[use], np.log(lat[use] - 0.2)
            o = np.argsort(xs)
            xs, ys = xs[o], ys[o]
            # average graphs of (almost) the same size, then interpolate in log-log;
            # standalone slow ops of one size cost the same (Elu 31.8 / Tanh 29.8 at 8.4M,
            # PRelu 69.4 / GELU chain 69.4 at 16.8M), so the curve is tight
            gx, gy = [], []
            for xv, yv in zip(xs, ys):
                if gx and abs(xv - gx[-1][0]) < 0.05:
                    gx[-1].append(xv)
                    gy[-1].append(yv)
                else:
                    gx.append([xv])
                    gy.append([yv])
            cx, cy = np.array([np.mean(v) for v in gx]), np.array([np.mean(v) for v in gy])
            sl = [(cy[b] - cy[a]) / (cx[b] - cx[a]) for a in range(len(cx)) for b in range(a + 1, len(cx))]
            p = float(np.clip(np.median(sl), 1.0, 1.3)) if sl else 1.15
            if qx[i] <= cx[0]:
                yq = cy[0] + p * (qx[i] - cx[0])
            elif qx[i] >= cx[-1]:
                yq = cy[-1] + p * (qx[i] - cx[-1])
            else:
                yq = float(np.interp(qx[i], cx, cy))
            slow_lat = 0.2 + np.exp(yq)
            fast_lat = 0.05 + EW_FAST_NS * 1e-6 * np.exp(qx[i])
            # which regime a new op falls into cannot be told from train (2 of 12 known
            # standalone elementwise ops are fast): mixture mean in log1p space
            out[i] = (1 - EW_P_FAST) * np.log1p(slow_lat) + EW_P_FAST * np.log1p(fast_lat)
            kind[i] = "ewlaw"
    return out, kind


def family_rule(known, query, min_known: int = 1):
    """Predictions of tiny_rules() (kept for the experiment scripts)."""
    return tiny_rules(known, query, min_known)[0]


def fam_mix(pred_model, rule, sig, kind=None):
    """Identity-only graphs and new standalone elementwise ops: the rule alone. Other tiny
    families: mean of rule and model (about equally good, the mean is the safer choice)."""
    ok = np.isfinite(rule)
    alone = ok & (np.asarray(sig).astype(str) == "noop")
    if kind is not None:
        alone = alone | (ok & np.isin(np.asarray(kind).astype(str), ["noop", "ewlaw"]))
    out = np.where(ok, 0.5 * (np.nan_to_num(rule) + pred_model), pred_model)
    return np.where(alone, np.nan_to_num(rule), out)


def g1_blend(pred_model, r, sigma_model=0.25):
    """Inverse-variance blend of the model with G1 (g1_conf = expected RMSLE of G1)."""
    ok = np.isfinite(r.g1_pred.values.astype(float))
    conf = r.g1_conf.fillna(1.0).values.astype(float)
    w = np.where(ok, sigma_model**2 / (sigma_model**2 + conf**2), 0.0)
    return np.where(ok, w * r.g1_pred.fillna(0).values + (1 - w) * pred_model, pred_model)


def external_preds(data_dir, tr_ids, te_ids):
    """Models produced by other scripts: <data>/preds/<name>_{oof,test}.csv with id,pred."""
    out = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "preds", "*_oof.csv"))):
        name = os.path.basename(p)[: -len("_oof.csv")]
        pt = p[: -len("_oof.csv")] + "_test.csv"
        if name.startswith("main_") or not os.path.exists(pt):
            continue
        o = pd.read_csv(p).drop_duplicates("id").set_index("id").pred
        t = pd.read_csv(pt).drop_duplicates("id").set_index("id").pred
        oo, tt = o.reindex(tr_ids), t.reindex(te_ids)
        if oo.isna().any() or tt.isna().any():
            print(f"  external model {name}: incomplete coverage (oof {oo.isna().sum()} / test {tt.isna().sum()} missing), skipped")
            continue
        out[name] = (oo.values, tt.values)
    return out


def summary(tr, te_all, pred, has):
    """Sanity report on the test rows: printed before every submission so that a broken
    last-hour run (missing graphs, new op types, failed measurement, flat predictions) is visible."""
    te = te_all[has]
    print("SUMMARY of test rows:", flush=True)
    print(f"  rows {len(te_all)} | with graph {int(has.sum())} | without graph {int((~has).sum())}")
    single_tr = tr[tr.groupby("file").file.transform("size") == 1]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    fmt = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
    print(f"  latency quantiles 5/25/50/75/95%: train single-row files {fmt(np.quantile(single_tr.latency, q))} | test predictions (graph rows) {fmt(np.quantile(np.expm1(pred[has]), q))}")
    cnt = [c for c in tr.columns if c.startswith("cnt_") and c != "cnt_other"]
    new_ops = [c[4:] for c in cnt if tr[c].fillna(0).sum() == 0 and te[c].fillna(0).sum() > 0]
    print(f"  op types present only in test: {new_ops if new_ops else 'none'}")
    if "sig1" in te.columns:
        known = set(tr.sig1.dropna().astype(str)) - {""}
        tiny = te[te.sig1.notna() & (te.sig1.astype(str) != "")]
        unseen = tiny[~tiny.sig1.astype(str).isin(known)]
        print(f"  tiny graphs (<= 3 nodes): {len(tiny)}, with a signature not seen in train: {len(unseen)}")
        for r in unseen.itertuples():
            io = (r.input_elems + r.output_elems) / 1e6
            p = float(np.expm1(pred[np.where(te_all.id.values == r.id)[0][0]]))
            print(f"    {r.sig1:30s} batch {int(r.batch):4d} io {io:8.3f}M -> {p:8.3f} ms ({1e6 * p / max(io * 1e6, 1):.2f} ns/elem)")
    bad_shape = int((te.shape_known_frac < 0.6).sum())
    no_ms = int(te.ms_cuda.isna().sum()) if "ms_cuda" in te.columns else -1
    print(f"  graphs with < 60% known shapes: {bad_shape} | graph rows without GPU measurement: {no_ms if no_ms >= 0 else 'measurement not used'}")
    if has.any() and float(np.std(pred[has])) < 0.05:
        print("  !!! predictions on graph rows are almost constant: something is broken !!!")


# Extra-trees on the residual and on the direct target + the outlier-down-weighted model.
# Hidden-set simulation: every 2-4 model subset is within 0.001 of each other (0.185-0.186);
# on the public part (74 new graphs) this trio gave 0.0947 against 0.1015 for the trio with
# the Huber model instead of the down-weighted one.
STACK_FIXED = ("lgbx_res", "lgbx_bf", "lgbw_res")
SHRINK_TAU, SHRINK_LAM, SHRINK_MAX_NODES = 0.15, 0.7, 3


def stack_fixed(preds: dict) -> np.ndarray:
    """Plain mean of the extra-trees and Huber models. On the hidden-set simulation
    (10 x 171 hidden single-row files) this is as good as fitted weights and stable:
    NNLS weights changed a lot between fold splits."""
    cols = [n for n in STACK_FIXED if n in preds]
    return np.mean([preds[n] for n in cols], axis=0)


def shrink_to_base(stack: np.ndarray, preds: dict, n_nodes: np.ndarray) -> np.ndarray:
    """Tiny graphs (<= 3 real nodes) on which the LightGBM-type models disagree (std > tau)
    are standalone ops unlike anything in train; there the analytic base, which prices
    unfused elementwise / normalisation elements, is better than the trees. Move 70% of
    the way to the base. Hidden-set simulation: 0.2044 -> 0.1973 overall, 0.63 -> 0.44 on
    the affected rows. Not applied to larger graphs: the base is poor on big nets."""
    core = ("lgb_dir", "lgb_res", "lgb_bf", "lgbx_res", "lgbx_bf", "lgbh_res")
    lg = [preds[n] for n in core if n in preds]
    if len(lg) < 3 or "base" not in preds:
        return stack
    m = (np.std(lg, axis=0) > SHRINK_TAU) & (np.asarray(n_nodes, float) <= SHRINK_MAX_NODES)
    return np.where(m, stack + SHRINK_LAM * (preds["base"] - stack), stack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="submissions/v1.csv")
    ap.add_argument("--blind", choices=["zero", "prior"], default="zero")
    ap.add_argument("--measure", default="none", help="measurement files to use: none | cuda | cpu,cuda")
    ap.add_argument("--stack", choices=["fixed", "nnls"], default="fixed")
    ap.add_argument("--zoo", choices=["lgb", "full"], default="lgb", help="full adds CatBoost and XGBoost")
    ap.add_argument("--external", action="store_true", help="add <data>/preds/<name>_{oof,test}.csv models (nnls stack only)")
    ap.add_argument("--stack-models", default=",".join(STACK_FIXED), help="models averaged by the fixed stack")
    ap.add_argument("--seeds", type=int, default=3, help="model seeds averaged for the test predictions")
    ap.add_argument("--cv-seeds", type=int, default=1, help="model seeds averaged inside the out-of-fold report")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--g6-weight", type=float, default=0.3, help="weight of the GNN (scripts/g6.py) in the blend; used only if <data>/preds/g6_gnn_test.csv covers every test graph row")
    ap.add_argument("--ew-p-fast", type=float, default=None, help="override EW_P_FAST (0 = pure slow law for new standalone ops)")
    ap.add_argument("--drop-cols", default="tr_max_rank,n_tr_hi,elems_tr_hi,n_tr6,f2_rs_*,f2_max_kernel,f2_peak_per_io",
                    help="feature columns to leave out (name or prefix*). Default: the Transpose-rank features (CV +0.001, "
                         "public 0.1070 -> 0.1015 without them) and the Resize / kernel / peak columns of feat2 "
                         "(same in the hidden-set simulation, public 0.0966 -> 0.0915 without them)")
    ap.add_argument("--no-base2", action="store_true", help="use scripts/base.py only (no kernel-level base)")
    ap.add_argument("--no-near-twin", action="store_true", help="G1 without the near-twin link of big nets")
    ap.add_argument("--no-g1", action="store_true")
    ap.add_argument("--no-cv", action="store_true", help="skip the out-of-fold report (fixed stack only)")
    ap.add_argument("--fast", action="store_true", help="one seed for test predictions")
    ap.add_argument("--no-external", action="store_true", help="(kept for old command lines, external models are off by default)")
    a = ap.parse_args()
    t0 = time.time()
    if a.ew_p_fast is not None:
        global EW_P_FAST
        EW_P_FAST = a.ew_p_fast
    need_oof = a.stack == "nnls" or not a.no_cv
    if not a.no_base2:
        try:
            import base2 as B2

            B2.DATA_DIR = a.data
            BASES[:] = [B, B2]
        except Exception as e:  # the second base is an improvement, not a requirement
            print(f"  base2 not used: {type(e).__name__}: {e}", flush=True)

    df = C.load(a.data, measure=[m for m in a.measure.split(",") if m and m != "none"])
    is_tr = df.split.values == "train"
    X = C.feature_matrix(df)
    if a.drop_cols:
        pats = [c for c in a.drop_cols.split(",") if c]
        X = X.drop(columns=[c for c in X.columns if any(c == q or (q.endswith("*") and c.startswith(q[:-1])) for q in pats)])
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    te_all, Xte_all = df[~is_tr].reset_index(drop=True), X[~is_tr].reset_index(drop=True)
    has = (te_all.has_graph == 1).values
    te, Xte = te_all[has].reset_index(drop=True), Xte_all[has].reset_index(drop=True)
    y = tr.y.values
    single = (tr.groupby("file").file.transform("size") == 1).values
    zoo = ZOO if a.zoo == "full" else [z for z in ZOO if z[1].startswith("lgb")]
    globals()["STACK_FIXED"] = tuple(a.stack_models.split(","))
    zoo = zoo + [z for z in EXP_ZOO if z[0] in a.stack_models.split(",") and z not in zoo]
    print(f"config: measure={a.measure} stack={a.stack}:{'+'.join(STACK_FIXED)} bases={len(BASES)} seeds={a.seeds} shrink={not a.no_shrink} g1={not a.no_g1} | train {len(tr)} rows, test {len(te_all)} ({int(has.sum())} with graph)", flush=True)

    def line(name, p):
        print(f"  {name:34s} all {C.rmse(p, y):.4f} | single {C.rmse(p[single], y[single]):.4f} | multi {C.rmse(p[~single], y[~single]):.4f}", flush=True)

    # out-of-fold predictions (report, and weights for the nnls stack)
    oof = {}
    if need_oof:
        folds = C.make_folds(tr)
        for k in range(5):
            va, tn = folds == k, folds != k
            p = fit_predict_all(tr[tn].reset_index(drop=True), Xtr[tn].reset_index(drop=True), y[tn],
                                tr[va].reset_index(drop=True), Xtr[va].reset_index(drop=True), zoo=zoo,
                                seeds=tuple(range(a.cv_seeds)))  # fmt: skip
            for n, v in p.items():
                oof.setdefault(n, np.zeros(len(tr)))[va] = v
    test = fit_predict_all(tr, Xtr, y, te, Xte, seeds=(0,) if a.fast else tuple(range(a.seeds)), zoo=zoo)

    os.makedirs(os.path.join(a.data, "preds"), exist_ok=True)
    for n in test:
        if n in oof:
            pd.DataFrame({"id": tr.id, "pred": oof[n]}).to_csv(os.path.join(a.data, "preds", f"main_{n}_oof.csv"), index=False)
        pd.DataFrame({"id": te.id, "pred": test[n]}).to_csv(os.path.join(a.data, "preds", f"main_{n}_test.csv"), index=False)
    if a.external and a.stack == "nnls":
        for n, (o, t) in external_preds(a.data, tr.id.values, te.id.values).items():
            oof[n], test[n] = o, t

    if oof:
        print("CV RMSLE (folds grouped by fp):")
        for n in oof:
            line(n, oof[n])
    st_oof = None
    if a.stack == "nnls":
        P = pd.DataFrame({n: v for n, v in oof.items() if n != "base"})
        w = stack_weights(P, y, single)
        st_oof = P.values @ w.values
        pred_g = pd.DataFrame({n: test[n] for n in P.columns}).values @ w.values
        print("  nnls weights:", {k: round(float(v), 3) for k, v in w.items() if v > 1e-3}, flush=True)
    else:
        pred_g = stack_fixed(test)
        if oof:
            st_oof = stack_fixed(oof)
    if st_oof is not None:
        line(f"STACK ({a.stack})", st_oof)
    if not a.no_shrink:
        n_shr = int((shrink_to_base(pred_g, test, te.n_nodes_nonconst.values) != pred_g).sum())
        pred_g = shrink_to_base(pred_g, test, te.n_nodes_nonconst.values)
        if st_oof is not None:
            st_oof = shrink_to_base(st_oof, oof, tr.n_nodes_nonconst.values)
            line("STACK + shrink to base", st_oof)
        print(f"  shrink to base applied to {n_shr} test graph rows", flush=True)

    # G6 (additive GNN, scripts/g6.py): CV on single-row files 0.2368 -> 0.2243 at weight 0.3
    if a.g6_weight > 0:
        pt, po = os.path.join(a.data, "preds", "g6_gnn_test.csv"), os.path.join(a.data, "preds", "g6_gnn_oof.csv")
        g6t = pd.read_csv(pt).drop_duplicates("id").set_index("id").pred.reindex(te.id.values).values if os.path.exists(pt) else None
        if g6t is not None and np.isfinite(g6t).any():
            okr = np.isfinite(g6t)  # rows without a G6 prediction keep the stack alone
            # G6 goes into every row. Restricting it to single-row files was tried: CV on
            # unseen multi-row files likes it (0.942 -> 0.906), but the public score got
            # worse (0.1066 -> 0.1121): for files that have other batches in train G6 is good.
            pred_g = np.where(okr, (1 - a.g6_weight) * pred_g + a.g6_weight * np.nan_to_num(g6t), pred_g)
            print(f"  G6 blended into {int(okr.sum())} of {len(g6t)} test graph rows with weight {a.g6_weight}" + ("" if okr.all() else f"  !!! {int((~okr).sum())} rows have no G6 prediction (stale g6 files?)"), flush=True)
            if st_oof is not None and os.path.exists(po):
                g6o = pd.read_csv(po).drop_duplicates("id").set_index("id").pred.reindex(tr.id.values).values
                if np.isfinite(g6o).all():
                    line("G6 alone", g6o)
                    st_oof = (1 - a.g6_weight) * st_oof + a.g6_weight * g6o
                    line(f"STACK + {a.g6_weight} G6", st_oof)
        else:
            print(f"  G6 NOT used: {'no file' if g6t is None else 'no test graph row has a G6 prediction'}", flush=True)

    # tiny non-conv graphs: rules (leave-one-file-out on train for the report)
    if st_oof is not None:
        fr_tr, fk_tr = tiny_rules(tr, tr)
        for kd in ("noop", "fam", "ewlaw"):
            hit = np.isfinite(fr_tr) & (fk_tr == kd)
            if hit.any():
                print(f"  rule {kd:5s} on train (leave-one-file-out): {int(hit.sum()):3d} rows, rule {C.rmse(fr_tr[hit], y[hit]):.4f} vs stack {C.rmse(st_oof[hit], y[hit]):.4f}", flush=True)
        st_oof = fam_mix(st_oof, fr_tr, tr.sig1.values, fk_tr)
        line("STACK + tiny-graph rules", st_oof)
    fr_te, fk_te = tiny_rules(tr, te)
    print(f"  tiny-graph rules on test graph rows: { {k: int((fk_te == k).sum()) for k in ('noop', 'fam', 'ewlaw')} }", flush=True)
    pred_g = fam_mix(pred_g, fr_te, te.sig1.values if "sig1" in te.columns else np.array([""] * len(te)), fk_te)
    # recurrent-only graphs (scripts/rules2.py): latency = law + ~100 ns per step*hidden unit
    try:
        import rules2

        if st_oof is not None:
            st_oof = rules2.apply(tr, tr, st_oof)
            line("STACK + RNN rule", st_oof)
        before = pred_g.copy()
        pred_g = rules2.apply(tr, te, pred_g)
        print(f"  RNN rule changed {int((np.abs(pred_g - before) > 1e-9).sum())} test graph rows", flush=True)
    except Exception as e:
        print(f"  RNN rule skipped: {type(e).__name__}: {e}", flush=True)
    if st_oof is not None:
        pd.DataFrame({"id": tr.id, "pred": st_oof}).to_csv(os.path.join(a.data, "preds", "main_STACK_oof.csv"), index=False)
    pd.DataFrame({"id": te.id, "pred": pred_g}).to_csv(os.path.join(a.data, "preds", "main_STACK_test.csv"), index=False)

    # G1: exact lookups / batch interpolation, blended with the model by expected error
    if not a.no_g1 and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1.py")):
        if st_oof is not None:
            try:
                import g1

                # honest check on train: hold out whole files (5 folds by file name), so a
                # file is matched only through twins / feature duplicates, like a new file
                files = np.unique(tr.file.values)
                np.random.RandomState(0).shuffle(files)
                ff = pd.Series(np.arange(len(files)) % 5, index=files).reindex(tr.file.values).values
                g_oof = st_oof.copy()
                n_hit = 0
                for k in range(5):
                    r = g1.g1_predict(tr[ff != k], tr[ff == k], data_dir=a.data)
                    g_oof[ff == k] = g1_blend(st_oof[ff == k], r)
                    n_hit += int(r.g1_pred.notna().sum())
                line(f"STACK+G1 (new-file sim, {n_hit} hits)", g_oof)
            except Exception as e:  # the report must not cost us G1 itself
                print(f"  G1 report skipped: {type(e).__name__}: {e}", flush=True)
        try:
            import g1

            if a.no_near_twin:
                g1.CFG["near_macs"] = 0.0
            r = g1.g1_predict(tr, te, data_dir=a.data)
            assert len(r) == len(te), "G1 returned a frame of another length"
            print(f"  G1 on test graph rows: {dict(r.g1_kind.value_counts())}", flush=True)
            # the model is less reliable than G1 on files whose other batches are in train
            # (simulation scripts/exp_mr.py: 0.106 -> 0.096 with sigma 0.45 for such rows)
            sig = np.where(te.file.isin(set(tr.file)).values, 0.45, 0.25)
            pred_g = g1_blend(pred_g, r, sigma_model=sig)
        except Exception as e:  # never let an optional part break the submission
            print(f"  G1 skipped: {type(e).__name__}: {e}", flush=True)

    # last guard: no prediction outside what the device has ever shown (x3)
    hi = float(np.log1p(3 * tr.latency.max()))
    n_clip = int(((pred_g < 0) | (pred_g > hi)).sum())
    if n_clip:
        print(f"  !!! {n_clip} predictions clipped to [0, {np.expm1(hi):.0f} ms]", flush=True)
    pred_g = np.clip(pred_g, 0.0, hi)

    pred = np.full(len(te_all), np.nan)
    pred[has] = pred_g
    if a.blind == "prior":
        prior = C.blind_prior(tr)
        pred[~has] = [prior.get(float(b), prior[None]) for b in te_all.batch[~has]]
    else:
        pred[~has] = 0.0
    C.write_submission(te_all, pred, a.out, os.path.join(a.data, "sample_submission.csv"))
    try:
        summary(tr, te_all, pred, has)
    except Exception as e:  # the report must never block a submission
        print(f"  summary skipped: {type(e).__name__}: {e}", flush=True)
    print(f"wrote {a.out}: {len(te_all)} rows, {int(has.sum())} from graphs, {int((~has).sum())} blind ({a.blind}); {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
