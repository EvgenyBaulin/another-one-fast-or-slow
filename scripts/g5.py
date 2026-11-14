#!/usr/bin/env python3
"""G5: two extra ensemble members - TabPFN regression and structural kNN.

Usage:
  python scripts/g5.py --data data [--no-tabpfn] [--explore] [--variants top60,all]

Writes (pred in log1p space, columns id,pred):
  <data>/preds/g5_knn_oof.csv     <data>/preds/g5_knn_test.csv
  <data>/preds/g5_tabpfn_oof.csv  <data>/preds/g5_tabpfn_test.csv
oof  = out-of-fold predictions for every train row (folds from common.make_folds);
test = test rows with has_graph == 1, model trained on all train rows.

TabPFN lives in a separate venv on the server (~/hacklab/fos/.venv-g5, see HANDOFF):
when `tabpfn` cannot be imported by the current interpreter and that venv exists,
the script re-executes itself with it. If TabPFN fails for any reason the kNN
files are still written and the script exits with code 0.
"""

import argparse
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))


def _reexec_into_g5_venv():
    if "--no-tabpfn" in sys.argv or os.environ.get("G5_REEXEC"):
        return
    try:
        import tabpfn  # noqa: F401

        return
    except Exception:
        pass
    alt = os.path.join(os.path.dirname(HERE), ".venv-g5", "bin", "python")
    if os.path.exists(alt):
        env = dict(os.environ, G5_REEXEC="1")
        os.execve(alt, [alt] + sys.argv, env)


_reexec_into_g5_venv()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU 0 is reserved for the measurement job

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

sys.path.insert(0, HERE)
import common as C  # noqa: E402

warnings.filterwarnings("ignore")

# TabPFN variants: name -> number of top features by LightGBM gain (None = all).
TABPFN_VARIANTS = {"all": None, "top150": 150, "top100": 100, "top60": 60, "top40": 40, "top25": 25}
TABPFN_DEFAULT = "top40,top60,all"  # run by default; best by single-row CV (or their average) is kept
SIZE_COLS = ("out_elems", "macs", "max_tensor_elems", "in_elems", "params", "file_bytes", "ms_cpu", "ms_cuda")
MEAS_PREFIXES = ("ms_", "create_", "scaled_")
RIDGE_COLS = SIZE_COLS + ("log_batch", "n_nodes", "input_elems", "output_elems", "ms_cpu_min", "ms_cuda_min")


# ---------------------------------------------------------------- LightGBM reference
def lgb_fit(Xtr, ytr, seed=0):
    """Same configuration as predict.fit_predict_lgb (one seed)."""
    m = lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=5,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed,
        verbose=-1, n_jobs=8,
    )  # fmt: skip
    m.fit(Xtr, ytr)
    return m


def gain(m) -> np.ndarray:
    g = m.booster_.feature_importance(importance_type="gain").astype(float)
    return g / max(g.sum(), 1e-12)


# ---------------------------------------------------------------- kNN
def standardize(A, B):
    """Median-impute and z-score with train statistics; drops constant columns."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    A = np.where(np.isfinite(A), A, np.nan)
    B = np.where(np.isfinite(B), B, np.nan)
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isnan(A), med, A)
    B = np.where(np.isnan(B), med, B)
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-9
    return (A[:, keep] - mu[keep]) / sd[keep], (B[:, keep] - mu[keep]) / sd[keep], keep


def knn_reps(Xtr: pd.DataFrame, Xte: pd.DataFrame, imp: np.ndarray) -> dict:
    """Structural representations: name -> (train matrix, test matrix)."""
    cols = list(Xtr.columns)
    A, B, keep = standardize(Xtr.values, Xte.values)
    w = imp[keep] / max(imp[keep].max(), 1e-12)
    reps = {"z": (A, B)}
    for g in (0.25, 0.5, 1.0):
        nz = w > 0
        reps[f"w{g}"] = (A[:, nz] * w[nz] ** g, B[:, nz] * w[nz] ** g)
    # op-count profile (log counts) + log sizes with double weight
    cnt = [c for c in cols if c.startswith(("cnt_", "n_"))]
    siz = [c for c in cols if c in SIZE_COLS or c == "log_batch"]
    if cnt and siz:
        Pa, Pb, _ = standardize(np.log1p(Xtr[cnt].clip(lower=0).values), np.log1p(Xte[cnt].clip(lower=0).values))
        Sa, Sb, _ = standardize(Xtr[siz].values, Xte[siz].values)
        reps["prof"] = (np.hstack([Pa, 2 * Sa]), np.hstack([Pb, 2 * Sb]))
    return reps


def knn_predict_all(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, imp: np.ndarray, only=None) -> dict:
    """Predictions of every kNN configuration (or just `only`) -> {config: pred}.

    config = (rep, k, power, mode); mode is "mean", "resid" (neighbours' residuals of
    a ridge fit on size features) or "ratio:<col>:<alpha>" (neighbour latency scaled
    by (size_i / size_j) ** alpha).
    """
    ytr = np.asarray(ytr, float)
    out = {}
    reps = knn_reps(Xtr, Xte, imp)
    size_cols = [c for c in SIZE_COLS if c in Xtr.columns and Xtr[c].notna().mean() > 0.5]
    rcols = [c for c in RIDGE_COLS if c in Xtr.columns]
    Ra, Rb, _ = standardize(Xtr[rcols].values, Xte[rcols].values)
    ridge = Ridge(alpha=1.0).fit(Ra, ytr)
    res_tr, base_te = ytr - ridge.predict(Ra), ridge.predict(Rb)
    modes = ["mean", "resid"] + [f"ratio:{c}:{a}" for c in size_cols for a in (0.5, 1.0)]
    kmax = min(15, len(ytr))
    for rep, (A, B) in reps.items():
        if only is not None and only[0] != rep:
            continue
        D = cdist(B, A)
        idx = np.argsort(D, axis=1)[:, :kmax]
        Ds = np.take_along_axis(D, idx, axis=1)
        scale = np.median(Ds[:, 0]) + 1e-9
        for k in (1, 3, 5, 8, 12):
            k_ = min(k, kmax)
            for p in (1, 2):
                wgt = 1.0 / (Ds[:, :k_] / scale + 0.05) ** p
                wgt = wgt / wgt.sum(1, keepdims=True)
                for mode in modes:
                    cfg = (rep, k, p, mode)
                    if only is not None and cfg != only:
                        continue
                    nb = idx[:, :k_]
                    if mode == "mean":
                        pred = (wgt * ytr[nb]).sum(1)
                    elif mode == "resid":
                        pred = base_te + (wgt * res_tr[nb]).sum(1)
                    else:
                        _, c, a = mode.split(":")
                        s_tr, s_te = Xtr[c].values.astype(float), Xte[c].values.astype(float)
                        delta = np.clip(s_te[:, None] - s_tr[nb], -4, 4)
                        delta = np.where(np.isfinite(delta), delta, 0.0)
                        pred = (wgt * np.log1p(np.expm1(ytr[nb]).clip(min=0) * np.exp(float(a) * delta))).sum(1)
                    out[cfg] = pred
    return out


# ---------------------------------------------------------------- TabPFN
def tabpfn_cols(Xtr: pd.DataFrame, imp: np.ndarray, top) -> list:
    ok = (Xtr.nunique(dropna=True) > 1).values
    order = [i for i in np.argsort(-imp) if ok[i]]
    if top is not None:
        order = [i for i in order[:top] if imp[i] > 0] or order[:top]
    return sorted(order)


def tabpfn_fit_predict(Xtr, ytr, Xte, n_estimators=8, seed=0):
    import torch
    from tabpfn import TabPFNRegressor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = np.asarray(Xtr, np.float32)
    B = np.asarray(Xte, np.float32)
    A[~np.isfinite(A)] = np.nan
    B[~np.isfinite(B)] = np.nan
    kw = dict(device=dev, n_estimators=n_estimators, random_state=seed, ignore_pretraining_limits=True)
    try:  # tabpfn >= 2.5: newer default weights need a licence token, v2 weights are open
        from tabpfn.constants import ModelVersion

        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, **kw)
    except ImportError:
        m = TabPFNRegressor(**kw)
    m.fit(A, np.asarray(ytr, np.float32))
    return np.asarray(m.predict(B), float)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-tabpfn", action="store_true")
    ap.add_argument("--explore", action="store_true", help="run every TabPFN variant (slow)")
    ap.add_argument("--variants", default=TABPFN_DEFAULT, help="comma list of TabPFN variants; best by single-row CV is kept")
    ap.add_argument("--n-estimators", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    df = C.load(a.data)
    trm, tem = (df.split == "train").values, (df.split == "test").values
    tr, te = df[trm].reset_index(drop=True), df[tem].reset_index(drop=True)
    X = C.feature_matrix(df)
    Xtr, Xte = X[trm].reset_index(drop=True), X[tem].reset_index(drop=True)
    y = tr.y.values.astype(float)
    folds = C.make_folds(tr)
    single = (tr.groupby("file").file.transform("size") == 1).values
    has = (te.has_graph == 1).values
    Xte_h = Xte[has].reset_index(drop=True)
    pdir = os.path.join(a.data, "preds")
    os.makedirs(pdir, exist_ok=True)
    print(f"train {len(tr)} (single-row {int(single.sum())}), test {len(te)} (with graph {int(has.sum())}), "
          f"features {X.shape[1]}", flush=True)  # fmt: skip

    def score(p):
        return C.rmse(p, y), C.rmse(p[single], y[single]), C.rmse(p[~single], y[~single]) if (~single).any() else float("nan")

    def save(name, oof, test):
        pd.DataFrame({"id": tr.id.values, "pred": oof}).to_csv(os.path.join(pdir, f"g5_{name}_oof.csv"), index=False)
        pd.DataFrame({"id": te.id.values[has], "pred": test}).to_csv(os.path.join(pdir, f"g5_{name}_test.csv"), index=False)
        assert np.isfinite(oof).all() and np.isfinite(test).all(), f"{name}: non-finite predictions"

    # LightGBM reference: OOF for the blend table, gain importances for kNN weights / TabPFN columns
    oof_lgb, imp_fold = np.zeros(len(tr)), []
    for k in range(5):
        va = folds == k
        m = lgb_fit(Xtr[~va], y[~va])
        oof_lgb[va] = m.predict(Xtr[va])
        imp_fold.append(gain(m))
    imp_full = gain(lgb_fit(Xtr, y))
    results = {"lightgbm": oof_lgb}
    # measurement columns (ms_*, create_*, scaled_*) may be partial or noisy: keep a
    # structure-only view as well, the kNN grid chooses between the two
    meas = [c for c in X.columns if c.startswith(MEAS_PREFIXES)]
    views = {"": (list(X.columns), imp_fold, imp_full)}
    if meas:
        scols = [c for c in X.columns if c not in meas]
        oof_s, imp_s = np.zeros(len(tr)), []
        for k in range(5):
            va = folds == k
            m = lgb_fit(Xtr[~va][scols], y[~va])
            oof_s[va] = m.predict(Xtr[va][scols])
            imp_s.append(gain(m))
        views["struct/"] = (scols, imp_s, gain(lgb_fit(Xtr[scols], y)))
        results["lightgbm, no measurement cols"] = oof_s
    print(f"lightgbm reference done ({len(meas)} measurement columns), {time.time() - t0:.0f}s", flush=True)

    # ---- kNN: every configuration out-of-fold, best one by single-row CV
    oofs = {}
    for view, (vc, vimp, _) in views.items():
        for k in range(5):
            va = folds == k
            for cfg, p in knn_predict_all(Xtr[~va][vc], y[~va], Xtr[va][vc], vimp[k]).items():
                oofs.setdefault((view + cfg[0],) + cfg[1:], np.zeros(len(tr)))[va] = p
    rank = sorted(oofs, key=lambda c: C.rmse(oofs[c][single], y[single]))
    best = rank[0]
    print(f"kNN: {len(oofs)} configs, best by single-row CV: rep={best[0]} k={best[1]} power={best[2]} mode={best[3]}")
    for c in rank[:5]:
        print("   %-42s all %.4f | single %.4f | multi %.4f" % (("%s k%d p%d %s" % c,) + score(oofs[c])))
    view = "struct/" if best[0].startswith("struct/") else ""
    vc, _, vimp_full = views[view]
    raw = (best[0][len(view):],) + best[1:]
    knn_test = knn_predict_all(Xtr[vc], y, Xte_h[vc], vimp_full, only=raw)[raw] if has.any() else np.zeros(0)
    save("knn", oofs[best], knn_test)
    results["g5_knn"] = oofs[best]
    print(f"kNN done, {time.time() - t0:.0f}s", flush=True)

    # ---- TabPFN
    if not a.no_tabpfn:
        try:
            names = list(TABPFN_VARIANTS) if a.explore else [v for v in a.variants.split(",") if v]
            tp = {}
            for name in names:
                t1 = time.time()
                oof = np.zeros(len(tr))
                for k in range(5):
                    va = folds == k
                    cols = tabpfn_cols(Xtr[~va], imp_fold[k], TABPFN_VARIANTS[name])
                    oof[va] = tabpfn_fit_predict(Xtr[~va].iloc[:, cols], y[~va], Xtr[va].iloc[:, cols], a.n_estimators)
                tp[name] = oof
                print("TabPFN %-7s all %.4f | single %.4f | multi %.4f | %.0fs" % ((name,) + score(oof) + (time.time() - t1,)), flush=True)
            if len(tp) > 1:  # the average of the variants is a candidate too
                tp["avg"] = np.mean([tp[n] for n in names], axis=0)
                print("TabPFN %-7s all %.4f | single %.4f | multi %.4f" % (("avg",) + score(tp["avg"])), flush=True)
            sc = {n: C.rmse(tp[n][single], y[single]) for n in tp}
            bestv = min(sc, key=sc.get)
            # single-row differences below 0.003 are noise on ~630 rows: then prefer the
            # average of the variants (steadier, much better on multi-row files)
            if "avg" in sc and sc["avg"] - sc[bestv] < 0.003:
                bestv = "avg"
            tp_test = np.zeros(0)
            if has.any():
                parts = []
                for name in names if bestv == "avg" else [bestv]:
                    cols = tabpfn_cols(Xtr, imp_full, TABPFN_VARIANTS[name])
                    parts.append(tabpfn_fit_predict(Xtr.iloc[:, cols], y, Xte_h.iloc[:, cols], a.n_estimators))
                tp_test = np.mean(parts, axis=0)
            save("tabpfn", tp[bestv], tp_test)
            results[f"g5_tabpfn[{bestv}]"] = tp[bestv]
            print(f"TabPFN done (kept {bestv}), {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # kNN files are already written; report and carry on
            print(f"TabPFN FAILED, g5_tabpfn_* not written: {type(e).__name__}: {str(e)[:500]}", flush=True)

    # ---- CV table
    print("\nCV RMSLE (log1p space)              all     single   multi  | 50/50 with lightgbm: all  single   multi")
    for name, p in results.items():
        line = "%-32s %.4f  %.4f  %.4f" % ((name,) + score(p))
        if not name.startswith("lightgbm"):
            line += "  |                      %.4f  %.4f  %.4f" % score(0.5 * p + 0.5 * oof_lgb)
        print(line)
    g5 = [p for n, p in results.items() if not n.startswith("lightgbm")]
    if len(g5) == 2:
        print("%-32s %.4f  %.4f  %.4f" % (("lgb+knn+tabpfn (1/3 each)",) + score((oof_lgb + g5[0] + g5[1]) / 3)))
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
