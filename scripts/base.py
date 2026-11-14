"""Analytic latency model: non-negative additive cost.

    latency ~= c0 + a*io + b*relu(io - IO_KNEE) + c*MACs + sum_op w_op * out_elems_op + d*nodes

where io = graph input elements + graph output elements. Seen in the data: for
graphs that run on the accelerator the transfer of inputs/outputs dominates
(~1.5 ns per element, ~2.5 ns above ~8M elements), compute is almost free, and
some standalone ops (normalisations, unfused activations) cost extra per element.

Weights are fitted with w >= 0 on the log1p error (the competition metric), so
the model extrapolates in size and batch, which tree models cannot do.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

IO_KNEE = 8e6


def design(df: pd.DataFrame, cols=None):
    """Raw (non-log) additive terms. Returns (matrix, column names)."""
    d = {}
    io = df.input_elems.fillna(0) + df.output_elems.fillna(0)
    d["const"] = np.ones(len(df))
    d["io"] = io
    d["io_excess"] = (io - IO_KNEE).clip(lower=0)
    d["in_only"] = df.input_elems.fillna(0)
    d["nodes"] = df.n_nodes_nonconst.fillna(0)
    d["out_elems_all"] = df.out_elems.fillna(0)
    # a handful of op-category terms only: per-op terms let NNLS explain single
    # outliers with absurd weights (tried: CV 0.97 on single-row files)
    d["macs"] = df.macs.fillna(0)
    for c in ("elems_elementwise", "elems_norm", "elems_movement", "elems_reduce", "elems_pool", "elems_other",
              "elems_ew_unfused", "elems_norm_unfused", "elems_traffic"):
        if c in df.columns:
            d[c] = df[c].fillna(0)
    A = pd.DataFrame(d, index=df.index)
    if cols is not None:
        A = A.reindex(columns=cols, fill_value=0.0)
    return A.values.astype(float), list(A.columns)


class AdditiveCost:
    def __init__(self, l2: float = 1e-4):
        self.l2 = l2

    def fit(self, df: pd.DataFrame, y_log1p: np.ndarray):
        A, self.cols = design(df)
        lat = np.expm1(np.asarray(y_log1p, float))
        live = A.mean(axis=0) > 0  # a term absent from train must stay at 0, not at a free weight
        self.scale = np.where(live, np.maximum(A.mean(axis=0), 1e-12), 1.0)
        As = A / self.scale
        # init: relative-error NNLS
        wgt = 1.0 / (lat + 0.3)
        w0, _ = nnls(As * wgt[:, None], lat * wgt, maxiter=20000)
        y = np.asarray(y_log1p, float)

        def loss(w):
            p = As @ w
            r = np.log1p(p) - y
            g = (2 * r / (1 + p)) @ As / len(y) + 2 * self.l2 * w
            return float(np.mean(r**2) + self.l2 * np.sum(w**2)), g

        res = minimize(loss, np.where(live, np.maximum(w0, 1e-6), 0.0), jac=True, method="L-BFGS-B",
                       bounds=[(0, None) if ok else (0, 0) for ok in live], options={"maxiter": 3000})  # fmt: skip
        self.w = np.where(live, res.x, 0.0)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """log1p of the predicted latency."""
        A, _ = design(df, self.cols)
        return np.log1p(np.maximum((A / self.scale) @ self.w, 0.0))

    def weights(self) -> pd.Series:
        """Cost per unit of each term (ms per element / MAC / node)."""
        return pd.Series(self.w / self.scale, index=self.cols)


def crossfit(df_tr: pd.DataFrame, y: np.ndarray, groups: np.ndarray, df_te: pd.DataFrame = None, n_splits: int = 5):
    """Out-of-fold base predictions for train rows (group-wise) and, if given,
    predictions for other rows from a model fitted on all train rows."""
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
