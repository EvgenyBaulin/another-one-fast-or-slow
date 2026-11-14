#!/usr/bin/env python3
"""Rules for two kinds of graphs the trees cannot price: recurrent-only graphs and graphs
built around a standalone normalisation (GroupNorm written as Reshape -> InstanceNorm ->
Reshape -> Mul -> Add and the like). `apply` runs the recurrent rule, then the norm rule.

1. Graphs made of recurrent layers (LSTM / GRU / RNN) and shape plumbing only.

Law seen on the five such single-row train graphs (batch 2, 64 time steps):
    latency = 0.25 ms + 2 ns * (input + output elements) + r * work,
    work = sum over recurrent nodes of  seq_len * directions * hidden_size   (f2_rnn_work),
with r = 95-104 ns for every one of them (1-3 GRU layers, 2-3 bidirectional LSTM layers).
The trees underpredict the LSTM graphs by 0.5-0.8 in log1p: the input and output are tiny
(49 thousand elements), so the io law says 0.3 ms while the graph runs 3.7-5.4 ms.

r is the median over the known recurrent-only graphs of other files, so nothing is fitted
on the query graph itself. `apply` needs the f2_* columns of scripts/feat2.py in both
frames; without them (or with fewer than MIN_KNOWN known graphs) it returns the prediction
unchanged.

2. Graphs without conv-like kernels whose kernels are one or more standalone (unfused)
normalisations plus elementwise / no-op nodes, and for which predict.tiny_rules has no
family (the op signature `sig1` is not in train, or its nearest graph is > 12x away in size).
On train a composite GroupNorm costs per input+output element what a single normalisation
of that size costs (4.8 ns at 4.7M elements against 3.8-4.6; 6.0 at 9.4M against 4.9; the
Mul and Add around the norm are free), so the law is taken from ALL known standalone-norm
graphs of other files: the NORM_K nearest in log(size), not farther than ~3x and only if
the nearest is within ~2x, are moved to the query size along
latency = c0 + a * size^p (p: Theil-Sen slope, 1.0-1.4) and the median is taken.
Size = input + output elements. `NORM_X = "norm"` uses 2 * elems_norm_unfused instead; on
train the two coincide for every graph the rule sees, so the choice cannot be validated.

  python scripts/rules2.py --data data     leave-one-file-out check on train
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

C0_MS = 0.25  # constant overhead of the device
IO_NS = 2.0  # ns per graph input / output element
MIN_KNOWN = 2
R_LO, R_HI = 30.0, 300.0  # ns per unit of work; a ratio outside is a broken graph, not a law
NORM_K = 3  # nearest known standalone-norm graphs used for a query
NORM_MIN_KNOWN = 5
NORM_MAX_DIST = 0.75  # in log(size): the nearest known graph must be within ~2x, else leave it to the model
NORM_NEIGH = 1.1  # neighbours farther than ~3x are not used: below ~0.3M elements the law is flat, above it
#                   grows faster than linearly, so a graph moved across that bend is off by 3x
NORM_X = "io"  # "io": input + output elements; "norm": 2 * elems_norm_unfused
NORM_W = 0.5  # weight of the norm rule against the model prediction. Rows the rule adds: hidden-set
#               simulation model 0.128, rule alone 0.106, mean 0.081; 5-fold CV 0.167 / 0.064 / 0.096.
#               The mean is never worse than the model and halves the worst miss of the rule
#               (BatchNorm is ~20% cheaper than Instance / Group norm of the same size).


def _mask(d: pd.DataFrame) -> np.ndarray:
    if "f2_rnn_only" not in d.columns or "f2_rnn_work" not in d.columns:
        return np.zeros(len(d), bool)
    io = (d.input_elems.fillna(0) + d.output_elems.fillna(0)).values
    ok = (d.f2_rnn_only.fillna(0).values == 1) & (d.f2_rnn_work.fillna(0).values > 0) & (io > 0)
    if "shape_known_frac" in d.columns:
        ok &= d.shape_known_frac.fillna(0).values >= 0.999
    return ok


def _base_ms(d: pd.DataFrame) -> np.ndarray:
    io = (d.input_elems.fillna(0) + d.output_elems.fillna(0)).values.astype(float)
    return C0_MS + IO_NS * 1e-6 * io


def rnn_rule(known: pd.DataFrame, query: pd.DataFrame):
    """(log1p prediction, n_known) per query row; NaN where the rule does not apply."""
    out = np.full(len(query), np.nan)
    nk = np.zeros(len(query), int)
    qm = _mask(query)
    if not qm.any():
        return out, nk
    k = known[_mask(known)]
    if "latency" not in k.columns or not len(k):
        return out, nk
    r = (k.latency.values.astype(float) - _base_ms(k)) * 1e6 / k.f2_rnn_work.values.astype(float)
    good = np.isfinite(r) & (r > R_LO) & (r < R_HI)
    kf, r = k.file.values[good], r[good]
    qb, qw, qf = _base_ms(query), query.f2_rnn_work.fillna(0).values.astype(float), query.file.values
    for i in np.where(qm)[0]:
        m = kf != qf[i]
        # one ratio per known file (a multi-row file must not outvote the others)
        rr = pd.Series(r[m]).groupby(kf[m]).median().values
        nk[i] = len(rr)
        if len(rr) < MIN_KNOWN:
            continue
        out[i] = np.log1p(qb[i] + float(np.median(rr)) * 1e-6 * qw[i])
    return out, nk


def _z(d: pd.DataFrame, c: str) -> np.ndarray:
    return d[c].fillna(0).values.astype(float) if c in d.columns else np.zeros(len(d))


def _norm_mask(d: pd.DataFrame) -> np.ndarray:
    """No conv-like kernels, at least one unfused normalisation, every other kernel is an
    unfused elementwise op (no-ops are not kernels), shapes fully known."""
    io = _z(d, "input_elems") + _z(d, "output_elems")
    nn = _z(d, "n_norm_unfused")
    ok = (_z(d, "has_graph") == 1) & (nn > 0) & (_z(d, "n_kernels") == _z(d, "n_ew_unfused") + nn) & (io > 0)
    if "macs" in d.columns:
        ok &= d.macs.fillna(-1).values == 0
    if "shape_known_frac" in d.columns:
        ok &= d.shape_known_frac.fillna(0).values >= 0.999
    return ok


def _norm_x(d: pd.DataFrame) -> np.ndarray:
    io = _z(d, "input_elems") + _z(d, "output_elems")
    if NORM_X == "norm":
        io = np.where(_z(d, "elems_norm_unfused") > 0, 2 * _z(d, "elems_norm_unfused"), io)
    return np.log(np.clip(io, 1, None))


def _fam_fires(known: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Rows of `query` for which predict.tiny_rules gives a 'fam' / 'noop' prediction (same
    conditions, kept in sync by hand: importing predict from here would be circular)."""
    out = np.zeros(len(query), bool)
    if "sig1" not in known.columns or "sig1" not in query.columns:
        return out

    def ok(d):
        io = _z(d, "input_elems") + _z(d, "output_elems")
        return (_z(d, "has_graph") == 1) & (d.macs.fillna(-1).values == 0) & (io > 0) & (d.shape_known_frac.fillna(0).values >= 0.999)

    lg = lambda d: np.log(np.clip(_z(d, "input_elems") + _z(d, "output_elems"), 1, None))  # noqa: E731
    k = known[ok(known)]
    kx, ks, kf = lg(k), k.sig1.fillna("").astype(str).values, k.file.values
    qx, qs, qf = lg(query), query.sig1.fillna("").astype(str).values, query.file.values
    for i in np.where(ok(query))[0]:
        m = (ks == qs[i]) & (kf != qf[i]) & (qs[i] != "")
        if m.any() and (qs[i] == "noop" or np.abs(kx[m] - qx[i]).min() <= 2.5):
            out[i] = True
    return out


def norm_rule(known: pd.DataFrame, query: pd.DataFrame, skip_fam: bool = True):
    """(log1p prediction, n_known) per query row; NaN where the rule does not apply.
    skip_fam=False also predicts the rows that predict.tiny_rules covers (for the check)."""
    out = np.full(len(query), np.nan)
    nk = np.zeros(len(query), int)
    qm = _norm_mask(query)
    if skip_fam and qm.any():
        qm &= ~_fam_fires(known, query)
    if not qm.any() or "latency" not in known.columns:
        return out, nk
    k = known[_norm_mask(known)]
    k = k[np.isfinite(k.latency.values.astype(float)) & (k.latency.values.astype(float) > 0)]
    kx, klat, kf = _norm_x(k), k.latency.values.astype(float), k.file.values
    qx, qf = _norm_x(query), query.file.values
    for i in np.where(qm)[0]:
        m = kf != qf[i]
        nk[i] = int(m.sum())
        if nk[i] < NORM_MIN_KNOWN:
            continue
        x, lat = kx[m], klat[m]
        c0 = min(0.2, 0.8 * lat.min())
        big = lat > 3 * c0
        yl = np.log(lat[big] - c0)
        xb = x[big]
        sl = [(yl[b] - yl[a]) / (xb[b] - xb[a]) for a in range(len(xb)) for b in range(a + 1, len(xb)) if abs(xb[b] - xb[a]) > 0.5]
        p = float(np.clip(np.median(sl), 1.0, 1.4)) if sl else 1.15
        d = np.abs(x - qx[i])
        if d.min() > NORM_MAX_DIST:
            continue
        j = np.argsort(d, kind="stable")[:NORM_K]
        j = j[d[j] <= NORM_NEIGH]
        out[i] = float(np.median(np.log1p(c0 + (lat[j] - c0) * np.exp(p * (qx[i] - x[j])))))
    return out, nk


def apply(known_df: pd.DataFrame, query_df: pd.DataFrame, pred_log1p, norm: bool = True, norm_w: float = None, norm_over_fam: bool = False) -> np.ndarray:
    """Recurrent-layer law where it applies, then the standalone-normalisation law (weight
    NORM_W against the incoming prediction; rows already set by the recurrent rule are kept).
    norm_over_fam=True lets the norm law also replace the family rule of predict.tiny_rules
    on single normalisations (experiment only)."""
    p = np.asarray(pred_log1p, float).copy()
    done = np.zeros(len(p), bool)
    try:
        rule, _ = rnn_rule(known_df, query_df)
        done = np.isfinite(rule)
        p[done] = rule[done]
    except Exception as e:  # an optional rule must never break a submission
        print(f"  rules2 (recurrent) skipped: {type(e).__name__}: {e}", flush=True)
    if norm:
        try:
            w = NORM_W if norm_w is None else norm_w
            rule, _ = norm_rule(known_df, query_df, skip_fam=not norm_over_fam)
            ok = np.isfinite(rule) & ~done
            p[ok] = w * rule[ok] + (1 - w) * p[ok]
        except Exception as e:
            print(f"  rules2 (norm) skipped: {type(e).__name__}: {e}", flush=True)
    return p


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    df = C.load(a.data, measure=[])
    if "f2_rnn_work" not in df.columns:  # common.load may already merge feat2.csv
        df = df.merge(pd.read_csv(os.path.join(a.data, "feat2.csv")).drop_duplicates("id"), on="id", how="left")
    tr = df[df.split == "train"].reset_index(drop=True)
    rule, nk = rnn_rule(tr, tr)  # the rule never uses rows of the query's own file
    hit = np.isfinite(rule)
    print(f"recurrent-only graphs in train: {int(_mask(tr).sum())}, rule applies to {int(hit.sum())} (leave-one-file-out)")
    t = tr[hit][["id", "batch", "latency", "f2_rnn_nodes", "f2_rnn_work"]].copy()
    t["rule_ms"], t["err_log1p"], t["known"] = np.expm1(rule[hit]), rule[hit] - tr.y.values[hit], nk[hit]
    t["io_law_ms"] = _base_ms(tr[hit])
    print(t.round(3).to_string(index=False))
    if hit.any():
        print(f"rule RMSLE {C.rmse(rule[hit], tr.y.values[hit]):.4f} | io law alone {C.rmse(np.log1p(_base_ms(tr[hit])), tr.y.values[hit]):.4f}")
    te = df[(df.split == "test") & (df.has_graph == 1)]
    print(f"test graph rows the rule would change: {int(np.isfinite(rnn_rule(tr, te)[0]).sum())}")

    # ---- standalone-normalisation law
    global NORM_X, NORM_K
    y = tr.y.values
    cov = _norm_mask(tr)
    fam = _fam_fires(tr, tr)
    print(f"\nnorm rule: {int(cov.sum())} train rows match the condition, tiny_rules covers {int((cov & fam).sum())} of them, the rule fires on {int(np.isfinite(norm_rule(tr, tr)[0]).sum())}")
    for xv in ("io", "norm"):
        for kk in (1, 3, 5):
            NORM_X, NORM_K = xv, kk
            r_all, _ = norm_rule(tr, tr, skip_fam=False)
            h = np.isfinite(r_all)
            new = h & ~fam
            print(f"  x={xv:4s} k={kk}: leave-one-file-out on all {int(h.sum())} rows {C.rmse(r_all[h], y[h]):.4f} | on the {int((h & fam).sum())} rows of tiny_rules {C.rmse(r_all[h & fam], y[h & fam]):.4f} | on the {int(new.sum())} rows the rule adds {C.rmse(r_all[new], y[new]):.4f}")
    NORM_X, NORM_K = "io", 3
    r_all, nk = norm_rule(tr, tr, skip_fam=False)
    h = np.isfinite(r_all)
    try:
        import predict as P

        fr, fk = P.tiny_rules(tr, tr)
        hf = h & np.isfinite(fr)
        print(f"  tiny_rules itself on its {int(hf.sum())} rows: {C.rmse(fr[hf], y[hf]):.4f} (norm rule there {C.rmse(r_all[hf], y[hf]):.4f})")
    except Exception as e:
        fr = np.full(len(tr), np.nan)
        print(f"  tiny_rules not available here: {type(e).__name__}: {e}")
    t = tr[h][["file", "batch", "sig1", "n_nodes_nonconst", "latency"]].copy()
    t["io_M"] = (_z(tr, "input_elems") + _z(tr, "output_elems"))[h] / 1e6
    t["rule_ms"], t["err"], t["fam_err"], t["adds"] = np.expm1(r_all[h]), r_all[h] - y[h], (fr - y)[h], (~fam)[h].astype(int)
    print(t.sort_values(["adds", "io_M"]).round(3).to_string(index=False))
    print(f"test graph rows the norm rule would change: {int(np.isfinite(norm_rule(tr, te)[0]).sum())}")


if __name__ == "__main__":
    main()
