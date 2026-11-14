"""Shrink-to-base variants on the hidden-set simulation (server)."""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P
for tag in ("none", "cuda"):
    d = pd.read_csv(f"data/hidden_sim_{tag}.csv")
    preds = {c[2:]: d[c].values for c in d.columns if c.startswith("m_")}
    st = P.stack_fixed(preds); sd = np.std([v for n, v in preds.items() if n.startswith("lgb")], axis=0)
    def fin(p):
        p = P.fam_mix(p, d.fam.values, d.sig1.fillna("").values)
        return P.g1_blend(p, pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values}))
    print(f"== {tag}: no shrink {C.rmse(fin(st), d.y):.4f}")
    for nmax, tau, lam in itertools.product([1, 3, 10, 1e9], [0.15, 0.25], [0.5, 0.7, 1.0]):
        m = (sd > tau) & (d.n_nodes.values <= nmax)
        p = np.where(m, st + lam * (preds["base"] - st), st)
        e0, e1 = fin(st)[m] - d.y.values[m], fin(p)[m] - d.y.values[m]
        print(f"  nodes<={nmax:<6g} tau {tau} lam {lam}: n {int(m.sum()):3d} -> {C.rmse(fin(p), d.y):.4f} | on those rows: before {np.sqrt(np.mean(e0**2)) if m.any() else 0:.3f} after {np.sqrt(np.mean(e1**2)) if m.any() else 0:.3f}")
