"""Compare stacking / rule / shrinkage variants on the hidden-set simulation tables (server)."""
import itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P

def rm(p, d): return C.rmse(p, d.y)
def per_seed(p, d): return " ".join(f"{C.rmse(p[d.seed.values == s], d.y[d.seed.values == s]):.3f}" for s in sorted(d.seed.unique()))

def finish(d, model, use_fam=True, use_g1=True):
    p = np.asarray(model, float).copy()
    if use_fam: p = P.fam_mix(p, d.fam.values, d.sig1.fillna("").values)
    if use_g1:
        r = pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values})
        p = P.g1_blend(p, r)
    return p

for tag in ("cuda", "none"):
    path = f"data/hidden_sim_{tag}.csv"
    if not os.path.exists(path): continue
    d = pd.read_csv(path); M = [c for c in d.columns if c.startswith("m_") and c != "m_base"]
    print(f"\n===== measure={tag}: {len(d)} hidden rows, {d.seed.nunique()} seeds; G1 hits {d.g1_pred.notna().mean():.3f}, family hits {d.fam.notna().mean():.3f}")
    for c in ["m_base"] + M: print(f"  {c[2:]:10s} raw {rm(d[c], d):.4f} | +fam+G1 {rm(finish(d, d[c]), d):.4f}")
    sets = {
        "all models": M,
        "res only": [c for c in M if c.endswith("_res")],
        "lgb_res+lgbx_res": ["m_lgb_res", "m_lgbx_res"],
        "lgb_res+lgbx_res+lgbh_res": ["m_lgb_res", "m_lgbx_res", "m_lgbh_res"],
        "lgbx_res+lgbh_res+xgb_res": ["m_lgbx_res", "m_lgbh_res", "m_xgb_res"],
        "lgbx_res+lgbh_res": ["m_lgbx_res", "m_lgbh_res"],
        "res + lgb_dir": [c for c in M if c.endswith("_res")] + ["m_lgb_dir"],
        "lgbx_res+lgbx_bf+lgbh_res": ["m_lgbx_res", "m_lgbx_bf", "m_lgbh_res"],
        "6 lgb-type models": [c for c in M if c.startswith("m_lgb")],
    }
    best = None
    for name, cols in sets.items():
        avg = d[cols].mean(axis=1).values; med = d[cols].median(axis=1).values
        f = finish(d, avg)
        print(f"  AVG {name:28s} raw {rm(avg, d):.4f} | +fam {rm(finish(d, avg, True, False), d):.4f} | +fam+G1 {rm(f, d):.4f} | median+fam+G1 {rm(finish(d, med), d):.4f} | per-seed {per_seed(f, d)}")
        if best is None or rm(f, d) < best[0]: best = (rm(f, d), name, cols)
    name, cols = "lgbx_res+lgbh_res", ["m_lgbx_res", "m_lgbh_res"]
    avg = d[cols].mean(axis=1).values; sd = d[[c for c in M if c.startswith("m_lgb")]].std(axis=1).values; base = d.m_base.values
    print(f"  shrink to base when models disagree (stack = AVG {name}); sd quantiles 50/90/99%: {np.quantile(sd, [.5, .9, .99]).round(3)}")
    for tau, lam in itertools.product([0.1, 0.15, 0.2, 0.25, 0.3], [0.3, 0.5, 0.7]):
        p = np.where(sd > tau, avg + lam * (base - avg), avg)
        print(f"    tau {tau} lam {lam}: n {int((sd > tau).sum())} -> +fam+G1 {rm(finish(d, p), d):.4f}  per-seed {per_seed(finish(d, p), d)}")
