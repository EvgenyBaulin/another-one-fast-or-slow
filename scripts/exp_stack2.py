"""Stack composition on the hidden-set simulation with the full rule chain (server)."""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/hidden_sim_none_exp.csv")
preds = {c[2:]: d[c].values for c in d.columns if c.startswith("m_")}
core6 = {k: preds[k] for k in ("base", "lgb_dir", "lgb_res", "lgb_bf", "lgbx_res", "lgbx_bf", "lgbh_res")}
def fin(st):
    st = P.shrink_to_base(st, core6, d.n_nodes.values)
    p = P.fam_mix(st, d.fam.values, d.sig1.fillna("").values, d.fam_kind.fillna("").values)
    return P.g1_blend(p, pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values}))
def show(name, cols):
    p = fin(np.mean([preds[c] for c in cols], axis=0))
    print(f"  {name:44s} {C.rmse(p, d.y):.4f}  per-seed " + " ".join(f"{C.rmse(p[d.seed.values == s], d.y[d.seed.values == s]):.3f}" for s in sorted(d.seed.unique())))
show("current: lgbx_res lgbx_bf lgbh_res", ["lgbx_res", "lgbx_bf", "lgbh_res"])
names = [n for n in preds if n != "base"]
for n in names: show(n, [n])
res = []
for k in (2, 3, 4):
    for cols in itertools.combinations(names, k):
        p = fin(np.mean([preds[c] for c in cols], axis=0)); res.append((C.rmse(p, d.y), cols))
res.sort()
print("best subsets:")
for r, cols in res[:12]: print(f"  {r:.4f}  {' '.join(cols)}")
show("lgbx_res lgbm_bf lgbw_res", ["lgbx_res", "lgbm_bf", "lgbw_res"])
show("lgbx_res lgbx_bf lgbm_bf lgbw_res", ["lgbx_res", "lgbx_bf", "lgbm_bf", "lgbw_res"])
show("lgbx_res lgbm_bf lgbm_dir lgbw_res", ["lgbx_res", "lgbm_bf", "lgbm_dir", "lgbw_res"])
