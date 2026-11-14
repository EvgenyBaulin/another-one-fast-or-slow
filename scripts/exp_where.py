"""Where is the remaining error? Hidden-set simulation rows by graph class (server)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
d = pd.read_csv("data/hidden_sim_none.csv")
f = pd.read_csv("data/features.csv").set_index("id")
for c in ("n_kernels", "n_ew_unfused", "n_norm_unfused", "n_nop", "cnt_Transpose", "cnt_Resize", "cnt_Reshape", "input_rank", "shape_known_frac", "elems_traffic", "input_elems", "output_elems"):
    d[c] = d.id.map(f[c])
preds = {c[2:]: d[c].values for c in d.columns if c.startswith("m_")}
st = P.shrink_to_base(P.stack_fixed(preds), preds, d.n_nodes.values)
p = P.fam_mix(st, d.fam.values, d.sig1.fillna("").values, d.fam_kind.fillna("").values)
p = P.g1_blend(p, pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values}))
d["p"] = p; d["e"] = p - d.y; d["se"] = d.e**2
def cls(r):
    if r.g1_kind != "none" and isinstance(r.g1_kind, str): return "0 G1 hit"
    if r.macs == 0 and r.n_kernels == r.n_ew_unfused + r.n_norm_unfused and r.n_kernels > 0: return "1 only unfused ew/norm kernels"
    if r.macs == 0: return "2 other, no conv-like"
    if r.n_nodes <= 3: return "3 conv-like, <=3 nodes"
    if r.n_nodes <= 12: return "4 conv-like, 4-12 nodes"
    if r.n_nodes <= 60: return "5 conv-like, 13-60 nodes"
    return "6 conv-like, >60 nodes"
d["cls"] = d.apply(cls, axis=1)
g = d.groupby("cls").agg(n=("se", "size"), rmse=("se", lambda x: float(np.sqrt(x.mean()))), sse=("se", "sum"), bias=("e", "mean"))
g["sse_share"] = g.sse / g.sse.sum(); g["n_per_171"] = g.n / 10
print(g.round(3).to_string()); print("total rmse", round(float(np.sqrt(d.se.mean())), 4))
for name, m in [("has Transpose", d.cnt_Transpose > 0), ("has Resize", d.cnt_Resize > 0), ("2 graph inputs+", d.id.map(f.n_inputs) > 1), ("shape_known<1", d.shape_known_frac < 0.999), ("batch>=16", d.batch >= 16), ("io>8M", d.io > 8e6)]:
    x = d[m & (d.cls != "0 G1 hit")]; print(f"  {name:18s} n {len(x):4d} rmse {np.sqrt(x.se.mean()):.3f} bias {x.e.mean():+.3f} sse share {x.se.sum() / d.se.sum():.3f}")
for name, m in [("io>8M", d.io > 8e6), ("io 2-8M", (d.io > 2e6) & (d.io <= 8e6)), ("has Transpose", d.cnt_Transpose > 0), ("conv-like 4-60 nodes", d.cls.str[0].isin(["4", "5"]))]:
    x = d[m & (d.cls != "0 G1 hit")]
    print(f"  {name:22s} n {len(x):4d}: stack {C.rmse(x.p, x.y):.3f} | base {C.rmse(x.m_base, x.y):.3f} | 0.7stack+0.3base {C.rmse(0.7 * x.p + 0.3 * x.m_base, x.y):.3f} | lgb_dir {C.rmse(x.m_lgb_dir, x.y):.3f} | lgbx_res {C.rmse(x.m_lgbx_res, x.y):.3f} | lgbh_res {C.rmse(x.m_lgbh_res, x.y):.3f}")
u = d[d.cls == "1 only unfused ew/norm kernels"].drop_duplicates("id")
print(u[["id", "batch", "sig1", "n_nodes", "n_norm_unfused", "io", "y", "p", "e", "fam", "fam_kind"]].round(3).sort_values("e").to_string())
w = d[d.cls.str[0].isin(["2", "4", "5"])].sort_values("se", ascending=False).drop_duplicates("id").head(25)
print(w[["id", "batch", "n_nodes", "io", "macs", "cnt_Transpose", "cnt_Resize", "y", "p", "e", "m_base"]].round(3).to_string())
