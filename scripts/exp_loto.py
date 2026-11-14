"""Leave-one-op-type-out check (server): how wrong are we on a tiny graph whose op
signature is absent from train? For every signature `sig1` of tiny non-conv graphs all its
rows are hidden, the pipeline (stack, shrink, family rule; no G1) is fitted on the rest."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P

def run(sig):
    P.N_JOBS = 2
    df = C.load("data", measure=[]); X = C.feature_matrix(df)
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    h = (tr.sig1.astype(str).values == sig) & (tr.macs.fillna(0).values == 0)
    k, Xk, q, Xq = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True), tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    pr = P.fit_predict_all(k, Xk, k.y.values, q, Xq, zoo=zoo)
    st = P.stack_fixed(pr); sh = P.shrink_to_base(st, pr, q.n_nodes_nonconst.values)
    fin = P.fam_mix(sh, P.family_rule(k, q), q.sig1.values)
    io = (q.input_elems + q.output_elems).values
    return pd.DataFrame({"sig1": sig, "batch": q.batch, "io_M": io / 1e6, "latency": q.latency, "y": q.y, "stk": st, "shrunk": sh, "final": fin, "base": pr["base"]})

if __name__ == "__main__":
    df = C.load("data", measure=[]); tr = df[df.split == "train"]
    sigs = sorted(set(tr.sig1[(tr.macs.fillna(0) == 0) & tr.sig1.notna() & (tr.sig1.astype(str) != "")].astype(str)))
    with Pool(11) as pool: d = pd.concat(pool.map(run, sigs), ignore_index=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    g = d.groupby("sig1").apply(lambda x: pd.Series(dict(n=len(x), stack=C.rmse(x.stk, x.y), shrunk=C.rmse(x.shrunk, x.y), base=C.rmse(x.base, x.y), bias=float((x.shrunk - x.y).mean()))))
    print(g.round(3).to_string())
    print(f"all {len(d)} rows: stack {C.rmse(d.stk, d.y):.4f} | shrunk {C.rmse(d.shrunk, d.y):.4f} | base {C.rmse(d.base, d.y):.4f} | mean bias of shrunk {float((d.shrunk - d.y).mean()):+.3f}")
    one = d[~d.sig1.str.contains(r"\+")]
    print(f"one-op signatures, {len(one)} rows: stack {C.rmse(one.stk, one.y):.4f} | shrunk {C.rmse(one.shrunk, one.y):.4f} | base {C.rmse(one.base, one.y):.4f}")
    for a in (0.0, 0.3, 0.5, 0.7, 1.0):
        print(f"   blend {a:.1f}*base + {1 - a:.1f}*stack on all tiny rows: {C.rmse(a * d.base + (1 - a) * d.stk, d.y):.4f}")
