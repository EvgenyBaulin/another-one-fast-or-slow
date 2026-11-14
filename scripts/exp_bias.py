"""Mean error (bias) of the out-of-fold stack by graph class (server). Run after predict.py."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
df = C.load("data", measure=[]); tr = df[df.split == "train"].reset_index(drop=True)
tr["p"] = tr.id.map(pd.read_csv("data/preds/main_STACK_oof.csv").set_index("id").pred); tr["e"] = tr.p - tr.y
single = tr.groupby("file").file.transform("size") == 1
s = tr[single].copy()
def rep(name, d):
    e = d.e.values; tm = np.sort(e)[int(len(e) * 0.05): int(len(e) * 0.95)]
    print(f"  {name:28s} n {len(d):4d} mean {e.mean():+.4f} (se {e.std() / np.sqrt(len(e)):.4f}) median {np.median(e):+.4f} trimmed-mean {tm.mean():+.4f} rmse {np.sqrt((e**2).mean()):.4f}")
rep("single-row all", s)
for m in ("main_lgbx_res", "main_lgbx_bf", "main_lgbw_res", "main_lgbh_res", "main_base", "g6_gnn"):
    f = f"data/preds/{m}_oof.csv"
    if os.path.exists(f):
        e = s.id.map(pd.read_csv(f).set_index("id").pred) - s.y
        print(f"  model {m:22s} mean {e.mean():+.4f} median {e.median():+.4f}")
s["cls"] = np.select([s.n_nodes_nonconst <= 3, s.n_nodes_nonconst <= 12, s.n_nodes_nonconst <= 60], ["<=3 nodes", "4-12", "13-60"], ">60")
for k, d in s.groupby("cls"): rep(k, d)
s["yb"] = pd.cut(s.y, [-1, 0.3, 0.7, 1.5, 3, 99])
for k, d in s.groupby("yb", observed=True): rep(f"y in {k}", d)
s["pb"] = pd.cut(s.p, [-1, 0.3, 0.7, 1.5, 3, 99])
for k, d in s.groupby("pb", observed=True): rep(f"pred in {k}", d)
for k, d in s.groupby("batch"): rep(f"batch {k}", d)
