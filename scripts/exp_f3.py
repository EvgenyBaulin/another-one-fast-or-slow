"""Hidden-set A/B for a group of feature columns (server): the full chain with and without
the columns matching --prefix (default f3_). Same splits as scripts/exp_hidden.py.

  python scripts/exp_f3.py --prefix f3_ --seeds 10
"""
import argparse, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, predict as P

DEFAULT_DROP = ["tr_max_rank", "n_tr_hi", "elems_tr_hi", "n_tr6", "f2_rs_*", "f2_max_kernel", "f2_peak_per_io"]

def run(args):
    seed, variant, prefix = args
    P.N_JOBS = 2
    import base2 as B2, g1, rules2
    B2.DATA_DIR = "data"; P.BASES[:] = [P.B, B2]
    df = C.load("data", measure=[]); X = C.feature_matrix(df)
    pats = DEFAULT_DROP + ([prefix + "*"] if variant == "base" else [])
    X = X.drop(columns=[c for c in X.columns if any(c == q or (q.endswith("*") and c.startswith(q[:-1])) for q in pats)])
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    files = tr.file[tr.groupby("file").file.transform("size") == 1].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, 171, replace=False))
    h = tr.file.isin(hide).values
    k, Xk, q, Xq = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True), tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    pr = P.fit_predict_all(k, Xk, k.y.values, q, Xq, zoo=zoo)
    st = P.shrink_to_base(P.stack_fixed(pr), pr, q.n_nodes_nonconst.values)
    fr, fk = P.tiny_rules(k, q)
    p = P.fam_mix(st, fr, q.sig1.values, fk)
    p = rules2.apply(k, q, p)
    p = P.g1_blend(p, g1.g1_predict(k, q, data_dir="data"))
    return seed, variant, C.rmse(p, q.y), C.rmse(st, q.y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="f3_"); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    jobs = [(s, v, a.prefix) for s in range(a.seed0, a.seed0 + a.seeds) for v in ("base", "with")]
    with Pool(10) as pool: res = pool.map(run, jobs)
    d = pd.DataFrame(res, columns=["seed", "variant", "chain", "stack"]).pivot(index="seed", columns="variant")
    d["diff"] = d[("chain", "with")] - d[("chain", "base")]
    print(d.round(4).to_string())
    ch = lambda v: float(np.sqrt((d[("chain", v)] ** 2).mean()))
    print(f"chain RMSLE pooled: without {ch('base'):.4f}  with {ch('with'):.4f}  | mean diff {d['diff'].mean():+.4f} ± {d['diff'].std() / np.sqrt(len(d)):.4f} | seeds better: {(d['diff'] < 0).sum()} of {len(d)}")
