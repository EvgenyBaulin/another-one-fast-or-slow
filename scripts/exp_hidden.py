"""Hidden-set simulation (server): repeat the last-hour situation many times.

For each seed: hide 171 random single-row train files, fit every model of the zoo on the
rest, predict the hidden rows, and store per-model predictions together with G1 and the
family rule. The stored table lets us compare stacking / shrinkage / rule variants on
~1700 hidden rows without re-running anything.

  python scripts/exp_hidden.py --seeds 10 --measure cuda  -> data/hidden_sim_cuda.csv
  python scripts/exp_hidden.py --seeds 10 --measure none  -> data/hidden_sim_none.csv
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402

N_HIDE = 171


def run(args):
    seed, measure, zoo_name = args
    P.N_JOBS = 2
    import g1

    df = C.load("data", measure=[m for m in measure.split(",") if m != "none"])
    X = C.feature_matrix(df)
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    single_files = tr.groupby("file").file.transform("size") == 1
    files = tr.file[single_files].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, N_HIDE, replace=False))
    h = tr.file.isin(hide).values
    known, Xk = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True)
    q, Xq = tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = {"all": P.ZOO, "exp": P.EXP_ZOO, "lgb": [z for z in P.ZOO if z[1].startswith("lgb")]}[zoo_name]
    preds = P.fit_predict_all(known, Xk, known.y.values, q, Xq, seeds=(0,), zoo=zoo)
    out = pd.DataFrame({"seed": seed, "id": q.id, "y": q.y, "batch": q.batch, "sig1": q.sig1,
                        "n_nodes": q.n_nodes_nonconst, "macs": q.macs,
                        "io": q.input_elems + q.output_elems})  # fmt: skip
    for n, v in preds.items():
        out[f"m_{n}"] = v
    r = g1.g1_predict(known, q, data_dir="data")
    out["g1_pred"], out["g1_conf"], out["g1_kind"] = r.g1_pred.values, r.g1_conf.values, r.g1_kind.values
    out["fam"], out["fam_kind"] = P.tiny_rules(known, q)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--measure", default="cuda")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--zoo", default="all", choices=["all", "exp", "lgb"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    with Pool(a.procs) as pool:
        res = pool.map(run, [(s, a.measure, a.zoo) for s in range(a.seeds)])
    d = pd.concat(res, ignore_index=True)
    path = f"data/hidden_sim_{a.measure.replace(',', '_')}{a.tag}.csv"
    d.to_csv(path, index=False)
    mcols = [c for c in d.columns if c.startswith("m_")]
    print(f"wrote {path}: {len(d)} rows")
    for c in mcols:
        print(f"  {c[2:]:10s} {C.rmse(d[c], d.y):.4f}   per-seed " + " ".join(f"{C.rmse(g[c], g.y):.3f}" for _, g in d.groupby("seed")))
