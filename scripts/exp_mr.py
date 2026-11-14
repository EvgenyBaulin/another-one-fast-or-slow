"""Multi-row files: how should G1 (batch curves) and the model be mixed? (server)

Test rows of files that also have rows in train are the bulk of the PUBLIC leaderboard
(54 of 128 rows). Simulation: hold out random rows of multi-row files (the other batches
of the same file stay in train), predict them with the model stack and with G1, and
compare mixes per G1 kind.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402

REPS = 2
K = 7  # the largest multi-row file has 7 rows


def run(args):
    rep, k = args
    P.N_JOBS = 2
    import g1

    df = C.load("data", measure=[])
    X = C.feature_matrix(df)
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    multi = np.where((tr.groupby("file").file.transform("size") > 1).values)[0]
    # like the test: at most one hidden row per file (fold = position of the row in a random
    # order of the rows of its file)
    rng = np.random.RandomState(100 + rep)
    fold = np.zeros(len(multi), int)
    fm = tr.file.values[multi]
    for f in np.unique(fm):
        ix = np.where(fm == f)[0]
        fold[ix] = rng.permutation(len(ix))
    h = np.zeros(len(tr), bool)
    h[multi[fold == k]] = True
    if not h.any():
        return None, None
    known, Xk, q, Xq = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True), tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    # one fit, predictions for the hidden rows and (in-sample) for the known rows: the
    # in-sample residual of a known batch of the same file is the file-level model bias
    nq = len(q)
    pr_all = P.fit_predict_all(known, Xk, known.y.values, pd.concat([q, known], ignore_index=True), pd.concat([Xq, Xk], ignore_index=True), zoo=zoo)
    pr = {n: v[:nq] for n, v in pr_all.items()}
    kn = pd.DataFrame({"rep": rep, "k": k, "id": known.id, "stk_in": P.stack_fixed({n: v[nq:] for n, v in pr_all.items()})})
    r = g1.g1_predict(known, q, data_dir="data")
    out = pd.DataFrame({"rep": rep, "k": k, "id": q.id, "file": q.file, "batch": q.batch, "shape": q["shape"], "y": q.y,
                        "stk": P.stack_fixed(pr), "g1": r.g1_pred.values, "kind": r.g1_kind.values, "src": r.g1_src.values,
                        "detail": r.g1_detail.values, "conf": r.g1_conf.values})  # fmt: skip
    for n, v in pr.items():
        out[f"m_{n}"] = v
    return out, kn


if __name__ == "__main__":
    with Pool(15) as pool:
        res = pool.map(run, [(r, k) for r in range(REPS) for k in range(K)])
    res = [x for x in res if x[0] is not None]
    d = pd.concat([x[0] for x in res], ignore_index=True)
    d.to_csv("data/mr_sim.csv", index=False)
    pd.concat([x[1] for x in res], ignore_index=True).to_csv("data/mr_sim_known.csv", index=False)
    pd.set_option("display.width", 220)
    rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))  # noqa: E731
    print(f"{len(d)} held-out rows of multi-row files; model stack alone {rm(d.stk, d.y):.4f}")
    for c in [c for c in d.columns if c.startswith("m_")]:
        print(f"   {c[2:]:10s} {rm(d[c], d.y):.4f}")
    cur = P.g1_blend(d.stk.values, pd.DataFrame({"g1_pred": d.g1.values, "g1_conf": d.conf.values}))
    print(f"current rule (inverse variance, sigma_model 0.25): {rm(cur, d.y):.4f}")
    rows = []
    d["detail"] = d.detail + "/" + d.src.fillna("")
    for kd, g in d.groupby("detail"):
        ok = g.g1.notna()
        row = {"detail": kd, "n": len(g), "model": rm(g.stk, g.y), "g1": rm(g.g1[ok], g.y[ok]) if ok.any() else np.nan,
               "current": rm(cur[g.index], g.y)}
        for w in (0.3, 0.5, 0.7, 0.85):
            row[f"w_g1={w}"] = rm(np.where(ok, w * g.g1.fillna(0) + (1 - w) * g.stk, g.stk), g.y)
        for r_ in range(REPS):
            gg = g[g.rep == r_]
            row[f"m/g1 rep{r_}"] = f"{rm(gg.stk, gg.y):.3f}/{rm(gg.g1.fillna(gg.stk), gg.y):.3f}" if len(gg) else ""
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("by batch (model vs g1):")
    print(d.groupby("batch").apply(lambda g: pd.Series({"n": len(g), "model": rm(g.stk, g.y), "g1": rm(g.g1.fillna(g.stk), g.y)})).round(3).to_string())
