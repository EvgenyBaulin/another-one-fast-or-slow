"""Hidden-set simulation with vs without the feat2.csv features and the rules2 rule (server).

Same split as exp_hidden.py: for each seed hide 171 random single-row train files, fit the
LightGBM-type zoo on the rest and run the final chain
    stack_fixed -> shrink_to_base -> fam_mix(tiny_rules) -> [rules2] -> g1_blend
on the hidden rows. One variant per run (each run stays under a few minutes), the report
compares the stored tables.

  python scripts/feat2.py --data data
  python scripts/exp_feat2.py --variant base          -> data/feat2_sim_base.csv
  python scripts/exp_feat2.py --variant f2            -> data/feat2_sim_f2.csv   (all f2_* columns)
  python scripts/exp_feat2.py --variant f2 --cols rnn,in_,out_ --tag sub  (subset by substring)
  python scripts/exp_feat2.py --variant base --seed0 10 --tag base_b   (ten other hidden splits)
  python scripts/exp_feat2.py --report base f2 ...
  python scripts/exp_feat2.py --cv                    (multi-row files must not get worse)
  python scripts/exp_feat2.py --norm sim [--seed0 10] (current predict.py config: RNN / norm rule variants)
  python scripts/exp_feat2.py --norm cv
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402
import rules2  # noqa: E402

N_HIDE = 171


def frame():
    df = C.load("data", measure=[])
    if any(c.startswith("f2_") for c in df.columns):  # common.load already merges feat2.csv
        return df
    f2 = pd.read_csv("data/feat2.csv").drop_duplicates("id")
    return df.merge(f2, on="id", how="left")


def run(args):
    seed, variant, cols, lgb_seed = args
    P.N_JOBS = 2
    import g1

    df = frame()
    f2c = [c for c in df.columns if c.startswith("f2_")]
    if variant == "base":
        drop = f2c
    else:
        drop = [c for c in f2c if cols and not any(s in c for s in cols)]
    X = C.feature_matrix(df.drop(columns=drop))
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    single_files = tr.groupby("file").file.transform("size") == 1
    files = tr.file[single_files].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, N_HIDE, replace=False))
    h = tr.file.isin(hide).values
    known, Xk = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True)
    q, Xq = tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    preds = P.fit_predict_all(known, Xk, known.y.values, q, Xq, seeds=(lgb_seed,), zoo=zoo)
    st = P.shrink_to_base(P.stack_fixed(preds), preds, q.n_nodes_nonconst.values)
    fam, kind = P.tiny_rules(known, q)
    p = P.fam_mix(st, fam, q.sig1.fillna("").values, kind)
    p_rule = rules2.apply(known, q, p)
    r = g1.g1_predict(known, q, data_dir="data")
    out = pd.DataFrame({"seed": seed, "id": q.id.values, "y": q.y.values, "n_x": X.shape[1]})
    out["p"] = P.g1_blend(p, r)
    out["p_rule"] = P.g1_blend(p_rule, r)
    out["rule_hit"] = (p_rule != p).astype(int)
    out["g1_kind"] = r.g1_kind.values
    return out


def cv_fold(args):
    """One fold of the fp-grouped 5-fold CV used by predict.py: stack + shrink on every train row."""
    k, variant = args
    P.N_JOBS = 2
    df = frame()
    f2c = [c for c in df.columns if c.startswith("f2_")]
    X = C.feature_matrix(df.drop(columns=f2c if variant == "base" else []))
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    folds = C.make_folds(tr)
    va, tn = folds == k, folds != k
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    a, b = tr[tn].reset_index(drop=True), tr[va].reset_index(drop=True)
    preds = P.fit_predict_all(a, Xtr[tn].reset_index(drop=True), a.y.values, b, Xtr[va].reset_index(drop=True), zoo=zoo)
    st = P.shrink_to_base(P.stack_fixed(preds), preds, b.n_nodes_nonconst.values)
    fam, kind = P.tiny_rules(a, b)
    st = P.fam_mix(st, fam, b.sig1.fillna("").values, kind)
    return pd.DataFrame({"variant": variant, "id": b.id.values, "y": b.y.values, "p": st, "p_rule": rules2.apply(a, b, st),
                         "single": (tr.groupby("file").file.transform("size") == 1).values[va]})


def cv():
    with Pool(10) as pool:
        d = pd.concat(pool.map(cv_fold, [(k, v) for v in ("base", "f2") for k in range(5)]), ignore_index=True)
    print("5-fold CV grouped by fp (stack + shrink + tiny rules, no G1):          all     single      multi")
    for v in ("base", "f2"):
        for c in ("p", "p_rule"):
            g = d[d.variant == v]
            print(f"  {v + ('+rule' if c == 'p_rule' else ''):60s} {C.rmse(g[c], g.y):.4f}     {C.rmse(g[c][g.single], g.y[g.single]):.4f}     {C.rmse(g[c][~g.single], g.y[~g.single]):.4f}")


# ----------------------------------------------------------------- current pipeline (norm rule check)
DROP_DEFAULT = "tr_max_rank,n_tr_hi,elems_tr_hi,n_tr6,f2_rs_*,f2_max_kernel,f2_peak_per_io"  # as in predict.py --drop-cols
NORM_COLS = ("p0", "p_rnn", "p_norm", "p_mix", "p_ovr")  # no rules2 | RNN rule | + norm rule alone | + norm 0.5 with model | + norm also over the family rule


def cur_setup():
    """Feature matrix and bases exactly as predict.main builds them by default."""
    P.N_JOBS = 2
    try:
        import base2 as B2

        B2.DATA_DIR = "data"
        P.BASES[:] = [P.B, B2]
    except Exception as e:
        print(f"  base2 not used: {type(e).__name__}: {e}", flush=True)
    df = frame()
    X = C.feature_matrix(df)
    pats = DROP_DEFAULT.split(",")
    X = X.drop(columns=[c for c in X.columns if any(c == q or (q.endswith("*") and c.startswith(q[:-1])) for q in pats)])
    is_tr = df.split.values == "train"
    return df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)


def cur_chain(known, Xk, q, Xq, lgb_seed=0):
    """Stack -> shrink -> tiny rules -> rules2 variants (before G1) on the query rows."""
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    zoo = zoo + [z for z in P.EXP_ZOO if z[0] in P.STACK_FIXED and z not in zoo]
    preds = P.fit_predict_all(known, Xk, known.y.values, q, Xq, seeds=(lgb_seed,), zoo=zoo)
    st = P.shrink_to_base(P.stack_fixed(preds), preds, q.n_nodes_nonconst.values)
    fam, kind = P.tiny_rules(known, q)
    p0 = P.fam_mix(st, fam, q.sig1.fillna("").values, kind)
    out = {"p0": p0, "p_rnn": rules2.apply(known, q, p0, norm=False), "p_norm": rules2.apply(known, q, p0, norm_w=1.0),
           "p_mix": rules2.apply(known, q, p0, norm_w=0.5), "p_ovr": rules2.apply(known, q, p0, norm_w=1.0, norm_over_fam=True)}
    out["stack"] = st
    out["norm_new"] = np.isfinite(rules2.norm_rule(known, q)[0]).astype(int)
    out["norm_any"] = np.isfinite(rules2.norm_rule(known, q, skip_fam=False)[0]).astype(int)
    return out


def cur_run(args):
    seed, lgb_seed = args
    import g1

    tr, Xtr = cur_setup()
    files = tr.file[tr.groupby("file").file.transform("size") == 1].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, N_HIDE, replace=False))
    h = tr.file.isin(hide).values
    known, q = tr[~h].reset_index(drop=True), tr[h].reset_index(drop=True)
    c = cur_chain(known, Xtr[~h].reset_index(drop=True), q, Xtr[h].reset_index(drop=True), lgb_seed)
    r = g1.g1_predict(known, q, data_dir="data")
    out = pd.DataFrame({"seed": seed, "id": q.id.values, "y": q.y.values, "n_x": Xtr.shape[1], "norm_new": c["norm_new"], "norm_any": c["norm_any"]})
    for n in NORM_COLS:
        out[n] = P.g1_blend(c[n], r)
        out[n + "_raw"] = c[n]
    out["stack"] = c["stack"]
    return out


def cur_cv_fold(k):
    tr, Xtr = cur_setup()
    folds = C.make_folds(tr)
    va, tn = folds == k, folds != k
    a, b = tr[tn].reset_index(drop=True), tr[va].reset_index(drop=True)
    c = cur_chain(a, Xtr[tn].reset_index(drop=True), b, Xtr[va].reset_index(drop=True))
    out = pd.DataFrame({"seed": k, "id": b.id.values, "y": b.y.values, "n_x": Xtr.shape[1], "norm_new": c["norm_new"], "norm_any": c["norm_any"],
                        "single": (tr.groupby("file").file.transform("size") == 1).values[va]})
    for n in NORM_COLS:
        out[n] = c[n]
        out[n + "_raw"] = c[n]
    out["stack"] = c["stack"]
    return out


def cur_report(d, title):
    print(f"{title}: {len(d)} rows, {int(d.n_x.iloc[0])} features")
    print(f"{'':34s} {'n':>5s} " + " ".join(f"{n:>8s}" for n in NORM_COLS))

    def line(name, m):
        if m.sum():
            print(f"{name:34s} {int(m.sum()):5d} " + " ".join(f"{C.rmse(d[n][m], d.y[m]):8.4f}" for n in NORM_COLS))

    line("ALL rows", np.ones(len(d), bool))
    if "single" in d.columns:
        line("  single-row files", d.single.values)
        line("  multi-row files", ~d.single.values)
    else:
        for s in sorted(d.seed.unique()):
            line(f"  seed {s}", (d.seed == s).values)
    line("rows the norm rule adds", d.norm_new.values == 1)
    line("rows of the norm condition", d.norm_any.values == 1)
    x = d[d.norm_new == 1]
    if len(x):
        t = x.assign(file=x.id.str.split("_").str[1], stack_err=x["stack"] - x.y, before=x.p_rnn_raw - x.y, rule=x.p_norm_raw - x.y, mix=x.p_mix_raw - x.y, final_rule=x.p_norm - x.y)
        print("  rows the rule adds (errors in log1p; before = stack + tiny rules, final_rule = after G1):")
        print(t[["seed", "file", "y", "stack_err", "before", "rule", "mix", "final_rule"]].round(3).to_string(index=False))
        print(f"  on them before G1: stack+tiny {C.rmse(x.p_rnn_raw, x.y):.4f} | rule alone {C.rmse(x.p_norm_raw, x.y):.4f} | mean {C.rmse(x.p_mix_raw, x.y):.4f}")
    x = d[(d.norm_any == 1) & (d.norm_new == 0)]
    if len(x):
        print(f"  single normalisations covered by tiny_rules ({len(x)} rows), before G1: family rule mixed with the stack {C.rmse(x.p_rnn_raw, x.y):.4f} | norm law instead {C.rmse(x.p_ovr_raw, x.y):.4f}")


CLUSTERS = [
    ("rank-6 Transpose", lambda d: d.n_tr6 > 0),
    ("rank-5 Transpose (max rank 5)", lambda d: d.tr_max_rank == 5),
    ("Resize, <= 40 nodes", lambda d: (d.cnt_Resize > 0) & (d.n_nodes_nonconst <= 40)),
    ("GroupNorm (InstanceNorm, no conv)", lambda d: (d.cnt_InstanceNormalization > 0) & (d.macs == 0)),
    ("SE block (Gemm>=2 + ReduceMean, <= 12 nodes)", lambda d: (d.cnt_Gemm >= 2) & (d.cnt_ReduceMean > 0) & (d.n_nodes_nonconst <= 12)),
    ("recurrent layers", lambda d: d.f2_rnn_nodes > 0),
    ("residual Add from graph input", lambda d: d.f2_res_in_per_io > 0),
    ("graph input read by a movement op", lambda d: d.f2_in_move_frac > 0),
    ("c9a970c39876 / 3d640dd7bbe5", lambda d: d.id.str.contains("c9a970c39876|3d640dd7bbe5")),
]  # fmt: skip


def report(names):
    df = frame().set_index("id")
    tabs = {n: pd.read_csv(f"data/feat2_sim_{n}.csv") for n in names}
    cols = [(n, c) for n in names for c in ("p", "p_rule")]
    b = tabs[names[0]]
    for n in names[1:]:
        assert (tabs[n].id.values == b.id.values).all() and (tabs[n].seed.values == b.seed.values).all(), "different hidden rows"
    head = " ".join(f"{n + ('+rule' if c == 'p_rule' else ''):>12s}" for n, c in cols)
    print(f"{'':46s} {'n':>5s} {head}")

    def line(name, m):
        if m.sum() == 0:
            return
        print(f"{name:46s} {int(m.sum()):5d} " + " ".join(f"{C.rmse(tabs[n][c][m], b.y[m]):12.4f}" for n, c in cols))

    line("ALL hidden rows", np.ones(len(b), bool))
    for s in sorted(b.seed.unique()):
        line(f"  seed {s}", (b.seed == s).values)
    print(f"features in X: " + ", ".join(f"{n} {int(tabs[n].n_x.iloc[0])}" for n in names) + f" | rows changed by the rule: {int(b.rule_hit.sum())}")
    d = df.reindex(b.id.values).reset_index()
    for name, fn in CLUSTERS:
        try:
            line(name, fn(d).fillna(False).values.astype(bool))
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}")
    # squared-error budget: what the rule / features change in total
    for n, c in cols[1:]:
        dse = (tabs[n][c] - b.y) ** 2 - (b.p - b.y) ** 2
        w = dse.groupby(b.id).sum().sort_values()
        print(f"{n + ('+rule' if c == 'p_rule' else ''):>12s}: sum of squared-error change {dse.sum():+.3f} of {((b.p - b.y) ** 2).sum():.2f}; best ids " +
              ", ".join(f"{i.split('_')[1]} {v:+.2f}" for i, v in w.head(4).items()) + " | worst " + ", ".join(f"{i.split('_')[1]} {v:+.2f}" for i, v in w.tail(4).items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "f2"])
    ap.add_argument("--cols", default="", help="comma-separated substrings of f2 columns to keep (variant f2); empty = all")
    ap.add_argument("--tag", default="", help="name of the stored table (default: the variant)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0, help="first hidden-split seed (10 = ten other splits)")
    ap.add_argument("--lgb-seed", type=int, default=0, help="random_state of the models")
    ap.add_argument("--report", nargs="*", default=None)
    ap.add_argument("--cv", action="store_true", help="fp-grouped 5-fold CV on all train rows, with vs without f2")
    ap.add_argument("--norm", choices=["sim", "cv", "report"], default=None, help="current pipeline with the rules2 variants: hidden-set simulation, 5-fold CV, or re-print a stored simulation (--tag)")
    a = ap.parse_args()
    if a.norm == "sim":
        t0 = time.time()
        with Pool(a.procs) as pool:
            d = pd.concat(pool.map(cur_run, [(s, a.lgb_seed) for s in range(a.seed0, a.seed0 + a.seeds)]), ignore_index=True)
        d.to_csv(f"data/feat2_sim_{a.tag or 'norm'}.csv", index=False)
        cur_report(d, f"hidden-set simulation, splits {a.seed0}-{a.seed0 + a.seeds - 1}, {time.time() - t0:.0f}s")
        sys.exit(0)
    if a.norm == "report":
        cur_report(pd.read_csv(f"data/feat2_sim_{a.tag or 'norm'}.csv"), f"stored simulation {a.tag or 'norm'}")
        sys.exit(0)
    if a.norm == "cv":
        with Pool(5) as pool:
            d = pd.concat(pool.map(cur_cv_fold, range(5)), ignore_index=True)
        cur_report(d, "5-fold CV grouped by fp (no G1)")
        sys.exit(0)
    if a.cv:
        cv()
        sys.exit(0)
    if a.report is not None:
        report(a.report or ["base", "f2"])
        sys.exit(0)
    t0 = time.time()
    cols = [c for c in a.cols.split(",") if c]
    with Pool(a.procs) as pool:
        res = pool.map(run, [(s, a.variant, cols, a.lgb_seed) for s in range(a.seed0, a.seed0 + a.seeds)])
    d = pd.concat(res, ignore_index=True)
    path = f"data/feat2_sim_{a.tag or a.variant}.csv"
    d.to_csv(path, index=False)
    print(f"wrote {path}: {len(d)} rows, {int(d.n_x.iloc[0])} features, RMSLE {C.rmse(d.p, d.y):.4f} (with rule {C.rmse(d.p_rule, d.y):.4f}), {time.time() - t0:.0f}s")
    print("  per seed " + " ".join(f"{C.rmse(g.p, g.y):.3f}" for _, g in d.groupby("seed")))
