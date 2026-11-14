"""Validation of scripts/base2.py against scripts/base.py (server).

  python scripts/exp_base2.py --cv       base alone, out-of-fold (common.make_folds), fitted rates
  python scripts/exp_base2.py --hidden   (resumable) hidden-set simulation: 10 seeds x 171 hidden single-row files,
                                         lgb-type zoo fitted twice (base.py / base2.py as P.B)
                                         -> data/hidden_sim_base2.csv, then the chain report
  python scripts/exp_base2.py --eval     chain report from the stored table
  python scripts/exp_base2.py --predict two --data D --out X.csv [predict.py args]
                                         predict.py as is, but every tree model is the mean of its
                                         versions on base.py and base2.py (--predict base2: base2 only)

Chain = P.stack_fixed -> P.shrink_to_base -> P.fam_mix (P.tiny_rules kinds) -> P.g1_blend,
the same as predict.py without G6. Needs data/base2_feats.csv (scripts/base2.py --build).
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base as B1  # noqa: E402
import base2 as B2  # noqa: E402
import common as C  # noqa: E402
import predict as P  # noqa: E402

N_HIDE = 171
TABLE = "data/hidden_sim_base2.csv"


# ------------------------------------------------------------------ 1. base alone
def cv_base(seeds=(0, 1, 2)):
    df = C.load("data", measure=[])
    tr = df[df.split.values == "train"].reset_index(drop=True)
    y = tr.y.values
    single = (tr.groupby("file").file.transform("size") == 1).values
    print("base alone, out-of-fold RMSLE (folds = common.make_folds, grouped by fp):")
    for s in seeds:
        folds = C.make_folds(tr, seed=s)
        for name, mod in (("base.py", B1), ("base2.py", B2)):
            oof = np.zeros(len(tr))
            for k in range(5):
                va = folds == k
                oof[va] = mod.AdditiveCost().fit(tr[~va], y[~va]).predict(tr[va])
            print(f"  fold seed {s}  {name:9s} all {C.rmse(oof, y):.4f} | single {C.rmse(oof[single], y[single]):.4f} | multi {C.rmse(oof[~single], y[~single]):.4f}", flush=True)
    m = B2.AdditiveCost().fit(tr, y)
    print(f"base2 fitted on all train rows ({len(m.cols)} terms, fitted={m.fitted}); ns per element, ms for const / dev / n_*:")
    for t, v in m.weights().items():
        unit = 1.0 if t in ("const", "dev") or t.startswith("n_") else 1e6
        print(f"  {t:12s} {v * unit:9.4f}")


# ------------------------------------------------------------------ 2. hidden-set simulation
def code_key(mode: str) -> str:
    """Parts are reused only while the code that produced them is unchanged (predict.py and
    g1.py are edited by other sessions): hash of the files, for base2 parts also of base2.py
    and of the BASE2_* experiment overrides."""
    import hashlib

    here = os.path.dirname(os.path.abspath(__file__))
    files = ["predict.py", "g1.py", "common.py", "base.py"] + (["base2.py"] if mode == "b" else [])
    h = hashlib.md5()
    for f in files:
        h.update(open(os.path.join(here, f), "rb").read())
    for f in ("features.csv", "ident.csv"):
        fp = os.path.join("data", f)
        h.update(str(os.path.getmtime(fp) if os.path.exists(fp) else 0).encode())
    if mode == "b":
        h.update("|".join(os.environ.get(k, "") for k in ("BASE2_TERMS", "BASE2_DROP", "BASE2_W_MULTI")).encode())
    return h.hexdigest()[:8]


def part_path(seed, mode, key):
    return os.path.join("data", "hsb2_parts", f"{mode}_{key}", f"seed{seed}.csv")


def run(args):
    """One (seed, base) job -> its own CSV, so a killed run is resumed, not repeated."""
    seed, mode, key = args
    path = part_path(seed, mode, key)
    if os.path.exists(path):
        return path
    P.N_JOBS = 2
    P.B = B2 if mode == "b" else B1
    df = C.load("data", measure=[])
    X = C.feature_matrix(df)
    is_tr = df.split.values == "train"
    tr, Xtr = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    single_files = tr.groupby("file").file.transform("size") == 1
    files = tr.file[single_files].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, N_HIDE, replace=False))  # same split as exp_hidden.py
    h = tr.file.isin(hide).values
    known, Xk = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True)
    q, Xq = tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    zoo = [z for z in P.ZOO if z[1].startswith("lgb")]
    preds = P.fit_predict_all(known, Xk, known.y.values, q, Xq, seeds=(0,), zoo=zoo)
    out = pd.DataFrame({"seed": seed, "id": q.id})
    for n, v in preds.items():
        out[f"{mode}_{n}"] = v
    if mode == "a":  # everything that does not depend on the base
        import g1

        out["y"], out["sig1"], out["n_nodes"] = q.y.values, q.sig1.values, q.n_nodes_nonconst.values
        r = g1.g1_predict(known, q, data_dir="data")
        out["g1_pred"], out["g1_conf"] = r.g1_pred.values, r.g1_conf.values
        out["fam"], out["fam_kind"] = P.tiny_rules(known, q)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path


def hidden(n_seeds: int, procs: int):
    t0 = time.time()
    key = {m: code_key(m) for m in "ab"}
    jobs = [(s, m, key[m]) for m in ("a", "b") for s in range(n_seeds)]
    todo = [j for j in jobs if not os.path.exists(part_path(*j))]
    print(f"code keys: base.py parts {key['a']}, base2 parts {key['b']}; {len(jobs) - len(todo)} of {len(jobs)} parts exist, running {len(todo)}", flush=True)
    if todo:
        with Pool(min(procs, len(todo))) as pool:
            for i, p in enumerate(pool.imap_unordered(run, todo, chunksize=1), 1):
                print(f"  [{i}/{len(todo)}] {p} {time.time() - t0:.0f}s", flush=True)
    a = pd.concat([pd.read_csv(part_path(s, "a", key["a"])) for s in range(n_seeds)], ignore_index=True)
    b = pd.concat([pd.read_csv(part_path(s, "b", key["b"])) for s in range(n_seeds)], ignore_index=True)
    d = a.merge(b, on=["seed", "id"], how="left", validate="one_to_one")
    d.to_csv(TABLE, index=False)
    print(f"wrote {TABLE}: {len(d)} rows, {time.time() - t0:.0f}s", flush=True)


def chain(d, pre, base=None, member=None, member_w=0.0, shrink=True):
    """Final prediction of the chain. pre: column prefix of the tree models ("a_" trees on
    base.py, "b_" trees on base2.py); base: column used by shrink_to_base (default: the
    base the trees were built on); member / member_w: column added to the stack and its weight."""
    preds = {c[len(pre):]: d[c].values for c in d.columns if c.startswith(pre)}
    if base is not None:
        preds["base"] = d[base].values
    p = P.stack_fixed(preds)
    if member is not None:
        p = (1 - member_w) * p + member_w * d[member].values
    if shrink:
        p = P.shrink_to_base(p, preds, d.n_nodes.values)
    p = P.fam_mix(p, d.fam.values, d.sig1.fillna("").values, d.fam_kind.fillna("").values)
    return P.g1_blend(p, pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values}))


def report():
    d = pd.read_csv(TABLE)
    seeds = sorted(d.seed.unique())
    print(f"hidden-set simulation: {len(d)} hidden rows, {len(seeds)} seeds; RMSLE overall | per seed")

    def line(name, p):
        ps = [C.rmse(p[d.seed.values == s], d.y[d.seed.values == s]) for s in seeds]
        print(f"  {name:58s} {C.rmse(p, d.y):.4f} | " + " ".join(f"{v:.3f}" for v in ps), flush=True)
        return np.array(ps)

    line("base.py alone", d.a_base.values)
    line("base2.py alone", d.b_base.values)
    line("stack_fixed only, trees on base.py", P.stack_fixed({c[2:]: d[c].values for c in d.columns if c.startswith("a_")}))
    line("stack_fixed only, trees on base2.py", P.stack_fixed({c[2:]: d[c].values for c in d.columns if c.startswith("b_")}))
    ref = line("CHAIN current (base.py everywhere)", chain(d, "a_"))
    full = line("CHAIN base2 everywhere (P.B = base2)", chain(d, "b_"))
    line("CHAIN trees on base2, shrink to base.py", chain(d, "b_", base="a_base"))
    v1 = line("CHAIN trees on base.py, shrink to base2", chain(d, "a_", base="b_base"))
    v2 = line("CHAIN trees on base.py, shrink to base2 + member 1/4", chain(d, "a_", base="b_base", member="b_base", member_w=0.25))
    line("CHAIN trees on base.py, shrink to base2 + member 0.1", chain(d, "a_", base="b_base", member="b_base", member_w=0.1))
    line("CHAIN trees on base.py, base2 member 1/4, shrink to base.py", chain(d, "a_", member="b_base", member_w=0.25))
    v3 = line("CHAIN mean of both chains", 0.5 * (chain(d, "a_") + chain(d, "b_")))
    # every tree model = mean of its two versions (trained on base.py and on base2.py); shrink to base.py.
    # This is what a two-line change of predict.fit_predict_all would do.
    both = d[["seed", "id", "y", "sig1", "n_nodes", "g1_pred", "g1_conf", "fam", "fam_kind"]].copy()
    for c in d.columns:
        if c.startswith("a_") and c != "a_base":
            both["m_" + c[2:]] = 0.5 * (d[c].values + d["b_" + c[2:]].values)
    both["m_base"] = d.a_base.values
    v4 = line("CHAIN per-model mean of both bases, shrink to base.py", chain(both, "m_"))
    both["m_base"] = d.b_base.values
    line("CHAIN per-model mean of both bases, shrink to base2", chain(both, "m_"))
    for name, v in (("base2 everywhere", full), ("shrink to base2", v1), ("shrink to base2 + member 1/4", v2), ("mean of both chains", v3),
                    ("per-model mean, shrink to base.py", v4)):
        print(f"  delta per seed vs current, {name:30s} " + " ".join(f"{x:+.3f}" for x in v - ref) + f" | seeds worse by > 0.005: {int((v - ref > 0.005).sum())}")


# ------------------------------------------------------------------ 3. predict.py with base2, without editing it
def two_base_fit_predict(orig):
    """fit_predict_all that fits every model once per base (base.py, base2.py) and averages
    the two versions; the 'base' entry (fallback of shrink_to_base) is base2. This is the
    variant "per-model mean of both bases, shrink to base2" of the report."""

    def f(tr, Xtr, y, te, Xte, seeds=(0,), zoo=None):
        outs = []
        for mod in (B1, B2):
            P.B = mod
            outs.append(orig(tr, Xtr, y, te, Xte, seeds=seeds, zoo=P.ZOO if zoo is None else zoo))
        P.B = B1
        res = {n: 0.5 * (outs[0][n] + outs[1][n]) for n in outs[0]}
        res["base"] = outs[1]["base"]
        return res

    return f


def run_predict(mode: str, argv: list):
    """predict.main() with base2 swapped in (mode "base2") or with both bases averaged ("two")."""
    if mode == "base2":
        P.B = B2
    else:
        P.fit_predict_all = two_base_fit_predict(P.fit_predict_all)
    sys.argv = ["predict.py"] + argv
    P.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", choices=["base2", "two"], help="run predict.main() with base2 / both bases; the remaining arguments go to predict.py")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    a, rest = ap.parse_known_args()
    if a.predict:
        run_predict(a.predict, rest)
        sys.exit(0)
    if a.cv:
        cv_base()
    if a.hidden:
        hidden(a.seeds, a.procs)
    if a.hidden or a.eval:
        report()
