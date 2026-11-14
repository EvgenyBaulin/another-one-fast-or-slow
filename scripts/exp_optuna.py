#!/usr/bin/env python3
"""Optuna search over the shared hyperparameters of the three stacked extra-trees models
(lgbx_res, lgbx_bf, lgbw_res) against the hidden-set simulation (server).

Setup is the one of predict.py: two analytic bases (base.py + base2.py, tree models averaged
over the two), feature matrix with the default --drop-cols, feat2 merged by common.load,
no measurements. For a split seed s, 171 random single-row train files are hidden (same
draw as scripts/exp_hidden.py), everything is fitted on the rest and the hidden rows go
through  stack_fixed -> shrink_to_base -> fam_mix(tiny_rules) -> rules2.apply -> g1_blend(0.25).

Per split seed, computed once (independent of the tuned parameters): the split, the
cross-fitted bases, the four non-tuned models that shrink_to_base looks at (lgb_dir,
lgb_res, lgb_bf, lgbh_res), G1, the tiny-graph rules and the rules2 output (affine in the
incoming prediction, stored as offset + slope). A trial refits only the three tuned models.

Search objective: mean RMSLE over split seeds 0-5, 1 model seed. Split seeds 6-15 are used
only at the end: defaults and the best 3 trials, 2 model seeds each.

  python scripts/exp_optuna.py --minutes 25 --procs 24
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402

N_HIDE = 171
DROP = "tr_max_rank,n_tr_hi,elems_tr_hi,n_tr6,f2_rs_*,f2_max_kernel,f2_peak_per_io"  # predict.py default
SEARCH_SEEDS = tuple(range(0, 6))
HOLD_SEEDS = tuple(range(6, 16))
DEFAULTS = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.0, max_depth=-1, min_split_gain=0.0,
    extra_trees=True, w_down=0.3, w_thr=1.5,
)  # fmt: skip
TUNED = ("lgbx_res", "lgbx_bf", "lgbw_res")
FIXED_ZOO = [z for z in P.ZOO if z[0] in ("lgb_dir", "lgb_res", "lgb_bf", "lgbh_res")]
G = {}  # parent: train frame and feature matrix (inherited by fork)
CACHE = {}  # split seed -> precomputed pieces (inherited by fork)


def _fit(p, Xa, ya, Xb, seed, weighted=False):
    """predict._lgbx / predict._lgbw with the hyperparameters taken from `p`."""
    m = lgb.LGBMRegressor(
        n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"], num_leaves=int(p["num_leaves"]),
        min_child_samples=int(p["min_child_samples"]), subsample=p["subsample"], subsample_freq=1,
        colsample_bytree=p["colsample_bytree"], extra_trees=bool(p["extra_trees"]), reg_lambda=p["reg_lambda"],
        reg_alpha=p["reg_alpha"], max_depth=int(p["max_depth"]), min_split_gain=p["min_split_gain"],
        random_state=seed, verbose=-1, n_jobs=1,
    )  # fmt: skip
    if weighted:
        w = np.where(np.abs(ya - np.median(ya)) > p["w_thr"], p["w_down"], 1.0)
        return m.fit(Xa, ya, sample_weight=w).predict(Xb)
    return m.fit(Xa, ya).predict(Xb)


def _tuned(d, p, mseeds):
    """The three tuned models, averaged over model seeds and then over the bases
    (same order of averaging as predict.fit_predict_all)."""
    acc = {n: [] for n in TUNED}
    for b_tr, b_te in d["bases"]:
        Xa = d["Xk"].copy()
        Xa["base"] = b_tr
        Xb = d["Xq"].copy()
        Xb["base"] = b_te
        res = d["y"] - b_tr
        acc["lgbx_res"].append(np.mean([b_te + _fit(p, Xa, res, Xb, s) for s in mseeds], axis=0))
        acc["lgbx_bf"].append(np.mean([_fit(p, Xa, d["y"], Xb, s) for s in mseeds], axis=0))
        acc["lgbw_res"].append(np.mean([b_te + _fit(p, Xa, res, Xb, s, weighted=True) for s in mseeds], axis=0))
    return {n: np.mean(v, axis=0) for n, v in acc.items()}


def _chain(d, preds):
    st = P.stack_fixed(preds)
    st = P.shrink_to_base(st, preds, d["n_nodes"])
    p = P.fam_mix(st, d["fam"], d["sig1"], d["fam_kind"])
    p = d["r2a"] + d["r2b"] * p  # rules2.apply(known, q, p)
    p = P.g1_blend(p, d["g1"], sigma_model=0.25)
    return C.rmse(p, d["yq"])


def evaluate(seed, p, mseeds):
    d = CACHE[seed]
    preds = dict(d["fixed"])
    preds.update(_tuned(d, p, mseeds))
    preds["base"] = d["bases"][-1][1]
    return _chain(d, preds)


def precompute(seed):
    t0 = time.time()
    import g1
    import rules2

    tr, Xtr = G["tr"], G["Xtr"]
    single_files = tr.groupby("file").file.transform("size") == 1
    files = tr.file[single_files].unique()
    hide = set(np.random.RandomState(1000 + seed).choice(files, N_HIDE, replace=False))
    h = tr.file.isin(hide).values
    known, Xk = tr[~h].reset_index(drop=True), Xtr[~h].reset_index(drop=True)
    q, Xq = tr[h].reset_index(drop=True), Xtr[h].reset_index(drop=True)
    y = known.y.values
    grp = known.fp.fillna(known.file).astype(str).values
    bases = []
    for Bm in P.BASES:
        b_tr, b_te, _ = Bm.crossfit(known, y, grp, q)
        bases.append((np.asarray(b_tr, float), np.asarray(b_te, float)))
    mseeds = (0,) if seed in SEARCH_SEEDS else (0, 1)
    outs = []
    for b_tr, b_te in bases:
        Xa = Xk.copy()
        Xa["base"] = b_tr
        Xb = Xq.copy()
        Xb["base"] = b_te
        out = {}
        for name, learner, kind in FIXED_ZOO:
            f = P.LEARNERS[learner]
            if kind == "dir" and outs:
                out[name] = outs[0][name]
            elif kind == "dir":
                out[name] = np.mean([f(Xk, y, Xq, s) for s in mseeds], axis=0)
            elif kind == "res":
                out[name] = np.mean([b_te + f(Xa, y - b_tr, Xb, s) for s in mseeds], axis=0)
            else:
                out[name] = np.mean([f(Xa, y, Xb, s) for s in mseeds], axis=0)
        outs.append(out)
    fixed = {n: np.mean([o[n] for o in outs], axis=0) for n in outs[0]}
    r = g1.g1_predict(known, q, data_dir="data")
    fam, fam_kind = P.tiny_rules(known, q)
    z = np.zeros(len(q))
    r2a = rules2.apply(known, q, z)
    r2b = rules2.apply(known, q, z + 1.0) - r2a
    t = np.random.RandomState(0).normal(2.0, 1.0, len(q))
    r2_err = float(np.abs(rules2.apply(known, q, t) - (r2a + r2b * t)).max())
    d = dict(
        Xk=Xk, Xq=Xq, y=y, yq=q.y.values, bases=bases, fixed=fixed, n_nodes=q.n_nodes_nonconst.values,
        fam=fam, fam_kind=fam_kind, sig1=q.sig1.values,
        g1=pd.DataFrame({"g1_pred": r.g1_pred.values, "g1_conf": r.g1_conf.values}),
        r2a=r2a, r2b=r2b,
    )  # fmt: skip
    msg = f"  split seed {seed}: known {len(known)} hidden {len(q)} feats {Xk.shape[1]} | rules2 affine err {r2_err:.1e} | rules2 changes {int((np.abs(r2a + r2b * t - t) > 1e-9).sum())} rows | g1 hits {int(r.g1_pred.notna().sum())} | fam {int(np.isfinite(fam).sum())}"
    if seed == 0:  # the hand-written pieces must reproduce predict.fit_predict_all exactly
        ref = P.fit_predict_all(known, Xk, y, q, Xq, seeds=(0,), zoo=[z_ for z_ in P.ZOO if z_[1].startswith("lgb")])
        mine = dict(fixed)
        mine.update(_tuned(d, DEFAULTS, (0,)))
        mine["base"] = bases[-1][1]
        diff = {n: float(np.abs(ref[n] - mine[n]).max()) for n in ref}
        msg += f"\n  SELF-CHECK vs predict.fit_predict_all (max |diff| per model): { {k: f'{v:.1e}' for k, v in diff.items()} }"
    return seed, d, msg + f" | {time.time() - t0:.0f}s"


def suggest(t):
    lr = t.suggest_float("learning_rate", 0.01, 0.1, log=True)
    lrn = t.suggest_float("lr_x_n", 20.0, 40.0)
    return dict(
        n_estimators=int(round(lrn / lr)), learning_rate=lr,
        num_leaves=t.suggest_int("num_leaves", 4, 48, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 2, 30),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.0, 20.0),
        reg_alpha=t.suggest_float("reg_alpha", 0.0, 2.0),
        max_depth=t.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8, 9, 10]),
        min_split_gain=t.suggest_float("min_split_gain", 0.0, 0.05),
        extra_trees=t.suggest_categorical("extra_trees", [True, False]),
        w_down=t.suggest_float("w_down", 0.1, 1.0),
        w_thr=t.suggest_float("w_thr", 1.0, 2.5),
    )  # fmt: skip


def fmt_params(p):
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def main():
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25.0, help="wall time of the search itself")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", default="logs/optuna")
    a = ap.parse_args()
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    P.N_JOBS = 1
    import base2 as B2

    B2.DATA_DIR = "data"
    P.BASES[:] = [P.B, B2]
    df = C.load("data", measure=[])
    X = C.feature_matrix(df)
    pats = [c for c in DROP.split(",") if c]
    X = X.drop(columns=[c for c in X.columns if any(c == q or (q.endswith("*") and c.startswith(q[:-1])) for q in pats)])
    is_tr = df.split.values == "train"
    G["tr"], G["Xtr"] = df[is_tr].reset_index(drop=True), X[is_tr].reset_index(drop=True)
    print(f"train rows {len(G['tr'])}, features {X.shape[1]}, bases {len(P.BASES)}", flush=True)

    ctx = mp.get_context("fork")
    all_seeds = SEARCH_SEEDS + HOLD_SEEDS
    with ctx.Pool(min(a.procs, len(all_seeds))) as pool:
        for seed, d, msg in pool.imap_unordered(precompute, all_seeds):
            CACHE[seed] = d
            print(msg, flush=True)
    print(f"precompute done in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
        seed=0, multivariate=True, constant_liar=True, n_startup_trials=30))  # fmt: skip
    study.enqueue_trial(dict(learning_rate=0.03, lr_x_n=30.0, num_leaves=15, min_child_samples=5, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.0, max_depth=-1, min_split_gain=0.0,
                             extra_trees=True, w_down=0.3, w_thr=1.5))  # fmt: skip
    n_conc = a.procs // len(SEARCH_SEEDS) + 1
    t_search = time.time()
    deadline = t_search + 60 * a.minutes
    flight = {}  # trial number -> (trial, params, {seed: AsyncResult})
    n_done = 0
    with ctx.Pool(a.procs) as pool:  # forked after CACHE is filled; the parent never fits a model
        while True:
            while len(flight) < n_conc and time.time() < deadline:
                t = study.ask()
                p = suggest(t)
                t.set_user_attr("params", p)
                flight[t.number] = (t, p, {s: pool.apply_async(evaluate, (s, p, (0,))) for s in SEARCH_SEEDS})
            if not flight:
                break
            for num in [n for n, (_, _, rs) in flight.items() if all(r.ready() for r in rs.values())]:
                t, p, rs = flight.pop(num)
                try:
                    sc = [rs[s].get() for s in SEARCH_SEEDS]
                    t.set_user_attr("per_seed", sc)
                    study.tell(t, float(np.mean(sc)))
                except Exception as e:
                    print(f"  trial {num} failed: {type(e).__name__}: {e}", flush=True)
                    study.tell(t, state=optuna.trial.TrialState.FAIL)
                n_done += 1
                if num == 0:
                    print(f"  trial 0 = defaults: {study.trials[0].value:.4f}  per-seed {' '.join(f'{v:.3f}' for v in sc)}", flush=True)
                if n_done % 10 == 0:
                    b = study.best_trial
                    print(f"  {n_done} trials, {time.time() - t_search:.0f}s, best {b.value:.4f} (trial {b.number}): {fmt_params(b.user_attrs['params'])}", flush=True)
            time.sleep(0.05)

        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        done.sort(key=lambda t: t.value)
        rows = [dict(trial=t.number, value=t.value, **t.user_attrs["params"], **{f"s{s}": v for s, v in zip(SEARCH_SEEDS, t.user_attrs["per_seed"])}) for t in done]
        pd.DataFrame(rows).to_csv(a.out + "_trials.csv", index=False)
        dflt = next(t for t in done if t.number == 0)
        print(f"SEARCH: {len(done)} trials in {time.time() - t_search:.0f}s; defaults {dflt.value:.4f}; values 10/50/90%: " + " ".join(f"{v:.4f}" for v in np.quantile([t.value for t in done], [0.1, 0.5, 0.9])), flush=True)
        for t in done[:10]:
            print(f"  trial {t.number:4d} {t.value:.4f} (vs defaults {t.value - dflt.value:+.4f}) per-seed {' '.join(f'{v:.3f}' for v in t.user_attrs['per_seed'])} | {fmt_params(t.user_attrs['params'])}", flush=True)
        try:
            imp = optuna.importance.get_param_importances(study)
            print("  importance: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()), flush=True)
        except Exception as e:
            print(f"  importance skipped: {type(e).__name__}: {e}", flush=True)

        # held-out split seeds: defaults and the best 3 trials, 2 model seeds
        cfgs = [("defaults", DEFAULTS)]
        for t in done:
            if t.number != 0 and len(cfgs) < 4 and all(t.user_attrs["params"] != c[1] for c in cfgs):
                cfgs.append((f"trial{t.number}", t.user_attrs["params"]))
        res = {n: {s: pool.apply_async(evaluate, (s, p, (0, 1))) for s in HOLD_SEEDS} for n, p in cfgs}
        hold = {n: np.array([rs[s].get() for s in HOLD_SEEDS]) for n, rs in res.items()}
    print(f"HELD-OUT split seeds {HOLD_SEEDS[0]}-{HOLD_SEEDS[-1]}, 2 model seeds:", flush=True)
    out = {"n_trials": len(done), "search_default": dflt.value, "configs": {}}
    for n, p in cfgs:
        v = hold[n]
        dd = v - hold["defaults"]
        se = float(np.std(dd, ddof=1) / np.sqrt(len(dd)))
        srch = next((t.value for t in done if f"trial{t.number}" == n), dflt.value)
        print(f"  {n:10s} search {srch:.4f} | held-out mean {v.mean():.4f} | per-seed {' '.join(f'{x:.3f}' for x in v)} | diff to defaults {dd.mean():+.4f} +- {se:.4f} | improved {int((dd < 0).sum())}/{len(dd)}", flush=True)
        out["configs"][n] = dict(params=p, search=srch, hold_mean=float(v.mean()), hold=list(map(float, v)), diff=float(dd.mean()), se=se, improved=int((dd < 0).sum()))
    for n, p in cfgs[1:]:
        print(f"  {n}: {fmt_params(p)}", flush=True)
    with open(a.out + "_result.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
