"""Fast analysis of the multi-row simulation (scripts/exp_mr.py must have run): recompute
G1 on the same hold-outs with config variants and compare with the saved model stack.
No model fitting here, a run takes seconds.  Usage: python scripts/exp_mr_g1.py [mode]
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import g1  # noqa: E402


def rm(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def load():
    df = C.load("data", measure=[])
    tr = df[df.split == "train"].reset_index(drop=True)
    sim = pd.read_csv("data/mr_sim.csv")
    kn = pd.read_csv("data/mr_sim_known.csv")
    return tr, sim, kn


def splits(tr, sim, kn):
    """Yields (rep, k, known, query, stk_query, stk_in_known) with frames in train order."""
    for (rep, k), s in sim.groupby(["rep", "k"]):
        h = tr.id.isin(set(s.id)).values
        q, known = tr[h], tr[~h]
        sq = s.set_index("id").stk.reindex(q.id).values
        sk = kn[(kn.rep == rep) & (kn.k == k)].set_index("id").stk_in.reindex(known.id).values
        yield rep, k, known, q, sq, sk


def run_cfg(tr, sim, kn, cfg=None, use_model=False, **kw):
    out = []
    for rep, k, known, q, sq, sk in splits(tr, sim, kn):
        extra = {"model_known": sk, "model_query": sq} if use_model else {}
        r = g1.g1_predict(known, q, data_dir="data", cfg={**g1.CFG, **(cfg or {})}, **extra, **kw)
        out.append(pd.DataFrame({"rep": rep, "k": k, "id": q.id.values, "file": q.file.values, "batch": q.batch.values, "shape": q["shape"].values,
                                 "y": q.y.values, "stk": sq, "g1": r.g1_pred.values, "detail": (r.g1_detail + "/" + r.g1_src.fillna("")).values, "conf": r.g1_conf.values, "spread": r.g1_spread.values}))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def report(d, name, ws=(0.5, 0.7, 0.85)):
    g1p = d.g1.fillna(d.stk)
    print(f"--- {name}: g1 alone {rm(g1p, d.y):.4f}  sse/rep {((g1p - d.y) ** 2).sum() / d.rep.nunique():.3f}")
    rows = []
    for kd, g in d.groupby("detail"):
        gp = g.g1.fillna(g.stk)
        row = {"detail": kd, "n": len(g), "model": rm(g.stk, g.y), "g1": rm(gp, g.y), "sse/rep": ((gp - g.y) ** 2).sum() / d.rep.nunique()}
        for w in ws:
            row[f"w={w}"] = rm(w * gp + (1 - w) * g.stk, g.y)
        row["g1 by rep"] = " ".join(f"{rm(gp[g.rep == r], g.y[g.rep == r]):.3f}" for r in sorted(d.rep.unique()))
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    tr, sim, kn = load()
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "base":
        report(run_cfg(tr, sim, kn), "current g1")
    elif mode == "rc":
        # quick check without touching g1: model + in-sample residual of the same file
        d = run_cfg(tr, sim, kn)
        rc = np.full(len(d), np.nan)
        pos = 0
        for rep, k, known, q, sq, sk in splits(tr, sim, kn):
            kf = known.assign(res=known.y.values - sk)
            for i, (f, b, s) in enumerate(zip(q.file.values, q.batch.values, q["shape"].values)):
                a = kf[(kf.file == f) & (kf["shape"] == s)]
                if len(a):
                    w = np.exp(-np.abs(np.log2(a.batch.values / b)) / 1.0)
                    rc[pos + i] = sq[i] + float(np.sum(w * a.res.values) / np.sum(w))
            pos += len(q)
        d["rc"] = rc
        rows = []
        for kd, g in d.groupby("detail"):
            ok = g.rc.notna() & g.g1.notna()
            g = g[ok]
            row = {"detail": kd, "n": len(g), "model": rm(g.stk, g.y), "g1": rm(g.g1, g.y), "rc": rm(g.rc, g.y)}
            for w in (0.3, 0.5, 0.7):
                row[f"g1*{w}+rc"] = rm(w * g.g1 + (1 - w) * g.rc, g.y)
            row["corr(e_g1,e_rc)"] = float(np.corrcoef(g.g1 - g.y, g.rc - g.y)[0, 1]) if len(g) > 2 and (g.g1 - g.y).std() > 0 else np.nan
            rows.append(row)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    elif mode == "curves":
        # leave-one-row-out (one missing batch per file, like the test), twin exact hits removed
        recs = g1.file_records(tr, g1._load_ident("data"))
        variants = [("old (down_max 1.3, prior)", {"down_max": 1.3, "one_kernel": False}), ("down_max 1.0", {"one_kernel": False}), ("down_max 1.1", {"down_max": 1.1, "one_kernel": False}),
                    ("down affine", {"down": "affine", "one_kernel": False}), ("down loglog cap1", {"down": "loglog", "one_kernel": False}),
                    ("kernel tau .7", {}), ("kernel tau .4", {"one_tau": 0.4}), ("kernel tau 1.2", {"one_tau": 1.2})]  # fmt: skip
        for name, over in variants:
            cfg = {**g1.CFG, **over}
            d = g1.loo(tr, recs, cfg, drop_identity_batch=True, only_multi=True)
            d = d[d.kind.isin(["interp", "extrap_up", "extrap_down"])]
            b1 = d[(d.kind == "extrap_down") & (d.batch == 1) & (d.n >= 3)]
            a = g1.loo(tr, recs, cfg)
            e1 = a[a.kind == "extrap1"]
            ex = a[(a.kind == "exact") & (a.nfile == 1)]
            # one-point setting on multi-row curves: keep one batch of the file, ask the next one
            print(f"{name:26s} down {g1._rmse(d.err[d.kind == 'extrap_down']):.4f} (b=1: n={len(b1)} {g1._rmse(b1.err):.4f} bias {b1.err.mean():+.3f}) up {g1._rmse(d.err[d.kind == 'extrap_up']):.4f} interp {g1._rmse(d.err[d.kind == 'interp']):.4f} | single-row: extrap1 n={len(e1)} {g1._rmse(e1.err):.4f}, exact dup n={len(ex)} {g1._rmse(ex.err):.4f}")
        # one-point simulation on multi-row curves
        rows = g1._rows(tr, True)
        multi = [r for r in rows if sum(1 for q in rows if q[0] == r[0] and q[2] == r[2]) > 1]
        for name, over in [("prior 0.7/0.8", {"one_kernel": False}), ("kernel .7", {}), ("kernel .4", {"one_tau": 0.4}), ("kernel 1.2", {"one_tau": 1.2})]:
            cfg = {**g1.CFG, **over}
            res = []
            for a_ in multi:
                for q in multi:
                    if q[0] == a_[0] and q[2] == a_[2] and q[1] in (a_[1] * 2, a_[1] / 2):
                        # known: everything except the rows of this latency curve (file and its twins), plus the anchor
                        lat = {(x[1], x[2], x[3]) for x in rows if x[0] == a_[0]}
                        kn = [x for x in rows if (x[1], x[2], x[3]) not in lat] + [a_]
                        o = g1._core(recs, kn, [q[:3]], cfg)[0]
                        res.append((q[1] > a_[1], q[1], q[3], o[0] - math.log1p(q[3]), o[1]))
            r = pd.DataFrame(res, columns=["up", "b", "lat", "err", "kind"])
            r = r[r.kind == "extrap1"]
            big = r.lat >= 50
            print(f"one-point sim {name:14s} n={len(r)} up {g1._rmse(r.err[r.up]):.4f} down {g1._rmse(r.err[~r.up]):.4f} | lat>=50ms: up {g1._rmse(r.err[r.up & big]):.4f} (n={int((r.up & big).sum())}) bias {r.err[r.up & big].mean():+.3f}; 16->32: {g1._rmse(r.err[r.up & (r.b == 32)]):.4f}")
    elif mode == "blend":
        def blend(d, sigma, conf=None):
            c = d.conf.values if conf is None else conf
            w = np.where(d.g1.notna(), sigma**2 / (sigma**2 + c**2), 0.0)
            return w * d.g1.fillna(0).values + (1 - w) * d.stk.values

        d = run_cfg(tr, sim, kn)
        old = {"interp": 0.08, "extrap_up": 0.10, "extrap_down": 0.18, "extrap1": 0.22, "shape/def": 0.20, "shape/ovr": 0.09}
        conf_old = np.array([0.005 if k.startswith("exact") else old.get(k.split("/")[0] if not k.startswith("shape") else k, 0.2) for k in d.detail])
        pols = {"model": d.stk.values, "g1 alone": d.g1.fillna(d.stk).values, "old conf s=.25": blend(d, 0.25, conf_old)}
        for sg in (0.25, 0.35, 0.45, 0.6):
            pols[f"new conf s={sg}"] = blend(d, sg)
        nr = d.rep.nunique()
        kinds = sorted(d.detail.unique())
        rows = []
        for name, p in pols.items():
            e2 = (p - d.y.values) ** 2
            row = {"policy": name, "rmse": float(np.sqrt(e2.mean())), "sse/rep": float(e2.sum() / nr), "by rep": " ".join(f"{np.sqrt(e2[(d.rep == r).values].mean()):.4f}" for r in sorted(d.rep.unique()))}
            for kd in kinds:
                row[kd.replace("extrap_", "ex_").replace("/same", "")] = float(np.sqrt(e2[(d.detail == kd).values].mean()))
            rows.append(row)
        print("n by kind:", d.groupby("detail").size().to_dict())
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    elif mode == "one":
        # one known batch -> neighbouring batch, on multi-row curves, error by the smaller batch
        recs = g1.file_records(tr, g1._load_ident("data"))
        rows = g1._rows(tr, True)
        cnt = {}
        for x in rows:
            cnt[(x[0], x[2])] = cnt.get((x[0], x[2]), 0) + 1
        multi = [r for r in rows if cnt[(r[0], r[2])] > 1]
        res = []
        for a_ in multi:
            lat = {(x[1], x[2], x[3]) for x in rows if x[0] == a_[0]}
            base = [x for x in rows if (x[1], x[2], x[3]) not in lat]
            for q in multi:
                if q[0] == a_[0] and q[2] == a_[2] and q[1] in (a_[1] * 2, a_[1] / 2):
                    o = g1._core(recs, base + [a_], [q[:3]], g1.CFG)[0]
                    res.append((q[1] > a_[1], min(q[1], a_[1]), o[0] - math.log1p(q[3]), o[1]))
        r = pd.DataFrame(res, columns=["up", "bmin", "err", "kind"])
        r = r[r.kind == "extrap1"]
        print(r.groupby(["up", "bmin"]).err.agg(n="size", rmse=g1._rmse, bias="mean").round(4).to_string())
    elif mode == "down":
        recs = g1.file_records(tr, g1._load_ident("data"))
        for name, over in [("default", {}), ("slope 1.05", {"reg_slope": 1.05}), ("slope 1.2", {"reg_slope": 1.2}), ("cost .8-1.5", {"reg_cost": (0.8, 1.5)}), ("cost .9-1.25", {"reg_cost": (0.9, 1.25)}), ("cost .7-2", {"reg_cost": (0.7, 2.0)})]:
            d = g1.loo(tr, recs, {**g1.CFG, **over}, drop_identity_batch=True, only_multi=True)
            g = d[d.kind == "extrap_down"]
            print(f"{name:14s} " + "  ".join(f"{'regular' if sp < 0.5 else 'irregular'}: n={len(x)} rmse {g1._rmse(x.err):.4f} med {x.err.abs().median():.3f} max {x.err.abs().max():.3f}" for sp, x in g.groupby("spread")))
        d = run_cfg(tr, sim, kn)
        g = d[d.detail.str.startswith("extrap_down")]
        for sp, x in g.groupby("spread"):
            print(f"sim {'regular' if sp < 0.5 else 'irregular'}: n={len(x)} model {rm(x.stk, x.y):.4f} g1 {rm(x.g1, x.y):.4f} " + " ".join(f"w={w}: {rm(w * x.g1 + (1 - w) * x.stk, x.y):.4f}" for w in (0.5, 0.7, 0.85, 0.95)) + f" | bias g1 {np.mean(x.g1 - x.y):+.3f} model {np.mean(x.stk - x.y):+.3f}")
    elif mode == "interp":
        recs = g1.file_records(tr, g1._load_ident("data"))
        for name, over in [("loglog", {"interp": "loglog"}), ("lin", {"interp": "lin"}), ("mix", {"interp": "mix"}), ("quad", {"interp": "quad"}), ("mixq .5", {"interp": "mixq"}), ("mixq .3", {"interp": "mixq", "quad_w": 0.3}), ("mixq .7", {"interp": "mixq", "quad_w": 0.7})]:
            d = g1.loo(tr, recs, {**g1.CFG, **over}, drop_identity_batch=True, only_multi=True)
            g = d[d.kind == "interp"]
            print(f"{name:8s} n={len(g)} rmse {g1._rmse(g.err):.4f} bias {g.err.mean():+.4f} med|e| {g.err.abs().median():.4f} | " + " ".join(f"b{int(b)}: {g1._rmse(x.err):.3f}/{x.err.mean():+.3f}" for b, x in g.groupby("batch") if len(x) > 5) + f" | spread<.03: {g1._rmse(g.err[g.spread < 0.03]):.4f} spread>=.03: {g1._rmse(g.err[g.spread >= 0.03]):.4f}")
    elif mode == "calib":
        recs = g1.file_records(tr, g1._load_ident("data"))
        d = g1.loo(tr, recs, g1.CFG, drop_identity_batch=True, only_multi=True)
        for kd in ("interp", "extrap_up", "extrap_down"):
            g = d[d.kind == kd]
            print(f"== {kd} n={len(g)} rmse {g1._rmse(g.err):.4f}; corr(|err|, spread) {np.corrcoef(g.err.abs(), g.spread)[0, 1]:.2f}")
            for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.12), (0.12, 9)):
                gg = g[(g.spread >= lo) & (g.spread < hi)]
                if len(gg):
                    print(f"   spread [{lo}, {hi}): n={len(gg):3d} rmse {g1._rmse(gg.err):.4f} med|e| {gg.err.abs().median():.4f} max {gg.err.abs().max():.3f}")
    elif mode == "rows":
        d = run_cfg(tr, sim, kn)
        d["err"] = d.g1 - d.y
        d["merr"] = d.stk - d.y
        print(d[d.err.abs() > 0.15].sort_values(["detail", "file", "batch"]).round(3).drop(columns=["id", "conf"]).to_string(index=False))
