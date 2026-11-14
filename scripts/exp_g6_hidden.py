#!/usr/bin/env python3
"""Blend weight of G6 on the hidden-set simulation (experiment, server).

  python scripts/exp_g6_hidden.py data/hidden_sim_none_exp.csv /tmp/g6x/hid_a.csv [/tmp/g6x/hid_b.csv ...]

Chain as in scripts/exp_stack2.py: mean of lgbx_res, lgbx_bf, lgbw_res -> shrink_to_base ->
[(1 - w) * stack + w * G6] -> fam_mix -> g1_blend, on the splits for which G6 predictions exist.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402

d = pd.read_csv(sys.argv[1])
g = pd.concat([pd.read_csv(p) for p in sys.argv[2:]], ignore_index=True).drop_duplicates(["seed", "id"])
d = d.merge(g, on=["seed", "id"], how="inner").reset_index(drop=True)
seeds = sorted(d.seed.unique())
print(f"{len(d)} hidden rows, splits {seeds}")
preds = {c[2:]: d[c].values for c in d.columns if c.startswith("m_")}
core = {k: preds[k] for k in ("base", "lgb_dir", "lgb_res", "lgb_bf", "lgbx_res", "lgbx_bf", "lgbh_res") if k in preds}
cols = [c for c in ("lgbx_res", "lgbx_bf", "lgbw_res") if c in preds]
st = P.shrink_to_base(np.mean([preds[c] for c in cols], axis=0), core, d.n_nodes.values)
g1 = pd.DataFrame({"g1_pred": d.g1_pred.values, "g1_conf": d.g1_conf.values})
fin = lambda p: P.g1_blend(P.fam_mix(p, d.fam.values, d.sig1.fillna("").values, d.fam_kind.fillna("").values), g1)  # noqa: E731
per = lambda p: " ".join(f"{C.rmse(p[d.seed.values == s], d.y[d.seed.values == s]):.3f}" for s in seeds)  # noqa: E731
print(f"stack models: {cols}")
print(f"  {'G6 alone (no rules, no G1)':34s} {C.rmse(d.g6, d.y):.4f}  per-split {per(d.g6.values)}")
print(f"  {'stack + shrink (no rules, no G1)':34s} {C.rmse(st, d.y):.4f}  per-split {per(st)}")
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    p = fin((1 - w) * st + w * d.g6.values)
    print(f"  {'full chain, G6 weight %.1f' % w:34s} {C.rmse(p, d.y):.4f}  per-split {per(p)}")
