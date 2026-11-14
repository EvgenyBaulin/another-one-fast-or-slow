#!/usr/bin/env python3
"""In-chain evaluation of G6 out-of-fold files (experiment, server).

Rebuilds the chain of predict.py from the per-model OOF files in <data>/preds/main_*_oof.csv:
  stack_fixed -> shrink_to_base -> [(1 - w) * stack + w * G6] -> tiny-graph rules -> RNN rule
and prints CV RMSLE (all / single-row / multi-row) for every G6 file and weight.

  python scripts/exp_g6_chain.py --data data --g6 old=data/preds/g6_gnn_oof.csv new=/tmp/x_oof.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import predict as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--g6", nargs="+", default=[], help="name=path of G6 OOF files (id,pred)")
ap.add_argument("--weights", default="0.2,0.3,0.4,0.5")
a = ap.parse_args()

df = C.load(a.data, measure=[])
tr = df[df.split == "train"].reset_index(drop=True)
y = tr.y.values
single = (tr.groupby("file").file.transform("size") == 1).values
sc = lambda p: (C.rmse(p, y), C.rmse(p[single], y[single]), C.rmse(p[~single], y[~single]))  # noqa: E731
rd = lambda p: pd.read_csv(p).drop_duplicates("id").set_index("id").pred.reindex(tr.id.values).values  # noqa: E731
oof = {}
for p in sorted(glob.glob(os.path.join(a.data, "preds", "main_*_oof.csv"))):
    n = os.path.basename(p)[len("main_") : -len("_oof.csv")]
    if n != "STACK" and np.isfinite(rd(p)).all():
        oof[n] = rd(p)
st = P.stack_fixed(oof)
print("%-34s all %.4f | single %.4f | multi %.4f" % (("stack_fixed (%s)" % "+".join(n for n in P.STACK_FIXED if n in oof),) + sc(st)))
st = P.shrink_to_base(st, oof, tr.n_nodes_nonconst.values)
print("%-34s all %.4f | single %.4f | multi %.4f" % (("+ shrink to base",) + sc(st)))
fr, fk = P.tiny_rules(tr, tr)


def chain(p):
    p = P.fam_mix(p, fr, tr.sig1.values, fk)
    try:
        import rules2

        p = rules2.apply(tr, tr, p)
    except Exception as e:
        print("RNN rule skipped:", e)
    return p


print("%-34s all %.4f | single %.4f | multi %.4f" % (("+ rules, no G6",) + sc(chain(st))))
for spec in a.g6:
    name, path = spec.split("=", 1)
    g = rd(path)
    if not np.isfinite(g).all():
        print(f"{name}: incomplete, skipped")
        continue
    print("%-34s all %.4f | single %.4f | multi %.4f" % ((f"G6 {name} alone",) + sc(g)))
    for w in [float(v) for v in a.weights.split(",")]:
        print("%-34s all %.4f | single %.4f | multi %.4f   before rules: single %.4f" % ((f"  w={w:.2f} G6 {name}, then rules",) + sc(chain((1 - w) * st + w * g)) + (sc((1 - w) * st + w * g)[1],)))
