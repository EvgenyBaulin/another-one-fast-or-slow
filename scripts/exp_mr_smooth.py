"""Idea 20: reconcile model predictions across the batches of a multi-row file that has no
anchor in train (g1.smooth_batch_curves). Check on train with the out-of-fold stack
(data/preds/main_STACK_oof.csv, folds by fp, G1 not included): RMSLE on the rows of multi-row
files before / after, per file and overall.   Usage: python scripts/exp_mr_smooth.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import g1  # noqa: E402

rm = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))  # noqa: E731

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    df = C.load("data", measure=[])
    tr = df[df.split == "train"].reset_index(drop=True)
    oof = pd.read_csv("data/preds/main_STACK_oof.csv").set_index("id").pred
    tr["p"] = oof.reindex(tr.id).values
    m = tr[(tr.groupby(["file", "shape"]).file.transform("size") >= 4) & tr.p.notna()].reset_index(drop=True)
    e0 = m.p.values - m.y.values
    lvl = m.assign(e=e0).groupby("file").e.transform("mean").values
    print(f"rows {len(m)}, files {m.file.nunique()}; before: rmsle {rm(e0):.4f} = file level {rm(lvl):.4f} + within file {rm(e0 - lvl):.4f}")
    # the best any within-file smoothing could do: remove all within-file error, keep the level
    rows = []
    variants = [(f, mx) for f in ("affine", "loglog") for mx in (1.0, 0.5)]
    for form, mx in variants:
        p = g1.smooth_batch_curves(m, m.p.values, None, min_rows=4, form=form, mix=mx)
        e = p - m.y.values
        d = m.assign(e0=e0, e=e)
        f = d.groupby("file").apply(lambda g: pd.Series({"sse0": (g.e0**2).sum(), "sse": (g.e**2).sum(), "within0": rm(g.e0 - g.e0.mean()), "within": rm(g.e - g.e.mean())}))
        dl = f.sse - f.sse0
        top = dl.sort_values()
        rows.append({"form": form, "mix": mx, "rmsle": rm(e), "delta_sse": float(dl.sum()), "files better": int((dl < -1e-4).sum()), "files worse": int((dl > 1e-4).sum()),
                     "within before": rm(e0 - lvl), "within after": rm(e - d.groupby("file").e.transform("mean").values),
                     "best 2 files": float(top.head(2).sum()), "worst 2 files": float(top.tail(2).sum()), "median file delta": float(dl.median())})  # fmt: skip
        if form == "affine" and mx == 1.0:
            keep = f.assign(delta=dl)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("per file, affine mix 1.0 (sorted by delta sse):")
    print(keep.sort_values("delta").round(4).to_string())
