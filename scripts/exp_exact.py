#!/usr/bin/env python3
"""Compare two features tables (before / after --exact-shapes).

Usage: python scripts/exp_exact.py data/features.csv data/features_exact.csv [--ident]
"""
import sys

import numpy as np
import pandas as pd

a = pd.read_csv(sys.argv[1]).set_index("id")
b = pd.read_csv(sys.argv[2]).set_index("id").loc[a.index]
if "--ident" in sys.argv:
    cols = [c for c in a.columns if c in b.columns]
    diff = [c for c in cols if not (a[c].fillna(-9e9).astype(str) == b[c].fillna(-9e9).astype(str)).all()]
    print("columns only in A:", sorted(set(a.columns) - set(b.columns)))
    print("columns only in B:", sorted(set(b.columns) - set(a.columns)))
    print("shared columns that differ:", diff)
    sys.exit(0)

print("missing columns in B:", sorted(set(a.columns) - set(b.columns)), "| new:", sorted(set(b.columns) - set(a.columns)))
g = a.has_graph == 1
num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]) and c not in ("latency",)]
ch = pd.Series((~np.isclose(a.loc[g, num].fillna(0).to_numpy(float), b.loc[g, num].fillna(0).to_numpy(float), rtol=1e-9, atol=1e-9)).any(axis=1), index=a.index[g])
ch = ch.reindex(a.index, fill_value=False)
print(f"rows changed: {ch.sum()} (train {(ch & (a.split == 'train')).sum()}, test {(ch & (a.split == 'test')).sum()}), graphs changed: {a.file[ch].nunique()}")
print("shape_src rows:", b.loc[g, "shape_src"].value_counts().to_dict(),
      "| graphs:", b.loc[g].groupby("shape_src").file.nunique().to_dict())
bins = [-0.01, 0.3, 0.6, 0.9, 0.999, 1.0]
for name, t in (("before", a), ("after", b)):
    print(f"shape_known_frac {name}:", pd.cut(t.loc[g, "shape_known_frac"], bins).value_counts().sort_index().to_dict(),
          "min %.3f" % t.loc[g, "shape_known_frac"].min())
for name, t in (("before", a), ("after", b)):
    print(f"infer_engine {name}:", t.loc[g, "infer_engine"].value_counts().to_dict(),
          "| macs_unknown_nodes>0 rows:", int((t.loc[g, "macs_unknown_nodes"] > 0).sum()))
inc = g & (a.shape_known_frac < 0.999)
print(f"incomplete before: {inc.sum()} rows / {a.file[inc].nunique()} graphs")
d = pd.DataFrame({
    "file": a.file, "split": a.split, "batch": a.batch, "mb": (a.file_bytes / 1e6).round(0), "eng": a.infer_engine,
    "src": b.shape_src, "ref0": a.ref_batch, "ref1": b.ref_batch,
    "kf0": a.shape_known_frac.round(3), "kf1": b.shape_known_frac.round(3),
    "r_out": (b.out_elems / a.out_elems.replace(0, np.nan)).round(3),
    "r_macs": (b.macs / a.macs.replace(0, np.nan)).round(3),
    "r_in": (b.in_elems / a.in_elems.replace(0, np.nan)).round(3),
    "mu0": a.macs_unknown_nodes, "mu1": b.macs_unknown_nodes,
})[inc]
pd.set_option("display.width", 250, "display.max_rows", 200)
per = d.groupby("file").agg(rows=("src", "size"), mb=("mb", "first"), eng=("eng", "first"), src=("src", "first"),
                            ref0=("ref0", "first"), ref1=("ref1", "first"), kf0=("kf0", "first"), kf1=("kf1", "first"),
                            r_out=("r_out", "first"), r_macs=("r_macs", "first"), r_in=("r_in", "first"),
                            mu0=("mu0", "first"), mu1=("mu1", "first"))
print(per.sort_values(["src", "kf0"]).to_string())
c = d[d.src == 1]
for k in ("r_out", "r_macs", "r_in"):
    q = c[k].dropna()
    print(f"{k} on completed rows: n={len(q)} min {q.min():.3f} q25 {q.quantile(.25):.3f} median {q.median():.3f} q75 {q.quantile(.75):.3f} max {q.max():.3f}")
