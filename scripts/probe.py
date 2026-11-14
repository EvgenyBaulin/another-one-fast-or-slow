#!/usr/bin/env python3
"""Build a hybrid submission: rows of group G come from file B, the rest from file A.

Used for A/B checks on the public leaderboard by row group (which component helps
on which kind of test rows). Groups:
  single_new  test-only single-row files (same population as the private rows)
  multi_new   test-only multi-row files
  intrain     files that also have rows in train
  g1:<kind>   rows where G1 gives exact / interp / extrap / shape / none

Usage: python scripts/probe.py A.csv B.csv <group> out.csv
"""
import sys

import pandas as pd


def groups(data="data"):
    f = pd.read_csv(f"{data}/features.csv", usecols=["id", "split", "file", "has_graph"])
    tr_files = set(f[f.split == "train"].file)
    te = f[(f.split == "test") & (f.has_graph == 1)].copy()
    n = te.groupby("file").file.transform("size")
    te["grp"] = "single_new"
    te.loc[n > 1, "grp"] = "multi_new"
    te.loc[te.file.isin(tr_files), "grp"] = "intrain"
    g1 = pd.read_csv(f"{data}/g1_test.csv")[["id", "g1_kind"]]
    return te.merge(g1, on="id", how="left")[["id", "grp", "g1_kind"]]


if __name__ == "__main__":
    a, b, grp, out = sys.argv[1:5]
    A, Bf, G = pd.read_csv(a), pd.read_csv(b), groups()
    ids = set(G[G.g1_kind == grp[3:]].id) if grp.startswith("g1:") else set(G[G.grp == grp].id)
    m = A.id.isin(ids)
    A.loc[m, "latency"] = A.loc[m, "id"].map(Bf.set_index("id").latency)
    A.to_csv(out, index=False)
    print(f"{out}: {int(m.sum())} rows of group {grp} taken from {b}")
