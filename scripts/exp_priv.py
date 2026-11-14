"""Sanity review of the predictions for the 171 private rows (server)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
df = C.load("data", measure=[])
old = pd.read_csv("data/archive/features_static_1737.csv", usecols=["id", "has_graph", "split"])
priv = set(old[(old.split == "test") & (old.has_graph == 0)].id)
te = df[df.id.isin(priv)].copy()
L = lambda n: pd.read_csv(f"data/preds/{n}_test.csv").set_index("id").pred
sub = pd.read_csv("submissions/final_slow.csv").set_index("id").latency
te["final"] = np.log1p(te.id.map(sub))
for n in ("main_base", "main_lgbx_res", "main_lgbx_bf", "main_lgbw_res", "main_lgb_dir", "main_STACK", "g6_gnn"):
    te[n.replace("main_", "")] = te.id.map(L(n))
g1 = pd.read_csv("data/g1_test.csv").set_index("id") if os.path.exists("data/g1_test.csv") else None
te["io_M"] = (te.input_elems + te.output_elems) / 1e6
te["sd"] = te[["lgbx_res", "lgbx_bf", "lgbw_res", "lgb_dir", "g6_gnn"]].std(axis=1)
te["d_g6"] = te.g6_gnn - te.STACK; te["d_base"] = te.base - te.STACK
print("private rows:", len(te), "| batch:", te.batch.value_counts().sort_index().to_dict())
print("nodes quantiles:", te.n_nodes_nonconst.quantile([.1, .5, .9, 1]).to_dict(), "| io_M quantiles:", te.io_M.quantile([.1, .5, .9, 1]).round(3).to_dict())
print("shape_known<0.999:", int((te.shape_known_frac < 0.999).sum()), "| tr_max_rank>=5:", int((te.tr_max_rank >= 5).sum()), "| rank 6:", int((te.n_tr6 > 0).sum()), "| Resize:", int((te.cnt_Resize > 0).sum()), "| rnn:", int((te.f2_rnn_nodes > 0).sum()), "| io==0:", int((te.io_M == 0).sum()))
print("final prediction quantiles (ms):", np.expm1(te.final).quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict())
cols = ["file", "batch", "n_nodes_nonconst", "sig1", "io_M", "tr_max_rank", "shape_known_frac", "base", "STACK", "g6_gnn", "final", "sd"]
print("\nlargest disagreement between models (sd):"); print(te.sort_values("sd", ascending=False)[cols].head(14).round(3).to_string(index=False))
print("\nG6 far from stack (|d| > 0.5):", int((te.d_g6.abs() > 0.5).sum()), "| base far from stack (|d| > 0.7):", int((te.d_base.abs() > 0.7).sum()))
x = te[(te.tr_max_rank >= 5) | (te.shape_known_frac < 0.999)]
print("\nhigh-rank transpose or incomplete shapes:"); print(x[cols].round(3).to_string(index=False))
te["ns_io"] = np.expm1(te.final) / te.io_M.clip(lower=1e-9) / 1e3 * 1e3
print("\nfastest per element (final, ns per io elem, io > 0.3M):"); print(te[te.io_M > 0.3].sort_values("ns_io")[["file", "batch", "sig1", "n_nodes_nonconst", "io_M", "final", "ns_io"]].head(8).round(3).to_string(index=False))
print("\nslowest per element (io > 0.3M):"); print(te[te.io_M > 0.3].sort_values("ns_io", ascending=False)[["file", "batch", "sig1", "n_nodes_nonconst", "io_M", "final", "ns_io"]].head(8).round(3).to_string(index=False))
