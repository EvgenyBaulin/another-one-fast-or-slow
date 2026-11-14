"""What do the components say about the unseen MAC-light CNN (test file 25ae2563c695) and
how do they do on comparable train nets? (server)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
df = C.load("data", measure=[])
L = lambda n, k: pd.read_csv(f"data/preds/{n}_{k}.csv").set_index("id").pred
te = df[(df.split == "test") & (df.file == "25ae2563c695.onnx")].sort_values("batch")
cols = {}
for n in ("main_base", "main_lgbx_res", "main_lgbx_bf", "main_lgbw_res", "main_lgb_dir", "main_STACK", "g6_gnn"):
    cols[n.replace("main_", "")] = np.expm1(te.id.map(L(n, "test")).values)
out = pd.DataFrame(cols, index=te.batch.astype(int).values)
out["io_law"] = 0.25 + 1.5e-6 * (te.input_elems + te.output_elems).values
print("test 25ae2563c695 (Conv52 HardSwish19 Relu14 GAP10 HardSigmoid9; 0.06 GMAC/sample), predictions in ms:"); print(out.round(2).to_string())
tr = df[df.split == "train"].copy()
tr["nrow"] = tr.groupby("file").file.transform("size"); tr["mps"] = tr.macs / tr.batch; tr["io"] = tr.input_elems + tr.output_elems
m = tr[(tr.nrow > 1) & (tr.cnt_Conv >= 15) & (tr.input_elems / tr.batch < 3e5)].copy()
for n in ("main_base", "main_STACK", "g6_gnn"):
    m[n.replace("main_", "")] = np.expm1(m.id.map(L(n, "oof")))
m["io_law"] = 0.25 + 1.5e-6 * m.io
m["excess_ms_per_sample"] = (m.latency - 0.25) / m.batch - 1.5e-6 * m.io / m.batch
print("train multi-row CNNs with input <= 300k elements per sample (OOF = whole file held out):")
print(m.sort_values(["mps", "file", "batch"])[["file", "batch", "n_nodes_nonconst", "cnt_Conv", "mps", "latency", "io_law", "excess_ms_per_sample", "base", "STACK", "g6_gnn"]].round(3).to_string())
