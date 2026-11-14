"""Worst hidden rows of the last rehearsal (server): exp_worst.py [dir] [csv]"""
import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
D = sys.argv[1] if len(sys.argv) > 1 else "data_rehearsal"
t = pd.read_csv(f"{D}/truth.csv"); lat = [c for c in t.columns if c != "id"][-1]
s = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else f"{D}/sub_phaseB.csv").set_index("id").latency
f = pd.read_csv(f"{D}/features.csv").set_index("id")
t["y"] = np.log1p(t[lat]); t["p"] = np.log1p(t.id.map(s)); t["e"] = t.p - t.y
for c in ("file", "batch", "n_nodes_nonconst", "sig1", "macs", "input_elems", "output_elems", "cnt_Transpose", "cnt_Resize", "n_norm_unfused", "n_ew_unfused"):
    t[c] = t.id.map(f[c])
t["io_M"] = (t.input_elems + t.output_elems) / 1e6; t["ns_io"] = t[lat] / t.io_M / 1e3 * 1e3
print(t.reindex(t.e.abs().sort_values(ascending=False).index)[["file", "batch", lat, "y", "p", "e", "n_nodes_nonconst", "sig1", "io_M", "ns_io", "macs", "cnt_Transpose", "cnt_Resize", "n_norm_unfused", "n_ew_unfused"]].head(12).round(3).to_string())
