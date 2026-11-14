"""Private rows vs train graphs with the same op multiset, same batch and same io size (server)."""
import os, sys, collections
from multiprocessing import Pool
import numpy as np, pandas as pd, onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

def sig(f):
    g = onnx.load(os.path.join("data/data", f), load_external_data=False).graph
    c = collections.Counter(n.op_type for n in g.node if n.op_type != "Constant")
    return f, "|".join(f"{k}:{v}" for k, v in sorted(c.items()))

if __name__ == "__main__":
    df = C.load("data", measure=[])
    old = pd.read_csv("data/archive/features_static_1737.csv", usecols=["id", "has_graph", "split"])
    priv = set(old[(old.split == "test") & (old.has_graph == 0)].id)
    with Pool(16) as p: S = dict(p.imap_unordered(sig, sorted(df.file.unique()), chunksize=16))
    df["sig"] = df.file.map(S); df["io"] = df.input_elems + df.output_elems
    tr = df[df.split == "train"]; te = df[df.id.isin(priv)].copy()
    sub = pd.read_csv("submissions/final_slow.csv").set_index("id").latency
    g1 = pd.read_csv("data/g1_test.csv").set_index("id")
    te["pred"] = te.id.map(sub); te["g1"] = te.id.map(g1.g1_kind)
    rows = []
    for r in te.itertuples():
        a = tr[(tr.sig == r.sig) & (tr.batch == r.batch) & ((tr.io - r.io).abs() <= 0.02 * max(r.io, 1))]
        if len(a):
            ll = np.log1p(a.latency.values)
            rows.append(dict(file=r.file[:12], batch=r.batch, nodes=int(r.n_nodes_nonconst), io_M=r.io / 1e6, g1=r.g1, n_analog=len(a), analog_ms=float(np.expm1(ll.mean())), analog_spread=float(ll.max() - ll.min()), pred_ms=r.pred, d=float(np.log1p(r.pred) - ll.mean())))
    A = pd.DataFrame(rows)
    print("private rows:", len(te), "| with a same-signature, same-batch, same-io train graph:", len(A), "| of them G1 exact:", int((A.g1 == "exact").sum()))
    B = A[A.g1 != "exact"].sort_values("d", key=abs, ascending=False)
    print("not G1-exact:", len(B), "| rms of (pred - analog):", round(float(np.sqrt((B.d**2).mean())), 3), "| median analog spread:", round(float(B.analog_spread.median()), 3))
    print(B.head(40).round(3).to_string(index=False))
