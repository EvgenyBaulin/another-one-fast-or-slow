"""Public multi-row test-only files: what are they, what do we predict, what are their nearest train nets? (server)"""
import os, sys, collections
import numpy as np, pandas as pd, onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 70)
df = C.load("data", measure=["cuda"]); tr = df[df.split == "train"]; te = df[(df.split == "test") & (df.has_graph == 1)].copy()
te = te[~te.file.isin(tr.file)]; te = te[te.groupby("file").file.transform("size") > 1]
sub = pd.read_csv("submissions/v11_slow_zero.csv").set_index("id").latency
g1 = pd.read_csv("data/g1_test.csv").set_index("id")
te["pred"] = te.id.map(sub); te["g1"] = te.id.map(g1.g1_kind); te["io_M"] = (te.input_elems + te.output_elems) / 1e6
te["ns_io"] = 1e6 * te.pred / (te.input_elems + te.output_elems)
for f, g in te.groupby("file"):
    m = onnx.load(os.path.join("data/data", f), load_external_data=False); c = collections.Counter(n.op_type for n in m.graph.node)
    meta = {p.key: p.value[:40] for p in m.metadata_props}
    r = g.iloc[0]
    print(f"== {f} producer={m.producer_name} {m.producer_version} nodes={int(r.n_nodes_nonconst)} params={r.params / 1e6:.1f}M macs/sample={r.macs / r.batch / 1e9:.2f}G shape_known={r.shape_known_frac:.2f} tr_max_rank={r.tr_max_rank} meta={meta.get('description', '')}")
    print("   ops:", " ".join(f"{k}{v}" for k, v in c.most_common(9)))
    print(g.sort_values("batch")[["batch", "shape", "io_M", "pred", "ns_io", "ms_cuda", "g1"]].round(3).to_string(index=False))
    # nearest train multi-row nets by params
    t = tr[tr.groupby("file").file.transform("size") > 1].drop_duplicates("file").copy()
    t["dp"] = (np.log(t.params + 1) - np.log(r.params + 1)).abs()
    near = t.sort_values("dp").head(3)
    for nf in near.file:
        tt = tr[tr.file == nf].sort_values("batch")
        print(f"   near train {nf}: params {tt.params.iloc[0] / 1e6:.1f}M nodes {int(tt.n_nodes_nonconst.iloc[0])} macs/sample {tt.macs.iloc[0] / tt.batch.iloc[0] / 1e9:.2f}G | " + " ".join(f"b{int(b)}:{l:.1f}" for b, l in zip(tt.batch, tt.latency)) + " | ms_cuda " + " ".join(f"{v:.1f}" for v in tt.ms_cuda))
