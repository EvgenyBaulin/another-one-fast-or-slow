"""Score a submission on the hidden rows of a rehearsal dir (server): exp_score.py <dir> <csv>..."""
import sys, numpy as np, pandas as pd
D = sys.argv[1]; t = pd.read_csv(f"{D}/truth.csv"); lat = [c for c in t.columns if c != "id"][-1]
for f in sys.argv[2:]:
    s = pd.read_csv(f).set_index("id").latency
    e = np.log1p(t.id.map(s)) - np.log1p(t[lat])
    print(f"{f}: hidden rows {len(t)} RMSLE {np.sqrt((e**2).mean()):.4f} | top-5 share {np.sort(e.values**2)[::-1][:5].sum() / (e**2).sum():.2f} | without top-5 {np.sqrt(np.sort(e.values**2)[:-5].mean()):.4f}")
