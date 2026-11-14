"""Shared helpers: data loading, feature matrix, CV folds, metric.

Conventions used by every model script:
- target space is log1p(latency); RMSLE == RMSE in that space;
- `data/features.csv` comes from scripts/features.py (train + test rows);
- optional `data/measure_cpu.csv` / `data/measure_cuda.csv` come from scripts/measure.py;
- CV folds are grouped by structure fingerprint `fp`, so twins of a graph never
  sit on both sides of a split.
"""

import os

import numpy as np
import pandas as pd

ID_COLS = ["id", "split", "file", "batch", "shape", "latency", "fp"]
LOG_PREFIXES = (
    "macs", "out_elems", "in_elems", "elems_", "params", "file_bytes",
    "max_tensor_elems", "input_elems", "output_elems", "ms_",
)  # fmt: skip


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load(data_dir: str = "data", measure=("cpu", "cuda")) -> pd.DataFrame:
    """features.csv joined with measurement files when they exist."""
    df = pd.read_csv(os.path.join(data_dir, "features.csv"))
    f2 = os.path.join(data_dir, "feat2.csv")
    if os.path.exists(f2):  # extra graph features (scripts/feat2.py): RNN work, io consumers, ...
        e = pd.read_csv(f2).drop_duplicates("id")
        df = df.merge(e[["id"] + [c for c in e.columns if c.startswith("f2_")]], on="id", how="left")
        miss = int(((df.has_graph == 1) & df.f2_in_fan.isna()).sum()) if "f2_in_fan" in df.columns else -1
        if miss:
            print(f"  !!! feat2.csv lacks {miss} graph rows: re-run scripts/feat2.py --data {data_dir}", flush=True)
    for prov in measure:
        p = os.path.join(data_dir, f"measure_{prov}.csv")
        if not os.path.exists(p):
            continue
        m = pd.read_csv(p)
        ok = m.status.astype(str).str.startswith("ok")
        m = m[ok].copy()
        # scale to the requested batch when a smaller/native batch was measured
        k = m.batch.astype(float) / m.measured_batch.astype(float)
        out = pd.DataFrame({"id": m.id})
        out[f"ms_{prov}"] = m.ms_median * k
        out[f"ms_{prov}_min"] = m.ms_min * k
        out[f"ms_{prov}_first"] = m.ms_first
        out[f"create_{prov}"] = m.create_s
        out[f"scaled_{prov}"] = (k != 1).astype(int)
        df = df.merge(out.drop_duplicates("id"), on="id", how="left")
    df["y"] = np.log1p(df.latency)
    df["batch"] = df.batch.astype(float)
    return df.copy()


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric model features; size-like columns go to log1p."""
    X = df.drop(columns=[c for c in ID_COLS + ["y"] if c in df.columns]).copy()
    X = X.select_dtypes(include=[np.number, bool]).astype(float)
    for c in X.columns:
        if c.startswith(LOG_PREFIXES):
            X[c] = np.log1p(X[c].clip(lower=0))
    # ratios the trees cannot build themselves (repeated CV: -0.010 on single-row files)
    z = lambda c: df[c].fillna(0).astype(float) if c in df.columns else pd.Series(0.0, index=df.index)  # noqa: E731
    io = z("input_elems") + z("output_elems")
    X["d_io"] = np.log1p(io)
    X["d_unf_share"] = (z("elems_ew_unfused") + z("elems_norm_unfused")) / (z("out_elems") + 1)
    X["d_ew_share"] = z("elems_elementwise") / (z("out_elems") + 1)
    X["d_convlike"] = (z("macs") > 0).astype(float)
    X["d_macs_per_io"] = np.log1p(z("macs") / (io + 1))
    X["d_out_per_io"] = np.log1p(z("out_elems") / (io + 1))
    X["d_io_per_batch"] = np.log1p(io / df.batch.astype(float))
    X["d_traffic_per_io"] = np.log1p(z("elems_traffic") / (io + 1))
    X.loc[df.has_graph.values != 1, [c for c in X.columns if c.startswith("d_")]] = np.nan
    return X


def make_folds(train: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per train row, grouped by fp (rows without fp get their own group)."""
    grp = train.fp.fillna(train.file).astype(str).values
    uniq = np.unique(grp)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in grp])


def blind_prior(train: pd.DataFrame) -> dict:
    """log1p-mean latency by batch over single-row train files (fallback for rows
    whose ONNX file is missing), shrunk towards the overall single-row mean."""
    single = train[train.groupby("file").file.transform("size") == 1]
    gm = float(single.y.mean())
    st = single.groupby("batch").y.agg(["mean", "size"])
    prior = {float(b): float((r["mean"] * r["size"] + gm * 5) / (r["size"] + 5)) for b, r in st.iterrows()}
    prior[None] = gm
    return prior


def write_submission(test: pd.DataFrame, pred_log1p: np.ndarray, path: str, sample_path: str) -> None:
    """Submission in sample order; values are latencies (>= 0)."""
    sub = pd.read_csv(sample_path)[["id"]]
    p = np.asarray(pred_log1p, float)
    assert np.isfinite(p).all(), f"{int((~np.isfinite(p)).sum())} non-finite predictions"
    assert test.id.is_unique and sub.id.is_unique, "duplicate ids"
    s = pd.Series(np.expm1(p).clip(min=0), index=test.id.values)
    sub["latency"] = sub.id.map(s)
    assert sub.latency.notna().all() and len(sub) == len(test), "submission has gaps"
    assert np.isfinite(sub.latency).all(), "non-finite latency"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sub.to_csv(path, index=False)
