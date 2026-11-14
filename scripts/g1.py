#!/usr/bin/env python3
"""G1: predictions from known latencies of the same model (lookup + batch curves).

Identity of a model (a set of files that share one latency curve):
  * same file name;
  * "latency twins" among known files: >= 2 identical (batch, shape, latency) triples
    (static vs dynamic export of one net; the organisers measured it once);
  * "feature duplicates": equal per-sample MACs (rel 1e-6) and
      - same structure (fp, or skey = multiset of weight shapes from scripts/ident.py)
        with per-sample activation elements within OUT_TOL, or
      - a big net (params >= BIG_PARAMS) with params within BIG_TOL.
    fp alone is NOT enough: it ignores Constant values (e.g. Resize target size), so graphs
    with different resolutions collide; equal MACs/out_elems removes those collisions;
  * big nets with params within TWIN_TOL and the same input size (_param_twin);
  * "near twins" of big nets: params within BIG_TOL, MACs within NEAR_MACS, same input
    size (_near_twin; source 'near', lower confidence).

Prediction for (identity, shape, batch), all in log1p(latency):
  exact   - a known row with this batch (mean in log1p);
  interp  - batch inside the known range: mean of the chord (linear and log-log) and of a
            quadratic in log-log through three points;
  extrap  - batch outside the range: linear-space line through the two nearest batches,
            never steeper than proportional below the smallest batch; with one known batch
            the slope comes from other curves of similar latency (_slope1);
  shape   - `images:1xCxHxW` override with no rows of that shape: from another override
            shape of the same identity (pixel exponent), else the default curve at the
            "effective batch" of the shape, learnt from the other identities.
g1_conf is per row: it grows with the irregularity of the curve (g1_spread).

Validation of curves and of the blend with the model: scripts/exp_mr.py (simulation with
the model stack) and scripts/exp_mr_g1.py (fast G1-only checks).

CLI:  python scripts/g1.py --data data [--tune]
"""

import argparse
import bisect
import collections
import math
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"

CFG = {
    "out_tol": 0.05,  # per-sample out_elems tolerance for structure duplicates
    "big_params": 1e6,
    "big_tol": 0.005,
    "near_macs": 1e-3,  # see _near_twin
    "twin_tol": 1e-4,  # big nets: params this close + same per-sample input size = same net (MACs may differ when shape inference is incomplete)
    "interp": "mixq",  # lin | loglog | mix | quad | mixq (mix + quadratic in log-log, half each)
    "up": "affine",  # loglog | slope1 | shrink | clip | affine | mixu
    "down": "mixd",  # loglog | k0.8 | k0.6 | affine | mixd | p<slope>
    "one_up": 0.7,  # log-log slope when only one batch is known, query batch above
    "one_down": 0.8,  # ... query batch below
    "alpha": 1.2,  # fallback exponent: latency ~ pixels^alpha between resolutions
    "down_max": 1.0,  # cap of the log-log slope when extrapolating below the smallest batch
    "one_kernel": True,  # one known batch: slope from other curves of similar latency (_slope1)
    "one_tau": 0.7,  # kernel width in log latency
    "one_min": 5,
}

# expected RMSLE by (kind, source), from the validation printed by the CLI
CONF = {
    ("exact", "same"): 0.005,
    ("exact", "twin"): 0.005,
    ("exact", "dup"): 0.02,
    ("interp", "same"): 0.08,
    ("interp", "twin"): 0.08,
    ("interp", "dup"): 0.08,
    ("extrap_up", "same"): 0.10,
    ("extrap_up", "twin"): 0.10,
    ("extrap_up", "dup"): 0.15,
    ("extrap_down", "same"): 0.20,  # no relation to the spread; heavy tail (median |err| 0.04)
    ("extrap_down", "twin"): 0.20,
    ("extrap_down", "dup"): 0.20,
    ("extrap1", "any"): 0.20,  # refined by the smaller batch of the step in _conf
    ("shape", "ovr"): 0.08,  # from another override shape of the same model (0.06-0.08 in validation)
    # from the default-shape curve: G1 itself is 0.28-0.31 and so is the model; the value is
    # set so that the blend with sigma_model 0.45 is about half and half (best in exp_mr)
    ("shape", "def"): 0.40,
    ("shape", "keff"): 0.09,  # default curve at the effective batch of the shape
}


# kinds whose expected error grows with the spread of the curve (see _curve):
# conf = sqrt(base^2 + (k * spread)^2); calibrated on leave-one-row-out of multi-row files
CONF_SPREAD = {"interp": (0.04, 1.5), "extrap_up": (0.05, 1.5)}
CONF_NEAR = 0.06  # extra error of a near twin (no train pair to measure it; a cautious guess)
# extrap_down: (regular curve, irregular curve), see the end of _curve
CONF_DOWN = (0.07, 0.35)

_EXTRAP = {"extrap1": "extrap", "extrap_up": "extrap", "extrap_down": "extrap"}


def _conf(kind, src, spread):
    if src == "near":  # near twin: not the same measurement, add the twin-to-twin difference
        c = _conf(kind, "dup", spread)
        return math.sqrt(c**2 + CONF_NEAR**2) if c == c else c
    if kind == "extrap1" and spread == spread:
        # spread slot = the smaller batch of the step: one-point steps are reliable only
        # between large batches (exp_mr_g1.py one: 0.20-0.23 below batch 8, 0.09 at 8, 0.06 at 16)
        return 0.20 if spread < 8 else 0.09 if spread < 16 else 0.07
    if kind == "extrap_down" and spread == spread:
        return CONF_DOWN[0] if spread < 0.5 else CONF_DOWN[1]
    if kind in CONF_SPREAD and spread == spread:
        base, k = CONF_SPREAD[kind]
        return math.sqrt(base**2 + (k * spread) ** 2)
    return CONF.get((kind, "any" if kind == "extrap1" else src), float("nan"))


def _rel(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _load_ident(data_dir):
    p = os.path.join(data_dir or DATA_DIR, "ident.csv")
    if os.path.exists(p):
        return pd.read_csv(p, keep_default_na=False)
    return None


def file_records(df: pd.DataFrame, ident: pd.DataFrame | None = None) -> dict:
    """file -> dict(fp, skey, params, macs, out): features of the default-shape row per sample."""
    skey = {}
    if "skey" in df.columns:
        skey = dict(zip(df.file, df.skey.fillna("")))
    elif ident is not None:
        skey = dict(zip(ident.file, ident.skey))
    recs = {}
    d = df[(df.has_graph == 1) & (df["shape"] == "default")].drop_duplicates("file")
    inp = d.input_elems if "input_elems" in d.columns else pd.Series(0.0, index=d.index)
    for f, fp, par, macs, out, b, ie in zip(d.file, d.fp, d.params, d.macs, d.out_elems, d.batch, inp):
        if not (macs == macs and out == out and b > 0):
            continue
        recs[f] = {
            "inp": float(ie) / float(b) if ie == ie else 0.0,
            "file": f,
            "fp": fp if isinstance(fp, str) else "",
            "skey": skey.get(f, "") or "",
            "params": float(par) if par == par else 0.0,
            "macs": float(macs) / float(b),
            "out": float(out) / float(b),
        }
    return recs


def _bucket(r):
    if r["macs"] > 0:
        return "%.5e" % r["macs"]
    return "z|%s|%.5e" % (r["fp"], r["out"])


def _feat_link(a, b, cfg):
    if a["macs"] <= 0 or b["macs"] <= 0:
        return a["macs"] == b["macs"] and a["fp"] == b["fp"] and _rel(a["out"], b["out"]) < 1e-6
    if _rel(a["macs"], b["macs"]) > 1e-6:
        return False
    same = (a["fp"] and a["fp"] == b["fp"]) or (a["skey"] and a["skey"] == b["skey"])
    if same and _rel(a["out"], b["out"]) <= cfg["out_tol"]:
        return True
    if min(a["params"], b["params"]) >= cfg["big_params"] and _rel(a["params"], b["params"]) <= cfg["big_tol"]:
        return True
    return False


def _param_twin(a, b, cfg):
    """Big nets with (almost) the same parameter count and the same per-sample input size."""
    return (
        min(a["params"], b["params"]) >= cfg["big_params"]
        and _rel(a["params"], b["params"]) <= cfg.get("twin_tol", 0)
        and a.get("inp", 0) > 0
        and _rel(a["inp"], b.get("inp", 0)) < 1e-6
    )


def _near_twin(a, b, cfg):
    """Big nets that differ by a few small tensors (e.g. the same transformer with and without
    a pooler / classifier head): params within big_tol, per-sample MACs within near_macs,
    same per-sample input size. Not the same measurement, so the match is marked 'near'.
    On train the closest non-twins are far outside: params 1.2% / MACs 0.9% and 2.1% / 0.55%."""
    return (
        min(a["params"], b["params"]) >= cfg["big_params"]
        and _rel(a["params"], b["params"]) <= cfg["big_tol"]
        and min(a["macs"], b["macs"]) > 0
        and _rel(a["macs"], b["macs"]) <= cfg.get("near_macs", 0)
        and a.get("inp", 0) > 0
        and _rel(a["inp"], b.get("inp", 0)) < 1e-6
    )


def _pixels(shape: str):
    """`images:1x3x800x800` -> 800*800*3 (per sample); None for default / unparsable."""
    try:
        dims = [int(x) for x in shape.split(":")[1].split("x")]
        return float(np.prod(dims[1:]))
    except Exception:
        return None


def curve_predict(pts: dict, b: float, cfg: dict = CFG):
    """pts: {batch: log(latency)}. Returns (log latency, kind); kind is exact | interp |
    extrap_up | extrap_down | extrap1 (only one known batch, slope prior)."""
    return _curve(pts, b, cfg)[:2]


def _curve(pts: dict, b: float, cfg: dict = CFG):
    """Like curve_predict, plus the spread: |difference| in log latency between the two
    estimates behind the prediction (linear-space vs log-log). A regular curve (latency
    proportional to batch) has spread ~ 0, an irregular one a large spread."""
    bs = sorted(pts)
    if b in pts:
        return pts[b], "exact", 0.0
    if len(bs) == 1:
        b0 = bs[0]
        s = cfg["one_up"] if b > b0 else cfg["one_down"]
        return pts[b0] + s * math.log(b / b0), "extrap1", float("nan")
    if bs[0] < b < bs[-1]:
        i = bisect.bisect_left(bs, b)
        b0, b1 = bs[i - 1], bs[i]
        l0, l1 = pts[b0], pts[b1]
        ll = l0 + math.log(b / b0) / math.log(b1 / b0) * (l1 - l0)
        lin = math.log(math.exp(l0) + (b - b0) / (b1 - b0) * (math.exp(l1) - math.exp(l0)))
        m = cfg["interp"]
        if m in ("quad", "mixq"):
            # quadratic in log-log through the bracket and one more point on either side
            # (mean of the two when both exist): keeps the curvature that the chord loses
            qs = []
            for j in (i - 2, i + 1):
                if 0 <= j < len(bs):
                    xs = [math.log(bs[i - 1]), math.log(bs[i]), math.log(bs[j])]
                    ys = [l0, l1, pts[bs[j]]]
                    x = math.log(b)
                    qs.append(sum(ys[a] * np.prod([(x - xs[c]) / (xs[a] - xs[c]) for c in range(3) if c != a]) for a in range(3)))
            q = float(np.mean(qs)) if qs else ll
            lo_, hi_ = min(l0, l1), max(l0, l1)
            q = min(max(q, lo_), hi_)  # stay inside the bracket
            v = q if m == "quad" else cfg.get("quad_w", 0.5) * q + (1 - cfg.get("quad_w", 0.5)) * 0.5 * (ll + lin)
            return v, "interp", abs(ll - lin)
        return {"loglog": ll, "lin": lin, "mix": 0.5 * (ll + lin)}[m], "interp", abs(ll - lin)
    if b > bs[-1]:
        b0, b1 = bs[-2], bs[-1]
        l0, l1 = pts[b0], pts[b1]
        local = (l1 - l0) / math.log(b1 / b0)
        e0, e1 = math.exp(l0), math.exp(l1)
        ll = l1 + local * math.log(b / b1)
        aff = math.log(e1 + (e1 - e0) / (b1 - b0) * (b - b1)) if e1 > e0 else l1 + math.log(b / b1)
        m = cfg["up"]
        if m == "affine":
            v = aff
        elif m == "mixu":
            v = 0.5 * (aff + ll)
        else:
            s = {"loglog": local, "slope1": 1.0, "shrink": 0.5 * local + 0.5, "clip": min(max(local, 0.9), 1.15)}[m]
            v = l1 + s * math.log(b / b1)
        return v, "extrap_up", abs(aff - ll)
    b0, b1 = bs[0], bs[1]
    l0, l1 = pts[b0], pts[b1]
    local = (l1 - l0) / math.log(b1 / b0)
    cap = cfg.get("down_max", 1.3)
    e0, e1 = math.exp(l0), math.exp(l1)
    aff = math.log(max(e0 - (e1 - e0) / (b1 - b0) * (b0 - b), e0 * b / b0))  # never steeper than proportional
    ll = l0 - min(max(local, 0.0), cap) * math.log(b0 / b)
    m = cfg["down"]
    if m == "affine":
        v = aff
    elif m == "mixd":
        v = 0.5 * (aff + ll)
    else:
        s = float(m[1:]) if m.startswith("p") else {"loglog": local, "k0.8": 0.8 * local, "k0.6": 0.6 * local}[m]
        v = l0 - min(max(s, 0.0), cap) * math.log(b0 / b)
    # regular curve: the two anchors are neighbouring batches, the start is not superlinear and
    # the per-sample cost at the anchor is close to the one at the largest batches. Only then
    # is the step below the smallest batch reliable (spread slot: 0 = regular, 1 = irregular).
    regular = len(bs) >= 3 and b1 / b0 <= 2.0001 and local <= cfg.get("reg_slope", 1.1)
    if regular:
        c_inf = float(np.median([math.exp(pts[k]) / k for k in bs[-2:]]))
        lo, hi = cfg.get("reg_cost", (0.85, 1.35))
        regular = lo <= (e0 / b0) / c_inf <= hi
    return v, "extrap_down", 0.0 if regular else 1.0


def _slope1(b0: float, l0: float, b: float, lib: list, cfg: dict):
    """Log-log slope from batch b0 to b when only one batch of the model is known: kernel
    average over the other curves that have both batches, weighted by closeness of their
    log latency at b0 (small graphs scale sublinearly, big nets about proportionally).
    Returns None when fewer than one_min curves qualify."""
    cand = [((c[b] - c[b0]) / math.log(b / b0), c[b0] - l0) for c in lib if b0 in c and b in c]
    if len(cand) < cfg["one_min"]:
        return None
    w = np.array([math.exp(-0.5 * (d / cfg["one_tau"]) ** 2) for _, d in cand]) + 1e-6
    return float(np.sum(w * np.array([sl for sl, _ in cand])) / np.sum(w))


def _solve_k(cur: dict, b: float, target: float, cfg: dict):
    """Effective-batch factor k with curve(k * b) = target (log latency), by bisection in
    log k over [1/8, 32]; None when the target is outside the curve on that range."""
    lo, hi = math.log(0.125), math.log(32.0)
    f = lambda t: _curve(cur, b * math.exp(t), cfg)[0] - target  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _core(recs: dict, krows: list, qrows: list, cfg: dict = CFG) -> list:
    """krows: (file, batch, shape, latency); qrows: (file, batch, shape).
    Returns per query (pred_log1p | nan, kind, src, n_batches, spread)."""
    kfiles = {r[0] for r in krows}
    uf, tw = _UF(), _UF()  # all links / latency-twin links only
    # latency twins
    by_triple = collections.defaultdict(set)
    for f, b, s, lat in krows:
        by_triple[(b, s, lat)].add(f)
    share = collections.Counter()
    for fs in by_triple.values():
        if len(fs) > 1:
            fs = sorted(fs)
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    share[(fs[i], fs[j])] += 1
    for (a, b), c in share.items():
        if c >= 2:
            uf.union(a, b)
            tw.union(a, b)
    # feature duplicates among known files
    buckets = collections.defaultdict(list)
    for f in kfiles:
        r = recs.get(f)
        if r is not None:
            buckets[_bucket(r)].append(r)
    for rs in buckets.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if _feat_link(rs[i], rs[j], cfg):
                    uf.union(rs[i]["file"], rs[j]["file"])
    bigs = [recs[f] for f in kfiles if f in recs and recs[f]["params"] >= cfg["big_params"]]
    for i in range(len(bigs)):
        for j in range(i + 1, len(bigs)):
            if _param_twin(bigs[i], bigs[j], cfg) or _near_twin(bigs[i], bigs[j], cfg):
                uf.union(bigs[i]["file"], bigs[j]["file"])
    # points per identity
    pts = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for f, b, s, lat in krows:
        pts[uf.find(f)][s][b].append((math.log(max(lat, 1e-6)), f))

    # library of batch curves (one per identity and shape) for _slope1
    lib = []
    if cfg.get("one_kernel"):
        for root, sh in pts.items():
            for s_, bm in sh.items():
                if len(bm) >= 2:
                    lib.append((root, {bb: float(np.mean([v for v, _ in vs])) for bb, vs in bm.items()}))

    # global shape statistics: exponent between two override shapes, offset override vs default
    alphas, deltas, keff = [], collections.defaultdict(list), collections.defaultdict(list)
    for root, sh in pts.items():
        ov = [s for s in sh if s != "default" and _pixels(s)]
        for i in range(len(ov)):
            for j in range(i + 1, len(ov)):
                p0, p1 = _pixels(ov[i]), _pixels(ov[j])
                if p0 == p1:
                    continue
                for b in set(sh[ov[i]]) & set(sh[ov[j]]):
                    l0 = np.mean([v for v, _ in sh[ov[i]][b]])
                    l1 = np.mean([v for v, _ in sh[ov[j]][b]])
                    alphas.append((root, (l1 - l0) / math.log(p1 / p0)))
        if "default" in sh:
            for s in ov:
                for b in set(sh[s]) & set(sh["default"]):
                    d = np.mean([v for v, _ in sh[s][b]]) - np.mean([v for v, _ in sh["default"][b]])
                    deltas[s].append((root, d))
            # a row with an override shape behaves like the default curve at an "effective
            # batch" k * batch (800x800 -> k ~ 1.56, i.e. more pixels per sample)
            if len(sh["default"]) >= 2:
                cdef = {bb: float(np.mean([v for v, _ in vs])) for bb, vs in sh["default"].items()}
                for s in ov:
                    for b, vs in sh[s].items():
                        k = _solve_k(cdef, b, float(np.mean([v for v, _ in vs])), cfg)
                        if k is not None:
                            keff[s].append((root, k))

    out = []
    for f, b, s in qrows:
        roots = set()
        near = False
        if f in kfiles:
            roots.add(uf.find(f))
        elif f in recs:
            r = recs[f]
            for o in buckets.get(_bucket(r), ()):
                if _feat_link(r, o, cfg):
                    roots.add(uf.find(o["file"]))
            for o in bigs:
                if _param_twin(r, o, cfg):
                    roots.add(uf.find(o["file"]))
            if not roots:  # nothing stricter matched: near twins of big nets
                for o in bigs:
                    if _near_twin(r, o, cfg):
                        roots.add(uf.find(o["file"]))
                        near = True
        if not roots:
            out.append((float("nan"), "none", "", 0, float("nan")))
            continue

        def pooled(shape):
            p = collections.defaultdict(list)
            for root in roots:
                for bb, vs in pts[root].get(shape, {}).items():
                    p[bb].extend(vs)
            return p

        def src_of(vs):
            if any(g == f for _, g in vs):
                return "same"
            if f in kfiles and any(tw.find(g) == tw.find(f) for _, g in vs):
                return "twin"
            return "near" if near else "dup"

        p = pooled(s)
        if p:
            if b in p:
                y = float(np.mean([math.log1p(math.exp(v)) for v, _ in p[b]]))
                out.append((y, "exact", src_of(p[b]), len(p), 0.0))
                continue
            cur = {bb: float(np.mean([v for v, _ in vs])) for bb, vs in p.items()}
            ll, kind, spread = _curve(cur, b, cfg)
            if lib and kind == "extrap1":
                (b0, l0), = cur.items()
                sl = _slope1(b0, l0, b, [c for root, c in lib if root not in roots], cfg)
                if sl is not None:
                    ll = l0 + sl * math.log(b / b0)
                spread = min(b0, b)
            src = "any" if kind == "extrap1" else src_of([x for vs in p.values() for x in vs])
            out.append((math.log1p(math.exp(ll)), kind, src, len(p), spread))
            continue
        # shape override without rows of this shape
        pix = _pixels(s)
        if pix is None:
            out.append((float("nan"), "none", "", 0, float("nan")))
            continue
        ll, via = None, "ovr"
        a_vals = [a for root, a in alphas if root not in roots]
        alpha = float(np.median(a_vals)) if a_vals else cfg["alpha"]
        for root in roots:  # 1) another override shape at the same batch
            for s2, bm in pts[root].items():
                if s2 != "default" and _pixels(s2) and b in bm:
                    ll = np.mean([v for v, _ in bm[b]]) + alpha * math.log(pix / _pixels(s2))
        if ll is None:  # 2) default curve at this batch + offset of the shape
            pd_ = pooled("default")
            via = "def"
            if pd_:
                cur = {bb: float(np.mean([v for v, _ in vs])) for bb, vs in pd_.items()}
                base, _ = curve_predict(cur, b, cfg)
                d = [x for root, x in deltas.get(s, ()) if root not in roots]
                ks = [x for root, x in keff.get(s, ()) if root not in roots]
                if cfg.get("shape_keff", True) and len(cur) >= 2 and ks:
                    via = "keff"
                    ll = _curve(cur, b * math.exp(float(np.median(ks))), cfg)[0]
                elif d:
                    ll = base + float(np.median(d))
                else:
                    for s2, ds in deltas.items():
                        d = [x for root, x in ds if root not in roots]
                        if d and _pixels(s2):
                            ll = base + float(np.median(d)) + alpha * math.log(pix / _pixels(s2))
                            break
        if ll is None:
            out.append((float("nan"), "none", "", 0, float("nan")))
        else:
            out.append((math.log1p(math.exp(ll)), "shape", via, 1, float("nan")))
    return out


def _rows(df, with_lat):
    b = df.batch.astype(float).values
    if with_lat:
        return list(zip(df.file.values, b, df["shape"].values, df.latency.astype(float).values))
    return list(zip(df.file.values, b, df["shape"].values))


def _frame(res, index):
    out = pd.DataFrame(res, columns=["g1_pred", "g1_kind", "g1_src", "g1_n", "g1_spread"], index=index)
    out["g1_conf"] = [_conf(k, s, sp) for k, s, sp in zip(out.g1_kind, out.g1_src, out.g1_spread)]
    out["g1_detail"] = out.g1_kind
    out["g1_kind"] = out.g1_kind.replace(_EXTRAP)
    return out[["g1_pred", "g1_kind", "g1_conf", "g1_src", "g1_n", "g1_detail", "g1_spread"]]


def g1_predict(known: pd.DataFrame, query: pd.DataFrame, ident: pd.DataFrame | None = None, data_dir: str | None = None, cfg: dict = CFG) -> pd.DataFrame:
    """known: rows with latency (columns of features.csv incl. file, fp, batch, shape, latency).
    query: rows to predict (same columns, latency ignored).
    Returns DataFrame indexed like query with columns:
      g1_pred  - prediction in log1p(latency) space, NaN where G1 does not apply
      g1_kind  - 'exact' | 'interp' | 'extrap' | 'shape' | 'none'
      g1_conf  - rough expected RMSLE of this prediction (from validation), NaN for 'none'
      g1_src   - 'same' (same file) | 'twin' (latency twin) | 'dup' (feature duplicate) |
                 'near' (big net differing by a few small tensors) | 'any';
                 for 'shape': 'ovr' (from another override shape) | 'keff' (default curve at the
                 effective batch of the shape) | 'def' (default curve + offset, fallback)
      g1_n     - number of known batches behind the prediction
      g1_detail - like g1_kind, with 'extrap_up' / 'extrap_down' / 'extrap1' (one known batch)
      g1_spread - |linear-space estimate - log-log estimate| in log latency (irregularity of
                 the curve; drives g1_conf for interp / extrap_up); for extrap_down 0 = regular
                 curve, 1 = irregular; for extrap1 the smaller batch of the step
    Weight-shape keys come from `<data_dir>/ident.csv` (scripts/ident.py) when the frames have
    no `skey` column; without that file only fp is used for structure equality.
    """
    if ident is None and "skey" not in known.columns:
        ident = _load_ident(data_dir)
    known = known[known.latency.notna()]
    cols = ["file", "batch", "shape", "has_graph", "fp", "params", "macs", "out_elems"] + (["input_elems"] if "input_elems" in known.columns and "input_elems" in query.columns else []) + (["skey"] if "skey" in known.columns and "skey" in query.columns else [])
    recs = file_records(pd.concat([known[cols], query[cols]], ignore_index=True), ident)
    res = _core(recs, _rows(known, True), _rows(query, False), cfg)
    return _frame(res, query.index)


def _fit_batch_curve(b: np.ndarray, lat: np.ndarray, form: str) -> np.ndarray:
    """One smooth curve through (batch, predicted latency) of a file; returns fitted latency.
    affine: lat = c0 + a * batch, c0 >= 0, a >= 0, least squares on the relative error;
    loglog: log lat = alpha + beta * log batch, beta clipped to [0, 1.2]."""
    if form == "loglog":
        x, y = np.log(b), np.log(lat)
        xm, ym = x.mean(), y.mean()
        beta = float(np.sum((x - xm) * (y - ym)) / max(np.sum((x - xm) ** 2), 1e-12))
        beta = min(max(beta, 0.0), 1.2)
        return np.exp(ym + beta * (x - xm))
    w = 1.0 / lat**2
    sw, sb, sbb, sp, sbp = w.sum(), (w * b).sum(), (w * b * b).sum(), (w * lat).sum(), (w * b * lat).sum()
    det = sw * sbb - sb * sb
    c0, a = ((sbb * sp - sb * sbp) / det, (sw * sbp - sb * sp) / det) if det > 1e-12 * sw * sbb else (-1.0, -1.0)
    if c0 < 0:
        c0, a = 0.0, sbp / sbb
    if a < 0:
        c0, a = sp / sw, 0.0
    return c0 + a * b


def smooth_batch_curves(query: pd.DataFrame, pred_log1p, g1_frame: pd.DataFrame | None = None, min_rows: int = 4, form: str = "affine", mix: float = 1.0) -> np.ndarray:
    """Reconcile model predictions across batches of one file. For every (file, shape) of
    `query` with >= min_rows rows of distinct batches, where G1 found nothing (all g1_kind ==
    'none' when g1_frame is given), fit one curve to the predictions (see _fit_batch_curve) and
    return mix * curve + (1 - mix) * prediction in log1p; other rows are returned unchanged.
    query, pred_log1p and g1_frame are aligned by position.
    NOT USED in predict.py: on train (scripts/exp_mr_smooth.py, out-of-fold stack, 222 rows of
    44 multi-row files, two versions of the out-of-fold file) it does not help: 0.8918 ->
    0.8938 (affine) / 0.8935 (loglog) / 0.8916-0.8919 (half mix); 0.9375 -> 0.9397 / 0.9397 /
    0.9380. The error is the level of the whole file (0.87-0.91), not the noise between
    batches (0.21-0.22, unchanged by smoothing), and superlinear curves get worse."""
    pred = np.asarray(pred_log1p, float)
    out = pred.copy()
    b = query.batch.values.astype(float)
    free = np.ones(len(query), bool) if g1_frame is None else (g1_frame.g1_kind.values == "none")
    groups = collections.defaultdict(list)
    for i, key in enumerate(zip(query.file.values, query["shape"].values)):
        groups[key].append(i)
    for idx in groups.values():
        idx = np.array(idx)
        if len(idx) < min_rows or len(set(b[idx])) < min_rows or not free[idx].all() or not np.isfinite(pred[idx]).all() or (b[idx] <= 0).any():
            continue
        lat = np.clip(np.expm1(pred[idx]), 1e-6, None)
        out[idx] = mix * np.log1p(_fit_batch_curve(b[idx], lat, form)) + (1 - mix) * pred[idx]
    return out


# ----------------------------------------------------------------------------- validation


def _rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")


def loo(train: pd.DataFrame, recs: dict, cfg: dict, drop_identity_batch: bool = False, only_multi: bool = False) -> pd.DataFrame:
    """Leave-one-row-out over train rows. For a single-row file this is leave-one-file-out
    (the file can only be matched through its features). With drop_identity_batch the rows of
    every other file with the same batch+shape that G1 would use as an exact hit are dropped
    too, which forces interpolation / extrapolation."""
    rows = _rows(train, True)
    nfile = collections.Counter(r[0] for r in rows)
    y = np.log1p(train.latency.values.astype(float))
    res = []
    for i, r in enumerate(rows):
        if only_multi and nfile[r[0]] < 2:
            continue
        kn = rows[:i] + rows[i + 1 :]
        o = _core(recs, kn, [r[:3]], cfg)[0]
        if drop_identity_batch and o[1] == "exact":
            # remove the latency twins of this row (same batch, shape, latency), predict again;
            # if a feature duplicate still gives an exact hit, remove the whole batch+shape
            kn = [k for k in kn if not (k[1] == r[1] and k[2] == r[2] and k[3] == r[3])]
            o = _core(recs, kn, [r[:3]], cfg)[0]
            if o[1] == "exact":
                kn = [k for k in kn if not (k[1] == r[1] and k[2] == r[2])]
                o = _core(recs, kn, [r[:3]], cfg)[0]
        res.append((r[0], r[1], r[2], nfile[r[0]], y[i], *o))
    d = pd.DataFrame(res, columns=["file", "batch", "shape", "nfile", "y", "pred", "kind", "src", "n", "spread"])
    d["err"] = d.pred - d.y
    return d


def table(d: pd.DataFrame, by) -> pd.DataFrame:
    d = d[d.kind != "none"]
    if not len(d):
        return pd.DataFrame()
    g = d.groupby(by).err
    return pd.DataFrame({"n": g.size(), "rmsle": g.apply(_rmse), "med_abs": g.apply(lambda e: float(np.median(np.abs(e)))), "max_abs": g.apply(lambda e: float(np.max(np.abs(e))))}).round(4)


def shape_validation(train, recs, cfg):
    """Hold out override-shape rows that are alone in their (file, shape): (a) one at a time
    (the other override shape of the file stays), (b) all override rows of the file at once
    (prediction has to come from the default curve)."""
    rows = _rows(train, True)
    cnt = collections.Counter((r[0], r[2]) for r in rows)
    res = []
    for i, r in enumerate(rows):
        if r[2] == "default" or cnt[(r[0], r[2])] != 1:
            continue
        for mode in ("one", "all"):
            kn = [k for j, k in enumerate(rows) if j != i and not (mode == "all" and k[0] == r[0] and k[2] != "default")]
            o = _core(recs, kn, [r[:3]], cfg)[0]
            res.append((mode, r[0], r[2], math.log1p(r[3]), o[0], o[1]))
    d = pd.DataFrame(res, columns=["mode", "file", "shape", "y", "pred", "kind"])
    d["err"] = d.pred - d.y
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tune", action="store_true", help="compare interpolation / extrapolation / linking variants")
    ap.add_argument("--out", default=None, help="write test predictions (id + g1 columns) to this csv")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    df = pd.read_csv(os.path.join(a.data, "features.csv")).copy()
    ident = _load_ident(a.data)
    print(f"ident.csv: {'found, ' + str(len(ident)) + ' files' if ident is not None else 'MISSING (fp only)'}")
    train, test = df[df.split == "train"], df[df.split == "test"]
    recs = file_records(df, ident)

    if a.tune:
        print("\n== linking variants, single-row train files (leave-one-file-out) ==")
        single = None
        for name, over in [("fp/skey dup only", {"big_params": 1e30}), ("default", {}), ("out_tol 0.01", {"out_tol": 0.01}), ("out_tol 0.1", {"out_tol": 0.1}), ("out_tol 0.2", {"out_tol": 0.2}), ("big_params 1e5", {"big_params": 1e5})]:
            d = loo(train, recs, {**CFG, **over})
            d = d[(d.nfile == 1) & (d.kind != "none")]
            print(f"{name:20s} " + "  ".join(f"{k}: n={len(g)} rmsle={_rmse(g.err):.4f} max={g.err.abs().max():.3f}" for k, g in d.groupby("kind")))
        print("\n== curve variants, multi-row files, exact hits removed (forced interp/extrap) ==")
        for key, vals in [("interp", ["lin", "loglog", "mix"]), ("up", ["loglog", "slope1", "shrink", "clip", "affine", "mixu"]), ("down", ["loglog", "k0.8", "k0.6", "affine", "mixd", "p0.7", "p0.9"])]:
            for v in vals:
                d = loo(train, recs, {**CFG, key: v}, drop_identity_batch=True, only_multi=True)
                d = d[d.kind.isin(["interp", "extrap_up", "extrap_down"])]
                d = d.assign(kind=d.kind.replace(_EXTRAP))
                print(f"{key:7s}={v:7s} " + "  ".join(f"{k}: n={len(g)} rmsle={_rmse(g.err):.4f}" for k, g in d.groupby("kind")) + "   extrap by batch: " + " ".join(f"b{int(b)}:{_rmse(g.err):.3f}({len(g)})" for b, g in d[d.kind == "extrap"].groupby("batch")))
        print("\n== one-point slope prior (single-row files, extrap1) ==")
        for su in (0.6, 0.7, 0.8, 1.0):
            for sd in (0.6, 0.8):
                d = loo(train, recs, {**CFG, "one_up": su, "one_down": sd})
                d = d[d.kind == "extrap1"]
                print(f"one_up={su} one_down={sd}: n={len(d)} rmsle={_rmse(d.err):.4f}")
        return

    print("\n== validation on train: leave-one-row-out (single-row file = leave-one-file-out) ==")
    d = loo(train, recs, CFG)
    d["files"] = np.where(d.nfile > 1, "multi-row", "single-row")
    print(table(d, ["files", "kind", "src"]).to_string())
    print("-- by kind --")
    print(table(d.assign(kind=d.kind.replace(_EXTRAP)), ["kind"]).to_string())
    cov = d.groupby("files").apply(lambda g: f"{(g.kind != 'none').sum()}/{len(g)}")
    print("covered rows:", cov.to_dict())

    print("\n== multi-row files with exact hits removed (forced interp / extrap) ==")
    d2 = loo(train, recs, CFG, drop_identity_batch=True, only_multi=True)
    print(table(d2, ["kind", "src"]).to_string())

    sv = shape_validation(train, recs, CFG)
    if len(sv):
        print("\n== shape override rule (one = other override shape known, all = only default curve) ==")
        print(table(sv, ["mode", "kind"]).to_string())
        print(sv.round(3).to_string(index=False))

    print("\n== coverage on test ==")
    p = g1_predict(train, test, ident=ident)
    t = test.assign(kind=p.g1_detail.values, src=p.g1_src.values)
    t["group"] = np.where(t.has_graph == 0, "no graph yet", np.where(t.file.isin(set(train.file)), "file in train", np.where(t.groupby("file").file.transform("size") > 1, "test-only multi-row", "test-only single-row")))
    print(t.groupby(["group", "kind", "src"]).size().to_string())
    print("by kind:", p.g1_kind.value_counts().to_dict())
    if a.out:
        p.insert(0, "id", test.id.values)
        p.to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
