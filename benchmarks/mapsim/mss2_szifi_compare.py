"""SZiFi's catalogues against our finder's on the MSS2 comparison maps.

    python benchmarks/mapsim/mss2_szifi_compare.py

Inputs: ``mss2_szifi_maps.py``'s ours/*.npz and meta.json, ``mss2_szifi_run.py``'s szifi/runs/*.npz.
Numpy and scipy only (runs in either environment). Writes szifi_compare/summary.txt and summary.json.

Conventions:
  * positions in our pixel indices (row iy, column ix). SZiFi reports the pixel half a pixel after
    the cluster (sz-poisson-blending's SZiFi check), so its (i, j) are shifted by -1/2 pixel; the
    offset is measured again here (iteration 0 of 'match' against our first pass);
  * interior: more than 60' (EDGE_DROP pixels) from the tile's edges, as our data vector;
  * merging: summaries.merge_local_max's rule (a peak is kept unless another one within 1.5' has a
    higher q; ties go to the earlier one in catalogue order), periodic distances;
  * confirmation: summaries.confirm's rule (of the halos with M >= 5e13 within 1.5', the one with
    the largest y0 gives z); the halos' positions come in the finder's 0.5' pixels;
  * data vector: the paper's 4 x 5 bins in q and z (open intervals).
The numpy merging and confirmation are first checked against the simulator's own (the counts saved
by mss2_szifi_maps.py) on our catalogues.

q scales: our finder's cut-out q ('ours raw') comes from a covariance of the tapered map, as
SZiFi's here (no correction for the taper's power loss); production divides it by
sigma_periodic / sigma_cut per scale ('ours prod', the periodic tiles' scale of the simulator).

Per depth and configuration, pooled over the tiles:
  sigma         SZiFi / ours per scale before (iteration 0) and after the masking step, and each
                code's after / before. The y0 normalisations of the two templates differ slightly
                at small scales (SZiFi's is pixel-sampled), so q is the robust comparison.
  masked        masked fraction of each code; overlap of the masks (intersection / union)
  match         kept (merged) interior detections with q > 5 in one catalogue that have a
                detection with q > 4 within 1' (any scale) in the other, both ways; for matched
                pairs, q_SZiFi - q_ours (mean, rms) and the median q_SZiFi / q_ours for q > 6,
                oriented SZiFi against ours in both directions
  final         SZiFi's last iteration (iteration 1; iteration 0 if SZiFi masked nothing and
                stopped: 'n_tiles_it1' counts the tiles with a second pass)
  counts        kept interior detections with q > 5, those confirmed, the data vector total, and
                the ratio SZiFi / ours, paired by tile (ours raw and ours prod)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2", "szifi_compare")
CFGS = ("match", "match4", "coll")
CFG_LABEL = {"match": "SZiFi set as ours", "match4": "same, 4 channels", "coll": "colleagues' settings"}
SZ_SHIFT = 0.5                    # pixels subtracted from SZiFi's (i, j)
R_MATCH_PX = 2.0                  # 1'
QC = 5.0


class Grid:
    def __init__(self, meta):
        self.N = int(meta["N"])
        self.pix = float(meta["pix_arcmin"])
        self.e = int(meta["edge_drop"])
        self.thetas = np.asarray(meta["thetas"], float)
        self.r_merge = float(meta["r_merge"])
        self.r_conf = float(meta["r_conf"])
        self.qe = np.asarray(meta["q_edges"], np.float32)
        self.ze = np.asarray(meta["z_edges"], np.float32)

    def inner(self, y, x):
        e, N = self.e, self.N
        return (y >= e) & (y < N - e) & (x >= e) & (x < N - e)


def tree(y, x, N):
    return cKDTree(np.c_[np.mod(y, N), np.mod(x, N)], boxsize=N)


def merge_keep(g, y, x, q):
    """summaries.merge_local_max on arrays in catalogue order."""
    n = q.size
    keep = np.ones(n, bool)
    if n < 2:
        return keep
    r = g.r_merge / g.pix
    pairs = tree(y, x, g.N).query_pairs(r=r + 1e-6, output_type="ndarray")
    if pairs.size:
        a, b = pairs[:, 0], pairs[:, 1]
        lo, hi = np.minimum(a, b), np.maximum(a, b)                 # lo comes first in catalogue order
        keep[lo[q[hi] > q[lo]]] = False                             # beaten by a later, higher peak
        keep[hi[q[lo] >= q[hi]]] = False                            # beaten by an earlier, higher or equal one
    return keep


def confirm(g, y, x, keep, hal):
    """summaries.confirm: ok and z per peak (positions in pixels)."""
    ok = np.zeros(y.size, bool)
    z = np.zeros(y.size, np.float32)
    if hal["h_y"].size == 0 or not keep.any():
        return ok, z
    t = tree(hal["h_y"].astype(np.float64), hal["h_x"].astype(np.float64), g.N)
    idx = np.flatnonzero(keep)
    lists = t.query_ball_point(np.c_[np.mod(y[idx], g.N), np.mod(x[idx], g.N)], r=g.r_conf / g.pix + 1e-6)
    for k, lst in zip(idx, lists):
        if lst:
            lst = np.sort(np.asarray(lst))                         # buffer order: ties go to the first, as argmax
            b = lst[np.argmax(hal["h_y0"][lst])]
            ok[k], z[k] = True, hal["h_z"][b]
    return ok, z


def histogram(g, q, z, ok):
    q, z = np.asarray(q, np.float32), np.asarray(z, np.float32)
    inside = ok & (q > g.qe[0]) & (q < g.qe[-1]) & (z > g.ze[0]) & (z < g.ze[-1])
    iq = np.sum(q[:, None] > g.qe[None, 1:-1], axis=1)
    iz = np.sum(z[:, None] > g.ze[None, 1:-1], axis=1)
    h = np.zeros((4, 5), int)
    np.add.at(h, (iq[inside], iz[inside]), 1)
    return h


def catalogue(g, y, x, q, hal, theta=None):
    """Merged, interior, confirmed: everything the counts need."""
    keep_all = merge_keep(g, y, x, q)
    keep = keep_all & g.inner(y, x)
    ok, z = confirm(g, y, x, keep, hal)
    return dict(y=y, x=x, q=q, theta=theta, keep=keep, ok=ok, z=z, hist=histogram(g, q, z, keep & ok),
                n_kept5=int(np.sum(keep & (q > QC))), n_conf5=int(np.sum(keep & ok & (q > QC))))


def match(A, B, qc=QC):
    """Kept interior detections of A with q > qc that have a detection of B (any) within 1'."""
    sel = np.flatnonzero(A["keep"] & (A["q"] > qc))
    out = dict(n=int(sel.size), n_matched=0, dq=np.zeros(0), qa=np.zeros(0), qb=np.zeros(0),
               ta=np.zeros(0), tb=np.zeros(0))
    if sel.size == 0 or B["q"].size == 0:
        return out
    lists = tree(B["y"], B["x"], N_GLOBAL).query_ball_point(
        np.c_[np.mod(A["y"][sel], N_GLOBAL), np.mod(A["x"][sel], N_GLOBAL)], r=R_MATCH_PX + 1e-6)
    qa, qb, ta, tb = [], [], [], []
    for k, lst in zip(sel, lists):
        if lst:
            lst = np.asarray(lst)
            b = lst[np.argmax(B["q"][lst])]
            qa.append(A["q"][k]); qb.append(B["q"][b])
            ta.append(A["theta"][k] if A["theta"] is not None else np.nan)
            tb.append(B["theta"][b] if B["theta"] is not None else np.nan)
    out.update(n_matched=len(qa), qa=np.asarray(qa, float), qb=np.asarray(qb, float), dq=np.asarray(qb, float) - np.asarray(qa, float),
               ta=np.asarray(ta, float), tb=np.asarray(tb, float))
    return out


def offset(ours0, sz0):
    """SZiFi's raw (i, j) minus our first-pass (iy, ix) for pairs with q > 6 in both (nearest within 2 px)."""
    a = ours0["q"] > 6
    b = sz0["q"] > 6
    if not a.any() or not b.any():
        return np.zeros((0, 2))
    t = tree(sz0["i"][b].astype(float), sz0["j"][b].astype(float), N_GLOBAL)
    d, k = t.query(np.c_[ours0["iy"][a], ours0["ix"][a]].astype(float), distance_upper_bound=2.0 + 1e-6)
    ok = np.isfinite(d)
    bi = np.flatnonzero(b)[k[ok]]
    d = np.c_[sz0["i"][bi] - ours0["iy"][a][ok], sz0["j"][bi] - ours0["ix"][a][ok]].astype(float)
    return (d + N_GLOBAL / 2) % N_GLOBAL - N_GLOBAL / 2                  # periodic


N_GLOBAL = 1200


def load():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    global N_GLOBAL
    N_GLOBAL = int(meta["N"])
    g = Grid(meta)
    ours = {}
    for f in sorted(glob.glob(os.path.join(OUT, "ours", "tile_*.npz"))):
        b = os.path.basename(f)[:-4]
        ti, dp = int(b[5:8]), b[9:]
        ours[(ti, dp)] = dict(np.load(f))
    sz = {}
    for f in sorted(glob.glob(os.path.join(OUT, "szifi", "runs", "*.npz"))):
        d = dict(np.load(f, allow_pickle=False))
        sz[(str(d["cfg"]), int(d["tile"]), str(d["depth"]))] = d
    return meta, g, ours, sz


def ours_catalogues(g, o):
    hal = {k: o[k] for k in ("h_y", "h_x", "h_z", "h_y0")}
    th = g.thetas
    q1 = o["pk1_q"].astype(np.float32)
    qr = (q1 / o["rs"].astype(np.float32)[o["pk1_j"]]).astype(np.float32)
    y1, x1 = o["pk1_iy"].astype(float), o["pk1_ix"].astype(float)
    y0, x0 = o["pk0_iy"].astype(float), o["pk0_ix"].astype(float)
    return dict(raw=catalogue(g, y1, x1, q1, hal, th[o["pk1_j"]]),
                prod=catalogue(g, y1, x1, qr, hal, th[o["pk1_j"]]),
                raw0=catalogue(g, y0, x0, o["pk0_q"].astype(np.float32), hal, th[o["pk0_j"]])), hal


def sz_catalogues(g, d, hal):
    out = {}
    for it in (0, 1):
        if f"it{it}_q" not in d:
            continue
        y = d[f"it{it}_i"].astype(float) - SZ_SHIFT
        x = d[f"it{it}_j"].astype(float) - SZ_SHIFT
        c = catalogue(g, y, x, d[f"it{it}_q"].astype(float), hal, d[f"it{it}_theta"].astype(float))
        c.update(i=d[f"it{it}_i"], j=d[f"it{it}_j"])
        out[it] = c
    return out


def sigma_on_ours(d, it, thetas):
    """SZiFi's sigma per scale on our 15 scales (log-log interpolation for another grid)."""
    th = np.asarray(d["thetas"], float)
    s = np.asarray(d[f"it{it}_sig"], float)
    if th.size == thetas.size and np.allclose(th, thetas, rtol=1e-9):
        return s
    ok = np.isfinite(s) & (s > 0)
    return np.exp(np.interp(np.log(thetas), np.log(th[ok]), np.log(s[ok]), left=np.nan, right=np.nan))


def unpack(m, N):
    return np.unpackbits(m)[:N * N].reshape(N, N).astype(bool)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, rms=np.nan, median=np.nan)
    return dict(n=int(x.size), mean=float(x.mean()), rms=float(x.std()), median=float(np.median(x)))


def main():
    meta, g, ours, sz = load()
    depths = meta["depths"]
    tiles = meta["tiles"]
    lines, res = [], dict(check={}, depths={})
    P = lines.append

    # 1. the numpy merging and confirmation against the simulator's own counts
    bad = []
    for (ti, dp), o in sorted(ours.items()):
        oc, _ = ours_catalogues(g, o)
        for name, ref in (("prod", "counts_prod"), ("raw", "counts_raw"), ("raw0", "counts_raw0")):
            if not np.array_equal(oc[name]["hist"], o[ref]):
                bad.append((ti, dp, name, int(oc[name]["hist"].sum()), int(np.asarray(o[ref]).sum())))
    res["check"]["numpy_vs_simulator"] = dict(n_maps=len(ours), mismatches=bad)
    sc = meta.get("selfcheck", {})
    res["check"]["maps_selfcheck"] = sc
    P("SZiFi against our finder on MSS2 cut-outs with the simulator's sky (mss2_szifi_compare.py)")
    P(f"tiles {tiles}, sky {meta['sky']}; interior > {g.e * g.pix:.0f}' from the edges")
    nbad = len({(t, dd) for t, dd, *_ in bad})
    P(f"check: numpy merging and confirmation reproduce the simulator's data vectors on {len(ours) - nbad} "
      f"of {len(ours)} maps" + (f"; mismatches {bad}" if bad else ""))
    P("check: the rebuilt maps give the data-vector pipeline's counts: "
      + ", ".join(f"{t}/{dp}: {v['inline']}/{v['stored']}" for t, dd in sc.items() for dp, v in dd.items()))

    for dp in depths:
        R = dict(configs={})
        res["depths"][dp] = R
        P("")
        P(f"=== depth {dp}")
        per_tile = {}
        for ti in tiles:
            o = ours.get((ti, dp))
            if o is None:
                continue
            oc, hal = ours_catalogues(g, o)
            per_tile[ti] = dict(o=o, oc=oc, hal=hal)
        # ours alone
        R["ours"] = dict(
            f_masked=stats([v["o"]["f_masked"] for v in per_tile.values()]),
            kept5_raw=int(sum(v["oc"]["raw"]["n_kept5"] for v in per_tile.values())),
            kept5_prod=int(sum(v["oc"]["prod"]["n_kept5"] for v in per_tile.values())),
            dv_raw=int(sum(v["oc"]["raw"]["hist"].sum() for v in per_tile.values())),
            dv_prod=int(sum(v["oc"]["prod"]["hist"].sum() for v in per_tile.values())),
            conf5_raw=int(sum(v["oc"]["raw"]["n_conf5"] for v in per_tile.values())),
            conf5_prod=int(sum(v["oc"]["prod"]["n_conf5"] for v in per_tile.values())))
        Ro = R["ours"]
        P(f"ours: masked {Ro['f_masked']['mean']:.3f}; kept q>5 {Ro['kept5_raw']} (cut-out q) / {Ro['kept5_prod']} "
          f"(production q); confirmed {Ro['conf5_raw']} / {Ro['conf5_prod']}; data vector {Ro['dv_raw']} / {Ro['dv_prod']}")
        for cfg in CFGS:
            rows = [(ti, v, sz.get((cfg, ti, dp))) for ti, v in per_tile.items()]
            rows = [r for r in rows if r[2] is not None]
            if not rows:
                continue
            C = dict(n_tiles=len(rows))
            R["configs"][cfg] = C
            s0r, s1r, r_sz, r_ou, fm, jac = [], [], [], [], [], []
            m0a, m0b, m1a, m1b, m1p = [], [], [], [], []
            offs, cnt = [], dict(kept5=[], conf5=[], dv=[], kept5_raw=[], kept5_prod=[], dv_raw=[], dv_prod=[],
                                 unconf5=[], big_theta5=[])
            hist_sz = np.zeros((4, 5), int)
            n_it1 = 0
            for ti, v, d in rows:
                o, oc, hal = v["o"], v["oc"], v["hal"]
                sc_ = sz_catalogues(g, d, hal)
                if not sc_:
                    continue
                fin = max(sc_)                                    # SZiFi's last iteration
                n_it1 += int(fin == 1)
                if 0 in sc_:
                    s0 = sigma_on_ours(d, 0, g.thetas)
                    s0r.append(s0 / o["s0"])
                s1 = sigma_on_ours(d, fin, g.thetas)
                s1r.append(s1 / o["s1"])
                if 0 in sc_ and fin == 1:
                    r_sz.append(s1 / s0)
                r_ou.append(o["s1"] / o["s0"])
                if d["rec_fsky"].size:
                    fm.append(1.0 - float(d["rec_fsky"][0]))           # the mask of the second pass
                if "mask" in d:
                    a = ~unpack(o["mask"], g.N)
                    b = ~unpack(d["mask"], g.N)
                    jac.append(float((a & b).sum() / max((a | b).sum(), 1)))
                if 0 in sc_:
                    m0a.append(match(oc["raw0"], sc_[0]))
                    m0b.append(match(sc_[0], oc["raw0"]))
                    if cfg == "match":
                        offs.append(offset(dict(q=o["pk0_q"], iy=o["pk0_iy"], ix=o["pk0_ix"]),
                                           dict(q=d["it0_q"], i=d["it0_i"], j=d["it0_j"])))
                if True:
                    c1 = sc_[fin]
                    m1a.append(match(oc["raw"], c1))
                    m1b.append(match(c1, oc["raw"]))
                    m1p.append(match(oc["prod"], c1))
                    cnt["kept5"].append(c1["n_kept5"]); cnt["conf5"].append(c1["n_conf5"])
                    cnt["dv"].append(int(c1["hist"].sum()))
                    cnt["unconf5"].append(int(np.sum(c1["keep"] & ~c1["ok"] & (c1["q"] > QC))))
                    cnt["big_theta5"].append(int(np.sum(c1["keep"] & (c1["q"] > QC) & (c1["theta"] > g.thetas[-1] * 1.0001))))
                    cnt["kept5_raw"].append(oc["raw"]["n_kept5"]); cnt["kept5_prod"].append(oc["prod"]["n_kept5"])
                    cnt["dv_raw"].append(int(oc["raw"]["hist"].sum())); cnt["dv_prod"].append(int(oc["prod"]["hist"].sum()))
                    hist_sz += c1["hist"]

            def pool(ms, szifi_first=False):
                """Pooled matches; dq = q_SZiFi - q_ours and ratio = q_SZiFi / q_ours whichever
                catalogue was looked up in the other."""
                n = sum(m["n"] for m in ms)
                k = sum(m["n_matched"] for m in ms)
                qa = np.concatenate([m["qa"] for m in ms]) if ms else np.zeros(0)
                qb = np.concatenate([m["qb"] for m in ms]) if ms else np.zeros(0)
                qs, qo = (qa, qb) if szifi_first else (qb, qa)
                hi = (qs > 6) & (qo > 6)
                return dict(n=n, matched=k, frac=k / n if n else np.nan, dq=stats(qs - qo),
                            ratio_q6=stats(qs[hi] / qo[hi]) if hi.any() else stats([]))

            def ratio(a, b):
                a, b = np.asarray(a, float), np.asarray(b, float)
                if b.sum() == 0:
                    return dict(ratio=np.nan, se=np.nan)
                d = a - b * a.sum() / b.sum()
                return dict(ratio=float(a.sum() / b.sum()),
                            se=float(np.sqrt(len(a)) * d.std(ddof=1) / b.sum()) if len(a) > 1 else np.nan)
            C["sigma_it0_ratio"] = np.nanmean(s0r, 0).tolist() if s0r else None
            C["sigma_it1_ratio"] = np.nanmean(s1r, 0).tolist() if s1r else None
            C["sigma_drop_szifi"] = np.nanmean(r_sz, 0).tolist() if r_sz else None
            C["sigma_drop_ours"] = np.nanmean(r_ou, 0).tolist() if r_ou else None
            C["masked"] = stats(fm)
            C["mask_overlap"] = stats(jac)
            C["n_tiles_it1"] = n_it1
            C["it0_ours_in_szifi"], C["it0_szifi_in_ours"] = pool(m0a), pool(m0b, szifi_first=True)
            C["it1_ours_in_szifi"], C["it1_szifi_in_ours"] = pool(m1a), pool(m1b, szifi_first=True)
            C["it1_prod_in_szifi"] = pool(m1p)
            C["counts"] = {k: int(np.sum(v)) for k, v in cnt.items()}
            C["kept5_vs_raw"] = ratio(cnt["kept5"], cnt["kept5_raw"])
            C["kept5_vs_prod"] = ratio(cnt["kept5"], cnt["kept5_prod"])
            C["dv_vs_raw"] = ratio(cnt["dv"], cnt["dv_raw"])
            C["dv_vs_prod"] = ratio(cnt["dv"], cnt["dv_prod"])
            C["hist"] = hist_sz.tolist()
            if offs:
                oo = np.concatenate(offs)
                C["offset_px"] = dict(n=int(len(oo)), mean=oo.mean(0).tolist() if len(oo) else None,
                                      median=np.median(oo, 0).tolist() if len(oo) else None)
            ts = [float(d["t_run"]) for _, _, d in rows]
            C["t_run_s"] = stats(ts)
            C["peak_rss_gb"] = stats([float(d["peak_rss_gb"]) for _, _, d in rows])

            P(f"--- {cfg} ({CFG_LABEL[cfg]}), {C['n_tiles']} tiles ({n_it1} with a second pass)")
            pick = [0, 4, 7, 10, 14]
            th = ", ".join(f"{g.thetas[k]:.1f}'" for k in pick)
            if C["sigma_it0_ratio"] and C["sigma_it1_ratio"]:
                P(f"  sigma SZiFi/ours at {th}: before masking "
                  + " ".join(f"{C['sigma_it0_ratio'][k]:.3f}" for k in pick)
                  + "; after " + " ".join(f"{C['sigma_it1_ratio'][k]:.3f}" for k in pick))
            if C["sigma_drop_szifi"] and C["sigma_drop_ours"]:
                P(f"  sigma after/before masking: SZiFi " + " ".join(f"{C['sigma_drop_szifi'][k]:.3f}" for k in pick)
                  + "; ours " + " ".join(f"{C['sigma_drop_ours'][k]:.3f}" for k in pick))
            P(f"  masked: SZiFi {C['masked']['mean']:.3f} (ours {Ro['f_masked']['mean']:.3f}); "
              f"mask overlap (intersection/union) {C['mask_overlap']['mean']:.3f}")
            for lab, a, b in (("before masking", "it0_ours_in_szifi", "it0_szifi_in_ours"),
                              ("final", "it1_ours_in_szifi", "it1_szifi_in_ours")):
                A, B = C[a], C[b]
                P(f"  {lab}: ours q>5 found by SZiFi {A['matched']}/{A['n']} ({A['frac']:.4f}); SZiFi q>5 found by ours "
                  f"{B['matched']}/{B['n']} ({B['frac']:.4f}); q_SZiFi - q_ours {A['dq']['mean']:+.3f} +- {A['dq']['rms']:.3f}; "
                  f"median q_SZiFi/q_ours (q>6) {A['ratio_q6']['median']:.4f}")
            Pp = C["it1_prod_in_szifi"]
            P(f"  against production q: median q_SZiFi/q_prod (q>6) {Pp['ratio_q6']['median']:.4f}; "
              f"q_SZiFi - q_prod {Pp['dq']['mean']:+.3f} +- {Pp['dq']['rms']:.3f}")
            c = C["counts"]
            P(f"  counts: kept q>5 {c['kept5']} (ours {c['kept5_raw']} cut-out q, {c['kept5_prod']} production); "
              f"ratio {C['kept5_vs_raw']['ratio']:.4f} +- {C['kept5_vs_raw']['se']:.4f} / "
              f"{C['kept5_vs_prod']['ratio']:.4f} +- {C['kept5_vs_prod']['se']:.4f}")
            P(f"  data vector {c['dv']} (ours {c['dv_raw']} / {c['dv_prod']}): ratio {C['dv_vs_raw']['ratio']:.4f} +- "
              f"{C['dv_vs_raw']['se']:.4f} / {C['dv_vs_prod']['ratio']:.4f} +- {C['dv_vs_prod']['se']:.4f}; "
              f"kept q>5 unconfirmed {c['unconf5']}; with theta above 15' {c['big_theta5']}")
            if "offset_px" in C and C["offset_px"]["mean"] is not None:
                P(f"  SZiFi position minus ours (q>6, before masking): mean {np.round(C['offset_px']['mean'], 3).tolist()}, "
                  f"{C['offset_px']['n']} pairs")
            P(f"  run time {C['t_run_s']['mean']:.0f} s (max {max(ts):.0f}), peak memory {C['peak_rss_gb']['mean']:.2f} GB")

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        return v
    json.dump(clean(res), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
