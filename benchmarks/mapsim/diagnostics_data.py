"""Numbers behind the mapsim progress page, gathered into one JSON file.

    python benchmarks/mapsim/diagnostics_data.py

Reads the outputs of counts_modes.py, paper_sigma.py, optbias.py,
hmf_compare.py, prior_corners.py, mask_check.py, lens_check.py, the MSS2
noise check (Perlmutter job, copied to $MAPSIM_OUT/mss2), and the MSS2 noise
tests on rusty (mss2_tiles.py: tile_diagnostics.json and the tiles;
mss2_finder_test.py --summary: finder_test/summary.json; mss2_datavector_test.py
--summary: datavector_test/summary.json) under $MAPSIM_OUT,
simulates one goal-depth tile at the fiducial point for a picture (with the
masking iteration), and writes $MAPSIM_OUT/diagnostics/diagnostics.json.

The earlier project's noise ratios quoted for comparison (sz-poisson-blending,
analysis/realistic/step3/t05_analysis.log, 324 tiles per depth, its own halo
model) are σ_finder/σ_analytic per filter scale before and after the same
masking iteration (whitened estimate).

The pixel-loss estimate quoted for comparison is an independent calculation
from the curvature of the matched filter's response at its peak, averaged over
a uniform sub-pixel offset. It was done in the literature study of 2026-11-18,
with no simulation.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("MAPSIM_OUT", ".")

PIXLOSS_THETA = [0.50, 1.04, 2.15, 4.45, 9.23, 15.0]          # arcmin
PIXLOSS = {"baseline": [2.65, 2.16, 1.64, 1.17, 0.75, 0.52],   # percent
           "goal": [2.97, 2.43, 1.85, 1.32, 0.86, 0.61]}

EARLIER = {   # t05_analysis.log (4a), same 15 scales
    "goal": dict(before=[1.1597, 1.1684, 1.1779, 1.1883, 1.2001, 1.2122, 1.2235, 1.2331, 1.2395, 1.2416,
                         1.2383, 1.2291, 1.2144, 1.1953, 1.1734],
                 after=[1.0531, 1.0555, 1.0580, 1.0606, 1.0635, 1.0661, 1.0682, 1.0693, 1.0691, 1.0674,
                        1.0638, 1.0584, 1.0515, 1.0436, 1.0352],
                 masked=0.239, masked_range=[0.153, 0.347], tiles=324),
    "baseline": dict(before=[1.1179, 1.1241, 1.1307, 1.1379, 1.1460, 1.1545, 1.1624, 1.1692, 1.1739, 1.1756,
                             1.1735, 1.1674, 1.1573, 1.1439, 1.1282],
                     after=[1.0394, 1.0412, 1.0430, 1.0449, 1.0469, 1.0488, 1.0503, 1.0511, 1.0510, 1.0496,
                            1.0468, 1.0426, 1.0371, 1.0308, 1.0240],
                     masked=0.212, masked_range=[0.134, 0.314], tiles=324),
}


def rnd(x, sig=5):
    """Round floats to `sig` significant digits, recursively, for a compact file."""
    if isinstance(x, dict):
        return {k: rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return float(f"{x:.{sig}g}") if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def hmf_part():
    import hmf_compare as hc
    jx = np.load(f"{OUT}/hmf/jax.npz")
    co = np.load(f"{OUT}/hmf/colossus.npz")
    lnM, lnMc = jx["lnM"], jx["lnM_cnc"]
    M = np.logspace(13, np.log10(3e15), 41)
    zsel = [0.1, 0.5, 1.0, 2.0]
    idx = [int(np.argmin(np.abs(hc.Z_EVAL - z))) for z in zsel]
    ref = hc.at_masses(lnM, jx["cllog_eval_fid"], M)       # production: log10 Δ, no ν in ρ_m
    ratios = {}
    for name, num in (("cosmocnc_jax", hc.at_masses(lnMc, jx["cnc_eval_fid"], M)),
                      ("colossus", hc.at_masses(lnM, co["col_eval_fid"], M)),
                      ("classy_szlite_linear", hc.at_masses(lnM, jx["cl_eval_fid"], M))):
        ratios[name] = (num / ref)[idx].tolist()
    counts = {}
    for n in ("fid", "low", "high"):
        series = {
            "classy_szlite": (jx["z_count"], hc._dNdz(lnM, jx[f"cllog_count_{n}"], jx[f"vol_cl_count_{n}"], 5e13)),
            "classy_szlite_linear": (jx["z_count"], hc._dNdz(lnM, jx[f"cl_count_{n}"], jx[f"vol_cl_count_{n}"], 5e13)),
            "cosmocnc_jax": (jx["z_count_cnc"], hc._dNdz(lnMc, jx[f"cnc_count_{n}"], jx[f"vol_cnc_count_{n}"], 5e13)),
            "colossus": (jx["z_count"], hc._dNdz(lnM, co[f"col_count_{n}"], co[f"vol_col_count_{n}"], 5e13)),
        }
        counts[n] = {c: hc._integrate(z, f, 0.01, 3.0) for c, (z, f) in series.items()}
    return dict(M=M.tolist(), z=zsel, ratios=ratios, counts_5e13=counts,
                points={k: list(v) for k, v in hc.POINTS.items()})


def counts_part():
    from classy_szlite.sim import mmf, summaries as sm
    cm = json.load(open(f"{OUT}/counts/counts_modes.json"))
    ps = json.load(open(f"{OUT}/counts/paper_sigma.json"))
    out = dict(theta=list(mmf.THETAS15), sigma_paper=ps["sigma_paper_at_thetas15"],
               q_edges=list(sm.Q_EDGES), z_edges=list(sm.Z_EDGES), tiles=cm["tiles"],
               tiles_40pc=cm["tiles_40pc"], modes={})
    for mode in ("goal", "baseline"):
        m = cm["modes"][mode]
        out["modes"][mode] = dict(
            sigma_analytic=m["sigma_analytic"], sigma_map=m["sigma_map"],
            map_counts=m["map_counts_mean"], map_total=m["map_total_mean"],
            map_total_sem=m["map_total_sem"],
            cat_analytic_k0=m["catalogue"]["analytic_k0"], cat_map_k0=m["catalogue"]["map_k0"],
            cat_map_k1=m["catalogue"]["map_k1"],
            totals={k: m["catalogue_total"][k] for k in ("analytic_k0", "map_k0", "map_k1", "map_k2")})
    return out


def optbias_part():
    out = {}
    for mode in ("goal", "baseline"):
        d = np.load(f"{OUT}/optbias/optbias_{mode}.npz")
        qt, th = d["q_true"], d["theta500"]
        res = {}
        for cov in ("an", "est"):
            qn = d[f"{cov}_qn_max"].astype(float)          # with noise: max over 7x7 pixels and 15 sizes
            qs = d[f"{cov}_qs_max"].astype(float)          # the same without noise
            qd = d[f"{cov}_q_det"].astype(float)           # matched detection (NaN if none)
            rows = []
            for a in np.arange(1.0, 12.0, 1.0):
                s = (qt >= a) & (qt < a + 1.0)
                n = int(s.sum())
                d1, d2 = qn[s] - qs[s], qn[s] - qt[s]
                rows.append(dict(q=qt[s].mean(), n=n,
                                 ex_peak=d1.mean(), ex_peak_se=d1.std(ddof=1) / np.sqrt(n),
                                 ex_true=d2.mean(), ex_true_se=d2.std(ddof=1) / np.sqrt(n),
                                 k2_peak=np.mean(np.sqrt(qs[s] ** 2 + 2) - qs[s]),
                                 k3_peak=np.mean(np.sqrt(qs[s] ** 2 + 3) - qs[s])))
            det = np.nan_to_num(qd, nan=-np.inf) > 5
            comp = []
            for a in np.arange(2.0, 9.0, 0.5):
                s = (qt >= a) & (qt < a + 0.5)
                n = int(s.sum())
                p = float(det[s].mean())
                comp.append(dict(q=qt[s].mean(), n=n, p=p, se=np.sqrt(max(p * (1 - p), 0.25 / n) / n)))
            loss = []
            e3 = np.geomspace(0.5, 6.0, 7)
            for a, b in zip(e3[:-1], e3[1:]):
                s = (th >= a) & (th < b)
                ls = 1.0 - qs[s] / qt[s]
                loss.append(dict(theta=np.sqrt(a * b), n=int(s.sum()), loss=100 * ls.mean(),
                                 se=100 * ls.std(ddof=1) / np.sqrt(s.sum())))
            res[cov] = dict(excess=rows, completeness=comp, grid_loss=loss)
        out[mode] = res
    qg = np.linspace(1.0, 12.0, 111)
    out["curves"] = dict(q=qg.tolist(),
                         **{f"k{k}": (np.sqrt(qg ** 2 + k) - qg).tolist() for k in (1, 2, 3)})
    qc = np.linspace(2.0, 9.0, 141)
    out["completeness_curves"] = dict(q=qc.tolist(),
                                      **{f"k{k}": norm.cdf(np.sqrt(qc ** 2 + k) - 5).tolist()
                                         for k in (0, 2, 3)})
    out["pixel_loss_estimate"] = dict(theta=PIXLOSS_THETA, **PIXLOSS)
    return out


def masking_part():
    from classy_szlite.sim import mmf
    mk = json.load(open(f"{OUT}/mask/mask_tiles.json"))
    out = dict(theta=list(mmf.THETAS15), tiles=mk["tiles"], earlier=EARLIER,
               finder_seconds=mk["finder_seconds"], modes={})
    for mode in ("goal", "baseline"):
        m = mk["modes"][mode]
        P = m["passes"]
        tot = [np.asarray(p["total"], float) for p in P]
        out["modes"][mode] = dict(
            ratio=[p["sigma_ratio_mean"] for p in P], sem=[p["sigma_ratio_sem"] for p in P],
            f_masked=P[1]["f_masked"], n_masked=P[1]["n_masked"], f_masked_2=P[2]["f_masked"],
            totals=[float(t.mean()) for t in tot],
            totals_sem=[float(t.std(ddof=1) / np.sqrt(t.size)) for t in tot],
            iter2_sigma_maxdev=m["iter2_vs_iter1_sigma_maxdev"],
            iter2_count_diffs=m["counts_change"]["iter2_minus_iter1_per_tile"])
    sc = json.load(open(f"{OUT}/mask/mask_scatter.json"))
    out["scatter"] = dict(tiles=sc["tiles"], modes={k: dict(first=v["first"], masked=v["masked"])
                                                    for k, v in sc["modes"].items()})
    return out


def lensing_part():
    """σ_κ(θ500): ours on the tSZ-deprojected curves (production, baseline and goal), the
    paper's baseline table, and ours on the standard (deproj0) curve; the 32-tile stacked p."""
    import lens_table as lt
    from classy_szlite.sim import lens_tables
    D = os.path.join(os.environ.get("MAPSIM_REF", "/mnt/ceph/users/bbolliet/mapsim/ref"), "so_lensing_noise")
    ref = np.load(os.path.join(D, "so_sim_lensing_mf_noise.npy"))
    th_ref = ref[0] / lt.ARCMIN
    d0 = np.loadtxt(os.path.join(D, "nlkk_v3_1_0_deproj0_SENS1_fsky0p4_qe_lT30-3000_lP30-5000.dat"))
    sel = (d0[:, 0] >= lt.LMIN) & (d0[:, 0] <= lt.LMAX)
    s0 = [lt.sigma_kappa(t, d0[sel, 0], d0[sel, 7]) for t in th_ref]
    keep = (lens_tables.THETA500_ARCMIN >= 0.3) & (lens_tables.THETA500_ARCMIN <= 15.0)
    return dict(theta=lens_tables.THETA500_ARCMIN[keep].tolist(),
                baseline=lens_tables.SIGMA_KAPPA["baseline"][keep].tolist(),
                goal=lens_tables.SIGMA_KAPPA["goal"][keep].tolist(),
                ref_theta=th_ref.tolist(), ref_sigma=ref[1].tolist(), deproj0_theta=th_ref.tolist(),
                deproj0_baseline=s0, stacked=json.load(open(f"{OUT}/lensing/lens_check.json")))


def mss2_part():
    """MSS2 noise maps on Perlmutter: white depth per band from the noise spectra and from
    the inverse-variance maps (coadd, split 0), SO's 5-year levels, and the mnms simulations
    against the raw split-0 spectra (job output copied to $MAPSIM_OUT/mss2)."""
    from classy_szlite.sim import instrument as ins
    m = json.load(open(f"{OUT}/mss2/mss2_noise_check.json"))
    edges = np.asarray(m["ell_edges"])
    lc = np.sqrt(edges[1:] * edges[:-1])
    out = dict(bands=m["bands"], freqs=ins.SO_LAT["freqs"].tolist(), so_baseline=ins.SO_LAT["white"]["baseline"].tolist(),
               so_goal=ins.SO_LAT["white"]["goal"].tolist(), patches=len(m["patches"]), per_band={})
    for b in m["bands"]:
        d = {}
        for split in ("coadd", "split0"):
            sp = m["spectra"][f"{b}_{split}"]
            d[f"{split}_spectra"] = [x["white_measured"] for x in sp]
            d[f"{split}_ivar"] = [x["white_from_ivar"] for x in sp]
        raw = np.mean([x["cl"] for x in m["spectra"][f"{b}_split0"]], axis=0)
        sim = np.mean([x["cl"] for x in m["sims"][b]], axis=0)
        d["sim_over_raw"] = {f"{lo}-{hi}": float(np.mean(sim[(lc >= lo) & (lc < hi)] / raw[(lc >= lo) & (lc < hi)]))
                             for lo, hi in ((300, 1000), (1000, 3000), (3000, 5000), (5000, 6000))}
        out["per_band"][b] = d
    return out


def noise_test_part(streak_tile=79, streak_split=1):
    """MSS2 noise through the finder against mnms and Gaussian noise (mss2_finder_test.py
    --summary writes finder_test/summary.json), the noise maps' non-Gaussianity (mss2_tiles.py,
    tile_diagnostics.json: each map against its own Gaussian copy, paired by tile), and one
    split's local noise level with its noise peaks, for the picture of a streak."""
    from scipy.ndimage import gaussian_filter
    from classy_szlite.sim.flatsky import N, ARCMIN
    s = json.load(open(f"{OUT}/mss2/finder_test/summary.json"))
    td = json.load(open(f"{OUT}/mss2/tile_diagnostics.json"))
    stats = td["stats"]
    ng = {}
    for b in ("f090", "f150"):
        for li, (lo, hi) in enumerate(td["lbands"]):
            row = {}
            for name in ("K", "tail", "K0"):
                si = stats.index(name)
                per = {src: np.array([np.mean([r[li][si] for r in t["diag"][b][src]]) for t in td["tiles"]])
                       for src in ("raw", "mnms", "gauss_raw", "gauss_mnms")}
                for lab, x in (("raw", per["raw"] - per["gauss_raw"]), ("mnms", per["mnms"] - per["gauss_mnms"])):
                    row[f"{name}_{lab}"] = [float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))]
                row[f"{name}_ref"] = float(per["gauss_raw"].mean())
            si = stats.index("bp")
            ratio = np.array([np.mean([r[li][si] for r in t["diag"][b]["mnms"]]) / np.mean([r[li][si] for r in t["diag"][b]["raw"]])
                              for t in td["tiles"]])
            row["power_mnms_over_raw"] = [float(ratio.mean()), float(ratio.std(ddof=1) / np.sqrt(ratio.size))]
            ng[f"{b}_{lo}-{hi}"] = row
    # the streak picture: local noise level (ell 1000-5000, 30' smoothing) of one split and of the coadd depth
    d = np.load(f"{OUT}/mss2/tiles/tile_{streak_tile:03d}.npz")
    meta = json.loads(str(d["meta"]))
    mh = meta["margin_hf"]
    k = 2 * np.pi * np.fft.fftfreq(N, d=0.5 * ARCMIN)
    L = np.hypot(k[:, None], 2 * np.pi * np.fft.rfftfreq(N, d=0.5 * ARCMIN)[None, :])
    tp = np.outer(*(2 * [np.minimum(1.0, np.minimum(np.arange(N) + 0.5, N - 0.5 - np.arange(N)) / 60.0)]))

    def local(m):
        f = np.fft.irfft2(np.fft.rfft2((m - m.mean()) * tp) * ((L > 1000) & (L < 5000)), s=(N, N))
        r = np.sqrt(gaussian_filter(f * f, 30.0 / np.sqrt(8 * np.log(2)) / 0.5))
        return r / np.median(r[120:-120, 120:-120])

    def down(a, f=8):
        return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean((1, 3))

    def u8(a, lo, hi):
        return base64.b64encode(np.clip(np.round((a - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8).tobytes()).decode()

    split_map = local(d["raw_hf"][streak_split, 1, mh:mh + N, mh:mh + N].astype(np.float64))
    iv = d["ivar2"][3, 60:360, 60:360].astype(np.float64)
    depth = np.kron(1.0 / np.sqrt(iv / np.median(iv)), np.ones((4, 4)))
    tiles = []
    for f in sorted(glob.glob(f"{OUT}/mss2/finder_test/part_*_of_*.json")):
        tiles += json.load(open(f))
    geo = []
    for f in sorted(glob.glob(f"{OUT}/mss2/finder_test/geometry_part_*_of_*.json")):
        geo += json.load(open(f))
    rs = {d: np.mean([np.mean(t[d]["iso_per"]["s1"], 0) / np.mean(t[d]["iso_cut"]["s1"], 0) for t in geo], 0)
          for d in ("split", "coadd")}
    tr = [t for t in tiles if t["tile"] == streak_tile][0]
    qtrue = lambda p, d="split": p[3] / rs[d][p[2]]
    pk = [p for p in tr["split"]["raw"]["peaks"][streak_split] if qtrue(p) > 4.5]
    lo_, hi_ = 0.6, 2.4
    s["streak"] = dict(tile=streak_tile, split=streak_split, ra=meta["ra"], dec=meta["dec"], size=150, lo=lo_, hi=hi_,
                       split_rms=u8(down(split_map)[::-1], lo_, hi_), depth=u8(down(depth)[::-1], lo_, hi_),
                       split_max=float(split_map[120:-120, 120:-120].max()),
                       peaks=[[round((N - 1 - p[0]) / 8.0, 1), round(p[1] / 8.0, 1), round(qtrue(p), 2)] for p in pk],
                       q_min=4.5, n_peaks=len(pk),
                       n_peaks_by_split=[sum(1 for p in tr["split"]["raw"]["peaks"][i] if qtrue(p) > 4.5) for i in range(4)])
    s["nongauss"] = ng
    # on the simulator's q scale: the geometry run gives sigma_periodic / sigma_cut per scale (rs). The
    # run with the masking threshold at 4.3 (finder_test_qmask4.3) masks less than 4 does on periodic
    # tiles (about 4.2 would match, from the first-pass ratio), and the run at 4 masks more
    v3 = []
    for f in sorted(glob.glob(f"{OUT}/mss2/finder_test_qmask4.3/part_*_of_*.json")):
        v3 += json.load(open(f))
    run = v3 if len(v3) == len(tiles) else tiles
    rng = np.random.default_rng(1)
    ix = rng.integers(0, len(run), (2000, len(run)))
    thr = (4.0, 4.5, 5.0)
    trueq = dict(source="q_mask 4.3" if run is v3 else "q_mask 4", n_tiles=len(run), thresholds=list(thr), depth={})
    for depth in ("split", "coadd"):
        # thresholds on the cut-outs that give the isotropic Gaussian noise the counts measured for
        # the same noise on periodic tiles (the per-scale sigma ratio alone, measured at q_mask 4,
        # leaves q 0.8-0.9% too low at q_mask 4.3, which is ~15% in the steep noise-peak counts)
        target = np.mean([np.mean(t[depth]["iso_per"]["counts"], 0)[:3] for t in geo], 0)
        iso_q = [[np.array([p[3] / rs[depth][p[2]] for p in m]) for m in t[depth]["iso"]["peaks"]] for t in run]
        count = lambda x: np.mean([np.mean([np.sum(q > x) for q in maps]) for maps in iso_q])
        teff = []
        for i_, x in enumerate(thr):
            lo, hi = x - 0.8, x + 0.8
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if count(mid) > target[i_] else (lo, mid)
            teff.append(0.5 * (lo + hi))
        per = {}
        for k in ("raw", "mnms", "gauss", "gaussmod", "iso"):
            per[k] = np.array([np.mean([[sum(1 for p in m if p[3] / rs[depth][p[2]] > x) for x in teff]
                                        for m in t[depth][k]["peaks"]], axis=0) for t in run])
        dd = {}
        for k, c in per.items():
            r1 = np.array([np.mean(t[depth][k]["s1"], 0) / np.mean(t[depth]["raw"]["s1"], 0) for t in run])
            r0 = np.array([np.mean(t[depth][k]["s0"], 0) / np.mean(t[depth]["raw"]["s0"], 0) for t in run])
            dd[k] = dict(counts=c.mean(0).tolist(), counts_err=c[ix].mean(1).std(0, ddof=1).tolist(),
                         over_raw=(c.sum(0) / per["raw"].sum(0)).tolist(),
                         over_raw_err=(c[ix].sum(1) / np.maximum(per["raw"][ix].sum(1), 1e-9)).std(0, ddof=1).tolist(),
                         sigma_final=r1.mean(0).tolist(), sigma_final_err=r1[ix].mean(1).std(0, ddof=1).tolist(),
                         sigma_first=r0.mean(0).tolist(),
                         f_masked=float(np.mean([np.mean(t[depth][k]["f_masked"]) for t in run])))
            if k + "_inj" in run[0][depth]:
                rq = np.array([np.mean(np.ravel(t[depth][k + "_inj"]["qs_ref"])) / np.mean(np.ravel(t[depth]["raw_inj"]["qs_ref"]))
                               for t in run])
                dd[k]["cluster_q_over_raw"] = [float(rq.mean()), float(rq[ix].mean(1).std(ddof=1))]
        ip = np.array([np.mean(t[depth]["iso_per"]["counts"], 0)[:3] for t in geo])
        trueq["depth"][depth] = dict(kinds=dd, iso_periodic_measured=ip.mean(0).tolist(),
                                     sigma_periodic_over_cut=rs[depth].tolist(), thresholds_on_cut=teff)
    s["true_q"] = trueq
    return s


def datavector_part():
    """The simulator's sky in MSS2 noise and in model noise: data vectors compared through the
    production pipeline (mss2_datavector_test.py --summary writes datavector_test/summary.json).
    Adds the 40%-sky Poisson error per q bin of MSS2's counts."""
    s = json.load(open(f"{OUT}/mss2/datavector_test/summary.json"))
    n40 = 0.4 * 41253.0 / 64.0                        # 64-deg2 interiors in 40% of the sky
    for dp in s["depth"].values():
        dp["raw"]["poisson40_q"] = [float(1.0 / np.sqrt(n40 * c)) for c in dp["raw"]["q_counts"]]
        dp["raw"]["poisson40_total"] = float(1.0 / np.sqrt(n40 * dp["raw"]["detections"]))
    return s


def szifi_part():
    """Our finder against SZiFi on the MSS2 comparison maps: mss2_szifi_compare.py's summary.json,
    and the follow-up's summary_ch4.json (our finder with the four high-frequency channels) and
    noise_summary.json (noise after masking against the maps without clusters) when present."""
    base = f"{OUT}/mss2/szifi_compare"
    meta = json.load(open(f"{base}/meta.json"))
    out = dict(thetas=meta["thetas"], tiles=meta["tiles"], summary=json.load(open(f"{base}/summary.json")))
    for name, fn in (("ch4", "summary_ch4.json"), ("noise", "noise_summary.json")):
        if os.path.exists(f"{base}/{fn}"):
            out[name] = json.load(open(f"{base}/{fn}"))
    return out


def npe_part(tag="n10000_s0"):
    """Catalogue-level NPE: catalogue_npe.py's results (bias at the true point, coverage, TARP)."""
    return json.load(open(f"{OUT}/catalogue_npe/npe_{tag}.json"))


def priors_part():
    pc = json.load(open(f"{OUT}/priors/prior_corners.json"))
    out = dict(fiducial=pc["fiducial"], boxes={})
    for name, b in pc["boxes"].items():
        rel = [c["halos"] / pc["fiducial"]["halos"] for c in b["corners"]]
        h = b["heaviest"]
        out["boxes"][name] = dict(box=b["box"], halos_rel_min=min(rel), halos_rel_max=max(rel),
                                  halos_5e13=[min(c["halos_5e13"] for c in b["corners"]),
                                              max(c["halos_5e13"] for c in b["corners"])],
                                  ln10As=[min(c["ln10As"] for c in b["corners"]),
                                          max(c["ln10As"] for c in b["corners"])],
                                  average=h.get("prior_average"),
                                  peaks_max=max(h["n_peaks"]), peak_cap=h["peak_cap"],
                                  detections=h["detections"], overflow=any(h["overflow"]))
    return out


def tile_part(width=240):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import spike
    from classy_szlite.sim import halos as hl, mmf, paint as pt, simulator as simu, summaries as sm
    from classy_szlite.sim.flatsky import N
    theta = simu.THETA_FID
    cells, tb, cs, cfg, st, info = spike.setup(theta)

    @jax.jit
    def run(key):
        k_h, k_m = jax.random.split(key)
        halos = hl.sample_halos(k_h, cs.lam, cells, cfg.halo_cap, cs.cg, theta.h, theta.A_SZ,
                                theta.alpha_SZ, theta.sigma_lnq, theta.one_minus_b, d2N=cs.d2N)
        tabs = tb._replace(tab=st.paint_tab, rP=st.paint_rP)
        yk, _ = pt.paint_y(halos, tabs, cfg.paint, st.T_tab)
        X = simu.channel_maps(k_m, yk, st)
        # first pass, then one masking iteration (as simulator.find_clusters), keeping the mask
        q, _, C0 = mmf.q_cube(X, st.fs)
        mask, _ = mmf.cluster_mask(mmf.find_peaks(q, cfg.peak_cap), st.fs, cfg.q_mask)
        C1, _ = mmf.masked_cov(X, mask, C0, st.fs)
        q, sigma, _ = mmf.q_cube(X, st.fs, C=C1)
        pk = mmf.find_peaks(q, cfg.peak_cap)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        ymap = jnp.fft.irfft2(yk * st.B_full[3], s=(N, N))      # 145 GHz beam, 1.4'
        t145 = jnp.fft.irfft2(X[3], s=(N, N))                    # 145 GHz map [μK]
        counts = sm.histogram(pk.q, cf.z, keep & cf.ok)
        return halos, ymap, t145, q.max(0), sigma, pk, keep, cf, counts, mask

    key = spike.tile_keys(1)[0]                                   # tile 0 of counts_modes.py
    halos, ymap, t145, qmax, sigma, pk, keep, cf, counts, mask = jax.device_get(run(key))
    valid = np.asarray(halos.valid)
    lnM = np.asarray(halos.lnM)
    big = np.flatnonzero(valid & (lnM >= np.log(5e13)))
    # centre the picture on the most massive halo
    i0 = big[np.argmax(lnM[big])]
    cy = int(round((halos.iy[i0] + halos.oy[i0]) / 2.0)) % N
    cx = int(round((halos.ix[i0] + halos.ox[i0]) / 2.0)) % N
    h = width // 2

    def cut(a):
        return np.roll(np.asarray(a), (h - cy, h - cx), axis=(0, 1))[:width, :width]

    def pos(py, px):
        return (np.asarray(py) - cy + h) % N, (np.asarray(px) - cx + h) % N

    def u8(a):
        return base64.b64encode(np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8).tobytes()).decode()

    y = cut(ymap)
    y_scale = 2e-6
    y_top = float(np.quantile(np.arcsinh(np.maximum(y, 0) / y_scale), 0.9995))
    t = cut(t145)
    t_lim = 250.0
    qm = cut(qmax)
    q_lim = 10.0
    sig = np.asarray(sigma)
    thetas = np.asarray(mmf.THETAS15)
    hy, hx = pos((halos.iy[big] + halos.oy[big]) / 2.0, (halos.ix[big] + halos.ox[big]) / 2.0)
    th5 = np.asarray(halos.theta500)[big]
    qexp = np.asarray(halos.y0)[big] / np.exp(np.interp(np.log(th5), np.log(thetas), np.log(sig)))
    inside = (hy < width) & (hx < width)
    halo_list = [dict(x=float(a), y=float(b), M=float(np.exp(m)), z=float(zz), theta500=float(tt),
                      q_expected=float(qq))
                 for a, b, m, zz, tt, qq in zip(hx[inside], hy[inside], lnM[big][inside],
                                                np.asarray(halos.z)[big][inside], th5[inside],
                                                qexp[inside])]
    ok = np.asarray(pk.valid) & np.asarray(keep)
    py, px = pos(np.asarray(pk.iy)[ok], np.asarray(pk.ix)[ok])
    ins = (py < width) & (px < width)
    conf = np.asarray(cf.ok)[ok]
    peak_list = [dict(x=float(b), y=float(a), q=float(q), theta=float(thetas[j]), confirmed=bool(c),
                      z=float(zz) if c else None)
                 for a, b, q, j, c, zz in zip(py[ins], px[ins], np.asarray(pk.q)[ok][ins],
                                              np.asarray(pk.j)[ok][ins], conf[ins],
                                              np.asarray(cf.z)[ok][ins])]
    return dict(width=width, pixel_arcmin=0.5,
                y=dict(data=u8(np.arcsinh(np.maximum(y, 0) / y_scale) / y_top), scale=y_scale,
                       top=y_top, max=float(y.max())),
                t145=dict(data=u8(0.5 + 0.5 * np.clip(t / t_lim, -1, 1)), lim=t_lim),
                q=dict(data=u8(np.clip(qm, 0, q_lim) / q_lim), lim=q_lim),
                mask=dict(data=u8(1.0 - cut(mask)), masked_tile=float(1.0 - np.asarray(mask).mean())),
                halos=halo_list, peaks=peak_list,
                tile=dict(halos=int(valid.sum()), halos_5e13=int(big.size), kept=int(ok.sum()),
                          in_data_vector=int(np.asarray(counts).sum())))


def main():
    out = dict(hmf=hmf_part(), counts=counts_part(), optbias=optbias_part(), priors=priors_part(),
               masking=masking_part(), lensing=lensing_part(), mss2=mss2_part(), noise_test=noise_test_part(),
               datavector=datavector_part())
    out["tile"] = tile_part()
    if os.path.exists(f"{OUT}/mss2/szifi_compare/summary.json"):
        out["szifi"] = szifi_part()
    if os.path.exists(f"{OUT}/catalogue_npe/npe_n10000_s0.json"):
        out["npe"] = npe_part()
    os.makedirs(f"{OUT}/diagnostics", exist_ok=True)
    path = f"{OUT}/diagnostics/diagnostics.json"
    with open(path, "w") as fh:
        json.dump(rnd(out), fh, separators=(",", ":"))
    print(f"saved {path} ({os.path.getsize(path) / 1e3:.0f} kB)")
    t = out["tile"]["tile"]
    print(f"tile: {t}; peaks in the picture {len(out['tile']['peaks'])}, "
          f"massive halos {len(out['tile']['halos'])}")


if __name__ == "__main__":
    main()
