"""Numbers behind the mapsim progress page, gathered into one JSON file.

    python benchmarks/mapsim/diagnostics_data.py

Reads the outputs of counts_modes.py, paper_sigma.py, optbias.py,
hmf_compare.py, prior_corners.py and mask_check.py under $MAPSIM_OUT,
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
            totals={k: m["catalogue_total"][k] for k in ("analytic_k0", "map_k0", "map_k1")})
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
    return out


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
               masking=masking_part())
    out["tile"] = tile_part()
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
