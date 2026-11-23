"""The figures of the cluster-count pages (docs/clusters/figures/*.png) and the numbers of their tables
(docs/clusters/figures/numbers.json), from the pipeline's outputs (public forecast-noise runs).

    python docs/clusters/make_figures.py RESULTS_DIR

RESULTS_DIR holds the published runs' files in this layout (the research scripts' output folders):

    maps_npe_forecast/          truth.npz, npe_t1.json, npe_t1_samples.npz, npe_e{0-4}_applied_{mocks,held}.npz
    maps_npe_forecast_t168/     npe_map_e{11-15}_applied_{mocks,held}.npz
    catalogue_npe/forecast41/   truth.npz, npe_t2.json, npe_t2_samples.npz           (published model)
    catalogue_npe/calibf41/     npe_t2cal*                                          (calibrated at one point)
    catalogue_npe/calibtf41/    npe_t2calt*                                         (calibrated, linear in theta)
    catalogue_npe/calibcf41/    truth.npz, npe_t2calc*                              (calibrated, noise level)
    catalogue_npe/*_t168/       npe_{plain,fixed,noise}_e*_applied_*.npz           (40% of the sky)
    qcal_grid_forecast/         calib_grid.npz, points/point_00.npz                 (the 25-point calibration)

The one-tile figure runs classy_szlite.sim.tile.stages (about 1 min on a laptop the first time).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
C = dict(map="#2a78d6", published="#eb6834", point="#1baf7a", noise="#6250d6", theta="#52514e", ink="#14171c")
LABEL = dict(map="map level", published="catalogue, as published", point="catalogue, calibrated at one point",
             theta=r"catalogue, calibrated (linear in $\theta$)", noise="catalogue, calibrated through the noise level")
P4 = [r"$\sigma_8$", r"$\omega_\mathrm{c}$", r"$A_\mathrm{SZ}$", r"$\alpha_\mathrm{SZ}$"]
NAMES10 = P4 + [r"$h$", r"$\omega_\mathrm{b}$", r"$n_\mathrm{s}$", r"$\sigma_\mathrm{SZ}$", r"$\beta_\mathrm{lens}$",
                r"$\sigma_\mathrm{lens}$"]
# numbers.json keys of the thirds (kept as they were) and their mathtext labels
THIRDS = (("ω_c", r"$\omega_\mathrm{c}$"), ("A_SZ", r"$A_\mathrm{SZ}$"), ("α_SZ", r"$\alpha_\mathrm{SZ}$"))
WIDTHS = "bias  [posterior widths]"
plt.rcParams.update({"figure.dpi": 220, "savefig.dpi": 220, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e8ec", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.facecolor": "white"})
NUM = {}


def plain_log_ticks(axis, ticks):
    """Explicit major ticks with plain labels on a log axis, and no minor labels."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def summary(f):
    d = np.load(f)
    return {k: d[k] for k in ("theta", "mu", "sd", "rank")}


def ensemble(files):
    S = [summary(f) for f in files]
    mus = np.stack([s["mu"] for s in S]); sds = np.stack([s["sd"] for s in S])
    return dict(theta=S[0]["theta"], mu=mus.mean(0), sd=np.sqrt((sds ** 2).mean(0) + mus.var(0)),
                rank=np.stack([s["rank"] for s in S]).mean(0))


def bias(s):
    b = (s["mu"] - s["theta"]) / s["sd"]
    return b.mean(0), b.std(0, ddof=1) / np.sqrt(len(b))


LEV = np.linspace(0.1, 0.9, 9)


def coverage(s):
    return np.array([np.mean(np.abs(s["rank"] - 0.5) < lv / 2, axis=0) for lv in LEV])


# ---------------------------------------------------------------------------------------------------
def fig_tile():
    import jax
    jax.config.update("jax_enable_x64", True)
    from classy_szlite.sim import tile
    o = tile.stages(log=lambda *a: None)
    down = lambda a: a.reshape(600, 2, 600, 2).mean((1, 3))
    o = dict(o, y=o["y"] * 1e5)
    panels = [("y", r"Compton-$y$ $\times 10^5$ (painted haloes)", "magma", 0, float(np.quantile(o["y"], 0.9995))),
              ("tsz145", "145 GHz: tSZ [μK]", "RdBu_r", -200, 200), ("cmb145", "145 GHz: CMB [μK]", "RdBu_r", -200, 200),
              ("noise145", "145 GHz: noise [μK]", "RdBu_r", -200, 200), ("m145", "145 GHz: sum [μK]", "RdBu_r", -200, 200),
              ("q_first", "S/N, first pass", "viridis", -3, 8), ("mask", "masking-pass discs", "gray", 0, 1),
              ("q_final", "S/N, after masking", "viridis", -3, 8)]
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.8))
    for a, (k, t, cm, lo, hi) in zip(ax.ravel(), panels):
        im = a.imshow(down(o[k]), cmap=cm, vmin=lo, vmax=hi, origin="lower", extent=[0, 10, 0, 10])
        a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=7)
        if k == "mask":
            cb.ax.set_visible(False)                  # keeps the panel the same size as the others
    fig.suptitle(r"One 10° $\times$ 10° tile at the true parameters, 2019 forecast noise (goal)", fontsize=10, y=0.95)
    save(fig, "tile_stages.png")
    # zoom with detections and haloes
    p, h = o["peaks"], o["halos"]
    S = 240
    best, b0 = -1, (0, 0)
    for y0 in range(0, 1200 - S + 1, 40):
        for x0 in range(0, 1200 - S + 1, 40):
            n = np.sum(p["confirmed"] & (p["q"] > 5) & (p["y"] >= y0) & (p["y"] < y0 + S) & (p["x"] >= x0) & (p["x"] < x0 + S))
            if n > best:
                best, b0 = n, (y0, x0)
    y0, x0 = b0
    fig, a = plt.subplots(figsize=(5.4, 5.4))
    a.imshow(o["q_final"][y0:y0 + S, x0:x0 + S], cmap="viridis", vmin=-3, vmax=10, origin="lower", extent=[0, 2, 0, 2])
    a.grid(False)
    s = 2 / S
    hm = (h["q_model"] > 2) & (h["x"] >= x0) & (h["x"] < x0 + S) & (h["y"] >= y0) & (h["y"] < y0 + S)
    for xx, yy, th in zip(h["x"][hm], h["y"][hm], h["theta500"][hm]):
        a.add_patch(plt.Circle(((xx - x0) * s, (yy - y0) * s), max(th / 60, 0.01), fill=False, ec="white", lw=0.7, ls="--"))
    pm = (p["x"] >= x0) & (p["x"] < x0 + S) & (p["y"] >= y0) & (p["y"] < y0 + S)
    a.scatter((p["x"][pm & p["confirmed"]] - x0) * s, (p["y"][pm & p["confirmed"]] - y0) * s, marker="s", s=40, facecolors="none",
              edgecolors="#ffd166", lw=1.2, label="detection, confirmed")
    a.scatter((p["x"][pm & ~p["confirmed"]] - x0) * s, (p["y"][pm & ~p["confirmed"]] - y0) * s, marker="s", s=40, facecolors="none",
              edgecolors="#ff5c8a", lw=1.2, label="detection, no halo within 1.5′")
    a.plot([], [], ls="--", color="gray", lw=0.8, label=r"halo with model $q > 2$ (radius $\theta_{500}$)")
    a.set_xlabel("degrees"); a.set_ylabel("degrees")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
    a.set_title("S/N after masking, 2° zoom: detections and haloes", fontsize=9)
    save(fig, "tile_zoom.png")
    NUM["tile"] = dict(halos=int(len(h["z"])), detections5=int(np.sum(p["confirmed"] & (p["q"] > 5))),
                       noise_ratio_u=float(o["sigma_final"][7] / o["sigma_model"][7]),
                       noise_ratio_first=float(o["sigma_first"][7] / o["sigma_model"][7]))
    return o


def fig_noise(R, o):
    """The finder's noise over the instrument-plus-CMB noise, against filter size."""
    g = np.load(os.path.join(R, "qcal_grid_forecast", "calib_grid.npz"))
    th = np.geomspace(0.5, 15, 15)
    ratio_fid = g["sigma"][0] / o["sigma_model"][None]
    fig, a = plt.subplots(figsize=(6.2, 3.6))
    a.fill_between(th, np.quantile(ratio_fid, 0.05, 0), np.quantile(ratio_fid, 0.95, 0), color=C["map"], alpha=0.2, lw=0)
    a.plot(th, ratio_fid.mean(0), color=C["map"], lw=2.2, label="after the masking pass (mean of 41 tiles, 5–95% band)")
    a.plot(th, o["sigma_first"] / o["sigma_model"], color=C["published"], lw=1.8, ls="--", label="first pass, before masking (one tile)")
    a.axhline(1, color="gray", ls=":", lw=1.2, label="instrument + CMB only (catalogue model)")
    a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4, 8, 15]); a.set_xticklabels(["0.5′", "1′", "2′", "4′", "8′", "15′"])
    a.set_xlabel(r"filter size $\theta$"); a.set_ylabel(r"$\sigma_\mathrm{finder}/\sigma_\mathrm{instrument+CMB}$"); a.set_ylim(0.97, 1.34)
    a.legend(fontsize=8.5, loc="center", bbox_to_anchor=(0.5, 0.47))     # between the two curves
    save(fig, "finder_noise.png")
    NUM["noise"] = dict(ratio_mean=ratio_fid.mean(0).round(3).tolist())


def fig_grid(R):
    """The calibration at the 25 points: search lift and scatter against the finder's noise level."""
    g = np.load(os.path.join(R, "qcal_grid_forecast", "calib_grid.npz"))
    u = g["sigma"][:, :, 7].mean(1) / g["sigma"][0, :, 7].mean()
    P = g["theta"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for a, (c, lab) in zip(ax, ((1, r"search lift $k$"), (2, r"scatter $w$"))):
        sc = a.scatter(u, g["rows"][:, 1, c], c=P[:, 2], cmap="cividis", s=36, edgecolors="k", linewidths=0.3)
        a.scatter(u[0], g["rows"][0, 1, c], s=140, facecolors="none", edgecolors="#d03b3b", lw=1.5, label="the true parameters")
        a.set_xlabel("finder noise at 2.7′ relative to the true parameters"); a.set_ylabel(lab + r"  ($0.31 < z < 0.61$)")
        a.legend(fontsize=8.5, loc="upper right" if c == 1 else "upper left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02); cb.set_label(r"$A_\mathrm{SZ}$ of the point")
    save(fig, "calibration_grid.png")
    NUM["grid"] = dict(noise_range=[float(u.min()), float(u.max())], k_range=[float(g["rows"][:, 1, 1].min()), float(g["rows"][:, 1, 1].max())],
                       w_range=[float(g["rows"][:, 1, 2].min()), float(g["rows"][:, 1, 2].max())])


def fig_relation(R):
    d = np.load(os.path.join(R, "qcal_grid_forecast", "points", "point_00.npz"))
    qm, qo = d["q_model"].astype(float), d["q_obs"].astype(float)
    from scipy.stats import norm
    edges = np.geomspace(1.5, 50, 26)
    rows = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (qm >= a_) & (qm < b_)
        det = m & np.isfinite(qo) & (qo > 4)
        if m.sum() < 50:
            continue
        r = [np.sqrt(a_ * b_), m.sum(), np.mean(m & np.isfinite(qo) & (qo > 5)) / np.mean(m)]
        r += list(np.quantile(qo[det], [0.16, 0.5, 0.84])) if det.sum() > 30 else [np.nan] * 3
        rows.append(r)
    rows = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    a = ax[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(np.flatnonzero((qm > 3) & np.isfinite(qo)), 2500, replace=False)
    a.scatter(qm[sel], qo[sel], s=2, color="gray", alpha=0.35, lw=0)
    ok = np.isfinite(rows[:, 4])
    a.fill_between(rows[ok, 0], rows[ok, 3], rows[ok, 5], color=C["map"], alpha=0.25, lw=0)
    a.plot(rows[ok, 0], rows[ok, 4], color=C["map"], lw=2, label="finder: median and 16–84%")
    a.plot([3, 50], [3, 50], color=C["published"], ls="--", label=r"catalogue model: $q_\mathrm{obs} = q$")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(3, 50); a.set_ylim(3, 50)
    plain_log_ticks(a.xaxis, [3, 4, 6, 10, 20, 40]); plain_log_ticks(a.yaxis, [3, 4, 6, 10, 20, 40])
    a.set_xlabel(r"catalogue model's $q$ of the halo"); a.set_ylabel(r"finder's $q_\mathrm{obs}$ (detected)"); a.legend(fontsize=8)
    a.set_title(r"Observed $q_\mathrm{obs}$ against the model's $q$, halo by halo", fontsize=9)
    a = ax[1]
    qs = np.geomspace(1.5, 12, 100)
    a.plot(qs, norm.cdf(qs - 5), color=C["published"], lw=1.8, label=r"catalogue model $\Phi(q - 5)$")
    m = rows[:, 0] < 12
    a.scatter(rows[m, 0], rows[m, 2], color=C["map"], s=18, zorder=3, label="finder on maps")
    a.set_xscale("log"); plain_log_ticks(a.xaxis, [2, 3, 4, 6, 8, 12])
    a.set_xlabel(r"catalogue model's $q$ of the halo"); a.set_ylabel(r"$P$(detected with $q_\mathrm{obs} > 5$)"); a.legend(fontsize=8)
    a.set_title(r"Chance of a detection above $q_\mathrm{obs} = 5$", fontsize=9)
    save(fig, "finder_relation.png")


def fig_counts(R):
    m = np.load(os.path.join(R, "maps_npe_forecast", "truth.npz"))["x"][:, :20]
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0))
    from classy_szlite.sim.summaries import Q_EDGES, Z_EDGES     # the data vector's bin edges
    zlab = [f"{e:.2f}" for e in Z_EDGES]
    qlab = [f"{e:.1f}".removesuffix(".0") for e in Q_EDGES]
    for a, (name, d, title) in zip(ax, [("published", "forecast41", "as published"), ("noise", "calibcf41", "calibrated through the noise level")]):
        c = np.load(os.path.join(R, "catalogue_npe", d, "truth.npz"))["x"][:, :20]
        r = (m.mean(0) / c.mean(0)).reshape(4, 5)
        out[name] = r.round(3).tolist()
        im = a.imshow(r, cmap="RdBu", vmin=0.8, vmax=1.2, origin="lower")
        for i in range(4):
            for j in range(5):
                a.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=8)
        a.set_xticks(np.arange(6) - 0.5); a.set_xticklabels(zlab, fontsize=8)      # ticks on the bin edges
        a.set_yticks(np.arange(5) - 0.5); a.set_yticklabels(qlab, fontsize=8)
        a.set_xlabel(r"redshift $z$"); a.set_ylabel(r"$q_\mathrm{obs}$"); a.grid(False)
        a.set_title(f"map-level mocks / catalogue model\n{title}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    save(fig, "counts_ratio.png")
    NUM["counts"] = out


def fig_bias(R):
    res = {}
    j = lambda p: json.load(open(os.path.join(R, p)))
    t2, t2cal, t2calc = j("catalogue_npe/forecast41/npe_t2.json"), j("catalogue_npe/calibf41/npe_t2cal.json"), j("catalogue_npe/calibcf41/npe_t2calc.json")
    m10 = ensemble(sorted(glob.glob(os.path.join(R, "maps_npe_forecast", "npe_e*_applied_mocks.npz"))))
    res["10"] = dict(published=[t2["applied"]["maps"]["bias"][:4], t2["applied"]["maps"]["bias_err"][:4]],
                     point=[t2cal["applied"]["maps"]["bias"][:4], t2cal["applied"]["maps"]["bias_err"][:4]],
                     noise=[t2calc["applied"]["maps"]["bias"][:4], t2calc["applied"]["maps"]["bias_err"][:4]],
                     map=[x.tolist()[:4] for x in bias(m10)])
    d40 = {"published": ("forecast41_t168", "plain"), "point": ("calibf41_t168", "fixed"), "noise": ("calibcf41_t168", "noise")}
    res["40"] = {}
    for k, (dd, tag) in d40.items():
        fs = sorted(glob.glob(os.path.join(R, "catalogue_npe", dd, f"npe_{tag}_e*_applied_mocks.npz")))
        res["40"][k] = [x.tolist()[:4] for x in bias(ensemble(fs))]
    res["40"]["map"] = [x.tolist()[:4] for x in bias(ensemble(sorted(glob.glob(os.path.join(R, "maps_npe_forecast_t168", "npe_map_e*_applied_mocks.npz")))))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, (sky, title) in zip(ax, [("10", "10% of the sky (41 tiles)"), ("40", "40.7% of the sky (168 tiles)")]):
        for off, k in zip((-0.27, -0.09, 0.09, 0.27), ("published", "point", "noise", "map")):
            b, e = res[sky][k]
            a.errorbar(np.arange(4) + off, b, yerr=e, fmt="o", ms=5, color=C[k], label=LABEL[k], capsize=0)
        a.axhline(0, color="gray", ls=":", lw=1)
        a.set_xticks(range(4)); a.set_xticklabels(P4); a.set_title(title, fontsize=9)
        a.set_ylim(-9, 3.5)
    ax[0].set_ylabel(WIDTHS)
    ax[0].legend(fontsize=7.5, loc="lower left")         # the left panel has room below the points
    save(fig, "bias.png")
    NUM["bias"] = res


def fig_coverage(R):
    out = {}
    sets = {"point": "catalogue_npe/calibf41/npe_t2cal_applied_mapheld.npz", "theta": "catalogue_npe/calibtf41/npe_t2calt_applied_mapheld.npz",
            "noise": "catalogue_npe/calibcf41/npe_t2calc_applied_mapheld.npz"}
    cov = {k: coverage(summary(os.path.join(R, v))) for k, v in sets.items()}
    cov["map"] = coverage(ensemble(sorted(glob.glob(os.path.join(R, "maps_npe_forecast", "npe_e*_applied_held.npz")))))
    for k, v in cov.items():
        out[k] = dict(c50=v[4, :4].round(2).tolist(), c90=v[8, :4].round(2).tolist())
    c40 = {"noise": ensemble(sorted(glob.glob(os.path.join(R, "catalogue_npe", "calibcf41_t168", "npe_noise_e*_applied_held.npz")))),
           "map": ensemble(sorted(glob.glob(os.path.join(R, "maps_npe_forecast_t168", "npe_map_e*_applied_held.npz"))))}
    f12 = os.path.join(R, "catalogue_npe", "calibf41_t168", "npe_fixed_e12_applied_held.npz")
    if os.path.exists(f12):
        c40["point"] = summary(f12)
    cov40 = {k: coverage(v) for k, v in c40.items()}
    out["40"] = {k: dict(c50=v[4, :4].round(2).tolist(), c90=v[8, :4].round(2).tolist()) for k, v in cov40.items()}
    fig, ax = plt.subplots(2, 4, figsize=(11.5, 6.2), sharex=True, sharey=True)
    for row, (cc, sky) in enumerate(((cov, "10% of the sky"), (cov40, "40.7% of the sky"))):
        for jj in range(4):
            a = ax[row, jj]
            a.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.2)
            for k in ("point", "theta", "noise", "map"):
                if k in cc:
                    a.plot(np.r_[0, LEV, 1], np.r_[0, cc[k][:, jj], 1], color=C[k], lw=2, ls="--" if k == "theta" else "-", label=LABEL[k])
            a.set_title(f"{P4[jj]}, {sky}", fontsize=9.5)
            if row == 1:
                a.set_xlabel(r"credible level $\gamma$")
        ax[row, 0].set_ylabel("fraction of skies covered")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    save(fig, "coverage.png")
    NUM["coverage"] = out
    # bias of sigma8 in thirds of the true parameters, calibrated models
    th = {}
    for k in ("point", "noise"):
        s = summary(os.path.join(R, sets[k]))
        b = (s["mu"] - s["theta"]) / s["sd"]
        th[k] = {}
        for jj, (nm, _) in enumerate(THIRDS, 1):
            e = np.quantile(s["theta"][:, jj], [0, 1 / 3, 2 / 3, 1])
            th[k][nm] = [float(b[(s["theta"][:, jj] >= lo) & (s["theta"][:, jj] <= hi), 0].mean()) for lo, hi in zip(e[:-1], e[1:])]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
    for a, (nm, tex) in zip(ax, THIRDS):
        for off, k in ((-0.1, "point"), (0.1, "noise")):
            a.plot(np.arange(3) + off, th[k][nm], "o", color=C[k], label=LABEL[k])
        a.axhline(0, color="gray", ls=":", lw=1); a.set_xticks(range(3)); a.set_xticklabels(["low", "middle", "high"])
        a.set_title(f"thirds of the true {tex}", fontsize=9)
    ax[0].set_ylabel(r"$\sigma_8$ " + WIDTHS)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, "bias_thirds.png")
    NUM["thirds"] = th


def fig_ensemble(R):
    """Bias of the map-level posterior for each of five training seeds and their ensemble, 10% of the sky."""
    files = sorted(glob.glob(os.path.join(R, "maps_npe_forecast", "npe_e*_applied_mocks.npz")))
    fig, a = plt.subplots(figsize=(6.6, 3.4))
    for n, f in enumerate(files):
        b, e = bias(summary(f))
        a.errorbar(np.arange(4) - 0.24 + 0.08 * n, b[:4], yerr=e[:4], fmt="o", ms=4, color=C["map"], alpha=0.45,
                   label="one network (five seeds)" if n == 0 else None)
    b, e = bias(ensemble(files))
    a.errorbar(np.arange(4) + 0.24, b[:4], yerr=e[:4], fmt="D", ms=6, color=C["ink"], label="ensemble of the five")
    a.axhline(0, color="gray", ls=":", lw=1.2)
    a.set_xticks(range(4)); a.set_xticklabels(P4); a.set_ylabel(WIDTHS); a.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "ensemble.png")


def corner(post, name, truth=None, rng=None):
    """Corner plot of sigma8, Omega_m, A_SZ, alpha_SZ for several posteriors {key: samples (n, 10)}."""
    nu = 3 * 0.02 / 93.14

    def four(s):
        om = (s[:, 1] + s[:, 5] + nu) / s[:, 4] ** 2
        return np.c_[s[:, 0], om, s[:, 2], s[:, 3]]
    post = {k: four(v) for k, v in post.items()}
    truth = truth or [0.811, (0.121 + 0.0222 + nu) / 0.674 ** 2, -4.3054, 1.1233]
    lab = [P4[0], r"$\Omega_\mathrm{m}$", P4[2], P4[3]]
    if rng is None:                                   # every posterior's 0.2-99.8% range and the truth, padded
        rng = []
        for i in range(4):
            lo = min(min(np.quantile(v[:, i], 0.002) for v in post.values()), truth[i])
            hi = max(max(np.quantile(v[:, i], 0.998) for v in post.values()), truth[i])
            pad = 0.08 * (hi - lo)
            rng.append((lo - pad, hi + pad))
    fig, ax = plt.subplots(4, 4, figsize=(7.6, 7.6))
    for i in range(4):
        for jj in range(4):
            a = ax[i, jj]
            if jj > i:
                a.axis("off"); continue
            a.grid(False)
            for k, s in post.items():
                if i == jj:
                    h, e = np.histogram(s[:, i], 60, range=rng[i], density=True)
                    a.plot(0.5 * (e[1:] + e[:-1]), gaussian_filter1d(h, 1.5), color=C[k], lw=1.6)
                else:
                    H, xe, ye = np.histogram2d(s[:, jj], s[:, i], 50, range=[rng[jj], rng[i]])
                    H = gaussian_filter(H, 1.3); fl = np.sort(H.ravel())[::-1]; cs = np.cumsum(fl) / fl.sum()
                    lv = sorted(fl[np.searchsorted(cs, l)] for l in (0.68, 0.95))
                    if lv[0] < lv[1]:
                        a.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lv, colors=C[k], linewidths=[0.8, 1.6])
            if i == jj:
                a.axvline(truth[i], color=C["ink"], ls=":", lw=1); a.set_yticks([])
            else:
                a.plot(truth[jj], truth[i], "o", color=C["ink"], ms=4)
                a.set_ylim(rng[i])
            a.set_xlim(rng[jj])
            if i == 3:
                a.set_xlabel(lab[jj]); a.tick_params(axis="x", labelrotation=35, labelsize=8)
            else:
                a.set_xticklabels([])
            if jj == 0 and i > 0:
                a.set_ylabel(lab[i]); a.tick_params(axis="y", labelsize=8)
            elif jj > 0:
                a.set_yticklabels([])
    for k in post:
        ax[0, 3].plot([], [], color=C[k], label=LABEL[k])
    ax[0, 3].plot([], [], "o", color=C["ink"], label="truth"); ax[0, 3].legend(loc="upper right", fontsize=8.5)
    save(fig, name)


def fig_corner(R):
    L = lambda p, k: np.load(os.path.join(R, p))[k]
    corner({"published": L("catalogue_npe/forecast41/npe_t2_samples.npz", "maps"),
            "noise": L("catalogue_npe/calibcf41/npe_t2calc_samples.npz", "maps"),
            "map": L("maps_npe_forecast/npe_t1_samples.npz", "truth")}, "corner.png",
           rng=[(0.77, 0.86), (0.27, 0.36), (-4.55, -4.05), (0.95, 1.30)])
    # 40.7% of the sky: ensembles of five networks where they exist (samples pooled), else one network
    pool = lambda pat: np.concatenate([np.load(f)["mocks"] for f in sorted(glob.glob(os.path.join(R, pat)))])
    post40 = {"published": L("catalogue_npe/forecast41_t168/npe_plain_local_samples.npz", "mocks"),
              "point": L("catalogue_npe/calibf41_t168/npe_fixed_e12_samples.npz", "mocks"),
              "noise": pool("catalogue_npe/calibcf41_t168/npe_noise_e1*_samples.npz"),
              "map": pool("maps_npe_forecast_t168/npe_map_e1*_samples.npz")}
    corner(post40, "corner_40.png")
    NUM["corner_40"] = {k: dict(mean=v[:, :4].mean(0).round(4).tolist(), sd=v[:, :4].std(0).round(4).tolist()) for k, v in post40.items()}


def main(R):
    o = fig_tile()
    fig_noise(R, o)
    fig_grid(R)
    fig_relation(R)
    fig_counts(R)
    fig_bias(R)
    fig_coverage(R)
    fig_ensemble(R)
    fig_corner(R)
    json.dump(NUM, open(os.path.join(FIG, "numbers.json"), "w"), indent=1, default=float)
    print("wrote numbers.json")


if __name__ == "__main__":
    main(sys.argv[1])
