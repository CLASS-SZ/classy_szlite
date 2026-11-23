"""The finder's q against the catalogue model's q, cluster by cluster, on the map-level mocks.

    python benchmarks/mapsim/qcal.py run [--mocks 20] [--noise mss2|forecast]   # GPU: per-halo table
    python benchmarks/mapsim/qcal.py analyse [--noise mss2|forecast]           # CPU: fits and counts

Why: on the 41-tile survey the catalogue model of the same survey (catalogue_trainset.py --variant
survey41) predicts 10-23% more clusters above q = 8.4 than the map-level mocks, and up to 7% fewer
at q 5-8.4 and high z; the catalogue-level posterior is then biased on the mocks (LOG 2026-11-21).
This measures, for every halo of the mocks, the two q's:

  q_model = y0 / sigma_model(theta500): the catalogue model's q_true (y0 with its log-normal scatter;
            sigma_model = catalogue.analytic_sigma_y0 of the tile's noise spectra plus the CMB, linear
            in ln-ln between the filter's 15 sizes, clamped), whose observed value the model draws as
            q_true + N(0, 1);
  q_fsig  = y0 / sigma_finder(theta500): the same with the finder's own final-pass noise of that tile
            (masking pass; tSZ of the unmasked clusters included);
  q_obs   = the highest q among the kept, confirmed peaks matched to that halo (summaries.confirm:
            within 1.5', the matched halo is the one with the largest y0), -inf if none (peaks are kept
            down to q = 4 here: TileConfig.q_final = 4; the q > 5 data vector is unchanged).

The skies are exactly those of the map-level mocks (maps_trainset.py --set truth: same keys, same
tiles, same spectra), and the q > 5 counts are checked against the mocks' data vectors.

Analysis: sigma_finder / sigma_model per size; per q_model bin the detection fraction and the mean
and scatter of q_obs; censored fits of q_obs ~ N(sqrt((s q)^2 + k), w) above q_obs = 4 with q =
q_model and q = q_fsig, for all halos, per redshift bin, per theta500 bin and for isolated halos
(no other halo with q_model > 1 within 3'); blends; confirmed detections of halos with q_model < 2;
and the 4 x 5 counts predicted by the fitted relation (each halo's q_model through the fit, with the
same selection) against the mocks' counts. Output: $MAPSIM_OUT/qcal[_forecast]/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "qcal")
CAP = 8192                      # halos above M_conf per tile (about 4,400 at the fiducial point)
R_NEIGH = 3.0                   # arcmin, for the isolation flag


def run(args):
    out = OUT + ("_forecast" if args.noise == "forecast" else "")
    os.makedirs(out, exist_ok=True)
    table(args.noise, args.mocks, save_dir=out)


_BUILT = {}


def _build(noise, setup_theta, cap, light):
    """Static inputs and the jitted tile function (cached: one compilation per configuration)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import spike
    import maps_trainset as mt
    from classy_szlite.sim import simulator as simu, noise as nz, instrument as ins, summaries as sm
    from classy_szlite.sim import catalogue as ct, mmf
    from classy_szlite.sim._util import compact
    from classy_szlite.sim.flatsky import PIXF, LBOX
    nt, _ = nz.load_mss2(mt.SPECTRA, "mean")
    specs = mt.survey_spectra(nt.F.shape[0])
    cells, tb, _, cfg0, st, _ = spike.setup(setup_theta)
    cosmo = jax.jit(lambda t: simu.cosmology_stage(t, cells))
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb, tb, mode="goal")
    if noise == "forecast":
        sig_model, lnth = ct.analytic_sigma_y0_forecast(cmb, "goal")
        sig_model = np.repeat(sig_model, mt.N_TILES, axis=0)
    else:
        sig_model, lnth = ct.analytic_sigma_y0(nt, specs, cmb)
    cfg = cfg0._replace(q_final=4.0, peak_cap=6144)      # more final-pass peaks at q > 4 than at q > 5
    lnth_j = jnp.asarray(lnth)
    lnMc = np.float32(np.log(cfg.M_conf))
    forecast = noise == "forecast"
    CAPn = cap

    def interp_ln(lnsig, theta):                          # catalogue.eval_table for one row
        return ct.eval_table(ct.PatchTable(lnth=lnth_j, lnsig=lnsig[None]), theta)

    @jax.jit
    def tile(key, cs, th, st, nt, spec, lnsig_model):
        noise = None if forecast else nz.NoiseInput(nt, spec, jnp.float32(1.0))
        halos, X, ovf_paint = simu.tile_maps(key, cs, th, st, tb, cells, cfg, noise)
        pk, sigma, mk = simu.find_clusters(X, st.fs, cfg)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        det = keep & cf.ok
        counts = sm.histogram(pk.q, cf.z, det)
        C = halos.y0.shape[0]
        tgt = jnp.where(det, cf.halo, C)                                   # C = dropped
        q_best = jnp.full(C, -jnp.inf, jnp.float32).at[tgt].max(pk.q, mode="drop")
        n_match = jnp.zeros(C, jnp.int32).at[tgt].add(1, mode="drop")
        is_best = det & (pk.q == q_best[cf.halo])
        j_best = jnp.full(C, -1, jnp.int32).at[jnp.where(is_best, cf.halo, C)].set(pk.j, mode="drop")
        # the halos above the confirmation mass
        massive = halos.valid & (halos.lnM >= lnMc)
        idx, n_mass = compact(massive, CAPn, C - 1)
        ok = jnp.arange(CAPn) < n_mass
        g = lambda a: jnp.take(a, idx)
        y0, t5 = g(halos.y0).astype(jnp.float64), g(halos.theta500).astype(jnp.float64)
        q_model = jnp.where(ok, y0 / interp_ln(lnsig_model, t5), 0.0)
        q_fsig = jnp.where(ok, y0 / interp_ln(jnp.log(sigma.astype(jnp.float64)), t5), 0.0)
        if light:
            q_nb, shadow = jnp.zeros(CAPn, jnp.float32), jnp.zeros(CAPn, bool)
        else:
            # neighbours: the largest q_model of the other massive halos within R_NEIGH
            hy = (g(halos.iy).astype(jnp.float32) + g(halos.oy)) * PIXF
            hx = (g(halos.ix).astype(jnp.float32) + g(halos.ox)) * PIXF
            dy = sm._wrap(hy[:, None] - hy[None, :], LBOX)
            dx = sm._wrap(hx[:, None] - hx[None, :], LBOX)
            near = (dy * dy + dx * dx <= R_NEIGH ** 2) & ok[None, :] & ~jnp.eye(CAPn, dtype=bool)
            q_nb = jnp.max(jnp.where(near, q_model[None, :].astype(jnp.float32), 0.0), axis=1)
            close = (dy * dy + dx * dx <= np.float32(cfg.r_conf) ** 2) & ok[None, :] & ~jnp.eye(CAPn, dtype=bool)
            shadow = jnp.any(close & (g(halos.y0)[None, :] > g(halos.y0)[:, None]), axis=1)   # loses the match
        # confirmed peaks above q = 5 whose matched halo is not in the list cannot happen (all massive)
        n_unconf5 = jnp.sum(keep & ~cf.ok & (pk.q > 5.0))
        ovf = jnp.concatenate([halos.overflow[None], ovf_paint, mk["peak_overflow"][None], cf.overflow[None],
                               (n_mass > CAPn)[None]])
        return dict(ok=ok, z=g(halos.z), lnM=g(halos.lnM), theta500=t5.astype(jnp.float32), y0=y0.astype(jnp.float32),
                    q_model=q_model.astype(jnp.float32), q_fsig=q_fsig.astype(jnp.float32),
                    q_obs=g(q_best), n_match=g(n_match), j_obs=g(j_best), q_nb=q_nb, shadow=shadow,
                    counts=counts, sigma=sigma, sigma0=mk["sigma0"], n_unconf5=n_unconf5,
                    n_det5=jnp.sum(det & (pk.q > 5.0)), n_massive=n_mass, overflow=jnp.any(ovf))

    return dict(tile=tile, cosmo=cosmo, nt=nt, st=st, specs=specs, sig_model=sig_model, lnth=lnth)


def table(noise, n_mocks, theta=None, base_key=None, setup_theta=None, cap=CAP, save_dir=None, light=False, log=print):
    """The per-halo table of n_mocks map-level mocks of the 41-tile survey at theta (default: the
    fiducial point, with the keys of maps_trainset.py --set truth, i.e. the mocks themselves).
    Buffers are sized at setup_theta (default theta). light=True skips the neighbour flags (q_nb,
    shadow), which cost a cap x cap comparison. Returns (H, tiles dict); with save_dir, also writes
    halos.npy, tiles.npz and counts_*.npy there."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import spike
    import maps_trainset as mt
    from classy_szlite.sim import simulator as simu, noise as nz, instrument as ins, summaries as sm
    from classy_szlite.sim import catalogue as ct, mmf
    from classy_szlite.sim._util import compact
    from classy_szlite.sim.flatsky import PIXF, LBOX

    theta = simu.THETA_FID if theta is None else theta
    bkey = (noise, theta if setup_theta is None else setup_theta, cap, light)
    if bkey not in _BUILT:
        _BUILT[bkey] = _build(*bkey)
    b = _BUILT[bkey]
    tile, nt, st, specs, sig_model, lnth = b["tile"], b["nt"], b["st"], b["specs"], b["sig_model"], b["lnth"]
    cs = b["cosmo"](theta)
    base = jax.random.key(mt.SEED["truth"]) if base_key is None else base_key
    rows, per_tile = [], []
    t0 = time.time()
    nt_d, st_d = jax.device_put(nt), jax.device_put(st)
    th = jax.tree_util.tree_map(jnp.asarray, theta)
    for i in range(n_mocks):
        km = jax.random.fold_in(base, i)
        counts_mock = np.zeros((4, 5), np.int64)
        for t in range(mt.N_TILES):
            o = jax.device_get(tile(jax.random.fold_in(km, t), cs, th, st_d, nt_d, jnp.int32(specs[t]),
                                    jnp.asarray(np.log(sig_model[t]))))
            if o["overflow"]:
                raise SystemExit(f"overflow in mock {i}, tile {t}")
            k = o["ok"]
            rows.append(np.rec.fromarrays([np.full(k.sum(), i, np.int16), np.full(k.sum(), t, np.int16)]
                                          + [o[n][k] for n in ("z", "lnM", "theta500", "y0", "q_model", "q_fsig",
                                                               "q_obs", "n_match", "j_obs", "q_nb", "shadow")],
                                          names="mock,tile,z,lnM,theta500,y0,q_model,q_fsig,q_obs,n_match,j_obs,q_nb,shadow"))
            per_tile.append(dict(mock=i, tile=t, sigma=o["sigma"], sigma0=o["sigma0"], sigma_model=sig_model[t],
                                 n_unconf5=int(o["n_unconf5"]), n_det5=int(o["n_det5"]), n_massive=int(o["n_massive"])))
            counts_mock += o["counts"]
        log(f"[{time.time() - t0:6.0f} s] mock {i}: q > 5 counts in the data vector {int(counts_mock.sum())}")
        if save_dir:
            np.save(os.path.join(save_dir, f"counts_{i}.npy"), counts_mock)
    H = np.concatenate(rows)
    T = {k: np.array([p[k] for p in per_tile]) for k in
         ("mock", "tile", "sigma", "sigma0", "sigma_model", "n_unconf5", "n_det5", "n_massive")}
    if save_dir:
        np.save(os.path.join(save_dir, "halos.npy"), H)
        np.savez(os.path.join(save_dir, "tiles.npz"), **T, specs=specs, lnth=lnth, noise=noise)
    log(f"done: {len(H)} halos above M_conf in {n_mocks} mocks ({time.time() - t0:.0f} s)")
    return H, dict(T, specs=specs, lnth=lnth)


# ------------------------------------------------------------------------------------------------
def censored_fit(q, qo, lo=4.0):
    """Maximum likelihood of: with probability f the halo is lost whatever its q (its peak merged into
    a brighter one, or given to a heavier halo within 1.5'); otherwise q_obs ~ N(sqrt((s q)^2 + k), w),
    observed above lo and censored below. Without f, a few lost bright halos would inflate w and move
    k (review, 2026-11-21)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    det = np.isfinite(qo) & (qo > lo)

    def nll(p):
        s, k, w, f = p
        if not (0.0 <= f < 1.0 and w > 0.05):
            return np.inf
        mu = np.sqrt(np.maximum((s * q) ** 2 + k, 1e-6))
        ll = np.where(det, np.log1p(-f) + norm.logpdf(qo, mu, w),
                      np.logaddexp(np.log(max(f, 1e-300)), np.log1p(-f) + norm.logcdf((lo - mu) / w)))
        return -np.sum(ll)
    x0 = [1.0, 1.0, 1.0, 0.01]
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-4, maxiter=8000, maxfev=12000))
    r = minimize(nll, r.x, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-4, maxiter=8000, maxfev=12000))
    h = np.array([1e-3, 2e-2, 1e-3, 2e-4])
    Hm = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            def fv(da, db):
                p = r.x.copy(); p[a] += da; p[b] += db
                return nll(p)
            Hm[a, b] = (fv(h[a], h[b]) - fv(h[a], -h[b]) - fv(-h[a], h[b]) + fv(-h[a], -h[b])) / (4 * h[a] * h[b])
    err = np.full(4, np.nan)
    use = [0, 1, 2, 3] if np.all(np.isfinite(Hm)) else [0, 1, 2]       # f at its boundary 0: errors of s, k, w only
    try:
        err[use] = np.sqrt(np.abs(np.diag(np.linalg.inv(Hm[np.ix_(use, use)]))))
    except np.linalg.LinAlgError:
        pass
    s_, k, w, f = r.x
    return dict(s=s_, k=k, w=w, f=f, s_err=err[0], k_err=err[1], w_err=err[2], f_err=err[3],
                n=int(q.size), n_det=int(det.sum()), converged=bool(r.success))


def analyse(args):
    out = OUT + ("_forecast" if args.noise == "forecast" else "")
    H = np.load(os.path.join(out, "halos.npy"))
    T = np.load(os.path.join(out, "tiles.npz"))
    from scipy.stats import norm
    Q_EDGES = np.logspace(np.log10(5.0), np.log10(40.0), 5)
    Z_EDGES = np.linspace(0.01, 1.5, 6)
    lines = []

    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    n_mocks = int(H["mock"].max()) + 1
    P(f"{len(H)} halos above 5e13 Msun in {n_mocks} mocks x 41 tiles ({args.noise} noise)")
    # 1. noise
    r = T["sigma"] / T["sigma_model"]
    r0 = T["sigma0"] / T["sigma_model"]
    th = np.exp(T["lnth"])
    P("\n1. the finder's noise over the model's (tile mean, 5-95% over tiles), at 0.5', 1', 2', 4', 8', 15'")
    for name, v in (("final pass (masked)", r), ("first pass", r0)):
        cols = [np.argmin(np.abs(th - x)) for x in (0.5, 1, 2, 4, 8, 15)]
        P(f"   {name:20s} " + "  ".join(f"{v[:, c].mean():.3f} ({np.quantile(v[:, c], 0.05):.3f}-{np.quantile(v[:, c], 0.95):.3f})"
                                         for c in cols))
    # 2. consistency with the mocks
    mocks_path = os.path.join(os.path.dirname(out), "maps_npe" + ("_forecast" if args.noise == "forecast" else ""), "truth.npz")
    cm = np.array([np.load(os.path.join(out, f"counts_{i}.npy")) for i in range(n_mocks)])
    if os.path.exists(mocks_path):
        x = np.load(mocks_path)["x"][:n_mocks, :20].reshape(-1, 4, 5)
        P(f"\n2. q > 5 counts against the mocks' data vectors: identical in {int(np.sum(np.all(cm == x, axis=(1, 2))))}"
          f" of {n_mocks} mocks (max abs difference {int(np.abs(cm - x).max())})")
    qm, qf, qo, z, t5, qnb = (H[k].astype(np.float64) for k in ("q_model", "q_fsig", "q_obs", "z", "theta500", "q_nb"))
    det5 = np.isfinite(qo) & (qo > 5)
    # 3. per q_model bin
    P("\n3. per q_model bin: halos; detected with q_obs > 4 and > 5; for detected (q_obs > 4): mean q_obs/q_model,"
      " mean and sd of q_obs - q_model; the same against q_fsig")
    for a, b in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30), (30, 60)):
        m = (qm >= a) & (qm < b)
        d = m & np.isfinite(qo) & (qo > 4)
        if not m.any():
            continue
        P(f"   {a:3d}-{b:<3d} {m.sum():7d}  P(>4) {d.sum() / m.sum():.3f}  P(>5) {np.sum(m & det5) / m.sum():.3f}"
          f"  ratio {np.mean(qo[d] / qm[d]):.3f}  diff {np.mean(qo[d] - qm[d]):+.2f} sd {np.std(qo[d] - qm[d]):.2f}"
          f"  | vs q_fsig: ratio {np.mean(qo[d] / qf[d]):.3f}  diff {np.mean(qo[d] - qf[d]):+.2f}")
    # 4. censored fits
    P("\n4. fits: lost with probability f, else q_obs ~ N(sqrt((s q)^2 + k), w), censored below q_obs = 4;"
      " halos with q > 2 (the catalogue model: s = 1, k = 0, w = 1, f = 0)")
    iso = qnb < 1.0
    shadow = H["shadow"].astype(bool)
    sets = [("all", np.ones_like(qm, bool)), ("isolated (no q_model > 1 within 3')", iso),
            ("not shadowed (no heavier halo within 1.5')", ~shadow)]
    sets += [(f"z {Z_EDGES[i]:.2f}-{Z_EDGES[i + 1]:.2f}", (z > Z_EDGES[i]) & (z <= Z_EDGES[i + 1])) for i in range(5)]
    sets += [(f"theta500 {a}-{b}'", (t5 >= a) & (t5 < b)) for a, b in ((0, 1), (1, 2), (2, 4), (4, 100))]
    fits = {}
    for name, m in sets:
        res = {}
        for qn, qq in (("q_model", qm), ("q_fsig", qf)):
            mm = m & (qq > 2)
            res[qn] = censored_fit(qq[mm], qo[mm])
        fits[name] = res
        a, b = res["q_model"], res["q_fsig"]
        P(f"   {name:44s} q_model: s {a['s']:.4f}+-{a['s_err']:.4f} k {a['k']:+.2f}+-{a['k_err']:.2f} w {a['w']:.3f}"
          f" f {a['f']:.4f}  | q_fsig: s {b['s']:.4f} k {b['k']:+.2f} w {b['w']:.3f} f {b['f']:.4f}  ({a['n']} halos)"
          + ("" if a["converged"] and b["converged"] else "  [not converged]"))
    # 5. blends and odd matches
    nb = (qm > 2) & (qnb > 2)
    P(f"\n5. halos with q_model > 2 and a neighbour with q_model > 2 within 3': {nb.sum()} of {np.sum(qm > 2)}"
      f" ({nb.sum() / np.sum(qm > 2):.3f}); detected above 5: {np.sum(nb & det5) / max(nb.sum(), 1):.3f}"
      f" against {np.sum(~nb & (qm > 2) & det5) / np.sum(~nb & (qm > 2)):.3f} without")
    P(f"   detections q_obs > 5 of halos with q_model < 2: {np.sum(det5 & (qm < 2))} of {det5.sum()}"
      f" ({np.sum(det5 & (qm < 2)) / det5.sum():.4f}); halos matched by two or more kept peaks: {np.sum(H['n_match'] > 1)}")
    sh2 = shadow & (qm > 5)
    P(f"   halos with q_model > 5 shadowed by a heavier halo within 1.5': {sh2.sum()} ({sh2.sum() / np.sum(qm > 5):.4f});"
      f" of them detected above 5: {np.sum(sh2 & det5) / max(sh2.sum(), 1):.3f}")
    P(f"   kept peaks with q > 5 not confirmed (per tile): {T['n_unconf5'].mean():.2f}; confirmed q > 5 per tile"
      f" {T['n_det5'].mean():.1f}")
    # 6. counts predicted from the fits against the mocks
    P("\n6. the 4 x 5 counts (sum over mocks): mocks, and predicted from each halo's q_model by (a) the catalogue"
      " model (every halo), (b) the fitted relation (halos with q_model > 2; below, the finder's own detections of"
      " those halos), (c) the same fitted per redshift bin. z-scores use the binomial variance of the same halos"
      " (the predictions share the mocks' halos and y0 draws, so Poisson errors would be too large).")

    def hist(q, zz):
        c = np.zeros((4, 5))
        m = np.isfinite(q) & (q > Q_EDGES[0]) & (q < Q_EDGES[-1]) & (zz > Z_EDGES[0]) & (zz < Z_EDGES[-1])
        np.add.at(c, (np.searchsorted(Q_EDGES[1:-1], q[m], side="left"),
                      np.searchsorted(Z_EDGES[1:-1], zz[m], side="left")), 1)
        return c
    low = qm <= 2

    def predict(s_, k, w, f, mask, fit_only=True):
        mm = mask & ~low if fit_only else mask
        mu = np.sqrt(np.maximum((s_ * qm[mm]) ** 2 + k, 0.0))
        zz = z[mm]
        inz = (zz > Z_EDGES[0]) & (zz < Z_EDGES[-1])
        iz = np.searchsorted(Z_EDGES[1:-1], zz[inz], side="left")
        c, v = np.zeros((4, 5)), np.zeros((4, 5))
        for b in range(4):
            pb_ = (1 - f) * (norm.cdf((Q_EDGES[b + 1] - mu[inz]) / w) - norm.cdf((Q_EDGES[b] - mu[inz]) / w))
            np.add.at(c[b], iz, pb_)
            np.add.at(v[b], iz, pb_ * (1 - pb_))
        if fit_only:
            lm = mask & low
            c = c + hist(qo[lm], z[lm])
        return c, v
    every = np.ones_like(qm, bool)
    obs = cm.sum(0)
    obs1 = hist(qo, z)                                   # one count per halo: its best kept, confirmed peak
    pa, va = predict(1.0, 0.0, 1.0, 0.0, every, fit_only=False)
    f = fits["all"]["q_model"]
    pb, vb = predict(f["s"], f["k"], f["w"], f["f"], every)
    pc, vc = np.zeros((4, 5)), np.zeros((4, 5))
    for i, (n, m) in enumerate(sets[3:8]):
        ff = fits[n]["q_model"]
        c, v = predict(ff["s"], ff["k"], ff["w"], ff["f"], m)
        pc, vc = pc + c, vc + v
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    P(f"   mocks: total {obs.sum():.0f}; one count per halo {obs1.sum():.0f} (duplicate peaks per bin, fraction:\n"
      f"{(obs - obs1) / np.maximum(obs, 1)})")
    for name, p, v in (("(a) catalogue model", pa, va), ("(b) fitted, all", pb, vb), ("(c) fitted per z", pc, vc)):
        zsc = (obs1 - p) / np.sqrt(np.maximum(v, 1))
        P(f"   {name}: total {p.sum():.0f}; one-per-halo mocks / predicted (rows q bins, columns z bins)\n{obs1 / p}\n"
          f"   z-scores (binomial): max |z| {np.abs(zsc).max():.1f}, chi2 {np.sum(zsc ** 2):.0f} for 20 bins")
    # 7. the calibration for the catalogue model, variant (ii) (catalogue_trainset.py --variant calib41):
    # the finder's noise per tile (mean over the mocks) and, per redshift bin (the five data-vector bins and
    # z > 1.5, which the stacked p uses), the fit against q_fsig on halos with q_fsig > 2, applied to every
    # halo (checked: within 0.97-1.04 of the mocks per bin, LOG 2026-11-21).
    zb = [0.01, 0.31, 0.61, 0.90, 1.20, 1.50, 99.0]
    cal = []
    for a, b in zip(zb[:-1], zb[1:]):
        m = (z > a) & (z <= b) & (qf > 2)
        r = censored_fit(qf[m], qo[m])
        cal.append((b, r["s"], r["k"], r["w"], r["f"]))
    cal = np.array(cal)
    sig_tile = np.array([T["sigma"][T["tile"] == t].mean(0) for t in range(int(T["tile"].max()) + 1)])
    np.savez(os.path.join(out, "calibration.npz"), z_upper=cal[:, 0], s=cal[:, 1], k=cal[:, 2], w=cal[:, 3], f=cal[:, 4],
             sigma=sig_tile, lnth=T["lnth"], specs=T["specs"], noise=args.noise, n_mocks=n_mocks)
    P("\n7. calibration for the catalogue model (calibration.npz): per z bin (z_upper, s, k, w, f) against q_fsig")
    for row in cal:
        P("   " + "  ".join(f"{v:.4f}" for v in row))
    json.dump(dict(fits=fits, counts_mocks=obs.tolist(), counts_mocks_one_per_halo=obs1.tolist(), pred_catalogue=pa.tolist(), pred_fit_all=pb.tolist(),
                   pred_fit_z=pc.tolist()), open(os.path.join(out, "qcal.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "qcal.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyse"))
    ap.add_argument("--mocks", type=int, default=20)
    ap.add_argument("--noise", choices=("mss2", "forecast"), default="mss2")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyse(args)


if __name__ == "__main__":
    main()
