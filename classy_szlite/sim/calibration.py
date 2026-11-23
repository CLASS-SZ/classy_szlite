"""Calibrating the catalogue model on map-level simulations.

The catalogue model draws each halo's observed signal-to-noise as q_obs = q_true + N(0, 1) with
q_true = y0 / sigma_y0(theta500) from a noise table. The finder on maps behaves differently: its noise
includes the tSZ of faint clusters (confusion), its search over positions and sizes lifts faint clusters,
and neighbours add scatter and steal matches. This module measures that behaviour halo by halo and fits
the calibrated observation model of catalogue.CatalogueConfig:

    q = y0 / sigma_finder(theta500);  lost with probability f, else q_obs = sqrt((s q)^2 + k) + w N(0, 1)

per redshift bin (optionally per redshift and size cell), by censored maximum likelihood.

Three calibrations, each a file that survey.catalogue_model reads:

* ``calibrate_point``: at one parameter point (default the fiducial one) -> "calibrated-point";
* ``calibrate_grid`` then ``fit_noise_level``: at a Latin hypercube of points over the prior, written
  through the finder's noise level u = ln(sigma_finder(2') / sigma_finder(2'; fiducial)), quadratic in
  the parameters -> "calibrated-noise";
* ``calibrate_grid`` then ``fit_theta``: the same points, linear in the parameters -> "calibrated-theta".

Typical cost on one GPU (41 tiles, forecast noise): the per-halo table of 20 mocks takes about 1 min after
compilation; a 25-point grid with 5 mocks each about 7 min plus the fits (about 1 min per point on CPU).
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from . import simulator as simu, catalogue as ct, noise as nz, summaries as sm, priors
from ._util import compact
from .flatsky import PIXF, LBOX
from .setup import tile_setup, default_cmb

ZB = [0.01, 0.31, 0.61, 0.90, 1.20, 1.50, 99.0]     # redshift bins of the calibration (the data vector's, and z > 1.5)
SB = [0.0, 1.0, 2.0, 4.0, 1e9]                       # theta500 bins [arcmin] for size cells
PARAMS = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ")
GRID_SEED = 20261127
N_MIN = 400                                          # halos with q > 2 needed to fit a (z, size) cell
R_NEIGH = 3.0                                        # arcmin, for the neighbour flag

_BUILT = {}


def _build(noise, spectra, setup_theta, cap, neighbours, n_tiles, cmb_key):
    """Static inputs and the jitted per-tile function (one compilation per configuration)."""
    from .survey import survey_spectra
    cmb = default_cmb()
    ts = tile_setup(setup_theta, cmb, log=lambda *a: None)
    cells, tb, st = ts.cells, ts.tables, ts.static
    cosmo = jax.jit(lambda t: simu.cosmology_stage(t, cells))
    forecast = noise == "forecast"
    if forecast:
        nt, specs = None, np.zeros(n_tiles, np.int32)
        sig_model, lnth = ct.analytic_sigma_y0_forecast(cmb, "goal")
        sig_model = np.repeat(sig_model, n_tiles, axis=0)
    else:
        nt, _ = nz.load_mss2(spectra, "mean")
        specs = survey_spectra(nt.F.shape[0], n_tiles)
        sig_model, lnth = ct.analytic_sigma_y0(nt, specs, cmb)
    cfg = ts.cfg._replace(q_final=4.0, peak_cap=6144)    # keep final-pass peaks down to q = 4
    lnth_j = jnp.asarray(lnth)
    lnMc = np.float32(np.log(cfg.M_conf))

    def interp_ln(lnsig, theta500):
        return ct.eval_table(ct.PatchTable(lnth=lnth_j, lnsig=lnsig[None]), theta500)

    @jax.jit
    def tile(key, cs, th, st, nt, spec, lnsig_model):
        ni = None if forecast else nz.NoiseInput(nt, spec, jnp.float32(1.0))
        halos, X, ovf_paint = simu.tile_maps(key, cs, th, st, tb, cells, cfg, ni)
        pk, sigma, mk = simu.find_clusters(X, st.fs, cfg)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        det = keep & cf.ok
        counts = sm.histogram(pk.q, cf.z, det)
        C = halos.y0.shape[0]
        tgt = jnp.where(det, cf.halo, C)
        q_best = jnp.full(C, -jnp.inf, jnp.float32).at[tgt].max(pk.q, mode="drop")
        n_match = jnp.zeros(C, jnp.int32).at[tgt].add(1, mode="drop")
        massive = halos.valid & (halos.lnM >= lnMc)
        idx, n_mass = compact(massive, cap, C - 1)
        ok = jnp.arange(cap) < n_mass
        g = lambda a: jnp.take(a, idx)
        y0, t5 = g(halos.y0).astype(jnp.float64), g(halos.theta500).astype(jnp.float64)
        q_model = jnp.where(ok, y0 / interp_ln(lnsig_model, t5), 0.0)
        q_fsig = jnp.where(ok, y0 / interp_ln(jnp.log(sigma.astype(jnp.float64)), t5), 0.0)
        if neighbours:
            hy = (g(halos.iy).astype(jnp.float32) + g(halos.oy)) * PIXF
            hx = (g(halos.ix).astype(jnp.float32) + g(halos.ox)) * PIXF
            dy = sm._wrap(hy[:, None] - hy[None, :], LBOX)
            dx = sm._wrap(hx[:, None] - hx[None, :], LBOX)
            eye = jnp.eye(cap, dtype=bool)
            near = (dy * dy + dx * dx <= R_NEIGH ** 2) & ok[None, :] & ~eye
            q_nb = jnp.max(jnp.where(near, q_model[None, :].astype(jnp.float32), 0.0), axis=1)
            close = (dy * dy + dx * dx <= np.float32(cfg.r_conf) ** 2) & ok[None, :] & ~eye
            shadow = jnp.any(close & (g(halos.y0)[None, :] > g(halos.y0)[:, None]), axis=1)
        else:
            q_nb, shadow = jnp.zeros(cap, jnp.float32), jnp.zeros(cap, bool)
        ovf = jnp.concatenate([halos.overflow[None], ovf_paint, mk["peak_overflow"][None], cf.overflow[None],
                               (n_mass > cap)[None]])
        return dict(ok=ok, z=g(halos.z), lnM=g(halos.lnM), theta500=t5.astype(jnp.float32), y0=y0.astype(jnp.float32),
                    q_model=q_model.astype(jnp.float32), q_fsig=q_fsig.astype(jnp.float32), q_obs=g(q_best),
                    n_match=g(n_match), q_nb=q_nb, shadow=shadow, counts=counts, sigma=sigma, sigma0=mk["sigma0"],
                    overflow=jnp.any(ovf))

    return dict(tile=tile, cosmo=cosmo, nt=nt, st=st, specs=specs, sig_model=sig_model, lnth=lnth)


def halo_table(n_mocks, theta=None, seed=priors.SEEDS_MAP["truth"], n_tiles=41, noise="forecast", spectra=None,
               setup_theta=None, cap=8192, neighbours=False, key=None, log=print):
    """Every halo above 5e13 Msun of ``n_mocks`` map-level mocks of the survey at ``theta`` (default the
    fiducial point; with the default seed, the mocks of the 'truth' set themselves), with three q's:

      q_model  y0 / sigma_y0(theta500): the catalogue model's q_true (analytic noise of the filter);
      q_fsig   y0 / sigma_finder(theta500): the same with the finder's own final-pass noise of that tile;
      q_obs    the highest q of the kept, confirmed peaks matched to the halo (-inf if none; peaks kept
               down to q = 4).

    ``key`` replaces key(seed) as the base key. Buffers are sized at ``setup_theta`` (default theta). ``neighbours`` adds the largest q_model of other
    halos within 3' (q_nb) and whether a heavier halo lies within 1.5' (shadow): a cap x cap comparison.
    Returns (H, T): a record array of halos and a dict of per-tile arrays (sigma, sigma0, sigma_model,
    counts) with the noise tables' ln theta grid and spectrum indices."""
    theta = simu.THETA_FID if theta is None else theta
    setup_theta = theta if setup_theta is None else setup_theta
    bkey = (noise, spectra, setup_theta, cap, neighbours, n_tiles, os.environ.get("MAPSIM_CMB_TABLE", ""))
    if bkey not in _BUILT:
        _BUILT[bkey] = _build(*bkey)
    b = _BUILT[bkey]
    tile, specs, sig_model = b["tile"], b["specs"], b["sig_model"]
    cs = b["cosmo"](theta)
    base = jax.random.key(seed) if key is None else key
    nt_d, st_d = b["nt"], jax.device_put(b["st"])
    th = jax.tree_util.tree_map(jnp.asarray, theta)
    rows, per_tile = [], []
    t0 = time.time()
    fields = ("z", "lnM", "theta500", "y0", "q_model", "q_fsig", "q_obs", "n_match", "q_nb", "shadow")
    for i in range(n_mocks):
        km = jax.random.fold_in(base, i)
        for t in range(n_tiles):
            o = jax.device_get(tile(jax.random.fold_in(km, t), cs, th, st_d, nt_d, jnp.int32(specs[t]),
                                    jnp.asarray(np.log(sig_model[t]))))
            if o["overflow"]:
                raise RuntimeError(f"buffer overflow in mock {i}, tile {t}: size the buffers at a heavier setup_theta")
            k = o["ok"]
            rows.append(np.rec.fromarrays([np.full(k.sum(), i, np.int16), np.full(k.sum(), t, np.int16)] + [o[n][k] for n in fields],
                                          names="mock,tile," + ",".join(fields)))
            per_tile.append(dict(mock=i, tile=t, sigma=o["sigma"], sigma0=o["sigma0"], sigma_model=sig_model[t], counts=o["counts"]))
        log(f"[{time.time() - t0:6.0f} s] mock {i + 1}/{n_mocks}")
    H = np.concatenate(rows)
    T = {k: np.array([p[k] for p in per_tile]) for k in ("mock", "tile", "sigma", "sigma0", "sigma_model", "counts")}
    return H, dict(T, specs=specs, lnth=b["lnth"])


def censored_fit(q, qo, lo=4.0):
    """Maximum likelihood of: lost with probability f whatever q (peak merged into a brighter one, or
    credited to a heavier halo within 1.5'), else q_obs ~ N(sqrt((s q)^2 + k), w), observed above ``lo``
    and censored below. Returns s, k, w, f, their errors (numerical Hessian), n and n_det."""
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
    opts = dict(xatol=1e-6, fatol=1e-4, maxiter=8000, maxfev=12000)
    r = minimize(nll, [1.0, 1.0, 1.0, 0.01], method="Nelder-Mead", options=opts)
    r = minimize(nll, r.x, method="Nelder-Mead", options=opts)
    h = np.array([1e-3, 2e-2, 1e-3, 2e-4])
    Hm = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            def fv(da, db):
                p = r.x.copy(); p[a] += da; p[b] += db
                return nll(p)
            Hm[a, b] = (fv(h[a], h[b]) - fv(h[a], -h[b]) - fv(-h[a], h[b]) + fv(-h[a], -h[b])) / (4 * h[a] * h[b])
    err = np.full(4, np.nan)
    use = [0, 1, 2, 3] if np.all(np.isfinite(Hm)) else [0, 1, 2]       # f at its boundary 0
    try:
        err[use] = np.sqrt(np.abs(np.diag(np.linalg.inv(Hm[np.ix_(use, use)]))))
    except np.linalg.LinAlgError:
        pass
    s_, k, w, f = r.x
    return dict(s=s_, k=k, w=w, f=f, s_err=err[0], k_err=err[1], w_err=err[2], f_err=err[3],
                n=int(q.size), n_det=int(det.sum()), converged=bool(r.success))


def fit_rows(H, q_name="q_fsig"):
    """(n_z, 5): s, k, w, f, n per redshift bin (ZB) against ``q_name`` on halos with q > 2."""
    q, qo, z = H[q_name].astype(float), H["q_obs"].astype(float), H["z"].astype(float)
    rows = []
    for a, b in zip(ZB[:-1], ZB[1:]):
        m = (z > a) & (z <= b) & (q > 2)
        r = censored_fit(q[m], qo[m])
        rows.append((r["s"], r["k"], r["w"], r["f"], r["n"]))
    return np.array(rows)


def fit_cells(H):
    """(n_z, n_size, 5): s, k, w, f, n per (redshift, size) cell; NaN where fewer than N_MIN halos."""
    qf, qo, z, th = (H[k].astype(float) for k in ("q_fsig", "q_obs", "z", "theta500"))
    out = np.full((len(ZB) - 1, len(SB) - 1, 5), np.nan)
    for i, (a, b) in enumerate(zip(ZB[:-1], ZB[1:])):
        for j, (c, d) in enumerate(zip(SB[:-1], SB[1:])):
            m = (z > a) & (z <= b) & (th > c) & (th <= d) & (qf > 2)
            out[i, j, 4] = m.sum()
            if m.sum() >= N_MIN:
                r = censored_fit(qf[m], qo[m])
                out[i, j, :4] = (r["s"], r["k"], r["w"], r["f"])
    return out


def tile_sigma(T):
    """The finder's final-pass noise per tile (n_tiles, 15), mean over the mocks."""
    return np.array([T["sigma"][T["tile"] == t].mean(0) for t in range(int(T["tile"].max()) + 1)])


def calibrate_point(H, T, path, noise="forecast"):
    """Write the one-point calibration (catalogue.load_qcal's format) from a halo table."""
    rows = fit_rows(H)
    np.savez(path, z_upper=np.array(ZB[1:]), s=rows[:, 0], k=rows[:, 1], w=rows[:, 2], f=rows[:, 3],
             sigma=tile_sigma(T), lnth=T["lnth"], specs=T["specs"], noise=noise, n_mocks=int(H["mock"].max()) + 1)
    return rows


def grid_points(n, box="map", seed=GRID_SEED):
    """The fiducial point followed by a Latin hypercube of n points in the flat box (4 parameters)."""
    box = priors.BOXES[box] if isinstance(box, str) else box
    rng = np.random.default_rng(seed)
    u = np.stack([(rng.permutation(n) + rng.uniform(size=n)) / n for _ in PARAMS], 1)
    lo = np.array([box[p][0] for p in PARAMS])
    hi = np.array([box[p][1] for p in PARAMS])
    fid = np.array([getattr(simu.THETA_FID, p) for p in PARAMS])
    return np.vstack([fid, lo + u * (hi - lo)])


def calibrate_grid(out_dir, n_points=24, n_mocks=5, size=False, n_tiles=41, noise="forecast", spectra=None,
                   cap=16384, box="map", log=print):
    """The calibration at the fiducial point and a Latin hypercube of n_points over the prior, each from
    n_mocks map-level mocks with their own keys (fold_in(key(GRID_SEED), point)); buffers sized at the
    heaviest point. Writes out_dir/calib_grid.npz (rows per z bin; with ``size``, cells per (z, size) too)
    and out_dir/points/point_XX.npz (the halos with q_fsig > 1.5, for refits). Resumes."""
    os.makedirs(os.path.join(out_dir, "points"), exist_ok=True)
    P = grid_points(n_points, box)
    thetas = [simu.THETA_FID._replace(**dict(zip(PARAMS, map(float, row)))) for row in P]
    from . import halos as hl
    cells = hl.default_cells()
    lam_f = jax.jit(lambda t: simu.cosmology_stage(t, cells).lam.sum())
    lam = [float(lam_f(t)) for t in thetas]
    heavy = thetas[int(np.argmax(lam))]
    log(f"{len(P)} points; halos per tile {min(lam):.0f}-{max(lam):.0f}")
    path = os.path.join(out_dir, "calib_grid.npz")
    rows = np.full((len(P), len(ZB) - 1, 5), np.nan)
    cells_ = np.full((len(P), len(ZB) - 1, len(SB) - 1, 5), np.nan)
    sig = np.full((len(P), n_tiles, 15), np.nan)
    flag = np.zeros(len(P), bool)
    if os.path.exists(path):
        old = np.load(path)
        if np.array_equal(old["theta"], P):
            rows, sig, flag = np.array(old["rows"]), np.array(old["sigma"]), np.array(old["done"])
            cells_ = np.array(old["cells"])
            log(f"resuming: {flag.sum()} points done")
    t0 = time.time()
    lnth = specs = None
    for i, th in enumerate(thetas):
        if flag[i]:
            continue
        H, T = halo_table(n_mocks, th, n_tiles=n_tiles, noise=noise, spectra=spectra, setup_theta=heavy, cap=cap,
                          key=jax.random.fold_in(jax.random.key(GRID_SEED), i), log=lambda m: None)
        rows[i] = fit_rows(H)
        if size:
            cells_[i] = fit_cells(H)
        sig[i] = tile_sigma(T)
        lnth, specs = T["lnth"], T["specs"]
        keep = H["q_fsig"] > 1.5
        np.savez_compressed(os.path.join(out_dir, "points", f"point_{i:02d}.npz"),
                            **{k: H[k][keep] for k in ("mock", "tile", "z", "lnM", "theta500", "y0", "q_model", "q_fsig", "q_obs")},
                            sigma=T["sigma"], sigma_tile=T["tile"], theta=P[i])
        flag[i] = True
        np.savez(path, theta=P, rows=rows, cells=cells_, sigma=sig, done=flag, params=np.array(PARAMS),
                 z_upper=np.array(ZB[1:]), size_upper=np.array(SB[1:]), lnth=lnth, specs=specs, noise=noise, mocks=n_mocks)
        log(f"[{time.time() - t0:6.0f} s] point {i}/{len(P) - 1}: {len(H)} halos; "
            f"noise at 2' / fiducial {np.mean(sig[i][:, 7]) / np.mean(sig[0][:, 7]) if flag[0] else float('nan'):.4f}")
    return path


def _features(P):
    fid = np.array([getattr(simu.THETA_FID, p) for p in PARAMS])
    width = np.array([priors.MAP[p][1] - priors.MAP[p][0] for p in PARAMS])
    x = (P - fid) / width
    lin = np.c_[np.ones(len(P)), x]
    quad = np.c_[lin, np.array([x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)]).T]
    return fid, width, lin, quad


def _loo(F, y):
    e = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        c, *_ = np.linalg.lstsq(F[m], y[m], rcond=None)
        e.append(y[i] - F[i] @ c)
    return float(np.sqrt(np.mean(np.square(e))))


def fit_noise_level(grid_path, out_path, size=False, log=print):
    """The calibration through the finder's noise level u (quadratic in the scaled parameters): ln sigma at
    each filter size linear in u, s, k, w, f per redshift bin (or per (z, size) cell, where a cell short of
    halos at a point takes that point's z-bin value) quadratic in u. Writes the file
    survey.catalogue_model("calibrated-noise") reads; returns the leave-one-out errors."""
    d = np.load(grid_path)
    assert d["done"].all(), "not all grid points are done"
    P, rows, sig = d["theta"], d["rows"], d["sigma"]
    fid, width, Fl, Fq = _features(P)
    lsig = np.log(sig.mean(1)) - np.log(sig[0].mean(0))[None, :]
    u = lsig[:, 7]
    if size:
        C = d["cells"]
        vals = np.where(np.isnan(C[..., :4]), rows[:, :, None, :4], C[..., :4])
        nz_, ns = vals.shape[1], vals.shape[2]
        vals = vals.reshape(len(P), nz_ * ns, 4)
        upper = [(ZB[i + 1], SB[j + 1]) for i in range(nz_) for j in range(ns)]
    else:
        vals = rows[:, :, :4]
        upper = [(ZB[b + 1],) for b in range(len(ZB) - 1)]
    u_coef, *_ = np.linalg.lstsq(Fq, u, rcond=None)
    Fu = np.c_[np.ones(len(P)), u]
    Fu2 = np.c_[Fu, u * u]
    sig_ab = np.array([np.linalg.lstsq(Fu, lsig[:, j], rcond=None)[0] for j in range(15)])
    out_rows, loo = [], {"u": (_loo(Fl, u), _loo(Fq, u))}
    for b, up in enumerate(upper):
        r = list(up)
        for c, nm in enumerate("skwf"):
            y = vals[:, b, c]
            r += list(np.linalg.lstsq(Fu2, y, rcond=None)[0])
            loo[f"{nm}{b}"] = (_loo(Fl, y), _loo(Fu2, y))
        out_rows.append(r)
    log(f"noise level u: leave-one-out rms {loo['u'][0]:.4f} (linear in the parameters) -> {loo['u'][1]:.4f} (quadratic)")
    np.savez(out_path, fid=fid, width=width, u_coef=u_coef, sig_ab=sig_ab, rows=np.array(out_rows), sigma=sig[0],
             lnth=d["lnth"], specs=d["specs"], noise=str(d["noise"]), size_upper=np.array(SB[1:]) if size else np.array([]))
    return loo


def fit_theta(grid_path, out_path, log=print):
    """The calibration linear in the scaled parameters: ln sigma (tile mean, relative to the fiducial
    per-tile table) and s, k, w, f per z bin. Writes the file survey.catalogue_model("calibrated-theta") reads."""
    d = np.load(grid_path)
    assert d["done"].all(), "not all grid points are done"
    P, rows, sig = d["theta"], d["rows"], d["sigma"]
    fid, width, X, _ = _features(P)
    lsig = np.log(sig.mean(1)) - np.log(sig[0].mean(0))[None, :]
    coef_sig, *_ = np.linalg.lstsq(X, lsig, rcond=None)
    coef_obs = np.zeros((len(ZB) - 1, 4, 5))
    for b in range(len(ZB) - 1):
        for c in range(4):
            coef_obs[b, c] = np.linalg.lstsq(X, rows[:, b, c], rcond=None)[0]
    np.savez(out_path, fid=fid, width=width, g_sig=coef_sig[1:].T, d0_sig=coef_sig[0],
             rows=np.concatenate([np.array(ZB[1:])[:, None], coef_obs.reshape(len(ZB) - 1, 20)], 1),
             sigma=sig[0], lnth=d["lnth"], specs=d["specs"], noise=str(d["noise"]))
    log(f"written {out_path}")


def _tup(a):
    return tuple(tuple(float(v) for v in r) for r in np.atleast_2d(a))


def load_noise_level(path):
    """(CatalogueConfig.qcal_conf, PatchTable of the fiducial per-tile noise, size edges or None)."""
    d = np.load(path)
    qc = (tuple(map(float, d["fid"])), tuple(map(float, d["width"])), tuple(map(float, d["u_coef"])), _tup(d["sig_ab"]), _tup(d["rows"]))
    size = tuple(map(float, d["size_upper"])) if "size_upper" in d.files and len(d["size_upper"]) else None
    return qc, ct.PatchTable(lnth=jnp.asarray(d["lnth"]), lnsig=jnp.asarray(np.log(d["sigma"]))), size


def load_theta(path):
    """(CatalogueConfig.qcal_theta, PatchTable of the fiducial per-tile noise)."""
    d = np.load(path)
    qt = (tuple(map(float, d["fid"])), tuple(map(float, d["width"])), _tup(d["g_sig"]), tuple(map(float, d["d0_sig"])), _tup(d["rows"]))
    return qt, ct.PatchTable(lnth=jnp.asarray(d["lnth"]), lnsig=jnp.asarray(np.log(d["sigma"])))
