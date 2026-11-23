"""Simulated surveys: sets of map-level and catalogue-level simulations of a survey of 10-degree tiles,
each reduced to the 21 numbers of the data vector (4 x 5 counts in q and z, and the mean CMB-lensing
signal p of the clusters with q > 5).

Map level (``simulate_map_set``): per simulation, one cosmology stage, then every tile through
simulator.simulate_tile (halos, painted tSZ, lensed CMB, noise, the matched-filter finder with its masking
pass, merging, confirmation), summed over tiles.

Catalogue level (``simulate_catalogue_set``): catalogue.simulate on the same survey area, with a noise
model chosen by name (``catalogue_model``).

Both write one .npz per set (theta, x, done, ...), save as they go and resume where they stopped. Keys:
simulation i of a set uses fold_in(key(seed), i), and tile t of it fold_in of that with t, so a survey
of N tiles contains the skies of every smaller survey with the same seed.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from . import simulator as simu, catalogue as ct, halos as hl, noise as nz, priors
from .setup import tile_setup, default_cmb

SURVEY_SEED = 20261121          # choice of MSS2 tiles (internal noise path only)
N_CLASSES = 3                   # buffer-size classes of the map-level sets
CATALOGUE_CAPACITIES = (2 ** 18, 2 ** 19, 2 ** 20, 2 ** 21, 2 ** 22, 2 ** 23)


def f_sky(n_tiles):
    return n_tiles * hl.OMEGA_TILE / ct.SR_FULL


def survey_spectra(n_spec, n_tiles):
    """Indices of the measured noise spectra used by the n_tiles tiles (numpy default_rng(SURVEY_SEED);
    without replacement when n_tiles <= n_spec)."""
    return np.sort(np.random.default_rng(SURVEY_SEED).choice(n_spec, n_tiles, replace=n_tiles > n_spec)).astype(np.int32)


def make_survey_sim(cells, tables, cfg, n_tiles, chunk=21, forecast=True):
    """jitted (key, theta vector, static, noise tables, spectrum indices) -> the survey's 21 numbers and
    diagnostics. Static arrays are arguments, not constants, so the compiled program stays small."""

    @jax.jit
    def sim(key, thv, st, nt, spec_arr):
        th = simu.THETA_FID._replace(**{n: thv[j] for j, n in enumerate(priors.NAMES)})
        cs = simu.cosmology_stage(th, cells)

        def tile(args):
            t, s = args
            ni = None if forecast else nz.NoiseInput(nt, s, jnp.float32(1.0))
            return simu.simulate_tile(jax.random.fold_in(key, t), cs, th, st, tables, cells, cfg, ni)
        o = jax.lax.map(tile, (jnp.arange(n_tiles), spec_arr), batch_size=chunk)
        n_p = jnp.sum(o["n_p"])
        x = jnp.concatenate([jnp.sum(o["counts"], axis=0).ravel().astype(jnp.float64),
                             (jnp.sum(o["p_sum"]) / jnp.maximum(n_p, 1))[None]])
        return dict(x=x, overflow=jnp.any(o["overflow"]), n_halos=jnp.sum(o["n_halos"]), n_det=n_p,
                    sigma=jnp.mean(o["sigma"], axis=0))
    return sim


def _load(path, theta, keys):
    if os.path.exists(path):
        old = np.load(path)
        if old["theta"].shape == theta.shape and np.array_equal(old["theta"], theta):
            return {k: np.array(old[k]) for k in keys}
    return None


def simulate_map_set(path, theta, seed, n_tiles=41, noise="forecast", spectra=None, chunk=21, save_every=200,
                     n_classes=N_CLASSES, cmb_cl=None, log=print):
    """Map-level simulations of the parameter vectors ``theta`` (n, 10), written to ``path`` (resumes).

    noise: "forecast" (the 2019 SO forecast, goal depth; public) or "mss2" with ``spectra`` the path to
    measured per-tile noise spectra (noise.load_mss2). Buffers come in ``n_classes`` classes by expected
    halo number, each sized at its heaviest draw; a simulation that overflows is rerun in the next class
    with the same keys. Returns the dict that was saved."""
    n = len(theta)
    forecast = noise == "forecast"
    if forecast:
        nt, specs = None, np.zeros(n_tiles, np.int32)
    else:
        nt, _ = nz.load_mss2(spectra, "mean")
        specs = survey_spectra(nt.F.shape[0], n_tiles)
    cmb = cmb_cl or default_cmb()
    ts = tile_setup(simu.THETA_FID, cmb, log=log)
    cells, tb, st = ts.cells, ts.tables, ts.static
    base = jax.random.key(seed)

    lam_tot = jax.jit(jax.vmap(lambda v: simu.cosmology_stage(
        simu.THETA_FID._replace(**{nm: v[j] for j, nm in enumerate(priors.NAMES)}), cells).lam.sum()))
    L = np.concatenate([np.asarray(lam_tot(jnp.asarray(theta[a:a + 128]))) for a in range(0, n, 128)])
    edges = np.quantile(L, np.linspace(0, 1, n_classes + 1)[1:-1]) if n > n_classes else np.array([])
    cls = np.searchsorted(edges, L, side="left")
    class_theta = [theta[cls == c][np.argmax(L[cls == c])] if np.any(cls == c) else theta[np.argmax(L)]
                   for c in range(n_classes)]
    cfgs = []
    for c in range(n_classes):
        cfgs.append(tile_setup(priors.to_theta(class_theta[c]), cmb, log=log).cfg)
        log(f"class {c}: {int(np.sum(cls == c))} draws, halos per tile up to "
            f"{L[cls == c].max() if np.any(cls == c) else 0:.0f}, halo buffer {cfgs[c].halo_cap}")

    keys = ("x", "overflow", "n_halos", "n_det", "sigma", "class", "done")
    r = _load(path, theta, keys) or dict(x=np.full((n, 21), np.nan), overflow=np.zeros(n, bool),
                                         n_halos=np.zeros(n, np.int64), n_det=np.zeros(n, np.int64),
                                         sigma=np.zeros((n, 15)), **{"class": np.full(n, -1)},
                                         done=np.zeros(n, bool))
    if r["done"].any():
        log(f"resuming: {r['done'].sum()} done")
    settings = json.dumps(dict(n_tiles=n_tiles, noise="2019 SO forecast, goal" if forecast else "MSS2 four-split mean, depth 1",
                               lens_table="goal", seed=seed, survey_seed=SURVEY_SEED, classes=n_classes))

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, theta=theta, names=np.array(priors.NAMES), specs=specs, settings=settings, **r)

    sims = {}
    t0 = time.time()
    count = 0
    todo = np.flatnonzero(~r["done"])
    while todo.size:
        for i in todo:
            c = int(cls[i])
            if c not in sims:
                sims[c] = make_survey_sim(cells, tb, cfgs[c], n_tiles, chunk, forecast=forecast)
            o = jax.device_get(sims[c](jax.random.fold_in(base, int(i)), jnp.asarray(theta[i]), st, nt, jnp.asarray(specs)))
            if bool(o["overflow"]) and c < n_classes - 1:
                cls[i] = c + 1
                continue
            for k in ("x", "overflow", "n_halos", "n_det", "sigma"):
                r[k][i] = o[k]
            r["class"][i], r["done"][i] = c, True
            count += 1
            if count % save_every == 0:
                save()
                log(f"[{time.time() - t0:7.0f} s] {r['done'].sum()}/{n} ({count / (time.time() - t0):.2f} per s)")
        todo = np.flatnonzero(~r["done"])
    save()
    nd = r["n_det"]
    log(f"done: {n} in {time.time() - t0:.0f} s; detections {nd.min()}-{nd.max()} (median {int(np.median(nd))}); "
        f"overflow left {int(r['overflow'].sum())}; mean p {np.nanmin(r['x'][:, 20]):.3f}-{np.nanmax(r['x'][:, 20]):.3f}")
    return r


# ---- catalogue level -----------------------------------------------------------------------------------

CATALOGUE_MODELS = ("paper", "published", "calibrated-point", "calibrated-noise", "calibrated-theta")


def catalogue_model(model, n_tiles=41, calibration=None, cmb_cl=None, mode="goal"):
    """(sigma_y0 table, sigma_kappa table, CatalogueConfig) of a catalogue model of the survey.

    paper              the paper itself: its sigma_y0 and sigma_kappa fits, 40% of the sky
    published          the paper's model of this survey: our filter's analytic sigma_y0 for the
                       forecast noise and the CMB (no tSZ), our goal sigma_kappa, f_sky of n_tiles
    calibrated-point   calibrated on map-level mocks at one point (calibration.calibrate_point's file)
    calibrated-noise   calibrated across the prior through the finder's noise level
                       (calibration.fit_noise_level's file; with size cells if it has them)
    calibrated-theta   calibrated across the prior, linear in the parameters (calibration.fit_theta's file)

    The calibrated models need ``calibration``, the path to the corresponding file. Their per-tile noise
    tables are averaged into one (forecast noise: the tiles differ only by sampling)."""
    from . import calibration as cal
    if model == "paper":
        sz, le = ct.paper_tables()
        return sz, le, ct.CatalogueConfig()
    cfg = ct.CatalogueConfig(f_sky=f_sky(n_tiles), capacity=2 ** 19)
    le = ct.lens_table(mode)
    if model == "published":
        sig, lnth = ct.analytic_sigma_y0_forecast(cmb_cl or default_cmb(), mode)
        return ct.PatchTable(lnth=jnp.asarray(lnth), lnsig=jnp.asarray(np.log(sig))), le, cfg
    if calibration is None:
        raise ValueError(f"model {model!r} needs a calibration file")
    if model == "calibrated-point":
        qc, tab = ct.load_qcal(calibration)
        cfg = cfg._replace(qcal=qc)
    elif model == "calibrated-noise":
        qc, tab, size = cal.load_noise_level(calibration)
        cfg = cfg._replace(qcal_conf=qc, qcal_size=size)
    elif model == "calibrated-theta":
        qt, tab = cal.load_theta(calibration)
        cfg = cfg._replace(qcal_theta=qt)
    else:
        raise ValueError(f"unknown catalogue model {model!r}; one of {CATALOGUE_MODELS}")
    tab = tab._replace(lnsig=jnp.log(jnp.mean(jnp.exp(tab.lnsig), axis=0, keepdims=True)))
    return tab, le, cfg


def simulate_catalogue_set(path, theta, seed, sz, le, cfg, batch=16, chunk=512, log=print, info=None):
    """Catalogue-level simulations of the parameter vectors ``theta`` (n, 10), written to ``path``
    (resumes). Each draw goes to the smallest buffer that holds its expected halo number plus 8 standard
    deviations; an overflowing draw is rerun in the next one."""
    n = len(theta)
    cells = ct.survey_cells()
    base = jax.random.key(seed)
    t0 = time.time()
    lam_tot = jax.jit(jax.vmap(lambda *v: ct.survey_rates(simu.THETA_FID._replace(**dict(zip(priors.NAMES, v))), cells)[0].sum()))
    N_exp = np.concatenate([np.asarray(lam_tot(*[jnp.asarray(theta[a:a + 256, j]) for j in range(len(priors.NAMES))]))
                            for a in range(0, n, 256)]) * cfg.f_sky
    caps = np.asarray(CATALOGUE_CAPACITIES)
    cls = np.searchsorted(caps, N_exp + 8.0 * np.sqrt(N_exp))
    if np.any(cls >= len(caps)):
        raise ValueError(f"expected halos up to {N_exp.max():.3g}: beyond the largest buffer")
    log(f"{n} draws; expected halos {N_exp.min():.3g}-{N_exp.max():.3g}; buffers {np.bincount(cls, minlength=len(caps)).tolist()}")

    keys = ("x", "n_halos", "n_sel", "capacity", "done")
    r = _load(path, theta, keys) or dict(x=np.full((n, 21), np.nan), n_halos=np.zeros(n, np.int64),
                                         n_sel=np.zeros(n, np.int64), capacity=np.zeros(n, np.int64),
                                         done=np.zeros(n, bool))
    if r["done"].any():
        log(f"resuming: {r['done'].sum()} done")
    sims = {}

    def sim_fn(c):
        if c not in sims:
            cfg_c = cfg._replace(capacity=int(caps[c]))

            def one(i, *v):
                th = simu.THETA_FID._replace(**dict(zip(priors.NAMES, v)))
                cat = ct.simulate(jax.random.fold_in(base, i), th, cells, sz, le, cfg_c)
                return ct.data_vector(cat), cat.n_halos, cat.n_sel, cat.overflow
            sims[c] = jax.jit(jax.vmap(one))
        return sims[c]

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, theta=theta, names=np.array(priors.NAMES), **r,
                 settings=json.dumps(dict(f_sky=cfg.f_sky, seed=seed, qcal=cfg.qcal, qcal_theta=cfg.qcal_theta,
                                          qcal_conf=cfg.qcal_conf, qcal_size=cfg.qcal_size, **(info or {}))))

    t0 = time.time()
    todo = np.flatnonzero(~r["done"])
    while todo.size:
        for c in np.unique(cls[todo]):
            idx = todo[cls[todo] == c]
            for a in range(0, idx.size, batch):
                b = idx[a:a + batch]
                pad = np.concatenate([b, np.repeat(b[-1:], batch - b.size)])
                xv, nh, ns, ov = jax.device_get(sim_fn(int(c))(jnp.asarray(pad, dtype=jnp.uint32),
                                                               *[jnp.asarray(theta[pad, j]) for j in range(len(priors.NAMES))]))
                for k, i in enumerate(b):
                    if ov[k]:
                        cls[i] = min(cls[i] + 1, len(caps) - 1)
                        continue
                    r["x"][i], r["n_halos"][i], r["n_sel"][i], r["capacity"][i], r["done"][i] = xv[k], nh[k], ns[k], caps[c], True
                if r["done"].sum() % chunk < batch:
                    save()
        todo = np.flatnonzero(~r["done"])
    save()
    log(f"done: {n} in {time.time() - t0:.0f} s; selected {r['n_sel'].min()}-{r['n_sel'].max()} "
        f"(median {int(np.median(r['n_sel']))}); mean p {np.nanmin(r['x'][:, 20]):.3f}-{np.nanmax(r['x'][:, 20]):.3f}")
    return r
