"""Simulations for catalogue-level NPE: parameters from the paper's prior and the 21 numbers.

    python benchmarks/mapsim/catalogue_trainset.py --set train [--n 10000] [--batch 16]
    python benchmarks/mapsim/catalogue_trainset.py --set heldout --n 500     # coverage tests
    python benchmarks/mapsim/catalogue_trainset.py --set truth --n 100       # the paper's validation

Variant (0), paper-exact (sim/catalogue.py): the paper's sigma_y0 and sigma_kappa fits,
extrapolated; 40% of the sky; dof = 0; r = 0; a_lens = 1; (1 - b) = 0.8.

Prior (paper Table 1), parameter order:
  sigma8 U(0.7, 0.9), omega_cdm U(0.0804, 0.200), A_SZ U(-4.41, -4.21), alpha_SZ U(1, 1.24),
  h N(0.674, 0.01), omega_b N(0.0222, 0.00015), n_s N(0.96, 0.0042), sigma_lnq N(0.173, 0.05),
  beta_lens N(0.92, 0.01), sigma_lnp N(0.22, 0.05)
omega_cdm is clipped at 0.200, the emulators' training edge (the paper's prior reaches 0.204);
sigma_lnq and sigma_lnp are truncated at 0 (redrawn), which removes 3e-4 and 5e-6 of the
Gaussian mass. 'truth' uses the fiducial point (simulator.THETA_FID).

--variant forecast41: the same with the 2019 SO forecast noise at goal depth (one sigma_y0 table
for every tile), output in catalogue_npe/forecast41/ (PLAN section 10, paper 1).

--variant calibcs41 / calibcsf41: the same per (redshift, theta500) cell (qcal_grid.py run --size, then
fit --model conf --size: calib_conf_zs.npz).

--variant calibc41 / calibcf41 (variant (iv)): the same 25 points, with the calibration written through the
finder's noise level (quadratic in the parameters; qcal_grid.py fit --model conf: calib_conf.npz).

--variant calibt41 / calibtf41 (variant (iii)): the calibration follows the parameters, fitted over
24 points of the prior (qcal_grid.py: calib_theta.npz; CatalogueConfig.qcal_theta), MSS2 or forecast noise.

--variant calibf41: the same calibrated model for the forecast-noise survey (qcal_forecast/).

--variant calib41 (variant (ii)): the same survey with the finder's measured noise per tile and its
calibrated observation per redshift bin (qcal.py: calibration.npz, from map-level mocks at the
fiducial point; CatalogueConfig.qcal), held fixed over the prior. Output in catalogue_npe/calib41/.

--variant survey41 (variant (i), the catalogue model of the 10%-sky map-level survey,
maps_trainset.py): 41 equal-area patches, one per survey tile, each with our filter's analytic
sigma_y0 from that tile's MSS2 spectra plus the CMB (catalogue.analytic_sigma_y0), our goal
sigma_kappa table, f_sky = 41 tiles; the map-level prior box (maps_trainset.py). Output in
catalogue_npe/survey41/.

Keys: simulation i of a set uses fold_in(key(SEED[set]), i); parameters come from numpy's
default_rng(SEED[set]). Capacity classes: each draw goes to the smallest buffer that holds its
expected halo number plus 8 standard deviations; an overflowing draw is rerun in the next class.
Output: $MAPSIM_OUT/catalogue_npe/{set}.npz with theta (n, 10), x (n, 21), n_halos, n_sel,
capacity, and the settings; written every chunk, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import catalogue as ct, simulator as simu, halos as hl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "catalogue_npe")
NAMES = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ", "h", "omega_b", "n_s", "sigma_lnq", "beta_lens", "sigma_lnp")
UNIFORM = {"sigma8": (0.7, 0.9), "omega_cdm": (0.0804, 0.200), "A_SZ": (-4.41, -4.21), "alpha_SZ": (1.0, 1.24)}
GAUSS = {"h": (0.674, 0.01), "omega_b": (0.0222, 0.00015), "n_s": (0.96, 0.0042), "sigma_lnq": (0.173, 0.05),
         "beta_lens": (0.92, 0.01), "sigma_lnp": (0.22, 0.05)}
POSITIVE = ("sigma_lnq", "sigma_lnp")
SEED = {"train": 20261121, "heldout": 20261122, "truth": 20261123}
CLASSES = (2 ** 18, 2 ** 19, 2 ** 20, 2 ** 21, 2 ** 22, 2 ** 23)


def draw_prior(n, rng):
    th = np.empty((n, len(NAMES)))
    for j, name in enumerate(NAMES):
        if name in UNIFORM:
            th[:, j] = rng.uniform(*UNIFORM[name], n)
        else:
            mu, sd = GAUSS[name]
            v = rng.normal(mu, sd, n)
            if name in POSITIVE:
                while np.any(v <= 0):
                    bad = v <= 0
                    v[bad] = rng.normal(mu, sd, bad.sum())
            th[:, j] = v
    return th


def to_theta(row):
    d = dict(zip(NAMES, (float(v) for v in row)))
    return simu.THETA_FID._replace(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SEED), required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--variant", choices=("paper", "survey41", "forecast41", "calib41", "calibf41", "calibt41", "calibtf41", "calibc41", "calibcf41", "calibcs41", "calibcsf41"), default="paper")
    args = ap.parse_args()
    n = args.n or {"train": 10000, "heldout": 500, "truth": 100}[args.set]
    out = OUT if args.variant == "paper" else os.path.join(OUT, args.variant)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{args.set}.npz")
    rng = np.random.default_rng(SEED[args.set])
    cells = ct.survey_cells()
    if args.variant == "paper":
        prior = draw_prior
        sz, le = ct.paper_tables()
        cfg0 = ct.CatalogueConfig()
    else:
        import maps_trainset as mt
        from classy_szlite.sim import noise as nz, instrument as ins
        import spike
        prior = mt.draw_prior
        nt, _ = nz.load_mss2(mt.SPECTRA, "mean")
        specs = mt.survey_spectra(nt.F.shape[0])
        cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
        qcal, qcal_theta, qcal_conf, qcal_size = None, None, None, None
        if args.variant in ("calibc41", "calibcf41", "calibcs41", "calibcsf41"):  # through the finder's noise level
            import qcal_grid
            fc = args.variant in ("calibcf41", "calibcsf41")
            cal_path = os.path.join(qcal_grid.out_dir("forecast" if fc else "mss2"),
                                    "calib_conf_zs.npz" if args.variant.startswith("calibcs") else "calib_conf.npz")
            qcal_conf, tab, qcal_size = qcal_grid.load_calib_conf(cal_path)
            sig, lnth = np.exp(np.asarray(tab.lnsig)), np.asarray(tab.lnth)
            assert np.array_equal(np.load(cal_path)["specs"], specs)
        elif args.variant in ("calibt41", "calibtf41"):  # calibration following the parameters (qcal_grid.py)
            import qcal_grid
            cal_path = os.path.join(qcal_grid.out_dir("forecast" if args.variant == "calibtf41" else "mss2"), "calib_theta.npz")
            qcal_theta, tab = qcal_grid.load_calib_theta(cal_path)
            sig, lnth = np.exp(np.asarray(tab.lnsig)), np.asarray(tab.lnth)
            assert np.array_equal(np.load(cal_path)["specs"], specs)
        elif args.variant == "forecast41":            # the same survey with the 2019 forecast noise (goal)
            sig, lnth = ct.analytic_sigma_y0_forecast(cmb, "goal")
        elif args.variant in ("calib41", "calibf41"):   # the finder's noise and observation, from qcal.py
            cal_path = os.path.join(os.environ.get("MAPSIM_OUT", "."), "qcal" if args.variant == "calib41" else "qcal_forecast",
                                    "calibration.npz")
            qcal, tab = ct.load_qcal(cal_path)
            sig, lnth = np.exp(np.asarray(tab.lnsig)), np.asarray(tab.lnth)
            assert np.array_equal(np.load(cal_path)["specs"], specs)
        else:
            sig, lnth = ct.analytic_sigma_y0(nt, specs, cmb)
        sz = ct.PatchTable(lnth=jnp.asarray(lnth), lnsig=jnp.asarray(np.log(sig)))
        le = ct.lens_table("goal")
        cfg0 = ct.CatalogueConfig(f_sky=mt.N_TILES * hl.OMEGA_TILE / ct.SR_FULL, capacity=2 ** 19, qcal=qcal,
                                  qcal_theta=qcal_theta, qcal_conf=qcal_conf, qcal_size=qcal_size)
        if qcal is not None:
            print(f"calibration (z_upper, s, k, w, f): {qcal}", flush=True)
        print(f"{args.variant}: spectra {specs.tolist()}; sigma_y0 at 0.5' {sig[:, 0].min():.3g}-{sig[:, 0].max():.3g}, "
              f"at 15' {sig[:, -1].min():.3g}-{sig[:, -1].max():.3g}; f_sky {cfg0.f_sky:.4f}", flush=True)
        np.savez(os.path.join(out, "sigma_y0.npz"), sigma=sig, lnth=lnth, specs=specs)
    theta = (np.tile([getattr(simu.THETA_FID, k) for k in NAMES], (n, 1)) if args.set == "truth" else prior(n, rng))
    base = jax.random.key(SEED[args.set])

    # expected halo numbers -> capacity classes
    t0 = time.time()
    lam_tot = jax.jit(jax.vmap(lambda *v: ct.survey_rates(simu.THETA_FID._replace(**dict(zip(NAMES, v))), cells)[0].sum()))
    N_exp = np.concatenate([np.asarray(lam_tot(*[jnp.asarray(theta[a:a + 256, j]) for j in range(len(NAMES))]))
                            for a in range(0, n, 256)]) * cfg0.f_sky
    need = N_exp + 8.0 * np.sqrt(N_exp)
    cls = np.searchsorted(np.asarray(CLASSES), need)
    if np.any(cls >= len(CLASSES)):
        raise SystemExit(f"expected halos up to {N_exp.max():.3g}: beyond the largest class")
    print(f"{args.set}: {n} draws; expected halos {N_exp.min():.3g}-{N_exp.max():.3g}; classes "
          f"{np.bincount(cls, minlength=len(CLASSES)).tolist()} ({time.time() - t0:.0f} s)", flush=True)

    x = np.full((n, 21), np.nan)
    n_halos = np.zeros(n, np.int64)
    n_sel = np.zeros(n, np.int64)
    cap = np.zeros(n, np.int64)
    done = np.zeros(n, bool)
    if os.path.exists(path):                                   # resume
        old = np.load(path)
        if old["theta"].shape == theta.shape and np.array_equal(old["theta"], theta):
            x, n_halos, n_sel, cap, done = (np.array(old[k]) for k in ("x", "n_halos", "n_sel", "capacity", "done"))
            print(f"resuming: {done.sum()} done", flush=True)

    sims = {}

    def sim_fn(c):
        if c not in sims:
            cfg = cfg0._replace(capacity=CLASSES[c])

            def one(i, *v):
                th = simu.THETA_FID._replace(**dict(zip(NAMES, v)))
                cat = ct.simulate(jax.random.fold_in(base, i), th, cells, sz, le, cfg)
                return ct.data_vector(cat), cat.n_halos, cat.n_sel, cat.overflow
            sims[c] = jax.jit(jax.vmap(one))
        return sims[c]

    def save():
        np.savez(path, theta=theta, x=x, n_halos=n_halos, n_sel=n_sel, capacity=cap, done=done, names=np.array(NAMES),
                 settings=json.dumps(dict(variant=args.variant, f_sky=cfg0.f_sky, qcal=cfg0.qcal, qcal_theta=cfg0.qcal_theta, qcal_conf=cfg0.qcal_conf, qcal_size=cfg0.qcal_size, dof=cfg0.dof, r=cfg0.corr_lnq_lnp,
                                          a_lens=cfg0.a_lens, one_minus_b=simu.THETA_FID.one_minus_b,
                                          omega_cdm_max=0.200, seed=SEED[args.set], classes=list(CLASSES))))

    t0 = time.time()
    todo = np.flatnonzero(~done)
    while todo.size:
        for c in np.unique(cls[todo]):
            idx = todo[cls[todo] == c]
            for a in range(0, idx.size, args.batch):
                b = idx[a:a + args.batch]
                pad = np.concatenate([b, np.repeat(b[-1:], args.batch - b.size)])     # fixed batch shape
                xv, nh, ns, ov = jax.device_get(sim_fn(int(c))(jnp.asarray(pad, dtype=jnp.uint32),
                                                               *[jnp.asarray(theta[pad, j]) for j in range(len(NAMES))]))
                for k, i in enumerate(b):
                    if ov[k]:
                        cls[i] = min(cls[i] + 1, len(CLASSES) - 1)              # rerun in the next class
                        continue
                    x[i], n_halos[i], n_sel[i], cap[i], done[i] = xv[k], nh[k], ns[k], CLASSES[c], True
                if done.sum() % args.chunk < args.batch:
                    save()
                    rate = done.sum() / (time.time() - t0)
                    print(f"[{time.time() - t0:7.0f} s] {done.sum()}/{n} done ({rate:.2f} per s)", flush=True)
        todo = np.flatnonzero(~done)
    save()
    print(f"done: {n} in {time.time() - t0:.0f} s; selected {n_sel.min()}-{n_sel.max()} "
          f"(median {int(np.median(n_sel))}); mean p {np.nanmin(x[:, 20]):.3f}-{np.nanmax(x[:, 20]):.3f}", flush=True)


if __name__ == "__main__":
    main()
