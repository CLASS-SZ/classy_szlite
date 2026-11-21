"""Map-level simulations of a 10%-sky survey for NPE: parameters from the map-level prior and the 21 numbers.

    python benchmarks/mapsim/maps_trainset.py --set truth     # 100 at the fiducial point (the mocks)
    python benchmarks/mapsim/maps_trainset.py --set heldout   # 500 prior draws (coverage tests)
    python benchmarks/mapsim/maps_trainset.py --set train     # 10^4 prior draws

The survey: 41 periodic 10-degree tiles (about 10% of the sky), each with its own MSS2 noise
(noise.py: the tile's measured 6 x 6 spectra, four-split mean, depth 1), the same 41 spectra in
every simulation (a fixed footprint), drawn without replacement from the 107 measured tiles
(numpy default_rng(SURVEY_SEED)). Each simulation: one cosmology stage, then
simulator.simulate_tile on every tile (halos, painted tSZ, lensed CMB, MSS2 noise, the finder with
its masking pass, merging, confirmation), summed: the 4 x 5 counts and the mean stacked p over
every confirmed detection with q > 5 (simulator.stacked_lensing). sigma_kappa: the goal table.

Prior (PLAN section 3.2, decided 2026-11-18, the map-level box): sigma8 U(0.770, 0.852),
omega_cdm U(0.0804, 0.169), A_SZ U(-4.53, -4.08), alpha_SZ U(1, 1.24); the paper's Gaussians for
h, omega_b, n_s, sigma_lnq, beta_lens, sigma_lnp (the two widths truncated at 0).

Keys: simulation i of a set uses fold_in(key(SEED[set]), i), tile t fold_in of that with t.
Capacity classes: the draws are split by expected halo number into CLASSES groups; each group's
buffers (halos, painter, peaks) come from spike.setup at its heaviest draw; a simulation that
overflows any buffer is rerun in the next class (same keys, so the same sky).
Output: $MAPSIM_OUT/maps_npe/{set}.npz (theta, x, overflow, n_halos, n_det, sigma, class),
written every chunk; a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spike  # noqa: E402
from classy_szlite.sim import simulator as simu, noise as nz, instrument as ins  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "maps_npe")
SPECTRA = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2", "spectra", "spectra.npz")
NAMES = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ", "h", "omega_b", "n_s", "sigma_lnq", "beta_lens", "sigma_lnp")
UNIFORM = {"sigma8": (0.770, 0.852), "omega_cdm": (0.0804, 0.169), "A_SZ": (-4.53, -4.08), "alpha_SZ": (1.0, 1.24)}
GAUSS = {"h": (0.674, 0.01), "omega_b": (0.0222, 0.00015), "n_s": (0.96, 0.0042), "sigma_lnq": (0.173, 0.05),
         "beta_lens": (0.92, 0.01), "sigma_lnp": (0.22, 0.05)}
POSITIVE = ("sigma_lnq", "sigma_lnp")
SEED = {"train": 20261124, "heldout": 20261125, "truth": 20261126}
SURVEY_SEED, N_TILES, N_CLASSES = 20261121, 41, 3


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


def survey_spectra(n_spec):
    return np.sort(np.random.default_rng(SURVEY_SEED).choice(n_spec, N_TILES, replace=False)).astype(np.int32)


def theta_of(row):
    return simu.THETA_FID._replace(**dict(zip(NAMES, (float(v) for v in row))))


def make_sim(cells, tb, cfg, st, nt, specs, chunk):
    spec_arr = jnp.asarray(specs)

    @jax.jit
    def sim(key, thv):
        th = simu.THETA_FID._replace(**{n: thv[j] for j, n in enumerate(NAMES)})
        cs = simu.cosmology_stage(th, cells)

        def tile(args):
            t, s = args
            ni = nz.NoiseInput(nt, s, jnp.float32(1.0))
            return simu.simulate_tile(jax.random.fold_in(key, t), cs, th, st, tb, cells, cfg, ni)
        o = jax.lax.map(tile, (jnp.arange(N_TILES), spec_arr), batch_size=chunk)
        n_p = jnp.sum(o["n_p"])
        x = jnp.concatenate([jnp.sum(o["counts"], axis=0).ravel().astype(jnp.float64),
                             (jnp.sum(o["p_sum"]) / jnp.maximum(n_p, 1))[None]])
        return dict(x=x, overflow=jnp.any(o["overflow"]), n_halos=jnp.sum(o["n_halos"]), n_det=n_p,
                    sigma=jnp.mean(o["sigma"], axis=0))
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SEED), required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=21, help="tiles per vectorised batch")
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()
    n = args.n or {"train": 10000, "heldout": 500, "truth": 100}[args.set]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.set}.npz")
    rng = np.random.default_rng(SEED[args.set])
    theta = (np.tile([getattr(simu.THETA_FID, k) for k in NAMES], (n, 1)) if args.set == "truth" else draw_prior(n, rng))
    nt, info = nz.load_mss2(SPECTRA, "mean")
    specs = survey_spectra(nt.F.shape[0])
    cells, tb, cs0, cfg0, st0, _ = spike.setup(simu.THETA_FID)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb, tb, mode="goal")
    base = jax.random.key(SEED[args.set])

    # expected halos per tile -> capacity classes (each sized at its heaviest draw)
    lam_tot = jax.jit(jax.vmap(lambda v: simu.cosmology_stage(
        simu.THETA_FID._replace(**{nm: v[j] for j, nm in enumerate(NAMES)}), cells).lam.sum()))
    L = np.concatenate([np.asarray(lam_tot(jnp.asarray(theta[a:a + 128]))) for a in range(0, n, 128)])
    edges = np.quantile(L, np.linspace(0, 1, N_CLASSES + 1)[1:-1]) if n > N_CLASSES else np.array([])
    cls = np.searchsorted(edges, L, side="left")
    class_theta = [theta[cls == c][np.argmax(L[cls == c])] if np.any(cls == c) else theta[np.argmax(L)]
                   for c in range(N_CLASSES)]
    cfgs = []
    for c in range(N_CLASSES):
        _, _, _, cfg_c, _, inf_c = spike.setup(theta_of(class_theta[c]))
        cfgs.append(cfg_c)
        print(f"class {c}: {int(np.sum(cls == c))} draws, halos per tile up to {L[cls == c].max() if np.any(cls == c) else 0:.0f}, "
              f"halo buffer {cfg_c.halo_cap}", flush=True)
    print(f"{args.set}: {n} draws; survey spectra {specs.tolist()}", flush=True)

    x = np.full((n, 21), np.nan)
    ovf = np.zeros(n, bool)
    n_halos = np.zeros(n, np.int64)
    n_det = np.zeros(n, np.int64)
    sig = np.zeros((n, 15))
    used = np.full(n, -1)
    done = np.zeros(n, bool)
    if os.path.exists(path):
        old = np.load(path)
        if old["theta"].shape == theta.shape and np.array_equal(old["theta"], theta):
            x, ovf, n_halos, n_det, sig, used, done = (np.array(old[k]) for k in
                                                       ("x", "overflow", "n_halos", "n_det", "sigma", "class", "done"))
            print(f"resuming: {done.sum()} done", flush=True)

    def save():
        np.savez(path, theta=theta, x=x, overflow=ovf, n_halos=n_halos, n_det=n_det, sigma=sig, done=done,
                 names=np.array(NAMES), specs=specs, **{"class": used},
                 settings=json.dumps(dict(n_tiles=N_TILES, noise="MSS2 four-split mean, depth 1", lens_table="goal",
                                          seed=SEED[args.set], survey_seed=SURVEY_SEED, classes=N_CLASSES)))

    sims = {}
    t0 = time.time()
    todo = np.flatnonzero(~done)
    count = 0
    while todo.size:
        for i in todo:
            c = int(cls[i])
            if c not in sims:
                sims[c] = make_sim(cells, tb, cfgs[c], st, nt, specs, args.chunk)
            o = jax.device_get(sims[c](jax.random.fold_in(base, int(i)), jnp.asarray(theta[i])))
            if bool(o["overflow"]) and c < N_CLASSES - 1:
                cls[i] = c + 1                                     # rerun in the next class
                continue
            x[i], ovf[i], n_halos[i], n_det[i], sig[i], used[i], done[i] = (
                o["x"], o["overflow"], o["n_halos"], o["n_det"], o["sigma"], c, True)
            count += 1
            if count % args.save_every == 0:
                save()
                print(f"[{time.time() - t0:7.0f} s] {done.sum()}/{n} ({count / (time.time() - t0):.2f} per s)", flush=True)
        todo = np.flatnonzero(~done)
    save()
    print(f"done: {n} in {time.time() - t0:.0f} s; detections {n_det.min()}-{n_det.max()} (median {int(np.median(n_det))}); "
          f"overflow left {int(ovf.sum())}; mean p {np.nanmin(x[:, 20]):.3f}-{np.nanmax(x[:, 20]):.3f}", flush=True)


if __name__ == "__main__":
    main()
