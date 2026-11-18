"""Stacked CMB lensing at the fiducial point: map level against catalogue level.

    python benchmarks/mapsim/lens_check.py --tiles 32

Map level: the tile simulator (masking pass on); the stacked mean p over the
data-vector detections of all tiles, Σ p_sum / Σ n_p, with its standard error
from the tile-to-tile scatter.

Catalogue level, from the same λ cells: the mean of p̄(M, z) e^{σ_lnp²/2} over
halos with M ≥ 5e13, weighted by the probability that they land in the data
vector, P(5 < q_obs < 40) with q_obs ~ N(sqrt(q_true² + k), 1) and q_true =
y0/σ(θ500) using the finder's own σ (k = 1, as in counts_modes.py), and
0.01 < z < 1.5. Expect agreement to a few per cent: the catalogue model
already matches the counts to 3-5%.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike  # noqa: E402

from classy_szlite.sim import instrument as ins, lensing, mmf, simulator as simu, summaries as sm  # noqa: E402
from classy_szlite.sz_scaling import y0_median, theta500_arcmin  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "lensing")


def catalogue_mean_p(cs, theta, cells, st, sig_j, k_dof=1.0, M_min=5e13, n_gh=32):
    ze, me = cells.z_edges, cells.lnM_edges
    zc, lnMc = 0.5 * (ze[1:] + ze[:-1]), 0.5 * (me[1:] + me[:-1])
    lam = np.asarray(cs.lam)
    cg = cs.cg
    Ez = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Hz)) / (theta.h * 100.0 / 299792.458)
    DA = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Da))
    M = np.exp(lnMc)
    ybar = np.asarray(y0_median(M[None, :], Ez[:, None], theta.h, theta.A_SZ, theta.alpha_SZ, theta.one_minus_b))
    th = np.asarray(theta500_arcmin(M[None, :], Ez[:, None], DA[:, None], theta.h, theta.one_minus_b))
    sig = np.exp(np.interp(np.log(th), np.log(mmf.THETAS15), np.log(sig_j)))
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    mu = np.sqrt(((ybar / sig)[..., None] * np.exp(theta.sigma_lnq * x)) ** 2 + k_dof)
    P = np.einsum("abg,g->ab", norm.cdf(sm.Q_EDGES[-1] - mu) - norm.cdf(sm.Q_EDGES[0] - mu), w)
    inz = (zc > sm.Z_EDGES[0]) & (zc < sm.Z_EDGES[-1])
    W = lam * P * (M >= M_min)[None, :] * inz[:, None]
    Zg, Mg = np.meshgrid(zc, M, indexing="ij")
    pbar = np.asarray(lensing.p_mean(jnp.asarray(Mg), jnp.asarray(Zg), cg, theta.h, theta.beta_lens,
                                     cs.chi_star, st.lens_lnth, st.lens_lnsig))
    return float(np.sum(W * pbar) / np.sum(W) * np.exp(theta.sigma_lnp ** 2 / 2)), float(W.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=32)
    args = ap.parse_args()
    theta = simu.THETA_FID
    cells, tb, cs, cfg, st_goal, info = spike.setup(theta)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    statics = {"goal": st_goal, "baseline": simu.tile_static(cmb, tb, mode="baseline")}
    f = jax.jit(jax.vmap(lambda key, st: simu.simulate_tile(key, cs, theta, st, tb, cells, cfg), in_axes=(0, None)))
    keys = spike.tile_keys(args.tiles)
    res = {}
    for mode, st in statics.items():
        o = jax.device_get(f(keys, st))
        ps, n = np.asarray(o["p_sum"]), np.asarray(o["n_p"]).astype(float)
        mean = ps.sum() / n.sum()
        per = ps / n
        sem = np.sqrt(np.sum(n ** 2 * (per - mean) ** 2) / (n.sum() ** 2) * len(n) / (len(n) - 1))
        tot = o["counts"].sum((1, 2))
        assert np.array_equal(tot, o["n_p"]), "stacked selection differs from the histogram"
        cat, ncat = catalogue_mean_p(cs, theta, cells, st, np.asarray(o["sigma"]).mean(0))
        res[mode] = dict(map_mean_p=float(mean), map_sem=float(sem), n_per_tile=float(n.mean()),
                         catalogue_mean_p=cat, catalogue_n_per_tile=ncat)
        print(f"{mode}: stacked p = {mean:.4f} ± {sem:.4f} over {int(n.sum())} detections ({n.mean():.1f} per tile); "
              f"catalogue {cat:.4f} ({ncat:.1f} per tile); map/catalogue {mean / cat:.3f}; "
              f"over 40% of the sky the error on the mean would be {1 / np.sqrt(n.mean() * 165.0) * np.sqrt(1 + 0.0):.4f} (noise only)")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "lens_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", os.path.join(OUT, "lens_check.json"))


if __name__ == "__main__":
    main()
