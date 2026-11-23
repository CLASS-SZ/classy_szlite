"""Cluster counts at SO goal and baseline depth: map level against catalogue level.

    python benchmarks/mapsim/counts_modes.py --tiles 32

Map level: the tile simulator at the fiducial point, the same tile keys in
both modes (same halos, CMB and noise seeds; only the white-noise level
differs). The data vector counts confirmed detections with 5 < q < 40 and
0.01 < z < 1.5.

Catalogue level, from the same λ cells:

    N = Σ_cells λ_c ⟨P(5 < q_obs < 40)⟩_scatter,   M ≥ M_conf,
    q_obs ~ N(sqrt(q_true² + k), 1),   q_true = y0 / σ(θ500),

with y0 log-normal about the median relation, and σ(θ) interpolated in ln θ
between the 15 filter scales, either from the analytic covariance (CMB + noise,
no tSZ) or from the tiles' own covariance (one pass, unmasked tSZ included).
k = 0 is the paper's model; k = 2, 3 add an optimisation-bias term.
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
from scipy.stats import norm as _norm

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike  # noqa: E402

from classy_szlite.sim import instrument as ins, mmf, simulator as simu  # noqa: E402
from classy_szlite.sim import summaries as sm, flatsky as fsk  # noqa: E402
from classy_szlite.sz_scaling import y0_median, theta500_arcmin  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "counts")
TILES_40PC = 0.4 * 4 * np.pi * (180 / np.pi) ** 2 / 100.0      # 10° tiles in 40% of the sky


def log(*a):
    print(*a, flush=True)


def analytic_binned_cov(cmb, mode):
    """Binned 6×6 covariance (rfft units) of CMB + noise, as the filter bins it."""
    G = fsk.grid()
    B = ins.beams(G.m_L)
    C = ins.noise_cov(G.m_L, mode) + B[:, None, :] * B[None, :, :] * cmb(G.m_L)[None, None, :]
    C *= G.Npx / G.Om
    return np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[i, j], minlength=G.n_bins)
                               / G.bin_count for j in range(6)], -1) for i in range(6)], 1)


def sigma_from_cov(Cb, fs):
    """σ_j (y0 units) of the 15 filter scales for a binned covariance."""
    _, aCa = mmf.weights(jnp.asarray(Cb), fs)
    nrm = jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)
    return np.asarray(1.0 / jnp.sqrt(nrm))


def catalogue_hist(cs, theta, cells, sig_j, k_dof=0.0, M_min=5e13, n_gh=32, q_edges=None,
                   z_edges=None):
    """Expected counts per tile in (q, z) bins, by default the paper's (4, 5)."""
    qe = sm.Q_EDGES if q_edges is None else np.asarray(q_edges)
    zedges = sm.Z_EDGES if z_edges is None else np.asarray(z_edges)
    ze, me = cells.z_edges, cells.lnM_edges
    zc, lnMc = 0.5 * (ze[1:] + ze[:-1]), 0.5 * (me[1:] + me[:-1])
    lam = np.asarray(cs.lam)
    cg = cs.cg
    H0c = theta.h * 100.0 / 299792.458
    Ez = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Hz)) / H0c
    DA = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Da))
    M = np.exp(lnMc)
    ybar = np.asarray(y0_median(M[None, :], Ez[:, None], theta.h, theta.A_SZ, theta.alpha_SZ,
                                theta.one_minus_b))
    th = np.asarray(theta500_arcmin(M[None, :], Ez[:, None], DA[:, None], theta.h,
                                    theta.one_minus_b))
    sig = np.exp(np.interp(np.log(th), np.log(mmf.THETAS15), np.log(sig_j)))
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    qt = (ybar / sig)[..., None] * np.exp(theta.sigma_lnq * x)            # (nz, nm, gh)
    mu = np.sqrt(qt ** 2 + k_dof)[..., None]
    P = _norm.cdf(qe[1:] - mu) - _norm.cdf(qe[:-1] - mu)                  # (nz, nm, gh, n_q)
    P = np.einsum("abgq,g->abq", P, w)
    ok = M >= M_min
    zb = np.digitize(zc, zedges) - 1
    H = np.zeros((qe.size - 1, zedges.size - 1))
    for b in range(H.shape[1]):
        s = zb == b
        H[:, b] = np.einsum("ab,abq->q", lam[s][:, ok], P[s][:, ok])
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    theta = simu.THETA_FID
    t0 = time.time()
    cells, tb, cs, cfg, st_goal, info = spike.setup(theta)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    statics = {"goal": st_goal, "baseline": simu.tile_static(cmb, tb, mode="baseline")}
    log(f"setup {time.time() - t0:.1f} s; backend {jax.default_backend()}; "
        f"lambda per tile {info['lambda_total']:.0f}")

    f = jax.jit(jax.vmap(lambda key, st: simu.simulate_tile(key, cs, theta, st, tb, cells, cfg),
                         in_axes=(0, None)))
    keys = spike.tile_keys(args.tiles)
    res = dict(tiles=args.tiles, theta=theta._asdict(), tiles_40pc=TILES_40PC, modes={})
    for mode, st in statics.items():
        t1 = time.time()
        outs = [jax.device_get(f(keys[a:a + args.batch], st))
                for a in range(0, args.tiles, args.batch)]
        o = {k: np.concatenate([np.asarray(r[k]) for r in outs]) for k in outs[0]}
        counts = o["counts"].astype(float)                                 # (T, 4, 5)
        tot = counts.sum((1, 2))
        sig_map = o["sigma"].mean(0)
        sig_an = sigma_from_cov(analytic_binned_cov(cmb, mode), st.fs)
        cat = {}
        for src, s in (("analytic", sig_an), ("map", sig_map)):
            for k in (0.0, 1.0, 2.0, 3.0):
                cat[f"{src}_k{k:g}"] = catalogue_hist(cs, theta, cells, s, k_dof=k)
        cat["analytic_k0_noMcut"] = catalogue_hist(cs, theta, cells, sig_an, 0.0, M_min=0.0)
        r = dict(map_counts_mean=counts.mean(0).tolist(), map_total_mean=float(tot.mean()),
                 map_total_sem=float(tot.std(ddof=1) / np.sqrt(len(tot))),
                 n_peaks_mean=float(o["n_peaks"].mean()), n_kept_mean=float(o["n_kept"].mean()),
                 n_conf_mean=float(o["n_conf"].mean()), any_overflow=bool(o["overflow"].any()),
                 sigma_map=sig_map.tolist(), sigma_analytic=sig_an.tolist(),
                 catalogue={k: v.tolist() for k, v in cat.items()},
                 catalogue_total={k: float(v.sum()) for k, v in cat.items()},
                 seconds=time.time() - t1)
        res["modes"][mode] = r
        log(f"\n=== {mode}: {args.tiles} tiles in {r['seconds']:.0f} s; overflow {r['any_overflow']}")
        log(f"map level: {r['map_total_mean']:.1f} ± {r['map_total_sem']:.1f} per tile "
            f"-> {r['map_total_mean'] * TILES_40PC:,.0f} in 40% of the sky "
            f"(peaks {r['n_peaks_mean']:.0f}, kept {r['n_kept_mean']:.0f}, confirmed {r['n_conf_mean']:.0f})")
        log("sigma_map / sigma_analytic per scale: " + " ".join(f"{a / b:.3f}" for a, b in
                                                                 zip(sig_map, sig_an)))
        for k, v in r["catalogue_total"].items():
            log(f"catalogue {k:20s}: {v:7.1f} per tile -> {v * TILES_40PC:8,.0f};  map / catalogue = "
                f"{r['map_total_mean'] / v:.3f}")
        log("map-level histogram per tile (rows q bins, columns z bins):")
        log(np.array2string(counts.mean(0), precision=1, suppress_small=True))
        log("catalogue analytic k=0:")
        log(np.array2string(cat["analytic_k0"], precision=1, suppress_small=True))
    g, b = res["modes"]["goal"], res["modes"]["baseline"]
    log(f"\ngoal / baseline: map level {g['map_total_mean'] / b['map_total_mean']:.3f}; catalogue "
        f"analytic k=0 {g['catalogue_total']['analytic_k0'] / b['catalogue_total']['analytic_k0']:.3f}")
    with open(os.path.join(OUT, "counts_modes.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"saved {os.path.join(OUT, 'counts_modes.json')}")


if __name__ == "__main__":
    main()
