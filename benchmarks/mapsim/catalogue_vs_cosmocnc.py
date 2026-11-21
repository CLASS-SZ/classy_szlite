"""Our catalogue-level simulator against cosmocnc_jax's SO-like survey (reference use only).

    PYTHONPATH=$CODE python benchmarks/mapsim/catalogue_vs_cosmocnc.py

1. Code test, identical inputs: ln qbar = ln(ybar0 / sigma_y0(theta500)) and ln pbar =
   ln(a_lens kappa(r_s) / sigma_kappa(theta500,lens)) for 5,000 random (z, M500) per parameter
   point (the fiducial and two prior corners), from classy_szlite (sim/catalogue.py) and from
   cosmocnc_jax's pure functions (surveys/survey_sr_so_sim.py: precompute_q_prefactors,
   sr_q_so_sim_layer0, precompute_p_prefactors, sr_p_so_sim_layer0), both given our E(z), D_A,
   chi(z), chi_star, rho_c(z) and G/c^2, and the same cubic noise fits. Expected agreement:
   float64 round-off.
2. Physics context: the expected 4 x 5 counts and number selected at the fiducial point with the
   paper's tables, beside the paper's ~16,000 clusters and SO_sim_0's counts (cosmocnc_jax's
   reference catalogue, made with other settings: M down to 1.2e13, z to 2.37).
3. Timing of one catalogue and of the expectation (CPU).

cosmocnc_jax is imported from a reference copy (COSMOCNC_DIR) without its package __init__
(which needs cobaya); no cosmocnc_jax code is copied here.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.cosmology import build, ln10As_from_sigma8  # noqa: E402
from classy_szlite.sim import catalogue as ct, lensing, simulator as simu  # noqa: E402

CNC_DIR = os.environ.get("COSMOCNC_DIR", "/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax")
SO_SIM_0 = dict(n=15798, counts_zq=[[2369, 945, 334, 84], [4096, 1468, 483, 102], [2626, 847, 231, 48],
                                    [1127, 289, 72, 14], [356, 85, 16, 2]])     # rows z bins, columns q bins


def load_cnc():
    pkg = types.ModuleType("cosmocnc_jax")
    pkg.__path__ = [CNC_DIR]
    sys.modules["cosmocnc_jax"] = pkg
    return importlib.import_module("cosmocnc_jax.surveys.survey_sr_so_sim")


def per_halo(sr, theta, sz, le, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.01, 2.99, n))
    lnM = rng.uniform(np.log(5e13), np.log(1e16), n)
    M = np.exp(lnM)
    p = {"omega_cdm": theta.omega_cdm, "H0": 100.0 * theta.h, "omega_b": theta.omega_b, "n_s": theta.n_s}
    p["ln10^{10}A_s"] = ln10As_from_sigma8(p, theta.sigma8, as_reference=simu.LN10AS_REFERENCE)
    cg = build(p, z_grid=z, as_reference=simu.LN10AS_REFERENCE)        # nodes at the halo redshifts
    chi_s = lensing.chi_star(theta.h, theta.omega_cdm, theta.omega_b)
    cfg = ct.CatalogueConfig()
    qbar, pbar = ct._qbar_pbar(jnp.asarray(M), jnp.asarray(z), theta, cg, chi_s, sz, le, cfg)
    # the same inputs for cosmocnc_jax
    H0 = 100.0 * theta.h
    Hc = np.asarray(cg.Hz)                                       # H/c [1/Mpc]
    E = Hc / (H0 / lensing.C_KMS)
    DA, chi = np.asarray(cg.Da), np.asarray(cg.chi)
    rho_c = 3.0 * Hc ** 2 / (8.0 * np.pi * lensing.G_OVER_C2)
    zc = 1089.9                                                  # cancels in D_CMB / D_l_CMB
    D_CMB = float(chi_s) / (1 + zc)
    D_l = (float(chi_s) - chi) / (1 + zc)
    x0 = np.log(M / 1e14)
    pl, pt = sr.precompute_q_prefactors(E, H0, DA, theta.A_SZ, theta.one_minus_b, theta.alpha_SZ)
    lnq_c, _ = sr.sr_q_so_sim_layer0(x0, pl, pt, sz.coef, theta.alpha_SZ)
    plens, pth = sr.precompute_p_prefactors(E, H0, DA, D_CMB, D_l, rho_c, lensing.G_OVER_C2, theta.beta_lens)
    lnp_c = sr.sr_p_so_sim_layer0(x0, plens, pth, le.coef, 1.0, theta.beta_lens)
    dq = np.abs(np.log(np.asarray(qbar)) - np.asarray(lnq_c))
    dp = np.abs(np.log(np.asarray(pbar)) - np.asarray(lnp_c))
    return float(dq.max()), float(dp.max()), float(np.median(np.asarray(qbar))), float(np.median(np.asarray(pbar)))


def main():
    sr = load_cnc()
    cells = ct.survey_cells()
    sz, le = ct.paper_tables()
    print(f"paper fits: sigma_y0 {np.round(np.asarray(sz.coef), 6).tolist()}, sigma_kappa {np.round(np.asarray(le.coef), 6).tolist()}")
    points = {"fiducial": simu.THETA_FID,
              "low corner": simu.THETA_FID._replace(sigma8=0.70, omega_cdm=0.0804, h=0.664, A_SZ=-4.41, alpha_SZ=1.0),
              "high corner": simu.THETA_FID._replace(sigma8=0.90, omega_cdm=0.200, h=0.684, A_SZ=-4.21, alpha_SZ=1.24,
                                                     beta_lens=0.95)}
    print("1. per halo, identical inputs (max |d ln|):")
    worst = 0.0
    for name, th in points.items():
        dq, dp, mq, mp = per_halo(sr, th, sz, le)
        worst = max(worst, dq, dp)
        print(f"   {name:11s}: ln qbar {dq:.1e}, ln pbar {dp:.1e} (median qbar {mq:.3f}, pbar {mp:.4f})")
    print(f"   {'PASS' if worst < 1e-10 else 'FAIL'}: largest difference {worst:.1e}")

    print("2. expected data vector at the fiducial point, paper tables, 40% of the sky:")
    t0 = time.time()
    ec, en, ep = jax.jit(lambda: ct.expected(simu.THETA_FID, cells, sz, le))()
    t_exp = time.time() - t0
    ec = np.asarray(ec)
    print("   counts, rows q bins (5-8.4-14-24-40), columns z bins (0.01-0.31-...-1.5):")
    for row in ec:
        print("   " + "  ".join(f"{v:8.1f}" for v in row))
    print(f"   in the grid {ec.sum():.0f}; selected (q > 5, any z, any q) {float(en):.0f}; mean p {float(ep) / float(en):.4f}")
    so = np.asarray(SO_sim_0["counts_zq"]).T
    print(f"   SO_sim_0 (other settings): in the grid {so.sum()}, selected {SO_SIM_0['n']}; paper: about 16,000 selected")
    print("   ratio to SO_sim_0 per cell (q rows, z columns):")
    for r1, r2 in zip(ec, so):
        print("   " + "  ".join(f"{a / b:8.3f}" for a, b in zip(r1, r2)))

    print("3. timing (CPU):")
    f = jax.jit(lambda k: ct.simulate(k, simu.THETA_FID, cells, sz, le))
    t0 = time.time(); c = jax.block_until_ready(f(jax.random.key(0))); t_first = time.time() - t0
    ts = []
    for i in range(1, 4):
        t0 = time.time(); c = jax.block_until_ready(f(jax.random.key(i))); ts.append(time.time() - t0)
    print(f"   one catalogue: first {t_first:.1f} s, then {min(ts):.2f} s; halos {int(c.n_halos)}, selected {int(c.n_sel)}, "
          f"overflow {bool(c.overflow)}; expectation {t_exp:.1f} s (with compilation)")
    print("   data vector:", np.round(np.asarray(ct.data_vector(c)), 4).tolist())
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
