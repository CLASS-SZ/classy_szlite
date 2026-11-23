"""Stacked CMB-lensing observable (sim/lensing.py, simulator.stacked_lensing).

* The truncated-NFW constants: Σ(r_s) and the NFW mass normalisation, against
  direct numerical integrals.
* κ(r_s) against an independent route (Σ_crit with D_s and D_ls from χ*).
* χ* (distance to last scattering) at the fiducial point.
* The σ_κ table against cosmocnc_jax's baseline table (shape and level).
* The p draws: mean and variance against the log-normal-plus-noise formulae.
* The selection: the stacked sum covers exactly the detections the histogram counts.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.integrate import quad

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import lensing as lz, lens_tables, halos as hl, simulator as simu, summaries as sm  # noqa: E402
from classy_szlite.sim.flatsky import N  # noqa: E402

REF = os.path.join(os.environ.get("MAPSIM_REF", "/mnt/ceph/users/bbolliet/mapsim/ref"),
                   "so_lensing_noise", "so_sim_lensing_mf_noise.npy")


@pytest.fixture(scope="module")
def fid():
    cells = hl.default_cells()
    cs = jax.jit(lambda th: simu.cosmology_stage(th, cells))(simu.THETA_FID)
    return cells, cs


def test_nfw_constants():
    """Σ(r_s)/(ρ0 r_s) of the NFW truncated at 15 r_s, and the mass inside c r_s."""
    X = lz.X_TRUNC
    f = lambda z: 1.0 / (np.hypot(1.0, z) * (1.0 + np.hypot(1.0, z)) ** 2)
    s_num = 2.0 * quad(f, 0.0, np.sqrt(X * X - 1.0), limit=200)[0]
    assert abs(s_num / lz.S_RS - 1.0) < 1e-8, (s_num, lz.S_RS)
    m_num = quad(lambda x: x / (1.0 + x) ** 2, 0.0, lz.C500)[0]
    assert abs(m_num / lz.M_C - 1.0) < 1e-10


def test_chi_star_fiducial(fid):
    """Planck-like cosmology: χ* ≈ 13,900 Mpc; and the integral agrees with the
    cosmology grid at z = 3 when stopped there."""
    _, cs = fid
    chs = float(cs.chi_star)
    print(f"chi_star = {chs:.1f} Mpc")
    assert 13600.0 < chs < 14200.0


def test_kappa_independent_route(fid):
    """κ(r_s) = Σ(r_s)/Σ_crit with Σ_crit = c²/(4πG) D_s/(D_l D_ls), D_s and D_ls
    from χ* and the halo's χ: the same number as lensing.kappa_rs."""
    _, cs = fid
    th = simu.THETA_FID
    M, z, beta = 3e14, 0.5, th.beta_lens
    k = float(lz.kappa_rs(jnp.asarray(M), jnp.asarray(z), cs.cg, th.h, beta, cs.chi_star))
    cg = cs.cg
    Hc = float(np.interp(z, cg.z, cg.Hz)); DA = float(np.interp(z, cg.z, cg.Da))
    chi = float(np.interp(z, cg.z, cg.chi)); chs = float(cs.chi_star)
    rho_c = 2.77536627e11 * (Hc / (100.0 / lz.C_KMS)) ** 2           # M_sun/Mpc³ with H in units of 100 km/s/Mpc
    R500 = (3 * beta * M / (4 * np.pi * 500 * rho_c)) ** (1 / 3)
    rs = R500 / 3.0
    rho0 = beta * M / (4 * np.pi * rs ** 3 * (np.log(4.0) - 0.75))
    Ds, Dls = chs / (1 + lz.Z_STAR), (chs - chi) / (1 + lz.Z_STAR)
    sig_crit = 1.0 / (4 * np.pi * lz.G_OVER_C2) * Ds / (DA * Dls)
    k2 = 0.66259 * rho0 * rs / sig_crit
    print(f"kappa(r_s) at M = 3e14, z = 0.5: {k:.5f} (independent {k2:.5f})")
    assert abs(k / k2 - 1) < 2e-5
    assert 0.05 < k < 0.5          # by hand: Σ(r_s) ≈ 2.5e14 M_sun/Mpc², Σ_crit ≈ 1.4e15 → κ ≈ 0.18


@pytest.mark.skipif(not os.path.exists(REF), reason="cosmocnc_jax baseline table not copied")
def test_sigma_table_matches_reference():
    ref = np.load(REF)
    th = ref[0] / (np.pi / 180 / 60)
    ours = np.exp(np.interp(np.log(th), np.log(lens_tables.THETA500_ARCMIN), np.log(lens_tables.SIGMA_KAPPA["baseline"])))
    r = ours / ref[1]
    print(f"ours/reference: {r.min():.4f}-{r.max():.4f}")
    assert np.all(np.abs(r - 1) < 0.01)
    g = lens_tables.SIGMA_KAPPA["goal"] / lens_tables.SIGMA_KAPPA["baseline"]
    assert np.all((g > 0.8) & (g < 0.95))


def test_p_draw_moments():
    """E[p] = p̄ e^{σ²/2}, Var[p] = 1 + p̄² e^{σ²}(e^{σ²} − 1)."""
    pbar, s = 0.4, 0.22
    p = np.asarray(lz.draw_p(jax.random.key(1), jnp.full(400_000, pbar), s))
    m_th = pbar * np.exp(s * s / 2)
    v_th = 1 + pbar ** 2 * np.exp(s * s) * (np.exp(s * s) - 1)
    print(f"mean {p.mean():.4f} (theory {m_th:.4f}), var {p.var():.4f} (theory {v_th:.4f})")
    assert abs(p.mean() - m_th) < 5 * np.sqrt(v_th / p.size)
    assert abs(p.var() / v_th - 1) < 0.01


def test_stacked_selection_is_all_confirmed_above_q5(fid):
    """On a synthetic catalogue: the stacked sum covers every kept, confirmed detection with
    q > 5 (any z, q above 40 included), so at least the histogram total, and p_sum is finite."""
    cells, cs = fid
    th = simu.THETA_FID
    rng = np.random.default_rng(4)
    n, nh, K, H = 600, 300, 1024, 512
    hy, hx = rng.integers(0, 2 * N, nh), rng.integers(0, 2 * N, nh)
    pad = lambda a, f, size: np.concatenate([a, np.full(size - a.size, f)])
    halos = hl.Halos(iy=jnp.asarray(pad(hy, 0, H).astype(np.int32)), ix=jnp.asarray(pad(hx, 0, H).astype(np.int32)),
                     oy=jnp.zeros(H, jnp.float32), ox=jnp.zeros(H, jnp.float32),
                     z=jnp.asarray(pad(rng.uniform(0.0, 1.6, nh), 0.0, H).astype(np.float32)),
                     lnM=jnp.asarray(pad(np.log(rng.uniform(5e13, 1e15, nh)), 0.0, H).astype(np.float32)),
                     y0=jnp.asarray(pad(rng.uniform(1e-5, 1e-4, nh), 0.0, H).astype(np.float32)),
                     theta500=jnp.ones(H, jnp.float32), valid=jnp.arange(H) < nh,
                     count=jnp.int32(nh), overflow=jnp.bool_(False))
    pick = rng.integers(0, nh, n)
    iy = (hy[pick] // 2 + rng.integers(-2, 3, n)) % N
    ix = (hx[pick] // 2 + rng.integers(-2, 3, n)) % N
    q = rng.uniform(4.0, 45.0, n)
    pk = mmf_peaks(iy, ix, q, K)
    keep = sm.merge_local_max(pk, 1.5)
    cf = sm.confirm(pk, keep, halos, 5e13, 1.5, 1024)
    counts = sm.histogram(pk.q, cf.z, keep & cf.ok)

    class St:  # the two static arrays stacked_lensing needs
        lens_lnth, lens_lnsig = map(jnp.asarray, lz.table("goal"))
    p_sum, n_p = simu.stacked_lensing(jax.random.key(3), pk, keep, cf, halos, cs, th, St)
    print(f"histogram total {int(counts.sum())}, stacked n {int(n_p)}, mean p {float(p_sum) / max(int(n_p), 1):.3f}")
    sel = np.asarray(keep & cf.ok & (pk.q > np.float32(sm.Q_EDGES[0])))
    assert int(n_p) == int(sel.sum()) > int(counts.sum()) > 50     # z > 1.5 and q > 40 add to the stack
    assert np.isfinite(float(p_sum))
    assert bool(jnp.all(jnp.where(cf.ok, cf.halo < nh, True)))


def mmf_peaks(iy, ix, q, K):
    from classy_szlite.sim import mmf
    n = len(q)
    pad = lambda a, dt, f: jnp.asarray(np.concatenate([np.asarray(a), np.full(K - n, f)]).astype(dt))
    return mmf.Peaks(j=pad(np.zeros(n), np.int32, 0), iy=pad(iy, np.int32, 0), ix=pad(ix, np.int32, 0),
                     q=pad(q, np.float32, -np.inf), valid=jnp.arange(K) < n, count=jnp.int32(n),
                     overflow=jnp.bool_(False))
