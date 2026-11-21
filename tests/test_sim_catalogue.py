"""Catalogue-level simulator (classy_szlite/sim/catalogue.py).

* The survey cells line up with the data vector's z edges.
* The paper's noise fits from cosmocnc_jax's tables reproduce the coefficients its SO-like
  survey uses (skipped without the tables: COSMOCNC_DATA).
* Per halo, with r = 0.5 and dof = 2: the log-normal scatter of q and p has the set widths and
  correlation, and q_obs, p_obs carry unit Gaussian noise about sqrt(q_true^2 + dof) and p_true.
* The halos drawn: their number against the summed cell rates, and their z distribution
  against the rates' z marginal.
* Per halo, the selected clusters' p_obs summed against each drawn halo's exact conditional
  expectation (the observable layers alone, without the halo sampler).
* The whole simulator: the mean over 120 catalogues of the 20 counts, the number selected and
  the mean p agree with catalogue.expected (an independent quadrature) within 4.5 standard
  errors; and the same key gives the same catalogue, batched or not.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy import stats

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import catalogue as ct, simulator as simu, summaries as sm  # noqa: E402

HAVE_TABLES = os.path.exists(os.path.join(ct.COSMOCNC_DATA, "so_sim_sz_mf_noise.npy"))
needs_tables = pytest.mark.skipif(not HAVE_TABLES, reason="the paper's noise tables (COSMOCNC_DATA) are absent")
TH = simu.THETA_FID


@pytest.fixture(scope="module")
def setup():
    cells = ct.survey_cells()
    sz, le = ct.paper_tables() if HAVE_TABLES else (None, None)
    return cells, sz, le


def test_cells_align_with_z_edges():
    z = ct.survey_cells().z_edges
    for e in sm.Z_EDGES:
        assert np.min(np.abs(z - e)) < 1e-12, e


@needs_tables
def test_paper_fit_coefficients(setup):
    _, sz, le = setup
    np.testing.assert_allclose(np.asarray(sz.coef), [-0.061973, 0.553944, -1.811945, -10.294771], atol=2e-6)
    np.testing.assert_allclose(np.asarray(le.coef), [-0.018787, 0.132758, -1.660727, 0.976163], atol=2e-6)


@needs_tables
def test_scatter_and_noise_layers(setup):
    cells, sz, le = setup
    cfg = ct.CatalogueConfig(corr_lnq_lnp=0.5, dof=2.0, f_sky=0.05, capacity=2 ** 18)
    h = jax.jit(lambda k: ct.draw(k, TH, cells, sz, le, cfg))(jax.random.key(1))
    v = np.asarray(h.valid)
    n = int(v.sum())
    assert int(h.total) <= cfg.capacity and n == int(h.total) and n > 50_000
    a = np.log(np.asarray(h.q_true) / np.asarray(h.qbar))[v]
    b = np.log(np.asarray(h.p_true) / np.asarray(h.pbar))[v]
    tol = 5.0 / np.sqrt(n)
    assert abs(a.mean()) < tol * TH.sigma_lnq and abs(b.mean()) < tol * TH.sigma_lnp
    assert abs(a.std() / TH.sigma_lnq - 1.0) < tol and abs(b.std() / TH.sigma_lnp - 1.0) < tol
    assert abs(np.corrcoef(a, b)[0, 1] - 0.5) < tol
    n1 = (np.asarray(h.q_obs) - np.sqrt(np.asarray(h.q_true) ** 2 + 2.0))[v]
    n2 = (np.asarray(h.p_obs) - np.asarray(h.p_true))[v]
    for x in (n1, n2):
        assert abs(x.mean()) < tol and abs(x.std() - 1.0) < tol
    assert abs(np.corrcoef(n1, n2)[0, 1]) < tol


@needs_tables
def test_halo_number_and_redshifts(setup):
    cells, sz, le = setup
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18)
    lam_all, _, _, _ = ct.survey_rates(TH, cells)
    lam = np.asarray(lam_all) * cfg.f_sky
    h = jax.jit(lambda k: ct.draw(k, TH, cells, sz, le, cfg))(jax.random.key(2))
    N, L = int(h.total), lam.sum()
    assert abs(N - L) < 5 * np.sqrt(L), (N, L)
    z = np.asarray(h.z)[np.asarray(h.valid)]
    edges = cells.z_edges[::10]                              # 30 bins of 3 cells
    obs = np.histogram(z, edges)[0]
    expz = lam.sum(1).reshape(-1, 10).sum(1) * N / L
    keep = expz > 20
    chi2 = float(np.sum((obs[keep] - expz[keep]) ** 2 / expz[keep]))
    assert stats.chi2.sf(chi2, keep.sum() - 1) > 1e-4, chi2


@needs_tables
def test_simulations_match_expectation(setup):
    cells, sz, le = setup
    cfg = ct.CatalogueConfig()
    ec, en, ep = jax.jit(lambda th: ct.expected(th, cells, sz, le, cfg))(TH)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    f = jax.jit(lambda k: ct.simulate(k, TH, cells, sz, le, cfg))
    R = 120
    C, Ns, Pm = [], [], []
    for r in range(R):
        c = f(jax.random.key(100 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel)); Pm.append(float(c.p_sum) / int(c.n_sel))
    C, Ns, Pm = np.array(C), np.array(Ns, float), np.array(Pm)
    zc = (C.mean(0) - ec) / np.sqrt(ec / R)
    print("counts z-scores:", np.round(zc, 2).tolist())
    assert np.all(np.abs(zc) < 4.5)
    assert stats.chi2.sf(float(np.sum(zc ** 2)), 20) > 1e-4
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5, (Ns.mean(), en)
    z_p = (Pm.mean() - ep / en) / (Pm.std(ddof=1) / np.sqrt(R))
    print(f"selected {Ns.mean():.1f} against {en:.1f}; mean p {Pm.mean():.5f} against {ep / en:.5f} (z {z_p:+.2f})")
    assert abs(z_p) < 4.5


@needs_tables
def test_same_key_same_catalogue_batched(setup):
    cells, sz, le = setup
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18)
    f = jax.jit(lambda k: ct.simulate(k, TH, cells, sz, le, cfg))
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(7), i))(jnp.arange(3))
    single = [f(keys[i]) for i in range(3)]
    batch = jax.jit(jax.vmap(lambda k: ct.simulate(k, TH, cells, sz, le, cfg)))(keys)
    for i in range(3):
        assert np.array_equal(np.asarray(single[i].counts), np.asarray(batch.counts[i]))
        assert int(single[i].n_sel) == int(batch.n_sel[i])
        np.testing.assert_allclose(float(single[i].p_sum), float(batch.p_sum[i]), rtol=1e-12)
    again = f(keys[0])
    assert np.array_equal(np.asarray(again.counts), np.asarray(single[0].counts))


@needs_tables
def test_selected_p_against_conditional_expectation(setup):
    """Sum of p_obs over the selected halos of one full-survey draw against the sum, over the
    same drawn halos, of E[p_obs 1(q_obs > 5) | M, z] (Gauss-Hermite over the q scatter, r = 0),
    and the same for the number selected; errors from the per-halo variances."""
    from scipy.stats import norm
    cells, sz, le = setup
    cfg = ct.CatalogueConfig()
    tot_p, tot_e, var_p, tot_n, tot_en, var_n = 0.0, 0.0, 0.0, 0, 0.0, 0.0
    f = jax.jit(lambda k: ct.draw(k, TH, cells, sz, le, cfg))
    gh, gw = np.polynomial.hermite_e.hermegauss(40)
    gw = gw / gw.sum()
    for r in range(4):
        h = f(jax.random.key(500 + r))
        v = np.asarray(h.valid)
        qbar, pbar = np.asarray(h.qbar)[v], np.asarray(h.pbar)[v]
        sel = (np.asarray(h.q_obs)[v] > cfg.q_min)
        P = sum(w * (1.0 - norm.cdf(cfg.q_min - qbar * np.exp(TH.sigma_lnq * g))) for g, w in zip(gh, gw))
        Ep = pbar * np.exp(0.5 * TH.sigma_lnp ** 2)
        Ep2 = pbar ** 2 * np.exp(2 * TH.sigma_lnp ** 2) + 1.0
        tot_p += float(np.asarray(h.p_obs)[v][sel].sum()); tot_e += float(np.sum(P * Ep))
        var_p += float(np.sum(P * Ep2 - (P * Ep) ** 2))
        tot_n += int(sel.sum()); tot_en += float(P.sum()); var_n += float(np.sum(P * (1 - P)))
    zp, zn = (tot_p - tot_e) / np.sqrt(var_p), (tot_n - tot_en) / np.sqrt(var_n)
    print(f"per-halo conditional: p sum z {zp:+.2f}, number selected z {zn:+.2f}; "
          f"mean p {tot_p / tot_n:.5f} against {tot_e / tot_en:.5f}")
    assert abs(zp) < 4.5 and abs(zn) < 4.5
