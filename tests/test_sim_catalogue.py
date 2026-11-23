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
    if not HAVE_TABLES:
        pytest.skip("the paper's noise tables (COSMOCNC_DATA) are absent")
    cells = ct.survey_cells()
    sz, le = ct.paper_tables()
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


# ---- per-patch noise (variant (i): each survey tile's own sigma_y0) ------------------------

SPECTRA = os.path.join(os.environ.get("MAPSIM_OUT", "/mnt/ceph/users/bbolliet/mapsim"), "mss2", "spectra", "spectra.npz")


def test_patch_table_interpolation():
    lnth = jnp.log(jnp.array([0.5, 1.0, 2.0, 4.0]))
    lnsig = jnp.log(jnp.array([[4.0, 2.0, 1.0, 0.5], [8.0, 4.0, 2.0, 1.0]]))
    tab = ct.PatchTable(lnth=lnth, lnsig=lnsig)
    th = jnp.array([0.5, 1.0, np.sqrt(2.0), 4.0, 10.0, 0.1])
    for p in (0, 1):
        got = np.asarray(ct.eval_table(tab, th, jnp.full(th.shape, p)))
        base = np.array([4.0, 2.0, np.sqrt(2.0), 0.5, 0.5, 4.0]) * (1 + p)      # power law inside, held outside
        np.testing.assert_allclose(got, base, rtol=1e-12)


@needs_tables
def test_patches_simulations_match_expectation(setup):
    """Three equal-area patches with different noise (the paper's sigma_y0 table times 0.8, 1, 1.3)."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.3, 20.0, 30)
    base = np.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes)))
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(np.outer([0.8, 1.0, 1.3], base))))
    cfg = ct.CatalogueConfig(f_sky=0.1, capacity=2 ** 19)
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg))(TH)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    f = jax.jit(lambda k: ct.simulate(k, TH, cells, tab, le, cfg))
    R = 60
    C, Ns = [], []
    for r in range(R):
        c = f(jax.random.key(900 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel))
    C, Ns = np.array(C), np.array(Ns, float)
    keep = ec > 1.0
    zc = (C.mean(0)[keep] - ec[keep]) / np.sqrt(ec[keep] / R)
    print("patch counts z-scores:", np.round(zc, 2).tolist(), f"selected {Ns.mean():.1f} against {en:.1f}")
    assert np.all(np.abs(zc) < 4.5)
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5


def test_calibrated_observation_simulations_match_expectation(setup):
    """Variant (ii): per redshift bin scale s, search lift k, scatter w and loss f (CatalogueConfig.qcal),
    with values far from the plain model so that a mistake shows: simulations against the expectation."""
    cells, sz, le = setup
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    cfg = ct.CatalogueConfig(f_sky=0.1, capacity=2 ** 19, qcal=qc)
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, sz, le, cfg))(TH)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    e0, n0, _ = jax.jit(lambda t: ct.expected(t, cells, sz, le, cfg._replace(qcal=None)))(TH)
    assert abs(en / float(n0) - 1) > 0.02                      # the calibration changes the selection
    f = jax.jit(lambda k: ct.simulate(k, TH, cells, sz, le, cfg))
    R = 60
    C, Ns, Ps = [], [], []
    for r in range(R):
        c = f(jax.random.key(1900 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel)); Ps.append(float(c.p_sum))
    C, Ns = np.array(C), np.array(Ns, float)
    keep = ec > 1.0
    zc = (C.mean(0)[keep] - ec[keep]) / np.sqrt(ec[keep] / R)
    print("calibrated counts z-scores:", np.round(zc, 2).tolist(), f"selected {Ns.mean():.1f} against {en:.1f};"
          f" mean p {np.sum(Ps) / Ns.sum():.5f} against {ep / en:.5f}")
    assert np.all(np.abs(zc) < 4.5)
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5
    assert abs(np.sum(Ps) / Ns.sum() - ep / en) < 0.01


def _qcal_theta(qc, g=0.0, gs=0.0):
    """A qcal_theta built from a constant table qc, with gradients g (observation) and gs (ln sigma)."""
    fid = (0.811, 0.121, -4.3054, 1.1233)
    width = (0.082, 0.0886, 0.45, 0.24)
    rows = tuple((r[0], r[1], g, -g, g, 0.0, r[2], 2 * g, 0, 0, 0, r[3], 0, g, 0, 0, r[4], 0, 0, 0.1 * g, 0)
                 for r in qc)
    g_sig = tuple((gs, 0.5 * gs, gs, -gs) for _ in range(15))
    return (fid, width, g_sig, tuple(0.0 for _ in range(15)), rows)


def test_calibration_following_theta_reduces_to_fixed(setup):
    """qcal_theta with zero gradients gives the same catalogue as qcal."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.5, 15.0, 15)
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes))))[None])
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18, qcal=qc)
    a = jax.jit(lambda k: ct.simulate(k, TH, cells, tab, le, cfg))(jax.random.key(11))
    b = jax.jit(lambda k: ct.simulate(k, TH, cells, tab, le, cfg._replace(qcal=None, qcal_theta=_qcal_theta(qc))))(jax.random.key(11))
    assert np.array_equal(np.asarray(a.counts), np.asarray(b.counts)) and float(a.p_sum) == float(b.p_sum)


def test_calibration_following_theta_simulations_match_expectation(setup):
    """qcal_theta with gradients, at a point away from the fiducial one: simulations against the
    expectation, and the expectation differs from the fixed calibration's."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.5, 15.0, 15)
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes))))[None])
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    th = TH._replace(sigma8=0.84, omega_cdm=0.14, A_SZ=-4.2)
    cfg = ct.CatalogueConfig(f_sky=0.1, capacity=2 ** 19, qcal_theta=_qcal_theta(qc, g=0.05, gs=-0.04))   # shifts that add, not cancel
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg))(th)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    _, n0, _ = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg._replace(qcal_theta=None, qcal=qc)))(th)
    assert abs(en / float(n0) - 1) > 0.02
    f = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg))
    R = 40
    C, Ns, Ps = [], [], []
    for r in range(R):
        c = f(jax.random.key(2900 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel)); Ps.append(float(c.p_sum))
    C, Ns = np.array(C), np.array(Ns, float)
    keep = ec > 1.0
    zc = (C.mean(0)[keep] - ec[keep]) / np.sqrt(ec[keep] / R)
    print("calibration following theta, counts z-scores:", np.round(zc, 2).tolist(),
          f"selected {Ns.mean():.1f} against {en:.1f} (fixed calibration {float(n0):.1f})")
    assert np.all(np.abs(zc) < 4.5)
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5
    assert abs(np.sum(Ps) / Ns.sum() - ep / en) < 0.01


def _qcal_conf(qc, a=0.0, b=0.0, uc=None):
    """A qcal_conf from a constant table qc: noise level u from coefficients uc (15), ln sigma shift a + b u,
    and per-z quadratic terms in u of 0.3 a for k and 0.1 a for w."""
    fid = (0.811, 0.121, -4.3054, 1.1233)
    width = (0.082, 0.0886, 0.45, 0.24)
    uc = tuple(0.0 for _ in range(15)) if uc is None else uc
    rows = tuple((r[0], r[1], 0.0, 0.0, r[2], 3 * a, 0.3 * a, r[3], a, 0.1 * a, r[4], 0.0, 0.0) for r in qc)
    return (fid, width, uc, tuple((0.0, b) for _ in range(15)), rows)


def test_calibration_through_noise_level_reduces_to_fixed(setup):
    """qcal_conf with u = 0 everywhere gives the same catalogue as qcal."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.5, 15.0, 15)
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes))))[None])
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18, qcal=qc)
    th = TH._replace(sigma8=0.84, A_SZ=-4.2)
    a = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg))(jax.random.key(13))
    b = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg._replace(qcal=None, qcal_conf=_qcal_conf(qc, a=0.5, b=1.0))))(jax.random.key(13))
    assert np.array_equal(np.asarray(a.counts), np.asarray(b.counts)) and float(a.p_sum) == float(b.p_sum)


def test_calibration_through_noise_level_simulations_match_expectation(setup):
    """qcal_conf with a noise level that grows with sigma8 and A_SZ (quadratic terms too), away from the
    fiducial point: simulations against the expectation, which differs from the fixed calibration's."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.5, 15.0, 15)
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes))))[None])
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    uc = [0.0] * 15
    uc[1], uc[3], uc[5] = 0.05, 0.1, 0.03                # x_sigma8, x_A_SZ, x_sigma8^2
    th = TH._replace(sigma8=0.84, omega_cdm=0.14, A_SZ=-4.2)
    cfg = ct.CatalogueConfig(f_sky=0.1, capacity=2 ** 19, qcal_conf=_qcal_conf(qc, a=-2.0, b=1.0, uc=tuple(uc)))
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg))(th)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    _, n0, _ = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg._replace(qcal_conf=None, qcal=qc)))(th)
    assert abs(en / float(n0) - 1) > 0.02
    f = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg))
    R = 40
    C, Ns, Ps = [], [], []
    for r in range(R):
        c = f(jax.random.key(3900 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel)); Ps.append(float(c.p_sum))
    C, Ns = np.array(C), np.array(Ns, float)
    keep = ec > 1.0
    zc = (C.mean(0)[keep] - ec[keep]) / np.sqrt(ec[keep] / R)
    print("calibration through the noise level, counts z-scores:", np.round(zc, 2).tolist(),
          f"selected {Ns.mean():.1f} against {en:.1f} (fixed calibration {float(n0):.1f})")
    assert np.all(np.abs(zc) < 4.5)
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5
    assert abs(np.sum(Ps) / Ns.sum() - ep / en) < 0.01


def test_calibration_per_size_cells(setup):
    """qcal_size: with identical rows in every size cell, the same catalogue as the redshift-only table;
    with different rows per size, simulations against the expectation, which differs."""
    cells, sz, le = setup
    th_nodes = np.geomspace(0.5, 15.0, 15)
    tab = ct.PatchTable(lnth=jnp.log(jnp.asarray(th_nodes)), lnsig=jnp.log(jnp.asarray(ct.eval_fit(sz, jnp.asarray(th_nodes))))[None])
    qc = ((0.5, 0.92, 2.0, 1.1, 0.05), (1.0, 1.05, 0.5, 0.9, 0.0), (9.9, 0.97, 1.3, 1.3, 0.02))
    base = _qcal_conf(qc)
    edges = (1.0, 3.0, 1e9)
    rows_same = tuple((r[0], e) + tuple(r[1:]) for r in base[4] for e in edges)
    th = TH._replace(sigma8=0.83)
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18, qcal_conf=base)
    a = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg))(jax.random.key(17))
    b = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg._replace(qcal_conf=base[:4] + (rows_same,), qcal_size=edges)))(jax.random.key(17))
    assert np.array_equal(np.asarray(a.counts), np.asarray(b.counts)) and float(a.p_sum) == float(b.p_sum)
    # different search lift and scatter per size: small clusters k = 0, w = 1.3; large k = 2.5, w = 1.0
    per = {1.0: (0.0, 1.3), 3.0: (1.2, 1.1), 1e9: (2.5, 1.0)}
    rows_diff = tuple((r[0], e, r[1], 0.0, 0.0, per[e][0], 0.0, 0.0, per[e][1], 0.0, 0.0, r[10], 0.0, 0.0) for r in base[4] for e in edges)
    cfg2 = ct.CatalogueConfig(f_sky=0.1, capacity=2 ** 19, qcal_conf=base[:4] + (rows_diff,), qcal_size=edges)
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg2))(th)
    ec, en, ep = np.asarray(ec), float(en), float(ep)
    _, n0, _ = jax.jit(lambda t: ct.expected(t, cells, tab, le, cfg))(th)
    assert abs(en / float(n0) - 1) > 0.02
    f = jax.jit(lambda k: ct.simulate(k, th, cells, tab, le, cfg2))
    R = 40
    C, Ns, Ps = [], [], []
    for r in range(R):
        c = f(jax.random.key(4900 + r))
        assert not bool(c.overflow)
        C.append(np.asarray(c.counts, float)); Ns.append(int(c.n_sel)); Ps.append(float(c.p_sum))
    C, Ns = np.array(C), np.array(Ns, float)
    keep = ec > 1.0
    zc = (C.mean(0)[keep] - ec[keep]) / np.sqrt(ec[keep] / R)
    print("calibration per size cell, counts z-scores:", np.round(zc, 2).tolist(),
          f"selected {Ns.mean():.1f} against {en:.1f} (redshift-only {float(n0):.1f})")
    assert np.all(np.abs(zc) < 4.5)
    assert abs(Ns.mean() - en) / np.sqrt(en / R) < 4.5
    assert abs(np.sum(Ps) / Ns.sum() - ep / en) < 0.01


def test_calibration_none_is_unchanged(setup):
    """qcal=None leaves the paper-exact simulator bit-identical."""
    cells, sz, le = setup
    cfg = ct.CatalogueConfig(f_sky=0.05, capacity=2 ** 18)
    a = jax.jit(lambda k: ct.simulate(k, TH, cells, sz, le, cfg))(jax.random.key(7))
    b = jax.jit(lambda k: ct.simulate(k, TH, cells, sz, le, cfg._replace(qcal=None)))(jax.random.key(7))
    assert np.array_equal(np.asarray(a.counts), np.asarray(b.counts)) and float(a.p_sum) == float(b.p_sum)


@pytest.mark.skipif(not os.path.exists(SPECTRA), reason="MSS2 spectra absent")
def test_analytic_sigma_y0_matches_the_finder_on_noise():
    """Our filter's sigma_y0 from a tile's MSS2 spectra plus the CMB (catalogue.analytic_sigma_y0)
    against the sigma our finder estimates on maps drawn from the same spectra (no tSZ): within 3%."""
    from classy_szlite.sim import noise as nz, mmf, paint as pt, instrument as ins, flatsky as fsk
    path = os.path.join(os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending"),
                        "data_real", "cmb_lensed_TT_camb.npz")
    cmb = ins.cmb_cl_from_table(path) if os.path.exists(path) else ins.cmb_cl_emulator()
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(cmb, tb)
    nt, info = nz.load_mss2(SPECTRA, "mean")
    fs = mmf.filter_static()
    for s in (0, 50):
        sig_an, _ = ct.analytic_sigma_y0(nt, [s], cmb, fs)
        ni = nz.noise_input(nt, s, 1.0)
        est = []
        for r in range(3):
            X = jax.jit(lambda k: simu.channel_maps(k, jnp.zeros(fsk.grid().L.shape, jnp.complex64), st, ni))(
                jax.random.key(40 + r))
            est.append(np.asarray(jax.jit(lambda X: mmf.q_cube(X, fs))(X)[1]))
        ratio = np.mean(est, 0) / sig_an[0]
        print(f"spectrum {s}: finder sigma / analytic {np.round(ratio, 4).tolist()}")
        assert np.all(np.abs(ratio - 1.0) < 0.03), ratio
