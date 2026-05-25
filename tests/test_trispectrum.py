"""Tests for the 1-halo trispectrum + bandpower covariance API."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import classy_szlite as csl


@pytest.fixture(scope="module")
def ell8():
    return jnp.geomspace(100.0, 5000.0, 8)


@pytest.fixture(scope="module")
def delta_ell(ell8):
    return ell8 * jnp.log(ell8[1] / ell8[0])


def test_trispectrum_shape_and_symmetric(cosmo, profile, ell8):
    T = csl.cl_yy_trispectrum(cosmo, profile, ell8)
    assert T.shape == (len(ell8), len(ell8))
    # Symmetric within floating-point round-off (the function symmetrises)
    np.testing.assert_allclose(np.asarray(T), np.asarray(T.T),
                                rtol=0, atol=0)


def test_trispectrum_positive_definite(cosmo, profile, ell8):
    """T is positive-definite because it's an outer-product integral."""
    T = csl.cl_yy_trispectrum(cosmo, profile, ell8)
    eigvals = np.linalg.eigvalsh(np.asarray(T))
    assert (eigvals > 0).all(), f"min eigval = {eigvals.min():.3e}"


def test_covariance_gaussian_only_is_diagonal(cosmo, profile, ell8, delta_ell):
    cov = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=0.6,
                                include_trispectrum=False)
    off = cov - jnp.diag(jnp.diag(cov))
    np.testing.assert_allclose(np.asarray(off), 0.0)
    assert (np.diag(np.asarray(cov)) > 0).all()


def test_covariance_with_trispectrum_is_positive_definite(cosmo, profile,
                                                            ell8, delta_ell):
    cov = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=0.6)
    np.testing.assert_allclose(np.asarray(cov), np.asarray(cov.T))
    eigvals = np.linalg.eigvalsh(np.asarray(cov))
    assert (eigvals > 0).all()
    # Cholesky factorisation must succeed
    L = jnp.linalg.cholesky(cov)
    np.testing.assert_allclose(np.asarray(L @ L.T), np.asarray(cov),
                                rtol=1e-10, atol=1e-30)


def test_covariance_trispectrum_dominates_for_tsz(cosmo, profile, ell8, delta_ell):
    """For tSZ at typical bandpowers, the 1h-trispectrum dominates the
    Gaussian variance by orders of magnitude (rare-event statistics)."""
    cov_g  = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=0.6,
                                   include_trispectrum=False)
    cov_gt = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=0.6,
                                   include_trispectrum=True)
    diag_ratio = np.diag(np.asarray(cov_gt)) / np.diag(np.asarray(cov_g))
    assert (diag_ratio > 100).all(), f"trispectrum should boost diag by >100×; got {diag_ratio}"


def test_covariance_fsky_scaling(cosmo, profile, ell8, delta_ell):
    """fsky enters as 1/fsky on both terms → cov_a / cov_b == fsky_b / fsky_a."""
    cov_full = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=1.0)
    cov_half = csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell, fsky=0.5)
    np.testing.assert_allclose(np.asarray(cov_half / cov_full), 2.0,
                                rtol=1e-12)


def test_cholesky_synthetic_data_matches_covariance(cosmo, profile, ell8, delta_ell):
    """Draw N realisations from N(0, cov) via Cholesky, check the empirical
    covariance recovers the analytic one (on the diagonal, where signal is
    large; off-diagonals are 4 orders of magnitude smaller and MC-dominated)."""
    cov = np.asarray(csl.cl_yy_covariance(cosmo, profile, ell8, delta_ell,
                                            fsky=0.6))
    L   = np.linalg.cholesky(cov)
    rng = np.random.default_rng(0)
    N = 50000
    noise = L @ rng.standard_normal((len(ell8), N))    # (n_ell, N)
    cov_emp = np.cov(noise)
    # MC std on variance estimates ~ sqrt(2/N); use 10σ tolerance
    rtol_diag = 10.0 * np.sqrt(2.0 / N)
    np.testing.assert_allclose(np.diag(cov_emp), np.diag(cov), rtol=rtol_diag)
    # Frobenius distance on the full matrix relative to the diagonal norm
    rel_frob = np.linalg.norm(cov_emp - cov) / np.linalg.norm(np.diag(cov))
    assert rel_frob < 0.02, f"Frobenius rel error = {rel_frob:.3e}"
