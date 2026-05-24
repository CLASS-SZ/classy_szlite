"""Halo-model Cl^yy tests: shape, physics monotonicity, factory roundtrip."""
from __future__ import annotations
import numpy as np
import pytest
import classy_szlite as csl


@pytest.fixture(scope="module")
def ell_bp():
    import jax.numpy as jnp
    return jnp.geomspace(100, 5000, 8)


def test_cl_yy_shape_and_positivity(cosmo, profile, ell_bp):
    cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell_bp)
    assert cl_1h.shape == (len(ell_bp),)
    assert cl_2h.shape == (len(ell_bp),)
    assert (np.asarray(cl_1h) >= 0).all()
    assert (np.asarray(cl_2h) >= 0).all()


def test_factory_matches_full_pipeline(cosmo, profile, ell_bp):
    cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell_bp)
    ev = csl.cl_yy_factory(cosmo, ell_bp)
    cl_1h_f, cl_2h_f = ev(profile)
    np.testing.assert_allclose(cl_1h, cl_1h_f, rtol=1e-10)
    np.testing.assert_allclose(cl_2h, cl_2h_f, rtol=1e-10)


def test_factory_is_deterministic(cosmo, profile, ell_bp):
    ev = csl.cl_yy_factory(cosmo, ell_bp)
    a1, a2 = ev(profile)
    b1, b2 = ev(profile)
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)


def test_one_halo_dominates_at_high_ell(cosmo, profile):
    """At ℓ > 1000 the 1-halo term should dominate the 2-halo term."""
    import jax.numpy as jnp
    ell = jnp.array([2000.0, 3000.0, 5000.0])
    c1, c2 = csl.cl_yy(cosmo, profile, ell)
    assert (np.asarray(c1) > 10.0 * np.asarray(c2)).all()


def test_P0_scaling(cosmo, ell_bp):
    """Cl^yy scales roughly as P0² (P_e ∝ P0 in 1-halo)."""
    p1 = csl.ProfileParamsA10(P0=8.0, beta=5.48, B=1.25)
    p2 = csl.ProfileParamsA10(P0=4.0, beta=5.48, B=1.25)
    ev = csl.cl_yy_factory(cosmo, ell_bp)
    c1a, c2a = ev(p1)
    c1b, c2b = ev(p2)
    # 1-halo: ~P0² ratio = 4 expected
    ratio = np.asarray(c1a[3:] / c1b[3:])
    assert (ratio > 3.5).all() and (ratio < 4.5).all()


def test_B_decreases_signal(cosmo, ell_bp):
    """Increasing hydrostatic mass bias B reduces Cl^yy (less true mass)."""
    p_lo = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.0)
    p_hi = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.5)
    ev = csl.cl_yy_factory(cosmo, ell_bp)
    c_lo, _ = ev(p_lo)
    c_hi, _ = ev(p_hi)
    assert (np.asarray(c_lo) > np.asarray(c_hi)).all()


def test_cl_yy_convergence_baseline(cosmo, profile):
    """At default settings (n_z=100, n_m=200) error vs n_z=300 < 10⁻³."""
    import jax.numpy as jnp
    ell = jnp.geomspace(100, 5000, 8)
    ref_1h, ref_2h = csl.cl_yy(cosmo, profile, ell, n_z=300, n_m=200)
    cur_1h, cur_2h = csl.cl_yy(cosmo, profile, ell, n_z=100, n_m=200)
    ref = np.asarray(ref_1h + ref_2h)
    cur = np.asarray(cur_1h + cur_2h)
    max_rel = np.max(np.abs(cur - ref) / np.abs(ref))
    assert max_rel < 1e-3, f"default-settings convergence: max |ΔC/C| = {max_rel:.2e}"
