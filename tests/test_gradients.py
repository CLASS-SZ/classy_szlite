"""JAX gradient tests: autodiff vs finite difference, pytree gradients."""
from __future__ import annotations
import numpy as np
import pytest
import classy_szlite as csl


def test_factory_grad_matches_finite_diff(cosmo, profile):
    """∂Cℓ/∂P0 via jax.grad should match central-difference to <1e-6 rel."""
    import jax, jax.numpy as jnp
    ell = jnp.geomspace(2, 5000, 10)
    ev = csl.cl_yy_factory(cosmo, ell)

    def loss(P0):
        c1, c2 = ev(csl.ProfileParamsA10(P0=P0, beta=5.48, B=1.25))
        return jnp.sum(c1 + c2)

    g_jax = float(jax.grad(loss)(8.13))
    eps = 1e-3
    g_fd  = (float(loss(8.13 + eps)) - float(loss(8.13 - eps))) / (2 * eps)
    rel = abs(g_jax - g_fd) / abs(g_jax)
    assert rel < 1e-6, f"jax.grad vs FD: |Δ|/|g| = {rel:.2e}"


def test_grad_through_full_pipeline_runs(cosmo):
    """Gradient through cl_yy (cosmology + profile) should at least run."""
    import jax, jax.numpy as jnp
    ell = jnp.geomspace(100, 3000, 6)

    def loss(omega_cdm, P0):
        c = csl.CosmoParams(omega_cdm=omega_cdm)
        p = csl.ProfileParamsA10(P0=P0, beta=5.48, B=1.25)
        c1, c2 = csl.cl_yy(c, p, ell)
        return jnp.sum(c1 + c2)

    g = jax.grad(loss, argnums=(0, 1))(0.118, 8.13)
    assert np.isfinite(float(g[0]))
    assert np.isfinite(float(g[1]))


def test_grad_w_r_t_cosmoparams_pytree(cosmo):
    """jax.grad on CosmoParams returns a CosmoParams of derivatives."""
    import jax, jax.numpy as jnp
    ell = jnp.geomspace(100, 3000, 6)
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

    def loss(c):
        c1, c2 = csl.cl_yy(c, profile, ell)
        return jnp.sum(c1 + c2)

    grads = jax.grad(loss)(cosmo)
    # Should be a CosmoParams instance with float-array fields
    assert type(grads).__name__ == "CosmoParams"
    for name in ("omega_b", "omega_cdm", "H0", "fEDE"):
        v = getattr(grads, name)
        assert np.isfinite(float(v))


def test_jacobian_shape(cosmo, profile):
    """jax.jacfwd of Cl(P0, β) should produce (n_ell,)-shaped tangents."""
    import jax, jax.numpy as jnp
    ell = jnp.geomspace(2, 5000, 10)
    ev = csl.cl_yy_factory(cosmo, ell)

    def Cl_vec(P0, beta):
        c1, c2 = ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
        return c1 + c2

    J = jax.jacfwd(Cl_vec, argnums=(0, 1))(8.13, 5.48)
    assert J[0].shape == (len(ell),)
    assert J[1].shape == (len(ell),)
