"""End-to-end smoke test: load each emulator, evaluate every public function.

Lives in this file for backwards compat — granular tests are in
`test_derived.py`, `test_cmb.py`, etc.
"""
from __future__ import annotations
import numpy as np
import classy_szlite as csl


def test_default_cosmo_params_loads(cosmo):
    assert cosmo.omega_b > 0 and cosmo.H0 > 0
    assert cosmo.fEDE == 0.001       # default LCDM-equivalent setting


def test_cl_yy_factory_matches_full_pipeline(cosmo, profile):
    import jax.numpy as jnp
    ell = jnp.geomspace(100, 5000, 8)
    cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)
    cl_1h_f, cl_2h_f = csl.cl_yy_factory(cosmo, ell)(profile)
    np.testing.assert_allclose(cl_1h, cl_1h_f, rtol=1e-10)
    np.testing.assert_allclose(cl_2h, cl_2h_f, rtol=1e-10)
