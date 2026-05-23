"""End-to-end smoke test: load each emulator, evaluate every public function."""
import os
import numpy as np
import pytest

try:
    import classy_szlite as csl
    from classy_szlite._emulator import default_data_dir
    default_data_dir()                              # raises if data missing
    DATA_OK = True
except Exception as exc:
    DATA_OK = False

pytestmark = pytest.mark.skipif(not DATA_OK,
    reason="classy_szlite emulator data not available")


def test_default_cosmo_params_loads():
    p = csl.CosmoParams()
    assert p.omega_b > 0 and p.H0 > 0


def test_derived():
    d = csl.derived(csl.CosmoParams())
    assert 0.7 < d["sigma_8"] < 1.0
    assert 0.2 < d["Omega_m"] < 0.4
    assert 0.7 < d["S8"] < 1.0


def test_cmb_cls():
    out = csl.cl_TTTEEE(csl.CosmoParams())
    assert all(k in out for k in ("tt", "te", "ee", "ell"))
    ell, tt = out["ell"], out["tt"]
    i220 = int(np.argmin(np.abs(ell - 220)))
    Tcmb_uK = 2.7255e6
    peak_uK2 = tt[i220] * Tcmb_uK ** 2
    assert 3000 < peak_uK2 < 10000, f"unrealistic TT peak: {peak_uK2:.0f} μK²"


def test_pk_and_pnl():
    cosmo = csl.CosmoParams()
    z = np.array([0.0, 0.5, 1.0, 2.0])
    k, pk = csl.Pk(cosmo, z)
    k2, pnl = csl.Pnl(cosmo, z)
    assert k.shape == k2.shape
    assert pk.shape == (len(z), len(k))
    slope = (np.log(float(pk[0, 1])) - np.log(float(pk[0, 0]))) / (np.log(float(k[1])) - np.log(float(k[0])))
    assert abs(slope - 0.97) < 0.05, f"low-k slope {slope:.3f} not near n_s"


def test_distances():
    z = np.array([0.005, 0.5, 1.0, 2.0])
    Hz, chi, Da = csl.distances(csl.CosmoParams(), z)
    assert 15 < float(chi[0]) < 30, f"chi(0.005) = {float(chi[0])} unphysical"
    assert 2500 < float(chi[2]) < 4000, f"chi(1.0) = {float(chi[2])} unphysical"


def test_cl_yy_and_factory():
    import jax.numpy as jnp
    cosmo = csl.CosmoParams()
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    ell = jnp.geomspace(100, 5000, 8)
    # Full pipeline
    cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)
    dl = np.asarray((ell * (ell + 1) / (2 * np.pi)) * (cl_1h + cl_2h) * 1e12)
    assert all(0 < d < 50 for d in dl), f"unphysical D_ell: {dl}"
    # Factory should agree with full pipeline to numerical precision
    ev = csl.cl_yy_factory(cosmo, ell)
    cl_1h_f, cl_2h_f = ev(profile)
    np.testing.assert_allclose(cl_1h, cl_1h_f, rtol=1e-10)
    np.testing.assert_allclose(cl_2h, cl_2h_f, rtol=1e-10)
