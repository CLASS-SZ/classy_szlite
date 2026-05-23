"""End-to-end smoke test: load each emulator, evaluate every public function.

Skips silently if the emulator data is not on this machine.
"""
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
    DATA_ERR = exc

pytestmark = pytest.mark.skipif(not DATA_OK,
    reason="classy_szlite emulator data not available")


def test_default_cosmo_params_loads():
    p = csl.CosmoParams()
    assert p.omega_b > 0 and p.H0 > 0


@pytest.mark.parametrize("model", ["ede-v2", "lcdm"])
def test_derived(model):
    if model == "lcdm":
        cosmo = csl.CosmoParams().for_lcdm()         # set ν convention
    else:
        cosmo = csl.CosmoParams()
    d = csl.derived(cosmo, cosmo_model=model)
    assert 0.7 < d["sigma_8"] < 1.0
    assert 0.2 < d["Omega_m"] < 0.4
    assert 0.7 < d["S8"] < 1.0


@pytest.mark.parametrize("model", ["ede-v2", "lcdm"])
def test_cmb_cls(model):
    cosmo = csl.CosmoParams() if model == "ede-v2" else csl.CosmoParams().for_lcdm()
    out = csl.cl_TTTEEE(cosmo, cosmo_model=model)
    assert "tt" in out and "te" in out and "ee" in out and "ell" in out
    # First-acoustic-peak amplitude sanity check (Cl_TT around ell~220)
    ell, tt = out["ell"], out["tt"]
    i220 = int(np.argmin(np.abs(ell - 220)))
    # dimensionless Cl × (2.7255e6 μK)² → expect O(few×1e3) μK² at peak
    Tcmb_uK = 2.7255e6
    peak_uK2 = tt[i220] * Tcmb_uK ** 2 * ell[i220] * (ell[i220] + 1) / (2 * np.pi)
    assert 3000 < peak_uK2 < 10000, f"unrealistic TT peak: {peak_uK2:.0f} μK²"


@pytest.mark.parametrize("model", ["ede-v2", "lcdm"])
def test_pk_and_pnl(model):
    cosmo = csl.CosmoParams() if model == "ede-v2" else csl.CosmoParams().for_lcdm()
    z = np.array([0.0, 0.5, 1.0, 2.0])
    k, pk = csl.Pk(cosmo, z, cosmo_model=model)
    k2, pnl = csl.Pnl(cosmo, z, cosmo_model=model)
    assert k.shape == k2.shape
    assert pk.shape == (len(z), len(k))
    # P(k) should be ~ k^n_s at low k
    slope = (np.log(float(pk[0, 1])) - np.log(float(pk[0, 0]))) / (np.log(float(k[1])) - np.log(float(k[0])))
    assert abs(slope - 0.97) < 0.05, f"low-k slope {slope:.3f} not near n_s"


@pytest.mark.parametrize("model", ["ede-v2", "lcdm"])
def test_distances(model):
    cosmo = csl.CosmoParams() if model == "ede-v2" else csl.CosmoParams().for_lcdm()
    z = np.array([0.005, 0.5, 1.0, 2.0])
    Hz, chi, Da = csl.distances(cosmo, z, cosmo_model=model)
    # chi(z=0.005) ≈ c·z/H0 ≈ 22 Mpc; chi(z=1) ≈ 3300 Mpc
    assert 15 < float(chi[0]) < 30
    assert 2500 < float(chi[2]) < 4000


@pytest.mark.parametrize("model", ["ede-v2", "lcdm"])
def test_cl_yy(model):
    import jax.numpy as jnp
    cosmo = csl.CosmoParams() if model == "ede-v2" else csl.CosmoParams().for_lcdm()
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    ell = jnp.geomspace(100, 5000, 8)
    cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell, cosmo_model=model)
    assert cl_1h.shape == (8,) and cl_2h.shape == (8,)
    # Total D_ell × 1e12 should be O(0.1 - 5) in the bandpower range
    dl = np.asarray((ell * (ell + 1) / (2 * np.pi)) * (cl_1h + cl_2h) * 1e12)
    assert all(0 < d < 50 for d in dl), f"unphysical D_ell: {dl}"
