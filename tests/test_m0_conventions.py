"""M0: neutrino and Tinker conventions of the halo mass function, and the
traceable σ8 → ln(10¹⁰ A_s) map.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import classy_szlite as csl
from classy_szlite.constants import MPC_M
from classy_szlite.cosmology import (build, Omega_m0, omega_cdm_from_Omega_m,
                                     sigma8_der, ln10As_from_sigma8)
from classy_szlite.hmf import build_halo_grids, rho_crit_si, tinker08_hmf


@pytest.fixture(scope="module")
def grids():
    cd = csl.cosmo_to_dict(csl.CosmoParams())
    cg = build(cd, z_grid=jnp.array([0.1, 0.5, 1.0, 2.0]))
    return cd, cg


# ---- the m_ncdm fix: Ω_m in the HMF equals derived() -----------------------

def test_hmf_omega_m_equals_derived(grids):
    cd, cg = grids
    hg = build_halo_grids(cg, cd, delta_crit=500.0, n_m=50)
    rho_c0 = rho_crit_si(cd["H0"] * 1e3 / MPC_M)
    Om = csl.derived(csl.CosmoParams())["Omega_m"]
    assert abs(float(hg.rho_m0 / rho_c0) / Om - 1.0) < 1e-14
    # and Σmν = 3 × 0.02 eV is in it: Ω_ν h² = 0.06 / 93.14
    h = cd["H0"] / 100.0
    assert abs(Om * h ** 2 - (cd["omega_b"] + cd["omega_cdm"] + 0.06 / 93.14)) < 1e-15


def test_nu_excluded_gives_cb_only(grids):
    cd, cg = grids
    hg = build_halo_grids(cg, cd, delta_crit=500.0, n_m=50, nu_in_rho_m=False)
    rho_c0 = rho_crit_si(cd["H0"] * 1e3 / MPC_M)
    Om_cb = (cd["omega_b"] + cd["omega_cdm"]) / (cd["H0"] / 100.0) ** 2
    assert abs(float(hg.rho_m0 / rho_c0) / Om_cb - 1.0) < 1e-14
    # Ω_m(z) for Δ_mean follows nu_in_delta (default: same as nu_in_rho_m)
    hg2 = build_halo_grids(cg, cd, delta_crit=500.0, n_m=50, nu_in_rho_m=False,
                           nu_in_delta=True)
    ratio = np.asarray(hg2.Omega_m_z / hg.Omega_m_z)
    np.testing.assert_allclose(ratio, Omega_m0(cd) / Om_cb, rtol=1e-14)


def test_omega_cdm_round_trip(grids):
    cd, _ = grids
    for nu in (True, False):
        Om = Omega_m0(cd, include_nu=nu)
        assert abs(omega_cdm_from_Omega_m(Om, cd, include_nu=nu) - cd["omega_cdm"]) < 1e-15


def test_round_delta_mean(grids):
    from classy_szlite.hmf import sigma_dndlnm_tinker08
    cd, cg = grids
    kw = dict(delta_crit=500.0, m_min=1e13, m_max=1e15, n_m=50)
    hg = build_halo_grids(cg, cd, round_delta_mean=True, **kw)
    hg0 = build_halo_grids(cg, cd, **kw)
    dm = 500.0 / np.asarray(hg0.Omega_m_z)
    assert np.all(np.abs(dm - np.round(dm)) > 1e-6)     # this grid has non-integral Δ_mean
    # plumbing: the core function called with the rounded Δ_mean
    _, ref = sigma_dndlnm_tinker08(hg0.lnM, cg.z, cg.R, cg.sigma, hg0.rho_m0,
                                   jnp.round(jnp.asarray(dm)), "log10")
    np.testing.assert_allclose(np.asarray(hg.dndlnm), np.asarray(ref), rtol=1e-14, atol=0)
    # |ΔΔ| ≤ 0.5 out of Δ_mean ≳ 500: a small change for 1e13-1e15 M_sun
    r = np.asarray(hg.dndlnm / hg0.dndlnm)
    print(f"rounding Δ_mean: max |ratio - 1| = {np.max(np.abs(r - 1)):.2e}")
    assert np.max(np.abs(r - 1.0)) < 5e-3 and np.max(np.abs(r - 1.0)) > 1e-9


# ---- Tinker 2008 interpolation in Δ: hand-computed reference ---------------

_T = dict(D=[200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.],
          A=[0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],
          a=[1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66],
          b=[2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41],
          c=[1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])


def _f_hand(sigma, Delta, mode):
    """f(σ)/2 at z = 0, parameters interpolated by hand between the two
    bracketing nodes (no jnp.interp)."""
    D = _T["D"]
    j = max(i for i in range(len(D) - 1) if D[i] <= Delta)
    if mode == "linear":
        t = (Delta - D[j]) / (D[j + 1] - D[j])
    else:
        t = np.log10(Delta / D[j]) / np.log10(D[j + 1] / D[j])
    p = {n: _T[n][j] + t * (_T[n][j + 1] - _T[n][j]) for n in "Aabc"}
    return 0.5 * p["A"] * ((sigma / p["b"]) ** (-p["a"]) + 1.0) * np.exp(-p["c"] / sigma ** 2)


@pytest.mark.parametrize("Delta", [400.0, 500.0, 1000.0, 1612.3])
def test_tinker_interpolation_modes(Delta):
    sigma = np.array([[0.4, 0.8, 1.5]])
    z = np.array([0.0])
    dm = np.array([Delta])
    for mode in ("linear", "log10"):
        got = np.asarray(tinker08_hmf(jnp.asarray(sigma), jnp.asarray(z), jnp.asarray(dm),
                                      interp=mode))[0]
        np.testing.assert_allclose(got, _f_hand(sigma[0], Delta, mode), rtol=1e-13)
    lin = np.asarray(tinker08_hmf(sigma, z, dm, interp="linear"))
    log = np.asarray(tinker08_hmf(sigma, z, dm, interp="log10"))
    if Delta in _T["D"]:
        np.testing.assert_allclose(lin, log, rtol=1e-14)
    else:
        assert np.max(np.abs(lin / log - 1.0)) > 1e-4


def test_tinker_interp_rejects_unknown():
    with pytest.raises(ValueError):
        tinker08_hmf(jnp.ones((1, 1)), jnp.zeros(1), jnp.full(1, 500.0), interp="cubic")


# ---- σ8 → ln(10¹⁰ A_s) ---------------------------------------------------------

P_FID = {"omega_cdm": 0.121, "H0": 67.4, "omega_b": 0.0222, "n_s": 0.96}


@pytest.mark.parametrize("s8", [0.70, 0.811, 0.95])
def test_sigma8_map_round_trip(s8):
    x = ln10As_from_sigma8(P_FID, s8)
    got = sigma8_der(dict(P_FID, **{"ln10^{10}A_s": x}))
    assert abs(float(got) - s8) < 1e-12


def test_sigma8_map_is_close_to_As_propto_sigma8_squared():
    x1 = float(ln10As_from_sigma8(P_FID, 0.80))
    x2 = float(ln10As_from_sigma8(P_FID, 0.90))
    assert abs((x2 - x1) - 2.0 * np.log(0.90 / 0.80)) < 2e-3


def test_sigma8_map_jit_vmap_equals_eager():
    s8 = jnp.array([0.72, 0.811, 0.93])
    f = jax.jit(jax.vmap(lambda s: ln10As_from_sigma8(P_FID, s)))
    got = np.asarray(f(s8))
    ref = np.array([float(ln10As_from_sigma8(P_FID, float(s))) for s in s8])
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)


def test_sigma8_der_vs_pk_tophat():
    """Physics comparison (two emulators): σ8 from the DER network against
    σ(R = 8/h Mpc) from the PKL network + FFTLog, at z = 0."""
    from classy_szlite.cosmology import _predict_pk, _compute_sigma
    from classy_szlite._registry import DEFAULT_COSMO
    full = dict(DEFAULT_COSMO)
    _, pk = _predict_pk(full, jnp.array([0.0]))
    R, sig, _ = _compute_sigma(pk)
    s8_pk = float(np.exp(np.interp(np.log(8.0 / (full["H0"] / 100.0)),
                                   np.log(np.asarray(R)), np.log(np.asarray(sig)[0]))))
    s8_der = float(sigma8_der(full))
    print(f"sigma8: DER {s8_der:.6f}, PKL+FFTLog {s8_pk:.6f}, ratio-1 = {s8_pk/s8_der-1:.2e}")
    assert abs(s8_pk / s8_der - 1.0) < 5e-3
