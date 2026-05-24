"""Derived-parameter accuracy + monotonicity tests."""
from __future__ import annotations
import numpy as np
import pytest
import classy_szlite as csl


def test_default_derived_values(cosmo):
    d = csl.derived(cosmo)
    assert pytest.approx(0.812, abs=0.01)  == d["sigma_8"]
    assert pytest.approx(0.311, abs=0.005) == d["Omega_m"]
    assert pytest.approx(0.827, abs=0.01)  == d["S8"]
    assert d["der_full"].shape == (17,)


@pytest.mark.parametrize("omega_cdm,sigma8_expected", [
    (0.10, 0.74),     # less CDM → less growth → lower σ8
    (0.12, 0.82),
    (0.14, 0.89),     # more CDM → higher σ8
])
def test_sigma8_monotonic_in_omega_cdm(omega_cdm, sigma8_expected):
    cosmo = csl.CosmoParams(omega_cdm=omega_cdm)
    d = csl.derived(cosmo)
    assert pytest.approx(sigma8_expected, abs=0.05) == d["sigma_8"]


def test_S8_definition(cosmo):
    d = csl.derived(cosmo)
    s8_predicted = d["sigma_8"] * (d["Omega_m"] / 0.3) ** 0.5
    assert pytest.approx(s8_predicted, rel=1e-12) == d["S8"]


def test_Omega_m_sums_correctly():
    # Σmν = 3 m_ncdm under ede-v2 ν convention; Ω_m h² = ω_b + ω_cdm + Σmν/93.14
    cosmo = csl.CosmoParams()
    d = csl.derived(cosmo)
    h2 = (cosmo.H0 / 100.0) ** 2
    sum_mnu = 3.0 * cosmo.m_ncdm
    expected = (cosmo.omega_b + cosmo.omega_cdm + sum_mnu / 93.14) / h2
    assert pytest.approx(expected, rel=1e-10) == d["Omega_m"]
