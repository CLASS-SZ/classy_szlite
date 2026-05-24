"""CMB angular power spectrum tests."""
from __future__ import annotations
import numpy as np
import pytest
import classy_szlite as csl

T_CMB_UK2 = (2.7255e6) ** 2


def test_cl_TTTEEE_shape(cosmo):
    out = csl.cl_TTTEEE(cosmo)
    assert set(out) >= {"ell", "tt", "te", "ee"}
    n = len(out["ell"])
    for k in ("tt", "te", "ee"):
        assert out[k].shape == (n,)


def test_TT_first_acoustic_peak(cosmo):
    """D_ell^TT peak near ell ~ 220 should be 5000-6500 μK²."""
    out = csl.cl_TTTEEE(cosmo)
    ell = out["ell"]
    i220 = int(np.argmin(np.abs(ell - 220)))
    peak_uK2 = out["tt"][i220] * T_CMB_UK2
    assert 5000 < peak_uK2 < 6500, f"TT peak at l=220 = {peak_uK2:.0f} μK²"


def test_EE_peak_smaller_than_TT(cosmo):
    out = csl.cl_TTTEEE(cosmo)
    ee_peak = out["ee"].max() * T_CMB_UK2
    tt_peak = out["tt"].max() * T_CMB_UK2
    assert ee_peak < 0.2 * tt_peak, f"EE ({ee_peak:.0f}) too large vs TT ({tt_peak:.0f})"


def test_TE_can_be_negative(cosmo):
    """TE has both positive and negative regions — it's not log-spaced output."""
    out = csl.cl_TTTEEE(cosmo)
    assert out["te"].min() < 0, "TE should have negative regions"
    assert out["te"].max() > 0, "TE should have positive regions"


def test_ell_factor_off(cosmo):
    """ell_factor=False returns raw Cl (smaller magnitude, smooth profile)."""
    out_d = csl.cl_TTTEEE(cosmo, ell_factor=True)
    out_c = csl.cl_TTTEEE(cosmo, ell_factor=False)
    ell = out_c["ell"]
    factor = ell * (ell + 1) / (2 * np.pi)
    np.testing.assert_allclose(out_d["tt"], out_c["tt"] * factor, rtol=1e-10)
