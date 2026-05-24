"""Matter Pk + cosmological distance tests."""
from __future__ import annotations
import numpy as np
import pytest
import classy_szlite as csl


def test_pk_linear_shape_and_units(cosmo):
    z = np.array([0.0, 0.5, 1.0, 2.0])
    k, pk = csl.Pk(cosmo, z)
    assert pk.shape == (len(z), len(k))
    assert k[0] < 1e-3 and k[-1] > 1.0
    assert (pk > 0).all()


def test_pk_low_k_slope_matches_ns(cosmo):
    """In the deep low-k extrapolation regime, Pk ∝ k^n_s."""
    k, pk = csl.Pk(cosmo, np.array([0.0, 0.5]))
    # Only the deeply-extrapolated zone (k <~ 5e-4) is pure k^n_s;
    # in between (5e-4 to 5e-2) the power spectrum turns over before
    # the BAO bump and the effective slope is shallower than n_s.
    mask = k < 5e-4
    slope = np.polyfit(np.log(k[mask]), np.log(pk[0, mask]), 1)[0]
    assert pytest.approx(cosmo.n_s, abs=0.02) == slope


def test_pk_decreasing_with_z(cosmo):
    """P(k, z=0) > P(k, z=2) at all k."""
    z = np.array([0.0, 2.0])
    k, pk = csl.Pk(cosmo, z)
    assert (pk[0] > pk[1]).all()


def test_pnl_close_to_pk_at_low_k(cosmo):
    """Non-linear and linear Pk agree at k ≲ 0.05 h/Mpc."""
    z = np.array([0.0, 0.5])
    k, pk  = csl.Pk(cosmo, z)
    _, pnl = csl.Pnl(cosmo, z)
    mask = (k > 1e-3) & (k < 5e-2)
    rel = np.abs(pnl[0, mask] - pk[0, mask]) / pk[0, mask]
    assert rel.max() < 0.05, f"Pnl/Pk agreement at low-k: max rel = {rel.max():.3f}"


def test_pnl_above_pk_at_high_k(cosmo):
    """At k ~ 1 h/Mpc, non-linear power exceeds linear."""
    z = np.array([0.0, 0.5])
    k, pk  = csl.Pk(cosmo, z)
    _, pnl = csl.Pnl(cosmo, z)
    mask = (k > 0.5) & (k < 2.0)
    assert (pnl[0, mask] > pk[0, mask]).all()


# Distances ------------------------------------------------------------------

def test_distances_anchored_at_zero(cosmo):
    """chi(z) ~ 0 as z → 0 (numerically, ~22 Mpc at z=0.005)."""
    z = np.array([0.005, 0.1])
    Hz, chi, Da = csl.distances(cosmo, z)
    assert 15 < float(chi[0]) < 30
    assert pytest.approx(float(Da[0]), rel=1e-2) == float(chi[0]) / (1 + 0.005)


def test_distance_monotonicity(cosmo):
    z = np.linspace(0.005, 3.0, 50)
    Hz, chi, Da = csl.distances(cosmo, z)
    assert (np.diff(chi) > 0).all(), "chi(z) must increase monotonically"
    assert (np.diff(Hz) > 0).all(), "H(z) must increase monotonically"
    # Da peaks near z~1.5, so must rise then fall
    assert Da.argmax() > 5
    assert Da.argmax() < len(z) - 5


def test_distance_h0_scaling():
    """chi(z) scales as 1/H0 to leading order."""
    z = np.array([1.0, 2.0])
    c1 = csl.CosmoParams(H0=67.0)
    c2 = csl.CosmoParams(H0=72.0)
    _, chi1, _ = csl.distances(c1, z)
    _, chi2, _ = csl.distances(c2, z)
    # Higher H0 → smaller distances
    assert chi2[0] < chi1[0]
