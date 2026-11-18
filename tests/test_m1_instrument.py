"""M1 code tests: SO LAT instrument model, tile grid and binned analytic
covariance against szsim (reference dumped by benchmarks/mapsim/refgen_szsim.py).

Identical inputs, so the tolerances are round-off. Marked ``refdata``: the
reference file lives in $MAPSIM_REF (default on rusty ceph).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

REF = os.path.join(os.environ.get("MAPSIM_REF", "/mnt/ceph/users/bbolliet/mapsim/ref"),
                   "szsim_instrument.npz")
REFDATA = os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending")

pytestmark = [pytest.mark.refdata,
              pytest.mark.skipif(not os.path.exists(REF), reason="szsim reference not generated")]


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF))


def test_constants(ref):
    from classy_szlite.sim import instrument as ins
    for k, v in (("freqs", ins.SO_LAT["freqs"]), ("fwhm", ins.SO_LAT["fwhm"]),
                 ("white_goal", ins.SO_LAT["white"]["goal"]),
                 ("white_baseline", ins.SO_LAT["white"]["baseline"]),
                 ("N_red", ins.SO_LAT["N_red"])):
        np.testing.assert_array_equal(v, ref[k], err_msg=k)
    assert ins.t_eff() == ref["t_eff"]


def test_sed_and_beams(ref):
    from classy_szlite.sim import instrument as ins
    np.testing.assert_allclose(ins.tsz_f_uK(ins.SO_LAT["freqs"]), ref["f_uK"], rtol=1e-14)
    np.testing.assert_allclose(ins.tsz_g(ins.SO_LAT["freqs"]), ref["g_x"], rtol=1e-14)
    np.testing.assert_allclose(ins.beams(ref["ell"]), ref["beams"], rtol=1e-12, atol=0)


def test_sed_null_frequency():
    """Independent check: g(x) = 0 at x = 3.8300, i.e. ν ≈ 217.5 GHz."""
    from scipy.optimize import brentq
    from classy_szlite.sim import instrument as ins
    nu0 = brentq(lambda nu: ins.tsz_g(nu), 150.0, 300.0, xtol=1e-12)
    x0 = ins.H_PLANCK * nu0 * 1e9 / (ins.K_BOLTZ * ins.T_CMB)
    assert abs(x0 - 3.8300161) < 1e-6
    assert abs(nu0 - 217.5) < 0.05


@pytest.mark.parametrize("mode,atm", [("goal", True), ("baseline", True), ("goal", False)])
def test_noise_cov(ref, mode, atm):
    from classy_szlite.sim import instrument as ins
    key = f"noise_{mode}" + ("" if atm else "_noatm")
    got = ins.noise_cov(ref["ell"], mode, atm=atm)
    np.testing.assert_allclose(got, ref[key], rtol=1e-12, atol=0)


def test_cmb_table(ref):
    from classy_szlite.sim import instrument as ins
    cl = ins.cmb_cl_from_table(os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    np.testing.assert_allclose(cl(ref["ell"]), ref["cmb_cl"], rtol=1e-14, atol=0)


def test_grid_and_bins(ref):
    from classy_szlite.sim.flatsky import grid, DK
    G = grid()
    assert DK == ref["dk"] and abs(DK - 36.0) < 1e-12
    np.testing.assert_array_equal(G.n2, ref["grid_n2"])
    np.testing.assert_allclose(G.L, ref["grid_L"], rtol=1e-15)
    np.testing.assert_array_equal(G.sel, ref["grid_sel"])
    np.testing.assert_array_equal(G.ib, ref["grid_ib"])
    np.testing.assert_array_equal(G.W, ref["grid_W"])
    np.testing.assert_array_equal(G.bins_used, ref["bins_used"])
    # same in-band modes; ours are sorted by bin, szsim's row-major
    np.testing.assert_array_equal(np.sort(G.m_idx), ref["m_idx"])
    # full-plane mode count: Σ W over the half plane = N²
    assert G.W.sum() == G.n ** 2


def test_analytic_binned_cov(ref):
    """Our per-mode analytic C(k) averaged in the finder's bins = szsim's."""
    from classy_szlite.sim import instrument as ins
    from classy_szlite.sim.flatsky import grid
    G = grid()
    cl = ins.cmb_cl_from_table(os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    for mode in ("goal", "baseline"):
        B = ins.beams(G.m_L)                                          # (6, n_modes)
        Nn = ins.noise_cov(G.m_L, mode)                               # (6, 6, n_modes)
        C = Nn + B[:, None, :] * B[None, :, :] * cl(G.m_L)[None, None, :]
        C = C * (G.Npx / G.Om)
        Cb = np.zeros((G.n_bins, 6, 6))
        for i in range(6):
            for j in range(6):
                Cb[:, i, j] = np.bincount(G.m_binpos, weights=G.m_W * C[i, j],
                                          minlength=G.n_bins) / G.bin_count
        np.testing.assert_allclose(Cb, ref[f"abc_{mode}"][G.bins_used], rtol=1e-11, atol=0)


def test_cmb_emulator_vs_camb():
    """Physics comparison: the TT emulator (CLASS-EDE at fEDE = 0.001, three
    0.02 eV neutrinos) against szsim's CAMB table (one 0.06 eV neutrino) at
    the table's cosmology. Plan M1: ≤ 0.5% for ℓ ≤ 3000; measured 0.42%
    (2026-11-18). Higher ℓ is documented, not tested: 0.5-0.7% up to 9000."""
    import classy_szlite as csl
    from classy_szlite.sim import instrument as ins
    path = os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz")
    d = np.load(path)
    c = csl.CosmoParams(H0=67.66, omega_b=0.02243159044, omega_cdm=0.11934192793638221,
                        tau_reio=0.0544, ln10_10_As=float(np.log(2.1045375196677236e-09 * 1e10)),
                        n_s=0.9665)
    f = ins.cmb_cl_emulator(c)
    ell = np.arange(2, 3001, dtype=float)
    r = f(ell) / np.interp(ell, d["ell"], d["cl_uK2"]) - 1.0
    print(f"emulator vs CAMB, ell 2-3000: max |ratio-1| = {np.max(np.abs(r)):.2e}; "
          f"extension slope above 9000: {f.slope:.3f}")
    assert np.max(np.abs(r)) < 5e-3
    # the extension is a smooth, steeply falling tail
    assert -8.0 < f.slope < -2.0
