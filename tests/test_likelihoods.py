"""Smoke + best-fit χ² tests for the bundled JAX likelihoods.

Requires the data files extracted via
``python -m classy_szlite.likelihoods.extract_data``. Skipped automatically
if the data is not present.
"""
from __future__ import annotations
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import classy_szlite as csl
from classy_szlite.likelihoods._data import _find_data_file


# Skip the whole module if the data tables aren't installed.
try:
    _find_data_file()
    _HAVE_DATA = True
except FileNotFoundError:
    _HAVE_DATA = False

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA,
    reason="Run `python -m classy_szlite.likelihoods.extract_data` first.",
)


# Best-fit point from p-actbase_ede+n3_classsz.minimum.txt
BF_COSMO = csl.CosmoParams(
    omega_b=0.022751597, omega_cdm=0.13242228, H0=71.113655,
    ln10_10_As=3.0722822, n_s=0.98501417, tau_reio=0.05798782,
    fEDE=0.11678514, log10z_c=3.5517577, thetai_scf=2.6478792,
)
BF_NUIS = dict(
    a_tSZ=3.4983345, alpha_tSZ=-0.62001806, a_kSZ=1.2858443,
    a_p=7.6669211, beta_p=1.8798252, a_c=4.0215689,
    a_s=2.9096742, beta_s=-2.7694932,
    a_gtt=7.9189003, a_gte=0.41908216, a_gee=0.16687108,
    a_psee=0.0096106134, a_pste=-0.025321767, xi=0.08599627,
    calG_all=1.0013228, calE_dr6_pa4_f220=1.0,
    calE_dr6_pa5_f090=0.98883908, calE_dr6_pa5_f150=0.99891013,
    calE_dr6_pa6_f090=0.99875697, calE_dr6_pa6_f150=0.99889845,
    cal_dr6_pa4_f220=0.97804664, cal_dr6_pa5_f090=1.0004643,
    cal_dr6_pa5_f150=0.99900579, cal_dr6_pa6_f090=1.0001087,
    cal_dr6_pa6_f150=1.0011954, A_planck=1.0013228,
)

# Reference χ² from cobaya at the chain best-fit
_REF_TOTAL = 6515.21


def test_total_chi2_at_bestfit_v2():
    from classy_szlite.likelihoods import total_chi2
    val = float(total_chi2(BF_COSMO, BF_NUIS, use_v2_foregrounds=True))
    # We accept a small residual (~0.05) from the linearly-interpolated
    # sroll2 lookup. Anything bigger means an actual regression.
    assert abs(val - _REF_TOTAL) < 0.2, f"total χ² = {val:.3f} vs ref {_REF_TOTAL}"


def test_total_chi2_at_bestfit_fixed_fg():
    from classy_szlite.likelihoods import total_chi2
    val = float(total_chi2(BF_COSMO, BF_NUIS, use_v2_foregrounds=False))
    assert abs(val - _REF_TOTAL) < 0.2


def test_grad_through_total_chi2():
    """``jax.grad`` must traverse cosmology → Cls → χ² without error."""
    from classy_szlite.likelihoods import total_chi2
    cosmo_dict = dict(
        omega_b=BF_COSMO.omega_b, omega_cdm=BF_COSMO.omega_cdm,
        H0=BF_COSMO.H0, ln10_10_As=BF_COSMO.ln10_10_As,
        n_s=BF_COSMO.n_s, tau_reio=BF_COSMO.tau_reio,
        fEDE=BF_COSMO.fEDE, log10z_c=BF_COSMO.log10z_c,
        thetai_scf=BF_COSMO.thetai_scf,
        m_ncdm=0.02, N_ur=0.00441,
    )

    def loss(cosmo_d):
        return total_chi2(cosmo_d, BF_NUIS, use_v2_foregrounds=True)

    grad_fn = jax.jit(jax.grad(loss))
    g = grad_fn(cosmo_dict)
    # All cosmology entries should produce finite gradients
    for k in ("H0", "omega_b", "omega_cdm", "ln10_10_As", "n_s",
              "tau_reio", "fEDE", "log10z_c", "thetai_scf"):
        assert jnp.isfinite(g[k]), f"non-finite gradient for {k}"


def test_fg_linearity_at_far_amplitudes():
    """``chi2_mflike_v2`` is amplitude-linear by construction (with SED tilts
    fixed). This regression test fired when ``alpha_s`` was wrong in the
    extractor: at the BF point the bug is invisible (deviation = 0), but at
    perturbed amplitudes the chi² blew up by thousands. We compare against
    a hand-computed linear extrapolation that doesn't reach into cobaya."""
    from classy_szlite.likelihoods import chi2_mflike_v2, foreground
    from classy_szlite.likelihoods.core import _cosmo_to_cls

    # Build cosmology Cls once.
    cosmo_d = dict(omega_b=BF_COSMO.omega_b, omega_cdm=BF_COSMO.omega_cdm,
                    H0=BF_COSMO.H0, ln10_10_As=BF_COSMO.ln10_10_As,
                    n_s=BF_COSMO.n_s, tau_reio=BF_COSMO.tau_reio,
                    fEDE=BF_COSMO.fEDE, log10z_c=BF_COSMO.log10z_c,
                    thetai_scf=BF_COSMO.thetai_scf,
                    m_ncdm=0.02, N_ur=0.00441)
    cls = _cosmo_to_cls(cosmo_d)

    # χ² at BF — anchor
    chi2_bf = float(chi2_mflike_v2(cls, BF_NUIS))

    # Perturb amplitudes by ±20% one at a time and check the fg total matches
    # the linear scaling exactly.
    foreground._TABLES = None        # force a clean reload
    foreground._load()
    tt, te, ee, bf, fg_bf = foreground._TABLES

    # Amplitudes that don't enter the tSZ_and_CIB cross term — pure linear.
    for amp_key in ("a_kSZ", "a_p", "a_s", "a_gtt"):
        pert = dict(BF_NUIS); pert[amp_key] = 1.2 * pert[amp_key]
        fg_jax = foreground.fg_totals_jax(pert)
        comp = {"a_kSZ": "kSZ", "a_p": "cibp", "a_s": "radio",
                 "a_gtt": "dust"}[amp_key]
        i, j, ell_idx = 2, 2, 1498  # pa5_f150², ell ≈ 1500
        d = 0.2  # 1.2 / 1 − 1
        expected = float(fg_bf["tt"][i, j, ell_idx]
                          + d * tt[comp][i, j, ell_idx])
        actual = float(fg_jax["tt"][i, j, ell_idx])
        assert abs(actual - expected) < 1e-9, (
            f"{amp_key} 20%-perturb: jax={actual:.6e} expected={expected:.6e}"
        )

    # a_tSZ / a_c / xi all enter the cross via −ξ √(a_tSZ a_c). At 20%
    # perturbation of a_tSZ both tSZ self and cross move, so the test
    # accumulates both contributions.
    pert = dict(BF_NUIS); pert["a_tSZ"] = 1.2 * pert["a_tSZ"]
    fg_jax = foreground.fg_totals_jax(pert)
    i, j, ell_idx = 2, 2, 1498
    bf_amp_cross = -float(BF_NUIS["xi"]) * (float(BF_NUIS["a_tSZ"])
                                              * float(BF_NUIS["a_c"])) ** 0.5
    new_amp_cross = -float(BF_NUIS["xi"]) * (1.2 * float(BF_NUIS["a_tSZ"])
                                               * float(BF_NUIS["a_c"])) ** 0.5
    d_tSZ = 0.2
    d_cross = new_amp_cross / bf_amp_cross - 1.0
    expected = float(fg_bf["tt"][i, j, ell_idx]
                      + d_tSZ * tt["tSZ"][i, j, ell_idx]
                      + d_cross * tt["tSZxCIB"][i, j, ell_idx])
    actual = float(fg_jax["tt"][i, j, ell_idx])
    assert abs(actual - expected) < 1e-9, (
        f"a_tSZ 20%-perturb: jax={actual:.6e} expected={expected:.6e}"
    )


def test_ell_convention_round_trip():
    """Both conventions store the same emulator output, just under a shifted
    ℓ axis. ``Cl(ell) × ell²`` should therefore agree element-wise."""
    out_szfast = csl.cl_TTTEEE(BF_COSMO, ell_factor=False,
                                 ell_convention="classy_szfast")
    out_emul = csl.cl_TTTEEE(BF_COSMO, ell_factor=False,
                               ell_convention="emulator_modes")
    np.testing.assert_allclose(
        out_szfast["tt"] * out_szfast["ell"] ** 2,
        out_emul["tt"] * out_emul["ell"] ** 2,
        rtol=1e-12,
    )
