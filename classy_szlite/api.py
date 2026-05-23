"""High-level public API: derived params, CMB Cls, tSZ Cl^yy.

All functions take a :class:`classy_szlite.params.CosmoParams` and an
optional ``cosmo_model`` (default ``'ede-v2'``).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ._registry import get_emulator, DEFAULT_COSMO_MODEL, DEFAULT_COSMO
from .params import CosmoParams, ProfileParamsA10
from .cosmology import build as build_cosmo_grids, get_pk, get_pknl, get_distances
from .hmf import build_halo_grids
from .power_spectrum import cl_yy_1h_2h

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# CosmoParams ↔ dict
# ---------------------------------------------------------------------------

def cosmo_to_dict(cosmo: CosmoParams) -> dict:
    """Convert ``CosmoParams`` to the emulator-style dict (with curly-brace key)."""
    return {
        "omega_b":       float(cosmo.omega_b),
        "omega_cdm":     float(cosmo.omega_cdm),
        "H0":            float(cosmo.H0),
        "tau_reio":      float(cosmo.tau_reio),
        "ln10^{10}A_s":  float(cosmo.ln10_10_As),
        "n_s":           float(cosmo.n_s),
        "m_ncdm":        float(cosmo.m_ncdm),
        "N_ur":          float(cosmo.N_ur),
        "fEDE":          float(cosmo.fEDE),
        "log10z_c":      float(cosmo.log10z_c),
        "thetai_scf":    float(cosmo.thetai_scf),
        "r":             float(cosmo.r),
    }


# ---------------------------------------------------------------------------
# Derived params (σ8, Ω_m, S8, plus the full DER vector)
# ---------------------------------------------------------------------------

def derived(cosmo: CosmoParams, cosmo_model: str = DEFAULT_COSMO_MODEL) -> dict:
    """Return derived parameters: ``sigma_8``, ``Omega_m``, ``S8`` (and the
    full DER emulator output for the curious).

    The DER emulator outputs an array of derived params; index [1] is σ8
    in both lcdm and ede-v2 (matches classy_szfast.classy_sz.py:436).
    Ω_m is computed analytically from omega_b, omega_cdm, m_ncdm, H0.
    """
    em = get_emulator(cosmo_model, 'der')
    full = dict(DEFAULT_COSMO[cosmo_model])
    full.update(cosmo_to_dict(cosmo))
    p_in = {k: [full[k]] for k in em.parameters}
    # DER emulator outputs log10(derived); recover physical values via 10^.
    # (Matches classy_szfast.classy_sz.py:436 which uses ten_to_predictions_np.)
    der = 10.0 ** np.asarray(em.predict(p_in)).flatten()

    sigma_8 = float(der[1])
    h = float(cosmo.H0) / 100.0
    # Σmν depends on neutrino convention
    sum_mnu = float(cosmo.m_ncdm) * (3.0 if float(cosmo.N_ur) < 0.5 else 1.0)
    Omega_m = (float(cosmo.omega_b) + float(cosmo.omega_cdm)
               + sum_mnu / 93.14) / h ** 2
    S8 = sigma_8 * (Omega_m / 0.3) ** 0.5
    return {
        "sigma_8": sigma_8,
        "Omega_m": Omega_m,
        "S8":      S8,
        "der_full": der,    # full 14 (lcdm) or 17 (ede-v2) elements
    }


# ---------------------------------------------------------------------------
# CMB angular power spectra: TT, TE, EE, PP
# ---------------------------------------------------------------------------

# Output conventions: TT, EE, PP emulators were trained on log10[ Cl ]; TE
# was trained on Cl directly (since it can be negative). Convention mirrors
# classy_szfast/classy_szfast.py:509, 533, 536, 539.
_CMB_LOG_CONVENTION = {
    "tt": True, "ee": True, "pp": True,
    "te": False,
}


def cl_TTTEEE(cosmo: CosmoParams, cosmo_model: str = DEFAULT_COSMO_MODEL,
              spectra: tuple[str, ...] = ("tt", "te", "ee"),
              ell_factor: bool = True) -> dict:
    """CMB angular power spectra.

    Returns a dict with keys ``'ell'`` and the requested spectra
    (``'tt','te','ee'``). Values are **dimensionless** — multiply by
    ``Tcmb_uK² = (2.7255e6)²`` to convert to μK².

    Per-cosmo_model normalisation (mirrors classy_szfast.classy_szfast.py:522):
      * lcdm emulator output: ``log10[ ell(ell+1) Cl / (2π) ]``
        → recover Cl via × ``2π / [ell(ell+1)]``
      * ede-v2 emulator output: ``log10[ ell² Cl ]``
        → recover Cl via × ``1/ell²``

    ``ell_factor`` (default ``True``) — return ``D_ell = ell(ell+1) × Cl / (2π)``;
    ``False`` returns raw Cl.
    """
    full = dict(DEFAULT_COSMO[cosmo_model])
    full.update(cosmo_to_dict(cosmo))

    out = {}
    ell = None
    for spec in spectra:
        if spec not in _CMB_LOG_CONVENTION:
            raise ValueError(f"Unknown spectrum {spec!r}. Pick from "
                             f"{tuple(_CMB_LOG_CONVENTION)}.")
        em = get_emulator(cosmo_model, spec)
        p_in = {k: [full[k]] for k in em.parameters}
        pred = np.asarray(em.predict(p_in)).flatten()
        if _CMB_LOG_CONVENTION[spec]:
            pred = 10.0 ** pred
        out[spec] = pred
        if ell is None:
            ell = np.asarray(em.modes)

    # Per-cosmo_model factor that recovers raw Cl from the emulator output.
    # Same convention as classy_szfast.classy_szfast.py:522-527.
    if cosmo_model == 'ede-v2':
        factor_to_Cl = 1.0 / (ell ** 2)
    else:
        factor_to_Cl = 1.0 / (ell * (ell + 1) / (2.0 * np.pi))

    for s in spectra:
        out[s] = out[s] * factor_to_Cl                     # now raw Cl (dimensionless)

    out["ell"] = ell
    if ell_factor:
        # convert to D_ell = ell(ell+1) Cl / (2π)
        fac_dl = ell * (ell + 1) / (2.0 * np.pi)
        for s in spectra:
            out[s] = out[s] * fac_dl
    return out


# ---------------------------------------------------------------------------
# Pk, Pnl, distances (forward to cosmology module with proper defaults)
# ---------------------------------------------------------------------------

def Pk(cosmo: CosmoParams, z_arr, cosmo_model: str = DEFAULT_COSMO_MODEL):
    """Linear P(k, z) — returns ``(k, pk(z, k))``."""
    return get_pk(cosmo_to_dict(cosmo), z_arr, cosmo_model)


def Pnl(cosmo: CosmoParams, z_arr, cosmo_model: str = DEFAULT_COSMO_MODEL):
    """Non-linear P(k, z) (HMcode) — returns ``(k, pk(z, k))``."""
    return get_pknl(cosmo_to_dict(cosmo), z_arr, cosmo_model)


def distances(cosmo: CosmoParams, z_arr,
              cosmo_model: str = DEFAULT_COSMO_MODEL):
    """Returns ``(Hz, chi, Da)``. ``Hz`` is H(z)/c in 1/Mpc; distances in Mpc."""
    return get_distances(cosmo_to_dict(cosmo), z_arr, cosmo_model)


# ---------------------------------------------------------------------------
# tSZ Cl^yy (halo-model, Arnaud 2010 profile)
# ---------------------------------------------------------------------------

def cl_yy(cosmo: CosmoParams, profile: ProfileParamsA10,
          ell, cosmo_model: str = DEFAULT_COSMO_MODEL,
          z_grid: jax.Array | None = None,
          n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
          n_m: int = 200, delta_crit: float = 500.0):
    """Halo-model tSZ angular power spectrum.

    Returns ``(cl_1h, cl_2h)`` — dimensionless C_ell. Multiply by
    ``ell*(ell+1)/(2π)*1e12`` to get the conventional ``D_ell × 1e12`` form
    that matches Planck / ACT tSZ bandpower data.

    ``profile`` is :class:`~classy_szlite.params.ProfileParamsA10` (Arnaud 10
    gNFW + ``B`` = hydrostatic mass bias). Battaglia 12 not yet wired.
    """
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid, cosmo_model=cosmo_model)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    pp_dict = profile._asdict()
    cl_1h, cl_2h = cl_yy_1h_2h(jnp.asarray(ell), cg, hg, cosmo_dict,
                                profile='arnaud10', profile_params=pp_dict)
    return cl_1h, cl_2h
