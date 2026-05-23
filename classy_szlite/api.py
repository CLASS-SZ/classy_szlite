"""High-level public API: derived params, CMB Cls, matter Pk, distances,
tSZ Cl^yy. All functions take a :class:`classy_szlite.params.CosmoParams`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ._registry import get_emulator, DEFAULT_COSMO
from .params import CosmoParams, ProfileParamsA10
from .cosmology import build as build_cosmo_grids, get_pk, get_pknl, get_distances
from .hmf import build_halo_grids
from .power_spectrum import cl_yy_1h_2h

jax.config.update("jax_enable_x64", True)


def cosmo_to_dict(cosmo: CosmoParams) -> dict:
    """Convert ``CosmoParams`` to the emulator-style dict (with curly-brace key).

    Values pass through as-is (float or jax.Array) so the returned dict is
    JAX-traceable — pass a ``CosmoParams`` of tracers to ``jax.grad``.
    """
    return {
        "omega_b":       cosmo.omega_b,
        "omega_cdm":     cosmo.omega_cdm,
        "H0":            cosmo.H0,
        "tau_reio":      cosmo.tau_reio,
        "ln10^{10}A_s":  cosmo.ln10_10_As,
        "n_s":           cosmo.n_s,
        "m_ncdm":        cosmo.m_ncdm,
        "N_ur":          cosmo.N_ur,
        "fEDE":          cosmo.fEDE,
        "log10z_c":      cosmo.log10z_c,
        "thetai_scf":    cosmo.thetai_scf,
        "r":             cosmo.r,
    }


# ---------------------------------------------------------------------------
# Derived parameters (σ8, Ω_m, S8)
# ---------------------------------------------------------------------------

def derived(cosmo: CosmoParams) -> dict:
    """Derived parameters: ``sigma_8``, ``Omega_m``, ``S8``.

    Also returns the full 17-element DER emulator output as ``'der_full'``
    (σ8 is at index 1; consult the CosmoPower DER training script for the
    full list).
    """
    em = get_emulator('der')
    full = dict(DEFAULT_COSMO); full.update(cosmo_to_dict(cosmo))
    p_in = {k: [full[k]] for k in em.parameters}
    der = 10.0 ** np.asarray(em.predict(p_in)).flatten()

    sigma_8 = float(der[1])
    h = float(cosmo.H0) / 100.0
    # Σmν = 3 × m_ncdm for ede-v2 ν convention (3 degenerate ν)
    sum_mnu = 3.0 * float(cosmo.m_ncdm)
    Omega_m = (float(cosmo.omega_b) + float(cosmo.omega_cdm)
               + sum_mnu / 93.14) / h ** 2
    S8 = sigma_8 * (Omega_m / 0.3) ** 0.5
    return {"sigma_8": sigma_8, "Omega_m": Omega_m, "S8": S8, "der_full": der}


# ---------------------------------------------------------------------------
# CMB angular power spectra
# ---------------------------------------------------------------------------

# TT, EE, PP emulators output log10(prefactored Cl); TE outputs Cl directly
# (it can be negative). Recovery factor: ede-v2 uses 1/ell² to get raw Cl.
_CMB_LOG_CONVENTION = {"tt": True, "ee": True, "pp": True, "te": False}


def cl_TTTEEE(cosmo: CosmoParams,
              spectra: tuple[str, ...] = ("tt", "te", "ee"),
              ell_factor: bool = True) -> dict:
    """CMB angular power spectra.

    Returns a dict with keys ``'ell'`` and the requested spectra
    (``'tt','te','ee'``). Values are **dimensionless** — multiply by
    ``Tcmb_uK² = (2.7255e6)²`` to convert to μK².

    ``ell_factor`` (default ``True``) — return ``D_ell = ell(ell+1) Cl / (2π)``;
    ``False`` returns raw Cl.
    """
    full = dict(DEFAULT_COSMO); full.update(cosmo_to_dict(cosmo))

    out = {}
    ell = None
    for spec in spectra:
        if spec not in _CMB_LOG_CONVENTION:
            raise ValueError(f"Unknown spectrum {spec!r}. Pick from {tuple(_CMB_LOG_CONVENTION)}.")
        em = get_emulator(spec)
        p_in = {k: [full[k]] for k in em.parameters}
        pred = np.asarray(em.predict(p_in)).flatten()
        if _CMB_LOG_CONVENTION[spec]:
            pred = 10.0 ** pred
        out[spec] = pred
        if ell is None:
            ell = np.asarray(em.modes)

    # ede-v2 recovery factor: raw → Cl
    factor_to_Cl = 1.0 / (ell ** 2)
    for s in spectra:
        out[s] = out[s] * factor_to_Cl

    out["ell"] = ell
    if ell_factor:
        fac_dl = ell * (ell + 1) / (2.0 * np.pi)
        for s in spectra:
            out[s] = out[s] * fac_dl
    return out


# ---------------------------------------------------------------------------
# Pk, Pnl, distances
# ---------------------------------------------------------------------------

def Pk(cosmo: CosmoParams, z_arr):
    """Linear P(k, z) — returns ``(k, pk(z, k))``."""
    return get_pk(cosmo_to_dict(cosmo), z_arr)


def Pnl(cosmo: CosmoParams, z_arr):
    """Non-linear P(k, z) (HMcode) — returns ``(k, pk(z, k))``."""
    return get_pknl(cosmo_to_dict(cosmo), z_arr)


def distances(cosmo: CosmoParams, z_arr):
    """Returns ``(Hz, chi, Da)``. ``Hz`` is H(z)/c in 1/Mpc; distances in Mpc."""
    return get_distances(cosmo_to_dict(cosmo), z_arr)


# ---------------------------------------------------------------------------
# tSZ Cl^yy (halo-model, Arnaud 2010 profile)
# ---------------------------------------------------------------------------

def cl_yy(cosmo: CosmoParams, profile: ProfileParamsA10, ell,
          z_grid: jax.Array | None = None,
          n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
          n_m: int = 200, delta_crit: float = 500.0):
    """Halo-model tSZ angular power spectrum (full pipeline per call).

    Returns ``(cl_1h, cl_2h)`` — dimensionless C_ell. Multiply by
    ``ell*(ell+1)/(2π)*1e12`` to get ``D_ell × 1e12``.

    For MCMC sampling only profile parameters at fixed cosmology, use
    :func:`cl_yy_factory` instead — ~3× faster.
    """
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    pp_dict = profile._asdict()
    cl_1h, cl_2h = cl_yy_1h_2h(jnp.asarray(ell), cg, hg, cosmo_dict,
                                profile='arnaud10', profile_params=pp_dict)
    return cl_1h, cl_2h


def cl_yy_factory(cosmo: CosmoParams, ell,
                  z_grid: jax.Array | None = None,
                  n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
                  n_m: int = 200, delta_crit: float = 500.0):
    """Fixed-cosmology fast-path: precompute the heavy bits, get a closure.

    Builds ``CosmoGrids`` (emulators → P_lin, distances, σ(R)) and
    ``HaloGrids`` (Tinker 08 HMF, bias) **once**, then returns:

        ev(profile) -> (cl_1h, cl_2h)

    A subsequent ``ev(profile)`` call only runs the ``cl_yy_1h_2h``
    halo-model integration — typically ~5 ms per call. Intended for MCMC
    over profile / nuisance parameters with fixed cosmology.
    """
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    ell_jax = jnp.asarray(ell)

    def evaluate(profile: ProfileParamsA10):
        cl_1h, cl_2h = cl_yy_1h_2h(
            ell_jax, cg, hg, cosmo_dict,
            profile='arnaud10', profile_params=profile._asdict(),
        )
        return cl_1h, cl_2h

    return evaluate
