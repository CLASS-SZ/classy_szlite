"""
Halo mass function and linear bias (JAX).

Builds a ``HaloGrids`` container from ``CosmoGrids`` — one emulator
call, then all downstream integrals are pure JAX.

Unit system — *physical (h-free)* throughout
=============================================

========== ============================== ==============
Quantity   Symbol                          Unit
========== ============================== ==============
M          halo mass                       M_sun
R_L        Lagrangian radius               Mpc
dn/dlnM    halo mass function              1 / Mpc³
b          linear halo bias                dimensionless
rho_m      mean matter density (z = 0)     M_sun / Mpc³
rho_crit   critical density (z)            M_sun / Mpc³
========== ============================== ==============

Design for JAX performance
==========================
* ``build_halo_grids(cosmo_grids, params)`` interpolates σ(R) → σ(M)
  once and returns a ``HaloGrids`` named-tuple.
* Every field of ``HaloGrids`` is a plain ``jnp.array``.
* ``HaloGrids`` is a JAX pytree (NamedTuple), so it can be passed
  directly into ``@jax.jit`` / ``jax.grad`` / ``jax.vmap`` functions.

Tinker 2008 multiplicity function
----------------------------------
The HMF is

    dn/dlnM = (ρ̄_m / M) × f(σ) × |dlnσ⁻¹/dlnM|

where ``f(σ)`` is the Tinker 2008 fitting function (Table 2 + redshift
evolution).  The ``tinker08_hmf`` function returns ``f(σ) / 2`` (the
factor of ½ compensates for using ``dlnσ²/dlnR`` instead of
``2 × dlnσ⁻¹/dlnR`` in the chain rule), so the HMF formula becomes

    dn/dlnM = (ρ̄_m / M) × hmf_f × (1/3) × |dlnσ²/dlnR|

with ``dlnM = 3 dlnR``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cosmology import CosmoGrids

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Physical constants (shared copies in constants.py; old names kept)
# ===================================================================

import numpy as np

from .constants import (
    DELTA_C, G_SI as _G_SI, C_SI as _c_SI, MSUN_KG as _Msun_kg, MPC_M as _Mpc_m,
    THREE_OVER_8PI_G as _3_OVER_8PI_G, MPC3_OVER_MSUN as _MPC3_OVER_MSUN,
)
from .cosmology import Omega_m0
from ._registry import DEFAULT_COSMO


# ===================================================================
# Public data container
# ===================================================================

class HaloGrids(NamedTuple):
    """Precomputed HMF, bias, and support quantities on a (z, M) grid.

    All masses are in M_sun (physical, not M_sun/h).

    Axes convention
    ---------------
    ============== ================== ==============
    Field          Shape              Unit
    ============== ================== ==============
    lnM            (n_m,)             ln(M_sun)
    rho_m0         scalar             M_sun / Mpc³
    rho_crit_z     (n_z,)             M_sun / Mpc³
    Omega_m_z      (n_z,)             —
    sigma_m        (n_z, n_m)         —
    nu             (n_z, n_m)         δ_c / σ
    dndlnm         (n_z, n_m)         1 / Mpc³
    bias           (n_z, n_m)         —
    ============== ================== ==============
    """
    lnM:        jax.Array   # (n_m,)
    rho_m0:     jax.Array   # scalar
    rho_crit_z: jax.Array   # (n_z,)
    Omega_m_z:  jax.Array   # (n_z,)
    sigma_m:    jax.Array   # (n_z, n_m)
    nu:         jax.Array   # (n_z, n_m)
    dndlnm:     jax.Array   # (n_z, n_m)
    bias:       jax.Array   # (n_z, n_m)


# ===================================================================
# Tinker 2008 multiplicity function
# ===================================================================

_T08_DELTA = np.array([200., 300., 400., 600., 800.,
                       1200., 1600., 2400., 3200.])
_T08_DELTA_TAB = jnp.log10(jnp.array(_T08_DELTA))      # log10 Δ nodes (default)
_T08_A_TAB = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
_T08_a_TAB = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87,
                          2.13, 2.30, 2.53, 2.66])
_T08_b_TAB = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
_T08_c_TAB = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58,
                          1.80, 1.97, 2.24, 2.44])

TINKER_INTERP = ("log10", "linear")


def tinker08_hmf(sigma, z, delta_mean, interp: str = "log10"):
    """Tinker 2008 multiplicity function (including ½ prefactor).

    Parameters
    ----------
    sigma      : (n_z, n_m)  rms fluctuation
    z          : (n_z,)      redshift
    delta_mean : (n_z,)      overdensity w.r.t. mean matter density
    interp     : 'log10' interpolates the Table 2 parameters linearly in
                 log10 Δ (classy_szlite's default); 'linear' interpolates
                 linearly in Δ (cosmocnc's ``interp_tinker='linear'``).
                 Both agree at the table nodes.

    Returns
    -------
    f : (n_z, n_m) — f(σ)/2 (see module docstring for the convention)
    """
    if interp == "log10":
        x, nodes = jnp.log10(delta_mean), _T08_DELTA_TAB
    elif interp == "linear":
        x, nodes = delta_mean, jnp.asarray(_T08_DELTA)
    else:
        raise ValueError(f"interp must be one of {TINKER_INTERP}, got {interp!r}")

    Ap = (jnp.interp(x, nodes, _T08_A_TAB)
          * (1.0 + z) ** (-0.14))
    a = (jnp.interp(x, nodes, _T08_a_TAB)
         * (1.0 + z) ** (-0.06))
    b = (jnp.interp(x, nodes, _T08_b_TAB)
         * (1.0 + z) ** (-jnp.power(
             10.0, -jnp.power(0.75 / jnp.log10(delta_mean / 75.0), 1.2))))
    c = jnp.interp(x, nodes, _T08_c_TAB)

    return (0.5 * Ap[:, None]
            * (jnp.power(sigma / b[:, None], -a[:, None]) + 1.0)
            * jnp.exp(-c[:, None] / sigma ** 2))


# ===================================================================
# Tinker 2010 linear halo bias
# ===================================================================

def tinker10_bias(nu, delta):
    """Tinker et al. 2010 first-order halo bias b(ν).

    Parameters
    ----------
    nu    : array, peak height δ_c / σ
    delta : float or array, overdensity w.r.t. mean matter density

    Returns
    -------
    b : same shape as nu
    """
    y = jnp.log10(delta)
    A = 1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)
    c = 2.4
    dc = DELTA_C
    return 1.0 - A * nu ** a / (nu ** a + dc ** a) + B * nu ** b + C * nu ** c


# ===================================================================
# Builder
# ===================================================================

def rho_crit_si(H_si):
    """Critical density [M_sun / Mpc³] for H in 1/s."""
    # Grouped so every JAX intermediate stays in float32 range (TPU "f64"
    # silently clips at ~3.4e38).
    return (H_si ** 2 * _3_OVER_8PI_G) * _MPC3_OVER_MSUN


def sigma_dndlnm_tinker08(lnM, z, R, sigma_R, rho_m0, delta_mean,
                          tinker_interp: str = "log10"):
    """σ(M, z) and the Tinker 2008 dn/dlnM [1/Mpc³] from σ(R, z).

    The pure core of :func:`build_halo_grids`: no emulator, no
    cosmology dict, so code tests can inject σ(R, z), ρ̄_m and Δ_mean(z).

    lnM (n_m,) uniform in ln M_sun; z (n_z,); R (n_R,) Mpc; sigma_R
    (n_z, n_R); rho_m0 [M_sun/Mpc³] sets M = (4π/3) ρ̄_m R³; delta_mean
    (n_z,) is the overdensity w.r.t. the mean density.
    """
    M = jnp.exp(lnM)
    R_lag = jnp.power(3.0 * M / (4.0 * jnp.pi * rho_m0), 1.0 / 3.0)   # (n_m,) Mpc

    # σ(R, z) → σ(M, z): linear interpolation of ln σ in ln R
    log_R_m = jnp.log(R_lag)
    sigma_m = jnp.exp(jax.vmap(
        lambda lnsig_z: jnp.interp(log_R_m, jnp.log(R), lnsig_z)
    )(jnp.log(sigma_R)))                                               # (n_z, n_m)

    # dlnσ²/dlnR by finite differences on the mass grid
    # (lnM uniform ⟹ lnR uniform, R ∝ M^{1/3})
    lnsig2 = 2.0 * jnp.log(sigma_m)
    dlnR = (1.0 / 3.0) * (lnM[1] - lnM[0])
    dlnsig2_center = (lnsig2[:, 2:] - lnsig2[:, :-2]) / (2.0 * dlnR)
    dlnsig2_left = (lnsig2[:, 1:2] - lnsig2[:, 0:1]) / dlnR
    dlnsig2_right = (lnsig2[:, -1:] - lnsig2[:, -2:-1]) / dlnR
    dlnsig2_dlnR = jnp.concatenate(
        [dlnsig2_left, dlnsig2_center, dlnsig2_right], axis=1)         # (n_z, n_m)

    hmf_f = tinker08_hmf(sigma_m, z, delta_mean, interp=tinker_interp)  # includes ½

    # dn/dlnM = (ρ̄_m / M) × [f(σ)/2] × (1/3) × |dlnσ²/dlnR|
    #         = (ρ̄_m / M) × f(σ) × |dlnσ⁻¹/dlnM|
    dndlnm = ((rho_m0 / M[None, :]) * hmf_f * (1.0 / 3.0)
              * jnp.abs(dlnsig2_dlnR))
    return sigma_m, dndlnm


def build_halo_grids(cg: CosmoGrids,
                     params: dict,
                     delta_crit: float = 200.0,
                     m_min: float = 1e10,
                     m_max: float = 5e16,
                     n_m: int = 200,
                     *,
                     tinker_interp: str = "log10",
                     nu_in_rho_m: bool = True,
                     nu_in_delta: bool | None = None,
                     round_delta_mean: bool = False) -> HaloGrids:
    """Build HMF and bias grids from a ``CosmoGrids`` container.

    Parameters
    ----------
    cg : CosmoGrids
        Output of ``cosmology.build()``.
    params : dict
        Cosmological parameters (``omega_b``, ``omega_cdm``, ``H0``,
        ``m_ncdm``; missing entries take the emulator defaults).
    delta_crit : float
        Spherical-overdensity definition w.r.t. **critical** density
        (200 for M_200c, 500 for M_500c).  Converted to mean-density
        overdensity internally: Δ_mean(z) = Δ_crit / Ω_m(z).
    m_min, m_max : float
        Mass range in **M_sun** (physical).
    n_m : int
        Number of mass bins (uniform in ln M).
    tinker_interp : {'log10', 'linear'}
        Interpolation of the Tinker 2008 parameters in Δ (see
        :func:`tinker08_hmf`).
    nu_in_rho_m : bool
        Include massive neutrinos (ω_ν = 3 m_ncdm / 93.14) in ρ̄_m, which
        sets M(R). True: classy_szlite's convention. False: cb only, as in
        cosmocnc.
    nu_in_delta : bool or None
        Include them in Ω_m(z) for Δ_mean = Δ_crit / Ω_m(z). None: same as
        ``nu_in_rho_m``.
    round_delta_mean : bool
        Round Δ_mean to an integer, as colossus does (used only in the
        colossus code test).

    Returns
    -------
    HaloGrids
        Named-tuple of JAX arrays. ``rho_m0`` is the ρ̄_m used for M(R);
        ``Omega_m_z`` is the Ω_m(z) used for Δ_mean.
    """
    if nu_in_delta is None:
        nu_in_delta = nu_in_rho_m
    full = dict(DEFAULT_COSMO)
    full.update(params)

    # -- densities [M_sun / Mpc³] ------------------------------------------
    rho_crit_0 = rho_crit_si(full['H0'] * 1e3 / _Mpc_m)  # H0 in 1/s
    rho_m0 = Omega_m0(full, include_nu=nu_in_rho_m) * rho_crit_0
    # H(z) in 1/s: Hz [1/Mpc] × c [m/s] / Mpc [m]  (see cosmology.py)
    rho_crit_z = rho_crit_si(cg.Hz * _c_SI / _Mpc_m)       # (n_z,)

    # -- Ω_m(z) and Δ_mean(z) ---------------------------------------------
    Omega_m_z = (Omega_m0(full, include_nu=nu_in_delta) * rho_crit_0
                 * (1.0 + cg.z) ** 3 / rho_crit_z)          # (n_z,)
    delta_mean = delta_crit / Omega_m_z
    if round_delta_mean:
        delta_mean = jnp.round(delta_mean)

    # -- mass grid, σ(M, z) and the HMF -----------------------------------
    lnM = jnp.linspace(jnp.log(m_min), jnp.log(m_max), n_m)
    sigma_m, dndlnm = sigma_dndlnm_tinker08(lnM, cg.z, cg.R, cg.sigma, rho_m0,
                                            delta_mean, tinker_interp)

    # -- Peak height and bias ---------------------------------------------
    nu = DELTA_C / sigma_m                            # (n_z, n_m)
    bias = tinker10_bias(nu, delta_mean[:, None])     # (n_z, n_m)

    return HaloGrids(
        lnM=lnM,
        rho_m0=jnp.asarray(rho_m0, dtype=jnp.float64),
        rho_crit_z=rho_crit_z,
        Omega_m_z=Omega_m_z,
        sigma_m=sigma_m,
        nu=nu,
        dndlnm=dndlnm,
        bias=bias,
    )
