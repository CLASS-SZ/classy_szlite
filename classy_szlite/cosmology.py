"""Cosmological grids from the ede-v3 CosmoPower emulators (pure JAX).

Calls emulators once, returns a frozen :class:`CosmoGrids` container that
downstream JIT-compiled functions can consume without further Python
overhead.

Unit system — physical (h-free) throughout
==========================================
========= ========================== ==============
Quantity  Symbol                      Unit
========= ========================== ==============
k         wavenumber                  1 / Mpc
P(k,z)    matter power spectrum       Mpc³
R         TophatVar smoothing radius  Mpc
sigma     rms density fluctuation     dimensionless
H(z)      Hubble rate / c             1 / Mpc
chi(z)    comoving distance           Mpc
d_A(z)    angular-diameter distance   Mpc
========= ========================== ==============
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._registry import (
    PK_GRID, PKNL_ZMAX, DIST_ZMAX, DEFAULT_COSMO,
    get_emulator, output_is_log10,
)
from ._fftlog import FFTLog, Mellin_TophatSq_dim3
from .constants import N_NCDM_DEG, EV_PER_OMEGA_NU_H2

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Public data container
# ===================================================================

class CosmoGrids(NamedTuple):
    """Precomputed cosmological arrays. JAX pytree (NamedTuple)."""
    z:         jax.Array   # (n_z,)
    k:         jax.Array   # (n_k,)
    pk:        jax.Array   # (n_z, n_k)
    Hz:        jax.Array   # (n_z,)  — H(z)/c in 1/Mpc
    chi:       jax.Array   # (n_z,)
    Da:        jax.Array   # (n_z,)
    R:         jax.Array   # (n_R,)
    sigma:     jax.Array   # (n_z, n_R)
    dsigma2dR: jax.Array   # (n_z, n_R)


# ===================================================================
# Static grids (host-side numpy, built once at import)
# ===================================================================

def _k_grids():
    """Emulator k grid, its low-k extension and their union (all numpy).

    The extension continues the emulator's log-k step from ``kmin`` down to
    ``extrap_kmin``, so the union stays uniformly log-spaced, as FFTLog
    requires.
    """
    cfg = PK_GRID
    k_emu = np.geomspace(cfg['kmin'], cfg['kmax'], cfg['nk'])[::cfg['ndspl']]
    step = (np.log10(cfg['kmax']) - np.log10(cfg['kmin'])) / (cfg['nk'] - 1) * cfg['ndspl']
    n_extra = int(round((np.log10(cfg['kmin']) - np.log10(cfg['extrap_kmin'])) / step))
    k_extra = 10.0 ** (np.log10(cfg['kmin']) - np.arange(n_extra, 0, -1) * step)
    return k_emu, k_extra, np.concatenate([k_extra, k_emu])


K_EMU, K_EXTRA, K_GRID = _k_grids()


@lru_cache(maxsize=None)
def _tophat_var() -> FFTLog:
    """σ²(R) = ∫ dk/(2π²) k² P(k) W²(kR) on ``K_GRID``: pure-JAX FFTLog in
    float64, set up once (mcfit's algorithm and defaults: q = 1.5, low
    ringing, N = next power of two ≥ 2 n_k)."""
    with jax.ensure_compile_time_eval():
        return FFTLog(K_GRID, Mellin_TophatSq_dim3, q=1.5,
                      prefac=K_GRID ** 3 / (2.0 * np.pi ** 2), postfac=1.0,
                      lowring=True, dtype=np.float64)


# ===================================================================
# Emulators
# ===================================================================

def emulators(names=("pkl", "hz", "daz", "der")) -> dict:
    """The named emulators, loaded eagerly (cached). Each ``Emulator`` is a
    pytree, so the dict can be passed as an argument to jitted functions."""
    return {n: get_emulator(n) for n in names}


def _emu(emus, name):
    if emus is not None and name in emus:
        return emus[name]
    return get_emulator(name)


# ===================================================================
# Public builder
# ===================================================================

def build(params: dict,
          z_grid: jax.Array | None = None,
          n_z: int = 100,
          emus: dict | None = None) -> CosmoGrids:
    """Call the emulators once and return a ``CosmoGrids`` container.

    Pure JAX: ``jax.jit(jax.vmap(build))`` works, with ``params`` values as
    (batched) scalars. ``emus`` (from :func:`emulators`) may be passed in, so
    that the weights are jit arguments rather than baked-in constants; by
    default the cached emulators are used.
    """
    full = dict(DEFAULT_COSMO)
    full.update(params)

    if z_grid is None:
        z_grid = jnp.geomspace(5e-3, 5.0, n_z)
    z_grid = jnp.asarray(z_grid, dtype=jnp.float64)
    z_grid = jnp.where(z_grid < 5e-3, 5e-3, z_grid)

    k, pk = _predict_pk(full, z_grid, kind='linear', emus=emus)
    Hz, chi, Da = _predict_distances(full, z_grid, emus=emus)
    R, sigma, dsigma2dR = _compute_sigma(pk)

    return CosmoGrids(
        z=z_grid, k=k, pk=pk,
        Hz=Hz, chi=chi, Da=Da,
        R=R, sigma=sigma, dsigma2dR=dsigma2dR,
    )


# ===================================================================
# Pk emulator (linear / nonlinear) — with low-k extrapolation
# ===================================================================

def _predict_pk(full: dict, z_grid: jax.Array, kind: str = 'linear', emus=None):
    """Linear or nonlinear matter Pk. k in 1/Mpc, P in Mpc³.

    Returns ``(k, pk)`` with shapes ``(n_k,)`` and ``(n_z, n_k)``.

    Extends ``k`` below the emulator's native ``kmin = 5e-4`` down to
    ``extrap_kmin = 1e-4`` via ``P(k) ∝ k^n_s`` (valid for k ≪ k_eq), on the
    emulator's log-k step (see :func:`_k_grids`).
    """
    em = _emu(emus, 'pkl' if kind == 'linear' else 'pknl')
    z_grid = jnp.atleast_1d(jnp.asarray(z_grid, dtype=jnp.float64))

    inputs = {kn: full[kn] for kn in em.parameters if kn != 'z_pk_save_nonclass'}
    inputs['z_pk_save_nonclass'] = z_grid
    log10pk = em.predict(inputs)                                # (n_z, n_k_emu)
    if kind != 'linear':
        # P_nl = P_lin above PKNL_ZMAX
        log10pk_lin = _emu(emus, 'pkl').predict(inputs)
        log10pk = jnp.where((z_grid > PKNL_ZMAX)[:, None], log10pk_lin, log10pk)
    pk_emu = 10.0 ** log10pk * K_EMU ** -3                      # 'k3' convention

    pk_extra = pk_emu[:, 0:1] * (K_EXTRA / PK_GRID['kmin']) ** full['n_s']
    return jnp.asarray(K_GRID), jnp.concatenate([pk_extra, pk_emu], axis=1)


# ===================================================================
# Distances (Hz, chi, Da)
# ===================================================================

def _predict_distances(full: dict, z_grid: jax.Array, emus=None):
    """Hz/c [1/Mpc], chi [Mpc], Da [Mpc] interpolated to z_grid.

    Both Hz and Da emulators trained on z ∈ [0, 20], 5000 points; the
    Da emulator omits z=0 (returns 4999 values) — we prepend chi(0)=0
    to recover the proper z-grid.
    """
    h_em = _emu(emus, 'hz')
    d_em = _emu(emus, 'daz')
    Hz_fine = h_em.predict({kn: full[kn] for kn in h_em.parameters})[0]
    Da_fine = d_em.predict({kn: full[kn] for kn in d_em.parameters})[0]

    # Apply 10^ for log-output emulators (both Hz and Da are log10)
    if output_is_log10('hz'):
        Hz_fine = 10.0 ** Hz_fine
    if output_is_log10('daz'):
        Da_fine = 10.0 ** Da_fine

    # the DAZ emulator omits z=0; prepend chi(0)=0
    Da_fine = jnp.concatenate([jnp.zeros(1, dtype=Da_fine.dtype), Da_fine])

    z_fine = np.linspace(0.0, DIST_ZMAX, Hz_fine.shape[-1])

    chi_fine = Da_fine * (1.0 + z_fine)

    Hz  = jnp.interp(z_grid, z_fine, Hz_fine)
    chi = jnp.interp(z_grid, z_fine, chi_fine)
    chi = jnp.where(chi < 1.0, 1.0, chi)
    Da  = jnp.interp(z_grid, z_fine, Da_fine)
    return Hz, chi, Da


# ===================================================================
# σ(R, z) via the pure-JAX FFTLog TophatVar
# ===================================================================

def _compute_sigma(pk: jax.Array):
    """σ(R, z) and d(σ²)/dR on the FFTLog R grid, for P(k) on ``K_GRID``."""
    R, var = _tophat_var()(pk)
    R = jnp.asarray(R, dtype=jnp.float64)
    sigma = jnp.sqrt(var)
    log_R = jnp.log(R)
    dlogvar_dlogR = jnp.gradient(jnp.log(var), log_R, axis=-1)
    dsigma2dR = var * dlogvar_dlogR / R[None, :]
    return R, sigma, dsigma2dR


# ===================================================================
# Parameter maps (traceable)
# ===================================================================

def omega_nu(params: dict):
    """ω_ν = Σmν / 93.14 eV, with Σmν = 3 m_ncdm (the emulators' convention)."""
    m = params.get('m_ncdm', DEFAULT_COSMO['m_ncdm'])
    return N_NCDM_DEG * m / EV_PER_OMEGA_NU_H2


def Omega_m0(params: dict, include_nu: bool = True):
    """Ω_m today: (ω_b + ω_cdm [+ ω_ν]) / h²."""
    full = dict(DEFAULT_COSMO)
    full.update(params)
    om = full['omega_b'] + full['omega_cdm']
    if include_nu:
        om = om + omega_nu(full)
    return om / (full['H0'] / 100.0) ** 2


def omega_cdm_from_Omega_m(Omega_m, params: dict, include_nu: bool = True):
    """Inverse of :func:`Omega_m0`: ω_cdm for a given Ω_m, H0, ω_b, m_ncdm."""
    full = dict(DEFAULT_COSMO)
    full.update(params)
    om = Omega_m * (full['H0'] / 100.0) ** 2 - full['omega_b']
    if include_nu:
        om = om - omega_nu(full)
    return om


def sigma8_der(params: dict, emus: dict | None = None):
    """σ8 from the DER emulator (output index 1), as :func:`classy_szlite.derived`."""
    em = _emu(emus, 'der')
    full = dict(DEFAULT_COSMO)
    full.update(params)
    out = em.predict({kn: full[kn] for kn in em.parameters})[0]
    if output_is_log10('der'):
        out = 10.0 ** out
    return out[1]


def ln10As_from_sigma8(params: dict, sigma8, emus: dict | None = None,
                       n_newton: int = 8):
    """ln(10¹⁰ A_s) giving the DER-emulator σ8 = ``sigma8`` at fixed other
    parameters. Traceable and vmap-able.

    First step: A_s ∝ σ8², from the A_s in ``params`` (default 3.047). Then
    ``n_newton`` Newton steps with the exact (autodiff) derivative through the
    DER network, as in cosmocnc_jax's solver (8 fixed iterations).
    """
    full = dict(DEFAULT_COSMO)
    full.update(params)
    key = 'ln10^{10}A_s'

    def s8(x):
        return sigma8_der(dict(full, **{key: x}), emus)

    x0 = jnp.asarray(full[key], dtype=jnp.float64)
    x = x0 + 2.0 * jnp.log(sigma8 / s8(x0))

    def body(_, x):
        v, g = jax.value_and_grad(s8)(x)
        return x - (v - sigma8) / g

    return jax.lax.fori_loop(0, n_newton, body, x)


# ===================================================================
# Quick accessors
# ===================================================================

def get_pk(params, z_arr):
    """Linear P(k, z). k in 1/Mpc, P in Mpc³."""
    full = dict(DEFAULT_COSMO); full.update(params)
    return _predict_pk(full, jnp.asarray(z_arr), 'linear')


def get_pknl(params, z_arr):
    """Nonlinear P(k, z) (HMcode). k in 1/Mpc, P in Mpc³."""
    full = dict(DEFAULT_COSMO); full.update(params)
    return _predict_pk(full, jnp.asarray(z_arr), 'nonlinear')


def get_distances(params, z_arr):
    """Hz [1/Mpc], chi [Mpc], Da [Mpc]."""
    full = dict(DEFAULT_COSMO); full.update(params)
    return _predict_distances(full, jnp.atleast_1d(jnp.asarray(z_arr, dtype=jnp.float64)))
