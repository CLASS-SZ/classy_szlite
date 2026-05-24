"""Cosmological grids from the ede-v2 CosmoPower emulators (pure JAX).

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

import warnings as _warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._registry import (
    PK_GRID, DIST_ZMAX, DEFAULT_COSMO,
    get_emulator, output_is_log10,
)

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
# Public builder
# ===================================================================

def build(params: dict,
          z_grid: jax.Array | None = None,
          n_z: int = 100) -> CosmoGrids:
    """Call emulators once and return a ``CosmoGrids`` container."""
    full = dict(DEFAULT_COSMO)
    full.update(params)

    if z_grid is None:
        z_grid = jnp.geomspace(5e-3, 5.0, n_z)
    z_grid = jnp.asarray(z_grid)
    z_grid = jnp.where(z_grid < 5e-3, 5e-3, z_grid)
    n_z_val = int(z_grid.shape[0])

    k, pk = _predict_pk(full, z_grid, kind='linear')
    Hz, chi, Da = _predict_distances(full, z_grid)
    R, sigma, dsigma2dR = _compute_sigma(k, pk, n_z_val)

    return CosmoGrids(
        z=z_grid, k=k, pk=pk,
        Hz=Hz, chi=chi, Da=Da,
        R=R, sigma=sigma, dsigma2dR=dsigma2dR,
    )


# ===================================================================
# Pk emulator (linear / nonlinear) — with low-k extrapolation
# ===================================================================

def _predict_pk(full: dict, z_grid: jax.Array, kind: str = 'linear'):
    """Linear or nonlinear matter Pk. k in 1/Mpc, P in Mpc³.

    Returns ``(k, pk)`` with shapes ``(n_k,)`` and ``(n_z, n_k)``.

    Extends ``k`` below the emulator's native ``kmin = 5e-4`` down to
    ``extrap_kmin = 1e-4`` via ``P(k) ∝ k^n_s`` (valid for k ≪ k_eq).
    The extra points use the same log-k step as the emulator so the
    combined grid stays uniformly log-spaced (required by mcfit
    TophatVar σ(R)).
    """
    em = get_emulator('pkl' if kind == 'linear' else 'pknl')
    cfg = PK_GRID

    k_emu = jnp.geomspace(cfg['kmin'], cfg['kmax'], cfg['nk'])[::cfg['ndspl']]
    pk_power_fac = k_emu ** -3                                  # 'k3' convention

    n_z = int(z_grid.shape[0])
    params_pk = {kn: [v] * n_z for kn, v in full.items() if kn in em.parameters}
    params_pk['z_pk_save_nonclass'] = list(z_grid)

    log10pk = em.predict(params_pk)
    pk_emu = jnp.float64(10.0**log10pk * pk_power_fac)          # (n_z, n_k_emu)

    extrap_kmin = cfg['extrap_kmin']
    n_s = full['n_s']
    log10_step = (float(jnp.log10(cfg['kmax'])) - float(jnp.log10(cfg['kmin']))) / (cfg['nk'] - 1)
    log10_step_out = log10_step * cfg['ndspl']
    n_extra = int(round((float(jnp.log10(cfg['kmin'])) - float(jnp.log10(extrap_kmin)))
                        / log10_step_out))
    log10_k_extra = jnp.log10(cfg['kmin']) - jnp.arange(n_extra, 0, -1) * log10_step_out
    k_extra = jnp.power(10.0, log10_k_extra)
    pk_at_kmin = pk_emu[:, 0:1]
    pk_extra = pk_at_kmin * (k_extra[None, :] / cfg['kmin']) ** n_s
    k  = jnp.concatenate([k_extra, k_emu])
    pk = jnp.concatenate([pk_extra, pk_emu], axis=1)
    return k, pk


# ===================================================================
# Distances (Hz, chi, Da)
# ===================================================================

def _predict_distances(full: dict, z_grid: jax.Array):
    """Hz/c [1/Mpc], chi [Mpc], Da [Mpc] interpolated to z_grid.

    Both Hz and Da emulators trained on z ∈ [0, 20], 5000 points; the
    Da emulator omits z=0 (returns 4999 values) — we prepend chi(0)=0
    to recover the proper z-grid.
    """
    h_em = get_emulator('hz')
    d_em = get_emulator('daz')
    p_h = {kn: [full[kn]] for kn in h_em.parameters}
    p_d = {kn: [full[kn]] for kn in d_em.parameters}

    Hz_fine = h_em.predict(p_h)
    Da_fine = d_em.predict(p_d)

    # Apply 10^ for log-output emulators (both Hz and Da in ede-v2 are log10)
    if output_is_log10('hz'):
        Hz_fine = 10.0 ** Hz_fine
    if output_is_log10('daz'):
        Da_fine = 10.0 ** Da_fine

    # ede-v2 DAZ omits z=0; prepend chi(0)=0
    Da_fine = jnp.concatenate([jnp.zeros(1, dtype=Da_fine.dtype), Da_fine])

    z_fine = jnp.linspace(0.0, DIST_ZMAX, Hz_fine.shape[-1])

    chi_fine = Da_fine * (1.0 + z_fine)

    Hz  = jnp.interp(z_grid, z_fine, Hz_fine)
    chi = jnp.interp(z_grid, z_fine, chi_fine)
    chi = jnp.where(chi < 1.0, 1.0, chi)
    Da  = jnp.interp(z_grid, z_fine, Da_fine)
    return Hz, chi, Da


# ===================================================================
# σ(R, z) via mcfit TophatVar
# ===================================================================

def _compute_sigma(k: jax.Array, pk: jax.Array, n_z: int):
    """σ(R, z), d(σ²)/dR via pure-JAX FFTLog (TPU-safe)."""
    from ._fftlog import TophatVar
    tv = TophatVar(np.array(k, copy=False), lowring=True)
    R_all, var_all = tv(pk)
    R = jnp.asarray(R_all).flatten()
    sigma = jnp.sqrt(var_all)
    log_R = jnp.log(R)
    var = sigma ** 2
    log_var = jnp.log(var)
    dlogvar_dlogR = jnp.gradient(log_var, log_R, axis=-1)
    dsigma2dR = var * dlogvar_dlogR / R[None, :]
    return R, sigma, dsigma2dR


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
    return _predict_distances(full, jnp.asarray(z_arr))
