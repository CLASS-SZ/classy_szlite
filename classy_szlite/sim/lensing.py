"""Stacked CMB-lensing observable (catalogue level, the paper's model; PLAN §3.2).

For each detection in the data vector, from its matched halo (M500, z):

    p̄ = a_lens κ(r_s; β M500, z) / σ_κ(θ500(β M500, z)),
    ln p_true = ln p̄ + σ_lnp N(0,1),        p_obs = p_true + N(0,1),

with β = β_CMBlens (0.92), σ_lnp = σ_CMBlens (0.22), a_lens = 1. κ(r_s) is the
convergence at the scale radius of an NFW halo with c500 = 3 truncated in 3D at
5 R500 (Zubeldia & Challinor 2019, Eqs. 18-20):

    κ(r_s) = 4π (G/c²) Σ(r_s) D_l (1 − χ_l/χ*),   Σ(r_s) = S_RS ρ0 r_s,

and σ_κ(θ500) is the matched-filter noise of that amplitude (lens_tables.py,
SO baseline or goal). The summary is the mean of p_obs over the data-vector
sample: each tile returns the sum of p_obs and the number of detections.
Traceable (jnp).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..sz_scaling import theta500_arcmin
from . import lens_tables

C500 = 3.0
X_TRUNC = 5.0 * C500
S_RS = 2.0 * (2.0 - 3.0 * X_TRUNC + X_TRUNC ** 3) / (3.0 * (X_TRUNC ** 2 - 1.0) ** 1.5)   # 0.66259
M_C = np.log(1.0 + C500) - C500 / (1.0 + C500)        # NFW mass inside c500 r_s, units of 4π ρ0 r_s³
G_OVER_C2 = 4.78543e-20                                # Mpc / M_sun
C_KMS = 299792.458
Z_STAR = 1089.9                                        # last scattering
OMEGA_R_H2 = 4.15e-5                                   # photons + relativistic neutrinos
OMEGA_NU_H2 = 3 * 0.02 / 93.14                         # the simulator's three 0.02 eV neutrinos

_U, _WU = np.polynomial.legendre.leggauss(256)


def chi_star(h, omega_cdm, omega_b):
    """Comoving distance to last scattering [Mpc], flat ΛCDM with radiation
    (Gauss–Legendre in ln(1+z); ~0.1% accuracy is ample: κ depends on it only
    through 1 − χ_l/χ*)."""
    om = (omega_cdm + omega_b + OMEGA_NU_H2) / h ** 2
    orad = OMEGA_R_H2 / h ** 2
    a, b = 0.0, np.log(1.0 + Z_STAR)
    u = 0.5 * (b - a) * jnp.asarray(_U) + 0.5 * (b + a)
    zp1 = jnp.exp(u)
    E = jnp.sqrt(om * zp1 ** 3 + orad * zp1 ** 4 + (1.0 - om - orad))
    return C_KMS / (100.0 * h) * 0.5 * (b - a) * jnp.sum(jnp.asarray(_WU) * zp1 / E)


def kappa_rs(M500, z, cg, h, beta, chi_s):
    """Convergence at the scale radius for lensing mass β M500 at redshift z."""
    Ml = beta * M500
    Hc = jnp.interp(z, cg.z, cg.Hz)                      # H(z)/c [1/Mpc]
    rho_c = 3.0 * Hc ** 2 / (8.0 * np.pi * G_OVER_C2)    # M_sun / Mpc³
    R500 = (3.0 * Ml / (4.0 * np.pi * 500.0 * rho_c)) ** (1.0 / 3.0)
    rs = R500 / C500
    rho0 = Ml / (4.0 * np.pi * rs ** 3 * M_C)
    DA = jnp.interp(z, cg.z, cg.Da)
    chi = jnp.interp(z, cg.z, cg.chi)
    return 4.0 * np.pi * G_OVER_C2 * S_RS * rho0 * rs * DA * (1.0 - chi / chi_s)


def theta500_lens(M500, z, cg, h, beta):
    Ez = jnp.interp(z, cg.z, cg.Hz) / (h * 100.0 / C_KMS)
    return theta500_arcmin(M500, Ez, jnp.interp(z, cg.z, cg.Da), h, beta)


def table(mode="goal"):
    """(ln θ500 [arcmin], ln σ_κ) arrays of the SO table for ``mode``."""
    return (np.log(lens_tables.THETA500_ARCMIN), np.log(lens_tables.SIGMA_KAPPA[mode]))


def sigma_kappa(theta500, lnth_tab, lnsig_tab):
    """σ_κ at θ500 [arcmin]: linear in ln–ln, held at the end values outside the table."""
    return jnp.exp(jnp.interp(jnp.log(theta500), lnth_tab, lnsig_tab))


def p_mean(M500, z, cg, h, beta, chi_s, lnth_tab, lnsig_tab, a_lens=1.0):
    """p̄ = a_lens κ(r_s) / σ_κ(θ500), both at the lensing mass β M500."""
    return a_lens * kappa_rs(M500, z, cg, h, beta, chi_s) / sigma_kappa(
        theta500_lens(M500, z, cg, h, beta), lnth_tab, lnsig_tab)


def draw_p(key, pbar, sigma_lnp):
    """p_obs: log-normal intrinsic scatter about p̄, then unit Gaussian noise."""
    k1, k2 = jax.random.split(key)
    lnp = jnp.log(pbar) + sigma_lnp * jax.random.normal(k1, pbar.shape, dtype=jnp.float64)
    return jnp.exp(lnp) + jax.random.normal(k2, pbar.shape, dtype=jnp.float64)
