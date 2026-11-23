"""tSZ observable–mass relations for the map-level simulator (mapsim).

Production model: the paper's SO-like relation (Zubeldia, Bolliet, Challinor
& Handley 2025; cosmocnc_jax ``surveys/survey_sr_so_sim.py``), with the
**median** central Compton-y

    ȳ0 = 10^A_SZ E(z)² [h70 (1-b) M500 / 3e14 M_sun]^α_SZ h70^(-1/2),

and the angular scale

    θ500 = 6.997′ h70^(-2/3) [(1-b) M500 / 3e14 M_sun]^(1/3) E(z)^(-2/3) (500 Mpc / D_A).

Each halo draws y0 = ȳ0 exp(σ_lnq N(0,1)). Masses in M_sun (h-free), D_A in
Mpc, θ500 in arcmin. Traceable (jnp).
"""
from __future__ import annotations

import jax.numpy as jnp


def y0_median(M500, Ez, h, A_SZ, alpha_SZ, one_minus_b):
    """Median central y of the production relation (see module docstring)."""
    h70 = h / 0.7
    return (10.0 ** A_SZ * Ez ** 2 * (h70 * one_minus_b * M500 / 3e14) ** alpha_SZ
            / jnp.sqrt(h70))


def theta500_arcmin(M500, Ez, DA_Mpc, h, one_minus_b):
    """θ500 [arcmin] of the production relation (cosmocnc's 6.997′ formula)."""
    h70 = h / 0.7
    return (6.997 * h70 ** (-2.0 / 3.0) * (one_minus_b * M500 / 3e14) ** (1.0 / 3.0)
            * Ez ** (-2.0 / 3.0) * (500.0 / DA_Mpc))
