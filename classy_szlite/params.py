"""Parameter containers (JAX pytrees via NamedTuple).

A single :class:`CosmoParams` covers the full parameter space supported by
the ede-v2 emulator suite — the standard 6 cosmological parameters plus the
EDE-specific fields (``fEDE``, ``log10z_c``, ``thetai_scf``, ``r``) and the
neutrino fields (``m_ncdm``, ``N_ur``). LCDM-equivalent runs use the
default ``fEDE = 0.001``; users typically don't need to touch the EDE or
neutrino fields at all.

:class:`ProfileParamsA10` carries the Arnaud 2010 GNFW pressure-profile
parameters plus the hydrostatic mass bias ``B`` = M_true / M_HSE.
"""
from __future__ import annotations

from typing import NamedTuple

import jax


class CosmoParams(NamedTuple):
    """Cosmological parameters for the ede-v2 emulator suite.

    Defaults give the LCDM-equivalent point (Planck 18 + ``fEDE = 0.001``).
    """
    # Standard six
    omega_b:      float | jax.Array = 0.02242
    omega_cdm:    float | jax.Array = 0.11933
    H0:           float | jax.Array = 67.66
    tau_reio:     float | jax.Array = 0.054
    ln10_10_As:   float | jax.Array = 3.047
    n_s:          float | jax.Array = 0.9665

    # EDE (silently used; default = LCDM-equivalent point)
    fEDE:         float | jax.Array = 0.001
    log10z_c:     float | jax.Array = 3.562
    thetai_scf:   float | jax.Array = 2.83
    r:            float | jax.Array = 0.0

    # ν convention (matches ede-v2 emulator training: 3 deg. ν × 0.02 eV)
    m_ncdm:       float | jax.Array = 0.02
    N_ur:         float | jax.Array = 0.00441


class ProfileParamsA10(NamedTuple):
    """Arnaud 2010 gNFW pressure-profile parameters (for tSZ Cl^yy).

    ``B`` is the hydrostatic mass bias (``B = M_true / M_HSE``);
    the profile is evaluated at the effective M_HSE = M_true / B and
    r_500c_HSE = r_500c_true / B^(1/3).
    """
    P0:    float | jax.Array = 8.130
    c500:  float | jax.Array = 1.156
    gamma: float | jax.Array = 0.3292
    alpha: float | jax.Array = 1.0620
    beta:  float | jax.Array = 5.4807
    B:     float | jax.Array = 1.0
