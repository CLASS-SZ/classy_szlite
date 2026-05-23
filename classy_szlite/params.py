"""Parameter containers (JAX pytrees via NamedTuple).

A single ``CosmoParams`` covers both lcdm and ede-v2 — EDE-specific fields
(``fEDE``, ``log10z_c``, ``thetai_scf``, ``r``) default to the ede-v2
LCDM-equivalent point (``fEDE=0.001`` etc.). When ``cosmo_model='lcdm'``
those fields are silently ignored. Neutrino fields (``m_ncdm``, ``N_ur``)
default to the ede-v2 training convention (3 degenerate ν × 0.02 eV =
0.06 eV); pass ``m_ncdm=0.06, N_ur=2.0328`` for the lcdm convention.

Profile params (Arnaud 2010) include ``B`` = hydrostatic mass bias =
M_true / M_HSE. Default 1.0 (no bias).
"""
from __future__ import annotations

from typing import NamedTuple

import jax


class CosmoParams(NamedTuple):
    """Cosmological parameters covering lcdm + ede-v2.

    Defaults: ede-v2 emulator's "LCDM-equivalent" point (Planck 18 + tiny EDE).
    """
    # Standard six
    omega_b:      float | jax.Array = 0.02242
    omega_cdm:    float | jax.Array = 0.11933
    H0:           float | jax.Array = 67.66
    tau_reio:     float | jax.Array = 0.054
    ln10_10_As:   float | jax.Array = 3.047
    n_s:          float | jax.Array = 0.9665

    # EDE (ignored when cosmo_model='lcdm')
    fEDE:         float | jax.Array = 0.001
    log10z_c:     float | jax.Array = 3.562
    thetai_scf:   float | jax.Array = 2.83
    r:            float | jax.Array = 0.0

    # Neutrinos. Default = ede-v2 convention (3 deg. ν of 0.02 eV).
    # For lcdm convention pass m_ncdm=0.06, N_ur=2.0328.
    m_ncdm:       float | jax.Array = 0.02
    N_ur:         float | jax.Array = 0.00441

    def for_lcdm(self, m_ncdm: float = 0.06, N_ur: float = 2.0328) -> "CosmoParams":
        """Return a copy with ν fields set to the lcdm convention."""
        return self._replace(m_ncdm=m_ncdm, N_ur=N_ur)


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
