"""Physical constants and conventions shared across classy_szlite.

SI units unless stated. The values are the ones classy_szlite already used
(G and M_sun as in CLASS), moved here from ``hmf.py`` and
``power_spectrum.py`` so that every module shares one copy.
"""
from __future__ import annotations

import math

# Linear collapse threshold, (3/20)(12π)^{2/3}.
DELTA_C = 1.686470199841145

G_SI = 6.67428e-11              # m³ kg⁻¹ s⁻²  (CLASS value)
C_SI = 2.99792458e8             # m s⁻¹
MSUN_KG = 1.98855e30            # kg  (CLASS value)
MPC_M = 3.085677581282e22       # m / Mpc
MPC_CM = 3.085677581282e24      # cm / Mpc
SIGMA_T_M2 = 6.6524587321e-29   # m²  (Thomson cross-section)
ME_C2_EV = 0.510998946e6        # eV  (electron rest energy)
EV_J = 1.602176487e-19          # J / eV

# Composite constants, combined in Python (true float64) so that no JAX
# intermediate leaves the float32 range on backends that cap it.
THREE_OVER_8PI_G = 3.0 / (8.0 * math.pi * G_SI)   # kg s² m⁻³
MPC3_OVER_MSUN = MPC_M ** 3 / MSUN_KG               # m³ M_sun⁻¹ per Mpc³

# Neutrino convention of the ede-v3 emulators: one massive species with
# degeneracy 3 (CLASS ``deg_ncdm = 3``), each of mass ``m_ncdm`` eV, so
# Σmν = 3 m_ncdm and ω_ν = Σmν / 93.14 eV.
N_NCDM_DEG = 3
EV_PER_OMEGA_NU_H2 = 93.14
