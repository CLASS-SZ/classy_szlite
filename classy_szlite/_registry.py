"""Emulator file IDs, grid configs, and lazy loading.

Single emulator suite: **ede-v2**. The ede-v2 CosmoPower emulators are
extremely accurate and cover the LCDM-equivalent point at ``fEDE = 0.001``;
they replace any need for a separate LCDM emulator path.
"""
from __future__ import annotations

import os
from functools import lru_cache

from ._emulator import Emulator, default_data_dir


# File IDs (paths relative to <data_root>/ede/)
_EMULATOR_FILES = {
    "tt":   "TTTEEE/TT_v2.npz",
    "te":   "TTTEEE/TE_v2.npz",
    "ee":   "TTTEEE/EE_v2.npz",
    "pp":   "PP/PP_v2.npz",
    "pkl":  "PK/PKL_v2.npz",
    "pknl": "PK/PKNL_v2.npz",
    "hz":   "growth-and-distances/HZ_v2.npz",
    "daz":  "growth-and-distances/DAZ_v2.npz",
    "s8z":  "growth-and-distances/S8Z_v2.npz",
    "der":  "derived-parameters/DER_v2.npz",
}

# Subdir under the data root
_DATA_SUBDIR = "ede"


# ---------------------------------------------------------------------------
# Pk emulator grid + extrapolation configuration
# ---------------------------------------------------------------------------
# Final n_k = nk // ndspl. Final k = geomspace(kmin, kmax, nk)[::ndspl].
# prefac:
#   'k3' — emulator output is log10[ k³ Pk ]; recover Pk via × k^-3
# extrap_kmin: extend the Pk grid down to this k via P(k) ∝ k^n_s
PK_GRID = dict(kmin=5e-4, kmax=10.0, nk=1000, ndspl=1,
               prefac="k3", extrap_kmin=1e-4)

# Distance emulator z-grid: HZ trained on linspace(0, 20, 5000); DAZ omits
# z=0 (returns 4999 values) — we prepend chi(0)=0 in cosmology._predict_distances.
DIST_ZMAX = 20.0


# ---------------------------------------------------------------------------
# Per-emulator log10 convention
# ---------------------------------------------------------------------------
# Whether the raw NN output is log10(physical) vs physical directly.
# True  → caller must apply 10^ to recover the physical quantity.
# False → raw output IS the physical quantity (or the prefactored-Pk form).
LOG10_OUTPUT = {
    "tt": True, "ee": True, "pp": True, "te": False,    # CMB
    "pkl": True, "pknl": True,                          # Pk (handled in cosmology)
    "hz": True, "daz": True, "s8z": True,               # distances + sigma8(z)
    "der": True,                                        # derived (sigma_8 = der[1])
}


# ---------------------------------------------------------------------------
# Default cosmology parameters (matches ede-v2 emulator training default)
# ---------------------------------------------------------------------------
DEFAULT_COSMO = {
    "ln10^{10}A_s": 3.047, "n_s": 0.9665,
    "H0": 67.66, "omega_b": 0.02242, "omega_cdm": 0.11933,
    "tau_reio": 0.054,
    # EDE-specific defaults (LCDM-equivalent point)
    "fEDE": 0.001, "log10z_c": 3.562, "thetai_scf": 2.83, "r": 0.0,
    # ν convention: 3 degenerate ncdm of 0.02 eV → Σmν = 0.06 eV
    "m_ncdm": 0.02, "N_ur": 0.00441,
}


# ---------------------------------------------------------------------------
# Lazy emulator loading + cache
# ---------------------------------------------------------------------------

def _plain_variant(rel: str) -> str:
    """foo/TT_v2.npz -> foo/TT_v2_plain.npz."""
    return rel[:-len(".npz")] + "_plain.npz"


@lru_cache(maxsize=16)
def get_emulator(name: str) -> Emulator:
    """Load (and cache) the named emulator.

    Prefers the **pickle-free** ``_plain.npz`` variant if present, so users
    don't need tensorflow installed to deserialise the file.  Falls back to
    the original pickled ``v2.npz`` for backwards compatibility.
    """
    if name not in _EMULATOR_FILES:
        raise ValueError(f"Unknown emulator name {name!r}. "
                         f"Supported: {list(_EMULATOR_FILES)}")
    root = default_data_dir()
    base = os.path.join(root, _DATA_SUBDIR, _EMULATOR_FILES[name])
    plain = os.path.join(root, _DATA_SUBDIR, _plain_variant(_EMULATOR_FILES[name]))
    if os.path.isfile(plain):
        return Emulator.from_npz(plain)
    if os.path.isfile(base):
        return Emulator.from_npz(base)
    raise FileNotFoundError(
        f"classy_szlite: emulator file not found.  Looked for:\n  {plain}\n  {base}"
    )


def output_is_log10(name: str) -> bool:
    return LOG10_OUTPUT[name]
