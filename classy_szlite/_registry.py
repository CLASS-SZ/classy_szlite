"""Per-cosmo_model emulator file IDs, grid configs, and lazy loading.

Keeps everything CosmoPower-version-specific in one place so the rest of
the package only knows about logical names ('pkl', 'tt', etc.) and cosmo
models ('lcdm', 'ede-v2').
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

from ._emulator import Emulator, default_data_dir


# Currently supported cosmo_model labels and the file IDs to load.
# (Mirrors classy_szfast/emulators_meta_data.py for the subset we ship.)
_EMULATOR_FILES = {
    "lcdm": {
        "tt":   "TTTEEE/TT_v1.npz",
        "te":   "TTTEEE/TE_v1.npz",
        "ee":   "TTTEEE/EE_v1.npz",
        "pp":   "PP/PP_v1.npz",
        "pkl":  "PK/PKL_v1.npz",
        "pknl": "PK/PKNL_v1.npz",
        "hz":   "growth-and-distances/HZ_v1.npz",
        "daz":  "growth-and-distances/DAZ_v1.npz",
        "s8z":  "growth-and-distances/S8Z_v1.npz",
        "der":  "derived-parameters/DER_v1.npz",
    },
    "ede-v2": {
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
    },
}

# Subdirectory under the data root for each cosmo_model
_COSMO_MODEL_SUBDIRS = {
    "lcdm":   "lcdm",
    "ede-v2": "ede",        # historical: ede-v2 lives under ede/
}


# ---------------------------------------------------------------------------
# Pk-emulator grid + low-k extrapolation configuration
# ---------------------------------------------------------------------------
# Mirrors classy_szfast/classy_szfast.py:99-118, 206-214, 267-275.
# Final n_k = nk // ndspl. Final k = geomspace(kmin, kmax, nk)[::ndspl].
# prefac:
#   'ell' — emulator output is log10[ ell(ell+1) Pk / (2π) ] with labels
#           ls = arange(2, nk+2)[::ndspl]; recover Pk via × 1/(ell(ell+1)/(2π))
#   'k3'  — emulator output is log10[ k³ Pk ]; recover Pk via × k^-3
# extrap_kmin (optional): extend the Pk grid down to this k via P(k) ∝ k^n_s
PK_GRID_CONFIG = {
    "ede-v2": dict(kmin=5e-4, kmax=10.0, nk=1000, ndspl=1,
                   prefac="k3",  extrap_kmin=1e-4),
    "lcdm":   dict(kmin=1e-4, kmax=50.0, nk=5000, ndspl=10,
                   prefac="ell", extrap_kmin=None),
}

# Distance-emulator z-grid: both ede-v2 and lcdm emulators were trained on
# linspace(0, 20, 5000); ede-v2 DAZ omits z=0 (returns 4999 values) so we
# prepend a chi(0)=0 anchor. See classy_szfast.py:1110.
DIST_ZMAX = 20.0
DIST_NZ   = 5000

# Cosmologies whose DAZ emulator needs the z=0 prepend
_DAZ_NEEDS_Z0_PREPEND = {"ede-v2"}

# Whether each emulator's raw NN output is log10(physical) vs physical-directly.
# True  → caller must apply 10^ to recover the physical quantity.
# False → raw output IS the physical quantity (or the prefactored-Pk form).
# Empirically determined per (cosmo_model, name). Pkl/Pknl always log10 and
# handled separately in cosmology._predict_pk (with the prefactor).
LOG10_OUTPUT = {
    "lcdm": {
        "tt": True, "ee": True, "pp": True, "te": False,    # CMB
        "pkl": True, "pknl": True,                          # Pk (handled in cosmology)
        "hz": True, "daz": False, "s8z": False,             # distances + sigma8(z)
        "der": True,                                        # derived (sigma_8 = der[1])
    },
    "ede-v2": {
        "tt": True, "ee": True, "pp": True, "te": False,
        "pkl": True, "pknl": True,
        "hz": True, "daz": True, "s8z": True,
        "der": True,
    },
}


# ---------------------------------------------------------------------------
# Lazy emulator loading + cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_emulator(cosmo_model: str, name: str) -> Emulator:
    """Load (and cache) the emulator for ``(cosmo_model, name)``.

    ``name``: one of ``'tt','te','ee','pp','pkl','pknl','hz','daz','s8z','der'``.
    """
    if cosmo_model not in _EMULATOR_FILES:
        raise ValueError(f"Unknown cosmo_model {cosmo_model!r}. "
                         f"Supported: {list(_EMULATOR_FILES)}")
    if name not in _EMULATOR_FILES[cosmo_model]:
        raise ValueError(f"Unknown emulator name {name!r}. "
                         f"Supported: {list(_EMULATOR_FILES[cosmo_model])}")
    root = default_data_dir()
    subdir = _COSMO_MODEL_SUBDIRS[cosmo_model]
    path = os.path.join(root, subdir, _EMULATOR_FILES[cosmo_model][name])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"classy_szlite: emulator file not found: {path}")
    return Emulator.from_npz(path)


def daz_needs_z0_prepend(cosmo_model: str) -> bool:
    return cosmo_model in _DAZ_NEEDS_Z0_PREPEND


def output_is_log10(cosmo_model: str, name: str) -> bool:
    """Whether the raw NN output is log10(physical) for this emulator."""
    return LOG10_OUTPUT[cosmo_model][name]


SUPPORTED_COSMO_MODELS: tuple[str, ...] = ("ede-v2", "lcdm")
DEFAULT_COSMO_MODEL: str = "ede-v2"


# ---------------------------------------------------------------------------
# Default cosmology parameters per model
# ---------------------------------------------------------------------------
# Used for emulator-default fills when the user doesn't pass a parameter
# (matches emulator_dict[cosmo_model]['default'] in classy_szfast).
DEFAULT_COSMO = {
    "lcdm": {
        "ln10^{10}A_s": 3.047, "n_s": 0.9665,
        "H0": 67.66, "omega_b": 0.02242, "omega_cdm": 0.11933,
        "tau_reio": 0.054, "m_ncdm": 0.06, "N_ur": 2.0328,
    },
    "ede-v2": {
        # base cosmology (matches lcdm at fEDE=0 if ν-convention matched)
        "ln10^{10}A_s": 3.047, "n_s": 0.9665,
        "H0": 67.66, "omega_b": 0.02242, "omega_cdm": 0.11933,
        "tau_reio": 0.054,
        # EDE-specific (fEDE=0.001 = effectively LCDM)
        "fEDE": 0.001, "log10z_c": 3.562, "thetai_scf": 2.83, "r": 0.0,
        # ν convention: 3 degenerate ncdm of 0.02 eV → Σmν = 0.06 eV
        "m_ncdm": 0.02, "N_ur": 0.00441,
    },
}
