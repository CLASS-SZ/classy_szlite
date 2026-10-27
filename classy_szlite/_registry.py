"""Emulator file IDs, grid configs, and lazy loading.

Emulator suites **ede-v3** (default) and **ede-v2**, both trained on the same
CLASS-EDE spectra and covering the LCDM-equivalent point at ``fEDE = 0.001``.
ede-v3 (October 2026) retrains every network on the full training set; its TE
emulator predicts ``atanh(r_l)`` with ``r_l = C_l^TE / sqrt(C_l^TT C_l^EE)``.
Select the suite with :func:`set_emulator_version` or the environment variable
``CLASSY_SZLITE_EMULATORS`` (``v3`` or ``v2``). If the v3 files are not present
and no version was requested, ede-v2 is used.
"""
from __future__ import annotations

import os
from functools import lru_cache

from ._emulator import Emulator, default_data_dir


# File IDs (paths relative to <data_root>/ede/).
# Pickle-free files from cosmopower-organization/ede@main.
_SUITES = {
    "v2": {
        "tt":   "TTTEEE/TT_v2_plain.npz",
        "te":   "TTTEEE/TE_v2_plain.npz",
        "ee":   "TTTEEE/EE_v2_plain.npz",
        "pp":   "PP/PP_v2_plain.npz",
        "pkl":  "PK/PKL_v2_plain.npz",
        "pknl": "PK/PKNL_v2_plain.npz",
        "hz":   "growth-and-distances/HZ_v2_plain.npz",
        "daz":  "growth-and-distances/DAZ_v2_plain.npz",
        "s8z":  "growth-and-distances/S8Z_v2_plain.npz",
        "der":  "derived-parameters/DER_v2_plain.npz",
    },
    "v3": {
        "tt":   "TTTEEE/TT_v3_plain.npz",
        "te":   "TTTEEE/rTE_v3_plain.npz",
        "ee":   "TTTEEE/EE_v3_plain.npz",
        "pp":   "PP/PP_v3_plain.npz",
        "pkl":  "PK/PKL_v3_plain.npz",
        "pknl": "PK/PKNL_v3_plain.npz",
        "hz":   "growth-and-distances/HZ_v3_plain.npz",
        "daz":  "growth-and-distances/DAZ_v3_plain.npz",
        "s8z":  "growth-and-distances/S8Z_v3_plain.npz",
        "der":  "derived-parameters/DER_v3_plain.npz",
    },
}
# suites whose TE emulator predicts atanh(r_l)
_TE_FROM_R = {"v3"}

# every file of every suite (used by the data fetcher)
_EMULATOR_FILES = {f"{name}@{v}": rel for v, files in _SUITES.items() for name, rel in files.items()}

_version_requested = os.environ.get("CLASSY_SZLITE_EMULATORS")
_version = _version_requested or "v3"


def set_emulator_version(version: str) -> None:
    """Select the emulator suite (``"v3"`` or ``"v2"``) for subsequent calls."""
    global _version, _version_requested
    if version not in _SUITES:
        raise ValueError(f"Unknown emulator version {version!r}; expected one of {list(_SUITES)}")
    _version = _version_requested = version
    get_emulator.cache_clear()


def emulator_version() -> str:
    """The emulator suite in use (``"v3"`` falls back to ``"v2"`` if its files are missing)."""
    global _version
    if _version_requested is None and _version == "v3":
        root = default_data_dir()
        if not all(os.path.isfile(os.path.join(root, _DATA_SUBDIR, rel)) for rel in _SUITES["v3"].values()):
            import warnings
            warnings.warn("classy_szlite: ede-v3 emulator files not found, using ede-v2. "
                          "Run classy-szlite-get-data to download ede-v3.", stacklevel=2)
            _version = "v2"
    return _version


def te_from_r() -> bool:
    """True if the TE emulator of the active suite predicts atanh(r_l)."""
    return emulator_version() in _TE_FROM_R


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

# Redshift above which the nonlinear Pk is set to the linear one: CLASS-EDE
# does not apply HMcode beyond z ~ 5-10 (cosmology dependent) and the PKNL
# emulator is trained on HMcode spectra only (same convention as classy_szfast).
PKNL_ZMAX = 5.0

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

@lru_cache(maxsize=16)
def get_emulator(name: str) -> Emulator:
    """Load (and cache) the named emulator."""
    files = _SUITES[emulator_version()]
    if name not in files:
        raise ValueError(f"Unknown emulator name {name!r}. "
                         f"Supported: {list(files)}")
    rel  = files[name]
    root = default_data_dir()
    path = os.path.join(root, _DATA_SUBDIR, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"classy_szlite: emulator file not found: {path}\n"
            f"Get the emulator files with classy-szlite-get-data, or from "
            f"https://github.com/cosmopower-organization/ede"
        )
    return Emulator.from_npz(path)


def output_is_log10(name: str) -> bool:
    return LOG10_OUTPUT[name]
