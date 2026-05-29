"""Cached data-table loader for :mod:`classy_szlite.likelihoods`.

Everything heavy (cobaya / sacc / fgspectra) is intentionally kept *out* of
this module so that ``import classy_szlite.likelihoods`` works on a JAX-only
machine, as long as someone has already run
``python -m classy_szlite.likelihoods.extract_data`` on a machine with
``cobaya`` + ``act_dr6_mflike`` installed and copied the resulting npz over.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import jax.numpy as jnp


_DATA_ENV = "CLASSY_SZLITE_LIKELIHOOD_DATA"


def _candidate_paths() -> list[Path]:
    """Return the list of paths checked, in order, for the data npz."""
    candidates: list[Path] = []
    if os.environ.get(_DATA_ENV):
        candidates.append(Path(os.environ[_DATA_ENV]).expanduser())
    candidates.append(Path.home() / ".classy_szlite" / "likelihood_data.npz")
    # Convenient location when both bundles live alongside cosmopower emulator
    # data:
    candidates.append(Path.home() / "class_sz_data" / "likelihood_data.npz")
    return candidates


def _find_data_file() -> Path:
    for p in _candidate_paths():
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find the JAX-likelihood data npz. Tried:\n"
        + "\n".join(f"  {p}" for p in _candidate_paths())
        + f"\n\nSet the {_DATA_ENV} environment variable, or run\n"
        f"    python -m classy_szlite.likelihoods.extract_data\n"
        f"on a machine where cobaya + act_dr6_mflike are installed."
    )


def _native(a: np.ndarray) -> np.ndarray:
    """Return ``a`` in native byte order, contiguous, float64 where applicable."""
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder not in ("=", "|", "<") and a.dtype.kind in ("f", "i", "u"):
        a = a.astype(a.dtype.newbyteorder("="))
    if a.dtype.kind == "f" and a.dtype != np.float64:
        a = a.astype(np.float64)
    return a


@lru_cache(maxsize=1)
def load() -> dict[str, object]:
    """Load the cached likelihood-data npz as a dict of native-endian arrays.

    Cached per-process so repeated calls are free. Scalar arrays are
    flattened to Python scalars so callers can do ``int(d["lowTT_lmin"])``.
    """
    path = _find_data_file()
    with np.load(path, allow_pickle=True) as f:
        out = {}
        for k in f.files:
            v = np.asarray(f[k])
            if v.shape == ():            # 0-d array → scalar
                out[k] = v.item()
            elif v.dtype.kind in ("U", "S", "O"):
                out[k] = v
            else:
                out[k] = _native(v)
    return out


# Optional second npz: per-component foreground templates for chi2_mflike_v2.
@lru_cache(maxsize=1)
def load_fg_components() -> dict[str, object] | None:
    """Load the per-component foreground templates if present, else ``None``."""
    for p in _candidate_paths():
        candidate = p.parent / "mflike_fg_components.npz"
        if candidate.is_file():
            with np.load(candidate, allow_pickle=True) as f:
                out = {}
                for k in f.files:
                    v = np.asarray(f[k])
                    if v.shape == ():
                        out[k] = v.item()
                    elif v.dtype.kind in ("U", "S", "O"):
                        out[k] = v
                    else:
                        out[k] = _native(v)
                return out
    warnings.warn(
        "mflike_fg_components.npz not found; chi2_mflike_v2 will be "
        "unavailable. Run classy_szlite.likelihoods.extract_data to produce it.",
        RuntimeWarning, stacklevel=2,
    )
    return None


def jnp_table(arr: np.ndarray) -> jnp.ndarray:
    """Convert a numpy array to a float64 jnp array, contiguous + native-endian."""
    return jnp.asarray(_native(arr), dtype=jnp.float64)
