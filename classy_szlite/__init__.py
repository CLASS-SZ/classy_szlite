"""classy_szlite — pure-JAX subset of classy_szfast.

LCDM + EDE-v2 CosmoPower emulators for:
  * CMB Cls (TT, TE, EE)
  * Linear / non-linear matter Pk
  * Distances (Hz, chi, Da)
  * Derived params (σ8, Ω_m, S8)
  * Halo-model tSZ Cl^yy (Arnaud 10 GNFW)

No keras / tensorflow / scipy / class_sz dependencies — just JAX, numpy,
and mcfit (for the σ(R) TophatVar). EDE-v2 is the default cosmo_model.

Quick start:

>>> import classy_szlite as csl
>>> cosmo = csl.CosmoParams()                       # defaults (Planck-like + ede-v2 ν conv.)
>>> d = csl.derived(cosmo)                          # {'sigma_8', 'Omega_m', 'S8', ...}
>>> ell, cls = csl.cl_TTTEEE(cosmo)                 # dict with 'tt','te','ee','ell'
>>> k, pk = csl.Pk(cosmo, [0., 0.5, 1., 2.])        # linear P(k, z)
>>> Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])
>>> profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
>>> cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell=jnp.geomspace(2, 9000, 80))
"""
__version__ = "0.1.0"

# Public API
from .params import CosmoParams, ProfileParamsA10
from .api import (
    derived,
    cl_TTTEEE,
    Pk, Pnl,
    distances,
    cl_yy,
    cl_yy_factory,
    cosmo_to_dict,
)
from ._registry import (
    SUPPORTED_COSMO_MODELS,
    DEFAULT_COSMO_MODEL,
)

__all__ = [
    "__version__",
    # params
    "CosmoParams", "ProfileParamsA10",
    # observables
    "derived", "cl_TTTEEE", "Pk", "Pnl", "distances", "cl_yy", "cl_yy_factory",
    # utility
    "cosmo_to_dict",
    "SUPPORTED_COSMO_MODELS", "DEFAULT_COSMO_MODEL",
]
