"""classy_szlite — fast, differentiable cosmology in pure JAX.

A standalone Python package providing JIT-compiled, ``jax.grad``-friendly
access to:

  * CMB angular power spectra (TT, TE, EE)
  * Linear and non-linear matter Pk
  * Distances (H(z), chi, Da)
  * Derived parameters (σ8, Ω_m, S8)
  * Halo-model tSZ Cl^yy (Arnaud 2010 GNFW)

Backed by the high-accuracy ede-v2 CosmoPower emulators (Bolliet et al.).
LCDM-equivalent cosmology is obtained by leaving ``fEDE = 0.001`` (the
default) — no separate emulator path needed.

Runtime deps: ``jax``, ``numpy``, ``mcfit``.

Quick start:

>>> import classy_szlite as csl
>>> cosmo = csl.CosmoParams()
>>> d = csl.derived(cosmo)
>>> ell_cls = csl.cl_TTTEEE(cosmo)
>>> k, pk = csl.Pk(cosmo, [0., 0.5, 1., 2.])
>>> Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])
>>> profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
>>> import jax.numpy as jnp
>>> cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell=jnp.geomspace(2, 9000, 80))
>>> ev = csl.cl_yy_factory(cosmo, ell=jnp.geomspace(2, 9000, 80))  # fast path
>>> cl_1h, cl_2h = ev(profile)
"""
__version__ = "0.2.12"

from .params import CosmoParams, ProfileParamsA10
from .api import (
    derived,
    cl_TTTEEE,
    cl_TTTEEE_jax,
    Pk, Pnl,
    distances,
    cl_yy,
    cl_yy_factory,
    cl_yy_trispectrum,
    cl_yy_covariance,
    cosmo_to_dict,
)

__all__ = [
    "__version__",
    "CosmoParams", "ProfileParamsA10",
    "derived", "cl_TTTEEE", "cl_TTTEEE_jax", "Pk", "Pnl", "distances",
    "cl_yy", "cl_yy_factory", "cl_yy_trispectrum", "cl_yy_covariance",
    "cosmo_to_dict",
]
