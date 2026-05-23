# classy_szlite

**Fast, differentiable cosmology in pure JAX.**

A standalone Python package providing JIT-compiled, `jax.grad`-friendly
access to:

- CMB angular power spectra (TT, TE, EE)
- Linear and non-linear matter power spectrum
- Distances (H(z), comoving, angular-diameter)
- Derived parameters (σ8, Ω_m, S8)
- Halo-model tSZ Cl^yy (Arnaud 2010 GNFW pressure profile)

Backed by the high-accuracy **ede-v2 CosmoPower emulators** (Bolliet et al.).
LCDM-equivalent cosmology is the default — `fEDE = 0.001` (the package's
default) reproduces standard ΛCDM to emulator precision, and the same
emulator covers the full early-dark-energy parameter space when you want to
explore it.

Runtime dependencies: just `jax`, `numpy`, and `mcfit`.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api
throughput
gradients
examples
```

## Quick example

```python
import jax.numpy as jnp
import classy_szlite as csl

cosmo = csl.CosmoParams()

# Derived
print(csl.derived(cosmo))            # → {'sigma_8': 0.812, 'Omega_m': 0.311, 'S8': 0.827, ...}

# CMB
cls = csl.cl_TTTEEE(cosmo)           # → {'ell','tt','te','ee'}, dimensionless D_ℓ

# Matter Pk
k, pk = csl.Pk(cosmo, [0., 0.5, 1., 2.])

# Distances
Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])

# tSZ Cl^yy
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
ell = jnp.geomspace(2, 9000, 80)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)

# MCMC fast path: precompute cosmology + halo grids once, then ~5 ms/call
ev = csl.cl_yy_factory(cosmo, ell)
cl_1h, cl_2h = ev(profile)
```

## Repository

[github.com/CLASS-SZ/classy_szlite](https://github.com/CLASS-SZ/classy_szlite)
