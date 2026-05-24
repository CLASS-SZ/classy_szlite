# classy_szlite

**Fast, differentiable cosmology in pure JAX.**

A standalone Python package providing JIT-compiled, `jax.grad`-friendly
access to:

- CMB angular power spectra (TT, TE, EE)
- Linear and non-linear matter power spectrum
- Distances (H(z), comoving, angular-diameter)
- Derived parameters (σ8, Ω_m, S8)
- Halo-model tSZ Cl^yy (Arnaud 2010 GNFW pressure profile)

Backed by the high-accuracy **`v2` CosmoPower emulators** — the same
emulators used in the
[ACT DR6 extended-cosmology analysis](https://arxiv.org/abs/2503.14454)
(2025) and the
[ACT DR6 + DESI DR2 analysis](https://arxiv.org/abs/2505.08051) by
Poulin et al. (2025). They match the CAMB-based
[Jense et al. (2024) emulators](https://github.com/cosmopower-organization/jense_2024_emulators)
to well under 0.1 $\sigma$ in $\Lambda$CDM (where $\sigma$ is the typical
68% CL on parameter constraints from ACT DR6).

Runtime dependencies: just `jax`, `numpy`, and `mcfit`. See
[Installation](installation.md) for the emulator-coverage details
(LCDM, $m_\nu$-LCDM, $w$CDM, $N_{\rm eff}$-LCDM, EDE).

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api
throughput
gradients
convergence
examples
```

```{note}
**Inference recipes shipped as runnable scripts**:
[`examples/nuts_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/nuts_clyy_profile.py)
(NumPyro NUTS + cobaya RW-MH overlay),
[`examples/profile_bands.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/profile_bands.py)
(GNFW profile posterior bands),
[`examples/fisher_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/fisher_clyy_profile.py)
(Fisher matrix via `jax.jacfwd`).
See [Tutorials & examples](examples.md) for full walkthroughs.
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
