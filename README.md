# classy_szlite

[![tests](https://github.com/CLASS-SZ/classy_szlite/actions/workflows/test.yml/badge.svg)](https://github.com/CLASS-SZ/classy_szlite/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/classy-szlite.svg?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/classy-szlite/)
[![Python versions](https://img.shields.io/pypi/pyversions/classy-szlite.svg?logo=python&logoColor=white)](https://pypi.org/project/classy-szlite/)
[![Documentation Status](https://readthedocs.org/projects/classy-szlite/badge/?version=latest)](https://classy-szlite.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, differentiable cosmology in pure JAX.**

`classy_szlite` provides JIT-compiled, `jax.grad`-friendly access to:

- **CMB angular power spectra** — TT, TE, EE
- **Linear and nonlinear matter Pk** — P(k, z), Pnl(k, z)
- **Distances** — H(z), comoving χ(z), angular-diameter D_A(z)
- **Derived parameters** — σ8, Ω_m, S8
- **Halo-model tSZ Cl^yy** — Arnaud 2010 GNFW pressure profile

Backed by the high-accuracy **`v2` CosmoPower emulators** — the same
emulators used in the [ACT DR6 extended-cosmology
analysis](https://arxiv.org/abs/2503.14454) (2025) and the
[ACT DR6 + DESI DR2 analysis](https://arxiv.org/abs/2505.08051) by
Poulin et al. (2025), matching the CAMB-based
[Jense et al. (2024) emulators](https://github.com/cosmopower-organization/jense_2024_emulators)
to well under 0.1 σ in ΛCDM. See
[Installation](https://classy-szlite.readthedocs.io/en/latest/installation.html)
for the emulator-coverage details.

Runtime dependencies: `jax`, `numpy`, `mcfit`.

## Install

```bash
pip install classy_szlite
```

Or from source:

```bash
git clone https://github.com/CLASS-SZ/classy_szlite
cd classy_szlite
pip install -e .
```

You also need the CosmoPower emulator `.npz` files at `~/class_sz_data/`
(or the path in `$CLASSY_SZLITE_DATA_DIR`). See
[Installation](https://classy-szlite.readthedocs.io/en/latest/installation.html).

## Quick start

```python
import jax.numpy as jnp
import classy_szlite as csl

cosmo = csl.CosmoParams()                      # Planck-18 ΛCDM defaults

# Derived parameters
csl.derived(cosmo)
# → {'sigma_8': 0.812, 'Omega_m': 0.311, 'S8': 0.827, 'der_full': ...}

# CMB Cls (dimensionless D_ℓ; × Tcmb² for μK²)
csl.cl_TTTEEE(cosmo)
# → {'ell', 'tt', 'te', 'ee'}

# Matter Pk at multiple z
k, pk  = csl.Pk(cosmo,  [0., 0.5, 1., 2.])
k, pnl = csl.Pnl(cosmo, [0., 0.5, 1., 2.])

# Distances
Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])

# Halo-model tSZ Cl^yy
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
ell = jnp.geomspace(2, 9000, 80)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)

# MCMC fast path: precompute cosmology + halo grids once → ~5 ms/call
ev = csl.cl_yy_factory(cosmo, ell)
cl_1h, cl_2h = ev(profile)
```

## Throughput

Warm-call timing, n = 100 calls per benchmark, freshly randomised inputs:

| Function | mean ± std (ms) | calls/s |
| --- | --- | --- |
| `derived` | 0.54 ± 0.04 | 1850 ± 150 |
| `cl_TTTEEE` | 2.52 ± 0.14 | 400 ± 25 |
| `Pk` | 1.49 ± 0.12 | 670 ± 55 |
| `distances` | 1.29 ± 0.09 | 770 ± 60 |
| `cl_yy` (full pipeline) | 17.84 ± 0.58 | 56 ± 2 |
| `cl_yy_factory` (fixed-cosmo) | 5.38 ± 0.42 | 185 ± 15 |
| `cl_yy_factory + jax.grad` | 17.12 ± 1.01 | 58 ± 3 |

Reference platform: macOS arm64 (M-series CPU), single-thread JAX. See
[Throughput](https://classy-szlite.readthedocs.io/en/latest/throughput.html)
for a more detailed table + reproduction script.

## JAX gradients

All public functions are JAX-traceable. The factory closure is the
recommended path for gradient-based inference at fixed cosmology:

```python
import jax
ev = csl.cl_yy_factory(cosmo, ell)

def loss(P0, beta):
    cl_1h, cl_2h = ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
    return jnp.sum(cl_1h + cl_2h)

d_loss = jax.grad(loss, argnums=(0, 1))(8.13, 5.48)
```

Gradients also work through the full pipeline (cosmology + profile) and
w.r.t. the whole `CosmoParams` pytree — see
[Gradients](https://classy-szlite.readthedocs.io/en/latest/gradients.html).

## Documentation

Full docs at **https://classy-szlite.readthedocs.io**.

## License

MIT.
