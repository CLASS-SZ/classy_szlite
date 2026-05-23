# Quick start

The public API is organised around a single :class:`~classy_szlite.CosmoParams`
container (a `NamedTuple`, JAX pytree) and a handful of top-level functions.

## The cosmology container

```python
import classy_szlite as csl
cosmo = csl.CosmoParams()        # defaults: Planck-18 + fEDE=0.001 (LCDM-equivalent)
```

To override any parameter:

```python
cosmo = csl.CosmoParams(
    omega_b   = 0.022,
    omega_cdm = 0.12,
    H0        = 68.0,
    n_s       = 0.97,
)
```

EDE-specific fields (`fEDE`, `log10z_c`, `thetai_scf`, `r`) and the
neutrino fields (`m_ncdm`, `N_ur`) default to the ede-v2 emulator's
LCDM-equivalent setup — most users never need to touch them.

## The five calculations

```python
import jax.numpy as jnp

# 1. Derived parameters
csl.derived(cosmo)
# → {'sigma_8': ..., 'Omega_m': ..., 'S8': ..., 'der_full': ndarray}

# 2. CMB angular power spectra
out = csl.cl_TTTEEE(cosmo)
# → {'ell', 'tt', 'te', 'ee'}  (dimensionless; × Tcmb² gives μK²)

# 3. Matter power spectrum
k, pk_linear = csl.Pk(cosmo, [0., 0.5, 1., 2.])
k, pk_nonlin = csl.Pnl(cosmo, [0., 0.5, 1., 2.])

# 4. Distances
Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])

# 5. Halo-model tSZ Cl^yy
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
ell = jnp.geomspace(2, 9000, 80)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)
```

## The fast path for MCMC

When sampling **profile parameters at fixed cosmology** (the dominant tSZ
Cl^yy use case), :func:`~classy_szlite.cl_yy_factory` precomputes the heavy
cosmology + halo-model setup once and returns a closure that takes only
the profile — typically ~5 ms per call:

```python
ev = csl.cl_yy_factory(cosmo, ell)            # ~1 s, runs the emulators once

# MCMC loop:
for P0, beta in chain:
    profile = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl_1h, cl_2h = ev(profile)                # ~5 ms / call
```

## JAX gradients

All public functions are JAX-traceable. Profile-only gradients via the
factory closure are particularly cheap (~17 ms / call, ~3× the forward
pass):

```python
import jax

def loss(P0, beta):
    profile = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl_1h, cl_2h = ev(profile)
    return jnp.sum(cl_1h + cl_2h)

d_loss = jax.grad(loss, argnums=(0, 1))(8.13, 5.48)
```

See [Gradients](gradients.md) for the full picture, including gradients
through the cosmology pipeline.
