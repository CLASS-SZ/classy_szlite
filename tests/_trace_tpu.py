"""Find where NaN enters the cl_yy pipeline on TPU."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

print("Backend:", jax.default_backend())

import classy_szlite as csl
from classy_szlite import cosmology as csl_cos, power_spectrum as csl_ps

cosmo = csl.CosmoParams()
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

# Step 1: Pk
z_arr = jnp.linspace(0.01, 4.0, 100)
k, pk = csl.Pk(cosmo, z_arr)
print(f"k shape={k.shape}, dtype={k.dtype}, any_nan={bool(jnp.any(jnp.isnan(k)))}, "
      f"range=[{float(jnp.min(k)):.3e}, {float(jnp.max(k)):.3e}]")
print(f"pk shape={pk.shape}, dtype={pk.dtype}, any_nan={bool(jnp.any(jnp.isnan(pk)))}, "
      f"range=[{float(jnp.min(pk)):.3e}, {float(jnp.max(pk)):.3e}]")

# Step 1b: σ(R, z) via FFTLog
R, sigma, dsigma2dR = csl_cos._compute_sigma(k, pk, 100)
print(f"\nσ(R): shape={sigma.shape}, dtype={sigma.dtype}, "
      f"any_nan={bool(jnp.any(jnp.isnan(sigma)))}, "
      f"range=[{float(jnp.nanmin(sigma)):.3e}, {float(jnp.nanmax(sigma)):.3e}]")
print(f"R: range=[{float(jnp.min(R)):.3e}, {float(jnp.max(R)):.3e}]")

# Inspect precomputed A10 FT table (built at module import time)
from classy_szlite.power_spectrum import _A10_G_TABLE, _TABLE_S_GRID, _B12_G_TABLE
print(f"\n_A10_G_TABLE: shape={_A10_G_TABLE.shape}, dtype={_A10_G_TABLE.dtype}, "
      f"any_nan={bool(jnp.any(jnp.isnan(_A10_G_TABLE)))}, "
      f"any_inf={bool(jnp.any(jnp.isinf(_A10_G_TABLE)))}, "
      f"range=[{float(jnp.nanmin(_A10_G_TABLE)):.3e}, {float(jnp.nanmax(_A10_G_TABLE)):.3e}]")
print(f"_TABLE_S_GRID: range=[{float(jnp.min(_TABLE_S_GRID)):.3e}, {float(jnp.max(_TABLE_S_GRID)):.3e}]")
print(f"_B12_G_TABLE: any_nan={bool(jnp.any(jnp.isnan(_B12_G_TABLE)))}, "
      f"any_inf={bool(jnp.any(jnp.isinf(_B12_G_TABLE)))}")

from classy_szlite.api import cosmo_to_dict
from classy_szlite.cosmology import build as build_cosmo_grids
from classy_szlite.hmf import build_halo_grids

cosmo_dict = cosmo_to_dict(cosmo)
z_grid = jnp.geomspace(0.005, 3.0, 100)
cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
print("\nCosmoGrids:")
for k in dir(cg):
    if not k.startswith('_'):
        v = getattr(cg, k)
        if isinstance(v, jnp.ndarray):
            nans = bool(jnp.any(jnp.isnan(v)))
            print(f"  cg.{k}: shape={v.shape}, dtype={v.dtype}, any_nan={nans}, "
                  f"range=[{float(jnp.nanmin(v)):.3e}, {float(jnp.nanmax(v)):.3e}]")

hg = build_halo_grids(cg, cosmo_dict, delta_crit=500.0, m_min=1e10, m_max=3.5e15, n_m=200)
print("\nHaloGrids:")
for k in dir(hg):
    if not k.startswith('_'):
        v = getattr(hg, k)
        if isinstance(v, jnp.ndarray):
            nans = bool(jnp.any(jnp.isnan(v)))
            infs = bool(jnp.any(jnp.isinf(v)))
            print(f"  hg.{k}: shape={v.shape}, dtype={v.dtype}, nan={nans}, inf={infs}, "
                  f"range=[{float(jnp.nanmin(v)):.3e}, {float(jnp.nanmax(v)):.3e}]")
