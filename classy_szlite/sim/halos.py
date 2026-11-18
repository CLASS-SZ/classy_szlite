"""Poisson halos on one periodic tile, in fixed-capacity buffers (traceable).

λ cells: expected counts per (z, ln M) cell of one tile,

    λ_c = ∫∫_cell d²N/dz dlnM dz dlnM,   d²N/dz dlnM = dn/dlnM (χ²/H) c Ω_tile,

from the corner values of the cell (exact for the bilinear interpolant).
Sampling: Poisson counts per cell, then each buffer slot s < Σ counts is
assigned to its cell through the cumulative counts. First version (M0 spike):
z and ln M uniform inside the cell. M2 replaces this by exact sampling of the
bilinear interpolant.

Positions are uniform on the tile, stored as an int32 painting-grid pixel plus
a float32 offset in [0, 1) pixel units. Overflow (Σ counts > capacity) is
flagged, never truncated silently: the caller reruns the draw in a larger
capacity class.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ..hmf import build_halo_grids
from ..sz_scaling import y0_median, theta500_arcmin
from .flatsky import NF, LBOX, ARCMIN

OMEGA_TILE = (LBOX * ARCMIN) ** 2      # sr


class CellGrid(NamedTuple):
    """Static (z, ln M) cell edges."""
    z_edges: np.ndarray        # (n_z + 1,)
    lnM_edges: np.ndarray      # (n_m + 1,)


def default_cells(zmin=0.01, zmax=3.0, n_z=300, Mmin=1e13, Mmax=1e16, n_m=150):
    return CellGrid(np.linspace(zmin, zmax, n_z + 1),
                    np.linspace(np.log(Mmin), np.log(Mmax), n_m + 1))


def lambda_cells(cg, params, cells: CellGrid, hmf_kwargs=None):
    """Expected counts per cell for one tile, shape (n_z, n_m), float64.

    ``cg`` must be a ``CosmoGrids`` built on z = ``cells.z_edges``; the HMF is
    evaluated at the cell corners. ``hmf_kwargs`` go to ``build_halo_grids``
    (production: ``tinker_interp='linear', nu_in_rho_m=False``).
    """
    kw = dict(delta_crit=500.0, tinker_interp="linear", nu_in_rho_m=False)
    kw.update(hmf_kwargs or {})
    n_m = cells.lnM_edges.size
    hg = build_halo_grids(cg, params, m_min=float(np.exp(cells.lnM_edges[0])),
                          m_max=float(np.exp(cells.lnM_edges[-1])), n_m=n_m, **kw)
    d2N = hg.dndlnm * (cg.chi ** 2 / cg.Hz)[:, None] * OMEGA_TILE     # per dz dlnM
    corner = 0.25 * (d2N[:-1, :-1] + d2N[1:, :-1] + d2N[:-1, 1:] + d2N[1:, 1:])
    dz = np.diff(cells.z_edges)[:, None]
    dlnM = np.diff(cells.lnM_edges)[None, :]
    return corner * dz * dlnM


class Halos(NamedTuple):
    """Halo buffer of one tile (capacity C). Padding slots have y0 = 0."""
    iy: jax.Array       # (C,) int32, painting-grid pixel (axis 0)
    ix: jax.Array       # (C,) int32
    oy: jax.Array       # (C,) float32, offset in [0, 1) pixels
    ox: jax.Array       # (C,) float32
    z: jax.Array        # (C,) float32
    lnM: jax.Array      # (C,) float32
    y0: jax.Array       # (C,) float32, with scatter; 0 for padding
    theta500: jax.Array  # (C,) float32, arcmin
    valid: jax.Array    # (C,) bool
    count: jax.Array    # () int32, number of halos drawn
    overflow: jax.Array  # () bool


def sample_halos(key, lam, cells: CellGrid, capacity: int, cg, h, A_SZ, alpha_SZ,
                 sigma_lnq, one_minus_b):
    """Draw the halos of one tile into a buffer of ``capacity`` slots."""
    k_pois, k_z, k_m, k_py, k_px, k_oy, k_ox, k_sc = jax.random.split(key, 8)
    n_z, n_m = lam.shape
    counts = jax.random.poisson(k_pois, lam.ravel(), dtype=jnp.int32)
    csum = jnp.cumsum(counts)
    total = csum[-1]
    slot = jnp.arange(capacity, dtype=jnp.int32)
    valid = slot < total
    cell = jnp.searchsorted(csum, slot, side="right").astype(jnp.int32)
    cell = jnp.minimum(cell, n_z * n_m - 1)
    iz, im = cell // n_m, cell % n_m

    ze = jnp.asarray(cells.z_edges)
    me = jnp.asarray(cells.lnM_edges)
    u_z = jax.random.uniform(k_z, (capacity,), dtype=jnp.float64)
    u_m = jax.random.uniform(k_m, (capacity,), dtype=jnp.float64)
    z = ze[iz] + u_z * (ze[iz + 1] - ze[iz])
    lnM = me[im] + u_m * (me[im + 1] - me[im])

    # E(z) and D_A(z) at the halo redshifts (cosmology on the cell z edges)
    H0_over_c = h * 100.0 / 299792.458
    Ez = jnp.interp(z, cg.z, cg.Hz) / H0_over_c
    DA = jnp.interp(z, cg.z, cg.Da)
    M = jnp.exp(lnM)
    y0 = y0_median(M, Ez, h, A_SZ, alpha_SZ, one_minus_b)
    y0 = y0 * jnp.exp(sigma_lnq * jax.random.normal(k_sc, (capacity,), dtype=jnp.float64))
    th = theta500_arcmin(M, Ez, DA, h, one_minus_b)

    iy = jax.random.randint(k_py, (capacity,), 0, NF, dtype=jnp.int32)
    ix = jax.random.randint(k_px, (capacity,), 0, NF, dtype=jnp.int32)
    oy = jax.random.uniform(k_oy, (capacity,), dtype=jnp.float32)
    ox = jax.random.uniform(k_ox, (capacity,), dtype=jnp.float32)
    f32 = jnp.float32
    return Halos(iy=iy, ix=ix, oy=oy, ox=ox, z=z.astype(f32), lnM=lnM.astype(f32),
                 y0=jnp.where(valid, y0, 0.0).astype(f32), theta500=th.astype(f32),
                 valid=valid, count=total, overflow=total > capacity)
