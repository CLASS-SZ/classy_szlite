"""One tile, end to end (first version, for the M0 throughput spike).

θ → cosmology → λ cells (once per θ), then per tile:
halos → y(k) → six channel maps (tSZ + lensed CMB + white and atmospheric
noise) → matched filter (covariance from the maps, one pass) → peaks →
merging → confirmation → (q, z) histogram.

No masking iteration yet (M5), no lensing summary yet (M6).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ..cosmology import build, ln10As_from_sigma8
from . import instrument as ins
from . import halos as hl
from . import paint as pt
from . import mmf
from . import summaries as sm
from .flatsky import N, PIX, grid


class Theta(NamedTuple):
    """Parameters of one simulation (the paper's SO-like set)."""
    sigma8: float
    omega_cdm: float
    h: float
    omega_b: float
    n_s: float
    A_SZ: float
    alpha_SZ: float
    sigma_lnq: float
    one_minus_b: float


THETA_FID = Theta(sigma8=0.811, omega_cdm=0.121, h=0.674, omega_b=0.0222, n_s=0.96,
                  A_SZ=-4.3054, alpha_SZ=1.1233, sigma_lnq=0.173, one_minus_b=0.8)


class TileConfig(NamedTuple):
    """Static configuration (hashable)."""
    halo_cap: int
    paint: pt.PaintConfig
    peak_cap: int = 2048
    conf_cap: int = 16384
    M_conf: float = 5e13
    r_conf: float = 1.5
    r_merge: float = 1.5


class TileStatic(NamedTuple):
    """Static arrays (jit arguments)."""
    fs: mmf.FilterStatic
    a_full: jax.Array       # (6, N, N/2+1) f32: f_i B_i
    B_full: jax.Array       # (6, N, N/2+1) f32: B_i
    cmb_amp: jax.Array      # (N, N/2+1) f32: sqrt(C_ℓ / Ω_pix) [μK sr^-1/2]
    white_pix: jax.Array    # (6,) f32: white noise per pixel [μK]
    atm_amp: jax.Array      # (6, N, N/2+1) f32: sqrt(N_red,i shape(ℓ) / Ω_pix)
    paint_tab: jax.Array    # (n_th, nr) f32
    paint_rP: jax.Array     # (n_th,) f32
    T_tab: jax.Array        # direct-path T(u) table, f32


def tile_static(cmb_cl, tables: pt.PaintTables, mode="goal"):
    G = grid()
    f = ins.tsz_f_uK(ins.SO_LAT["freqs"])
    B = ins.beams(G.L)                                        # (6, N, N/2+1)
    cl = cmb_cl(G.L.ravel()).reshape(G.L.shape)
    red = ins.red_levels()
    shp = ins.atm_shape(G.L)
    f32 = np.float32
    return TileStatic(
        fs=mmf.filter_static(),
        a_full=jnp.asarray((B * f[:, None, None]).astype(f32)),
        B_full=jnp.asarray(B.astype(f32)),
        cmb_amp=jnp.asarray(np.sqrt(np.where(G.L > 0, cl, 0.0) / G.Om).astype(f32)),
        white_pix=jnp.asarray((ins.SO_LAT["white"][mode] / PIX).astype(f32)),
        atm_amp=jnp.asarray(np.sqrt(red[:, None, None] * shp[None] / G.Om).astype(f32)),
        paint_tab=jnp.asarray(tables.tab.astype(f32)),
        paint_rP=jnp.asarray(tables.rP.astype(f32)),
        T_tab=jnp.asarray(pt.T_uniform_table().astype(f32)),
    )


def cosmology_stage(theta: Theta, cells: hl.CellGrid):
    """λ cells and the cosmology grids on the cell z edges (traceable)."""
    p = {"omega_cdm": theta.omega_cdm, "H0": 100.0 * theta.h, "omega_b": theta.omega_b,
         "n_s": theta.n_s}
    p["ln10^{10}A_s"] = ln10As_from_sigma8(p, theta.sigma8)
    cg = build(p, z_grid=cells.z_edges)
    lam = hl.lambda_cells(cg, p, cells)
    return lam, cg


def channel_maps(key, yk, st: TileStatic):
    """Six channel maps (6, N, N/2+1) c64 in rfft units [μK]."""
    k_cmb, k_w, k_a = jax.random.split(key, 3)
    f32 = jnp.float32
    cmb = jnp.fft.rfft2(jax.random.normal(k_cmb, (N, N), dtype=f32)) * st.cmb_amp
    white = jnp.fft.rfft2(jax.random.normal(k_w, (6, N, N), dtype=f32)
                          * st.white_pix[:, None, None])
    g = jnp.fft.rfft2(jax.random.normal(k_a, (6, N, N), dtype=f32))       # 2 per tube
    r = f32(ins.SO_LAT["atm_corr"])
    s = jnp.sqrt(f32(1.0) - r * r)
    atm = []
    for t, (i, j) in enumerate(ins.SO_LAT["tubes"]):
        xa, xb = g[2 * t], g[2 * t + 1]
        atm.append((i, st.atm_amp[i] * xa))
        atm.append((j, st.atm_amp[j] * (r * xa + s * xb)))
    atm = jnp.stack([v for _, v in sorted(atm, key=lambda iv: iv[0])])
    return st.a_full * yk[None] + st.B_full * cmb[None] + white + atm


def simulate_tile(key, lam, cg, theta: Theta, st: TileStatic, tables: pt.PaintTables,
                  cells: hl.CellGrid, cfg: TileConfig):
    """Data vector and diagnostics of one tile."""
    k_h, k_m = jax.random.split(key)
    halos = hl.sample_halos(k_h, lam, cells, cfg.halo_cap, cg, theta.h, theta.A_SZ,
                            theta.alpha_SZ, theta.sigma_lnq, theta.one_minus_b)
    tabs = tables._replace(tab=st.paint_tab, rP=st.paint_rP)
    yk, ovf_paint = pt.paint_y(halos, tabs, cfg.paint, st.T_tab)
    X = channel_maps(k_m, yk, st)
    q, sigma, _ = mmf.q_cube(X, st.fs)
    pk = mmf.find_peaks(q, cfg.peak_cap)
    keep = sm.merge_local_max(pk, cfg.r_merge)
    cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
    counts = sm.histogram(pk.q, cf.z, keep & cf.ok)
    overflow = jnp.concatenate([halos.overflow[None], ovf_paint, pk.overflow[None],
                                cf.overflow[None]])
    return dict(counts=counts, n_halos=halos.count, n_peaks=pk.count,
                n_kept=jnp.sum(keep), n_conf=jnp.sum(keep & cf.ok),
                overflow=overflow, sigma=sigma)
