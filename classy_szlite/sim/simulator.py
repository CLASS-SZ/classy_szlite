"""One tile, end to end (first version, for the M0 throughput spike).

θ → cosmology → λ cells (once per θ), then per tile:
halos → y(k) → six channel maps (tSZ + lensed CMB + white and atmospheric
noise) → matched filter (covariance from the maps) → peaks → masking
iteration (discs around the q > 4 peaks, covariance re-estimated outside
them, filter run again; ``TileConfig.n_iter`` times, 1 by default as SZiFi)
→ merging → confirmation → (q, z) histogram.

Stacked CMB lensing: the catalogue-level observable p of each data-vector detection,
from its matched halo (lensing.py); each tile returns Σ p and the count.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ..cosmology import build, ln10As_from_sigma8, LN10AS_REFERENCE
from . import instrument as ins
from . import halos as hl
from . import paint as pt
from . import mmf
from . import summaries as sm
from . import lensing
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
    beta_lens: float = 0.92      # CMB-lensing mass bias β_CMBlens
    sigma_lnp: float = 0.22      # intrinsic scatter of the lensing observable σ_CMBlens


THETA_FID = Theta(sigma8=0.811, omega_cdm=0.121, h=0.674, omega_b=0.0222, n_s=0.96,
                  A_SZ=-4.3054, alpha_SZ=1.1233, sigma_lnq=0.173, one_minus_b=0.8)


class TileConfig(NamedTuple):
    """Static configuration (hashable)."""
    halo_cap: int
    paint: pt.PaintConfig
    peak_cap: int = 3072     # final-pass peaks reach ~1,900 at the wide prior box's heaviest corner
    conf_cap: int = 16384
    M_conf: float = 5e13
    r_conf: float = 1.5
    r_merge: float = 1.5
    n_iter: int = 1          # masking iterations (SZiFi's max_it); 0 = one unmasked pass
    q_mask: float = mmf.Q_MASK
    # Peak threshold of the final pass: the data vector's lower q edge. Merging keeps
    # a peak unless a neighbour has a higher q, so peaks with q ≤ 5 never remove a
    # q > 5 peak, and the histogram counts q > 5 only: the data vector is identical
    # to keeping every q > 4 peak, with ~40% fewer peaks in the buffer.
    q_final: float = float(sm.Q_EDGES[0])


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
    lens_lnth: jax.Array    # ln θ500 [arcmin] of the σ_κ table
    lens_lnsig: jax.Array   # ln σ_κ (SO lensing depth of ``mode``)


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
        lens_lnth=jnp.asarray(lensing.table(mode)[0]),
        lens_lnsig=jnp.asarray(lensing.table(mode)[1]),
    )


class CosmoStage(NamedTuple):
    lam: jax.Array          # (n_z, n_m) expected halos per cell, one tile
    cg: object              # CosmoGrids on the cell z edges
    d2N: jax.Array          # (n_z + 1, n_m + 1) corner densities d²N/dz dlnM
    chi_star: jax.Array     # () comoving distance to last scattering [Mpc]


def cosmology_stage(theta: Theta, cells: hl.CellGrid) -> CosmoStage:
    """λ cells, corner densities and the cosmology grids on the cell z edges
    (traceable)."""
    p = {"omega_cdm": theta.omega_cdm, "H0": 100.0 * theta.h, "omega_b": theta.omega_b,
         "n_s": theta.n_s}
    # A_s by the linear-theory scaling about ln(10^10 A_s) = 3.0: exact, and safe
    # at the prior corners, which map outside the emulators' A_s box (2.5-3.5)
    p["ln10^{10}A_s"] = ln10As_from_sigma8(p, theta.sigma8, as_reference=LN10AS_REFERENCE)
    cg = build(p, z_grid=cells.z_edges, as_reference=LN10AS_REFERENCE)
    d2N = hl.corner_density(cg, p, cells)
    return CosmoStage(lam=hl.lambda_from_corners(d2N, cells), cg=cg, d2N=d2N,
                      chi_star=lensing.chi_star(theta.h, theta.omega_cdm, theta.omega_b))


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


def tile_maps(key, cs: CosmoStage, theta: Theta, st: TileStatic, tables: pt.PaintTables,
              cells: hl.CellGrid, cfg: TileConfig):
    """Halos, the six channel maps and the painter's overflow flags of one tile."""
    k_h, k_m = jax.random.split(key)
    halos = hl.sample_halos(k_h, cs.lam, cells, cfg.halo_cap, cs.cg, theta.h, theta.A_SZ,
                            theta.alpha_SZ, theta.sigma_lnq, theta.one_minus_b, d2N=cs.d2N)
    tabs = tables._replace(tab=st.paint_tab, rP=st.paint_rP)
    yk, ovf_paint = pt.paint_y(halos, tabs, cfg.paint, st.T_tab)
    return halos, channel_maps(k_m, yk, st), ovf_paint


def simulate_tile(key, cs: CosmoStage, theta: Theta, st: TileStatic, tables: pt.PaintTables,
                  cells: hl.CellGrid, cfg: TileConfig):
    """Data vector and diagnostics of one tile."""
    halos, X, ovf_paint = tile_maps(key, cs, theta, st, tables, cells, cfg)
    pk, sigma, mk = find_clusters(X, st.fs, cfg)
    keep = sm.merge_local_max(pk, cfg.r_merge)
    cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
    counts = sm.histogram(pk.q, cf.z, keep & cf.ok)
    overflow = jnp.concatenate([halos.overflow[None], ovf_paint, mk["peak_overflow"][None],
                                cf.overflow[None]])
    p_sum, n_p = stacked_lensing(jax.random.fold_in(key, 7), pk, keep, cf, halos, cs, theta, st)
    return dict(counts=counts, n_halos=halos.count, n_peaks=pk.count,
                n_kept=jnp.sum(keep), n_conf=jnp.sum(keep & cf.ok),
                overflow=overflow, sigma=sigma, sigma0=mk["sigma0"],
                f_masked=mk["f_masked"], n_masked=mk["n_masked"],
                p_sum=p_sum, n_p=n_p)


def stacked_lensing(key, pk, keep, cf, halos, cs: CosmoStage, theta: Theta, st: TileStatic):
    """Sum of the lensing observable p over the data-vector detections of one
    tile, and their number (the survey's stacked mean is Σ p_sum / Σ n_p)."""
    sel = sm.in_data_vector(pk.q, cf.z, keep & cf.ok)
    M = jnp.exp(halos.lnM[cf.halo].astype(jnp.float64))
    z = halos.z[cf.halo].astype(jnp.float64)
    pbar = lensing.p_mean(M, z, cs.cg, theta.h, theta.beta_lens, cs.chi_star, st.lens_lnth, st.lens_lnsig)
    p = lensing.draw_p(key, jnp.where(sel, pbar, 1.0), theta.sigma_lnp)
    return jnp.sum(jnp.where(sel, p, 0.0)), jnp.sum(sel.astype(jnp.int32))


def find_clusters(X, fs: mmf.FilterStatic, cfg: TileConfig):
    """Matched filter with the masking iteration: the final pass's peaks
    (q > cfg.q_final), its σ_j and diagnostics (first-pass σ_j, masked fraction
    and number of discs of the last iteration, and a peak-buffer overflow flag
    over all passes, since an overflowing pass masks only the peaks it holds).
    Passes that feed a mask keep every peak with q > cfg.q_mask."""
    q, sigma, C0 = mmf.q_cube(X, fs)
    pk = mmf.find_peaks(q, cfg.peak_cap, cfg.q_mask if cfg.n_iter else cfg.q_final)
    sigma0, ovf = sigma, pk.overflow
    f, n_masked = jnp.float64(1.0), jnp.int32(0)
    for it in range(cfg.n_iter):
        mask, n_masked = mmf.cluster_mask(pk, fs, cfg.q_mask)
        C, f = mmf.masked_cov(X, mask, C0, fs)
        q, sigma, _ = mmf.q_cube(X, fs, C=C)
        last = it == cfg.n_iter - 1
        pk = mmf.find_peaks(q, cfg.peak_cap, cfg.q_final if last else cfg.q_mask)
        ovf = ovf | pk.overflow
    return pk, sigma, dict(sigma0=sigma0, f_masked=1.0 - f, n_masked=n_masked,
                           peak_overflow=ovf)
