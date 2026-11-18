"""Band-limited y painting on a periodic tile (PLAN §3.3).

One y map per tile, then the channels in Fourier space:

1. scatter-add each halo's beam-convolved radial profile on the 0.25′ grid
   (2400²), with a 0.8′ painting beam (scheme A of the M0 spike);
2. rfft2, crop to the 0.5′ grid's modes, divide out the painting beam on the
   in-band modes (zero elsewhere);
3. halos with θ500 > θ_split are synthesised exactly on the in-band modes
   instead (few halos, large footprints).

The result is y(k) in coarse rfft units:
y(k) = Σ_h y0_h θ_h² T(|k| θ_h) e^{-ik·x_h} / Ω_pix, the continuous template
with no pixel window, as the finder's template.

Radial tables: P(r; θ) = (1/2π) ∫ dℓ ℓ θ² T(ℓθ) B_paint(ℓ) J0(ℓ r) on θ nodes
uniform in ln θ, r uniform on [0, 5θ + 6σ_paint]; cubic Lagrange in r,
linear in ln θ.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import j0

from .. import profiles
from .flatsky import N, NF, PIXF, ARCMIN, grid, fine_to_coarse_rows

FWHM_PAINT = 0.8                                       # arcmin
SIG_PAINT = FWHM_PAINT / np.sqrt(8.0 * np.log(2.0))    # arcmin
R_SIGMAS = 6.0                                         # table extent: 5θ + 6σ_paint


# ---------------------------------------------------------------------------
# Tables (host-side)
# ---------------------------------------------------------------------------

class PaintTables(NamedTuple):
    lnth0: float              # ln θ of node 0 [arcmin]
    dln: float                # node spacing in ln θ
    rP: np.ndarray            # (n_th,) table extent [arcmin]
    tab: np.ndarray           # (n_th, nr) P(r) per unit y0
    th_split: float           # θ500 above which halos take the direct path


def radial_profile(theta, rP, nr):
    """P(r; θ) on r = linspace(0, rP, nr), per unit y0 (θ, r in arcmin)."""
    lmax = 9.5 / SIG_PAINT                             # B_paint = e^-45
    npan = int(np.ceil(lmax / (3.0 / rP)))             # ℓ r changes ≤ 3 rad per panel
    xg, wg = np.polynomial.legendre.leggauss(10)
    edges = np.linspace(0.0, lmax, npan + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    ell = (0.5 * (hi - lo) * xg + 0.5 * (hi + lo)).ravel()
    w = (0.5 * (hi - lo) * wg).ravel()
    v = w * ell * theta ** 2 * profiles.T_interp(ell * theta) \
        * np.exp(-0.5 * (ell * SIG_PAINT) ** 2) / (2.0 * np.pi)
    r = np.linspace(0.0, rP, nr)
    out = np.empty(nr)
    for a in range(0, nr, 256):
        out[a:a + 256] = j0(np.outer(r[a:a + 256], ell)) @ v
    return out


def build_tables(th_min=0.1, th_split=6.0, dln=0.02, nr=2048) -> PaintTables:
    n_th = int(np.ceil(np.log(th_split / th_min) / dln)) + 2
    th = th_min * np.exp(dln * np.arange(n_th))
    rP = R_SIGMAS * SIG_PAINT + 5.0 * th
    tab = np.stack([radial_profile(t, r, nr) for t, r in zip(th, rP)])
    return PaintTables(lnth0=float(np.log(th_min)), dln=float(dln), rP=rP, tab=tab,
                       th_split=float(th_split))


# ---------------------------------------------------------------------------
# Footprint classes
# ---------------------------------------------------------------------------

SIDES = (17, 25, 33, 41, 49, 65, 81, 97, 129, 161, 193, 257, 321, 385)
BLOCK_BUDGET = 2 ** 21          # footprint pixels per scan step


def half_width(rP_arcmin):
    """Footprint half-width h [fine pixels] covering radius rP around a centre
    anywhere inside its pixel: offsets -h..h cover [iy - h, iy + h]."""
    return np.ceil(np.asarray(rP_arcmin) / PIXF).astype(np.int64) + 1


class PaintConfig(NamedTuple):
    """Static painting configuration (hashable)."""
    sides: tuple            # footprint side per class
    caps: tuple             # capacity per class (multiple of its block size)
    blocks: tuple           # halos per scan step per class
    cap_big: int            # capacity of the direct path


def make_config(tables: PaintTables, expected_per_class, cap_big, margin=1.5, extra=256):
    """Capacities from expected class occupations (host-side)."""
    caps, blocks = [], []
    for S, n in zip(SIDES, expected_per_class):
        B = max(1, BLOCK_BUDGET // (S * S))
        cap = int(np.ceil((margin * n + extra) / B)) * B if n > 0 else B
        caps.append(cap)
        blocks.append(B)
    return PaintConfig(sides=SIDES, caps=tuple(caps), blocks=tuple(blocks), cap_big=int(cap_big))


def theta_node(theta, tables):
    """Node index i0 (clipped) and linear weight w in ln θ, traceable."""
    n_th = tables.rP.shape[0]
    t = (jnp.log(theta) - tables.lnth0) / tables.dln
    t = jnp.clip(t, 0.0, n_th - 1.000001)
    i0 = jnp.floor(t).astype(jnp.int32)
    return i0, (t - i0).astype(jnp.float32)


def footprint_class(theta, tables, rP_j):
    """Class index of each halo (len(SIDES) for θ > θ_split or beyond)."""
    i0, _ = theta_node(theta, tables)
    R = rP_j[i0 + 1]
    need = 2 * (jnp.ceil(R / PIXF).astype(jnp.int32) + 1) + 1
    cls = jnp.searchsorted(jnp.asarray(SIDES, dtype=jnp.int32), need, side="left")
    return jnp.where(theta > tables.th_split, len(SIDES), cls).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Scatter-add painter (scheme A)
# ---------------------------------------------------------------------------

def _cubic_row(tab, i, u, nr):
    """4-point Lagrange interpolation of rows ``i`` of ``tab`` at fractional
    nodes ``u`` ≥ 0 (even extension at 0, zero beyond the last node).
    ``i`` broadcasts against ``u``."""
    j = jnp.floor(u).astype(jnp.int32)
    t = u - j

    def tv(jj):
        jj = jnp.abs(jj)
        return jnp.where(jj < nr, tab[i, jnp.minimum(jj, nr - 1)], 0.0)

    v0, v1, v2, v3 = tv(j - 1), tv(j), tv(j + 1), tv(j + 2)
    return (-t * (t - 1.0) * (t - 2.0) / 6.0 * v0 + (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0 * v1
            - (t + 1.0) * t * (t - 2.0) / 2.0 * v2 + (t + 1.0) * t * (t - 1.0) / 6.0 * v3)


def _paint_class(buf, h, tab, rP, dr_inv, S, B, cap):
    """Scatter-add the halos ``h`` (dict of (cap,) arrays) with footprint S."""
    nr = tab.shape[1]
    hw = (S - 1) // 2
    d = jnp.arange(-hw, hw + 1, dtype=jnp.int32)
    df = d.astype(jnp.float32)

    def body(buf, blk):
        iy, ix, oy, ox, y0, i0, w = blk
        ry = (df[None, :] - oy[:, None]) * PIXF                      # (B, S) arcmin
        rx = (df[None, :] - ox[:, None]) * PIXF
        r = jnp.sqrt(ry[:, :, None] ** 2 + rx[:, None, :] ** 2)      # (B, S, S)
        a = i0[:, None, None]
        v0 = _cubic_row(tab, a, r * dr_inv[a], nr)
        v1 = _cubic_row(tab, a + 1, r * dr_inv[a + 1], nr)
        val = y0[:, None, None] * ((1.0 - w[:, None, None]) * v0 + w[:, None, None] * v1)
        jy = (iy[:, None] + d[None, :]) % NF
        jx = (ix[:, None] + d[None, :]) % NF
        flat = jy[:, :, None] * NF + jx[:, None, :]
        return buf.at[flat.ravel()].add(val.ravel(), mode="drop"), None

    nblk = cap // B
    blocks = tuple(x.reshape(nblk, B) for x in (h["iy"], h["ix"], h["oy"], h["ox"],
                                                h["y0"], h["i0"], h["w"]))
    buf, _ = jax.lax.scan(body, buf, blocks)
    return buf


def paint_fine(halos, tables, cfg: PaintConfig):
    """The fine 2400² y map (painting beam included) of the halos with
    θ500 ≤ θ_split, and per-class overflow flags."""
    tab = jnp.asarray(tables.tab, dtype=jnp.float32)
    rP = jnp.asarray(tables.rP, dtype=jnp.float32)
    dr_inv = (tab.shape[1] - 1) / rP
    i0, w = theta_node(halos.theta500, tables)
    cls = footprint_class(halos.theta500, tables, rP)
    cls = jnp.where(halos.valid, cls, -1)
    buf = jnp.zeros((NF * NF,), dtype=jnp.float32)
    cap_n = halos.y0.shape[0]
    overflow = []
    for c, (S, cap, B) in enumerate(zip(cfg.sides, cfg.caps, cfg.blocks)):
        sel = cls == c
        overflow.append(jnp.sum(sel) > cap)
        idx = jnp.nonzero(sel, size=cap, fill_value=cap_n)[0]

        def take(x, fill=0):
            return jnp.take(x, idx, mode="fill", fill_value=fill)
        h = dict(iy=take(halos.iy), ix=take(halos.ix), oy=take(halos.oy), ox=take(halos.ox),
                 y0=take(halos.y0), i0=take(i0), w=take(w))
        buf = _paint_class(buf, h, tab, rP, dr_inv, S, B, cap)
    return buf.reshape(NF, NF), jnp.stack(overflow)


def fine_to_coarse(fine):
    """rfft2 of the fine map, cropped to the coarse modes, in coarse rfft
    units, painting beam divided out on the in-band modes (zero elsewhere)."""
    G = grid()
    F = jnp.fft.rfft2(fine)
    rows = jnp.asarray(fine_to_coarse_rows(), dtype=jnp.int32)
    Fc = F[rows][:, : N // 2 + 1] / 4.0
    dec = np.where(G.sel, np.exp(0.5 * (G.L * SIG_PAINT * ARCMIN) ** 2), 0.0)
    return Fc * jnp.asarray(dec, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Direct synthesis (θ500 > θ_split, and the reference for tests)
# ---------------------------------------------------------------------------

DU_T = 0.005                          # T(u) table step for the direct path
U_T_MAX = 1600.0                      # covers θ500 ≤ 270′ at |k| ≤ 20000


def T_uniform_table():
    u = np.arange(0.0, U_T_MAX + DU_T, DU_T)
    return profiles.T_interp(u)


def synth_direct(halos_sel, T_tab, dtype=jnp.float32):
    """y(k) of the given halos on the coarse rfft grid (in-band modes only),
    coarse rfft units. ``halos_sel``: dict with iy, ix (int32, fine pixels),
    oy, ox (offsets), y0, theta500 (arcmin), all (C,). Exact up to the T(u)
    table: phases are reduced in integer arithmetic, so float32 keeps
    ~1e-7 rad at any position."""
    G = grid()
    sel = jnp.asarray(G.sel)
    i0 = jnp.asarray(G.i0, dtype=jnp.int32)[:, None]                  # (N, 1)
    i1 = jnp.asarray(G.i1, dtype=jnp.int32)[None, :]                  # (1, N/2+1)
    L = jnp.asarray(G.L, dtype=dtype)
    Om = G.Om
    T_tab = jnp.asarray(T_tab, dtype=dtype)
    twopi_over_NF = dtype(2.0 * np.pi / NF)

    def one(acc, h):
        iy, ix, oy, ox, y0, th = h
        th_rad = th * dtype(ARCMIN)
        u = L * th_rad / dtype(DU_T)
        j = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, T_tab.shape[0] - 2)
        t = u - j
        T = (1.0 - t) * T_tab[j] + t * T_tab[j + 1]
        amp = y0 * th_rad ** 2 * T / dtype(Om)
        # phase = 2π (i0 y + i1 x) / LBOX with y = (iy + oy) PIXF: exact integer part mod NF
        ph_int = ((i0 * iy) % NF + (i1 * ix) % NF) % NF
        ph = twopi_over_NF * (ph_int.astype(dtype) + i0.astype(dtype) * oy
                              + i1.astype(dtype) * ox)
        term = amp * (jnp.cos(ph) - 1j * jnp.sin(ph))
        return acc + jnp.where(sel, term, 0.0), None

    cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    acc0 = jnp.zeros(G.L.shape, dtype=cdtype)
    xs = tuple(jnp.asarray(halos_sel[k]) for k in ("iy", "ix", "oy", "ox", "y0", "theta500"))
    xs = xs[:2] + tuple(x.astype(dtype) for x in xs[2:])
    acc, _ = jax.lax.scan(one, acc0, xs)
    return acc


def paint_y(halos, tables, cfg: PaintConfig, T_tab):
    """y(k) of all halos of a tile (coarse rfft units, in-band), and overflow
    flags (per class, then the direct path)."""
    fine, ovf = paint_fine(halos, tables, cfg)
    yk = fine_to_coarse(fine)
    big = halos.valid & (halos.theta500 > tables.th_split)
    cap_n = halos.y0.shape[0]
    idx = jnp.nonzero(big, size=cfg.cap_big, fill_value=cap_n)[0]
    hb = {k: jnp.take(getattr(halos, k), idx, mode="fill", fill_value=0)
          for k in ("iy", "ix", "oy", "ox", "y0")}
    hb["theta500"] = jnp.take(halos.theta500, idx, mode="fill", fill_value=1.0)
    yk = yk + synth_direct(hb, T_tab)
    return yk, jnp.concatenate([ovf, (jnp.sum(big) > cfg.cap_big)[None]])
