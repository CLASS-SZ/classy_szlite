"""Detections → data vector (PLAN §3.4), first version (M0 spike).

* Merging: a peak is kept unless another peak within r_merge has a higher q
  (ties: the earlier one in catalogue order wins). This is a local maximum
  within the merge radius, over all scales. M6 fixes the production rule
  against chains of overlapping peaks.
* Confirmation: a kept peak is confirmed if a halo with M500 ≥ M_conf lies
  within r_conf; it takes the redshift of the matched halo with the largest
  y0. Unconfirmed peaks are dropped.
* Histogram: four q bins with edges logspace(log10 5, log10 40, 5) and five
  z bins with edges linspace(0.01, 1.5, 6), open intervals; q ≥ 40 and z
  outside 0.01-1.5 are dropped.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .flatsky import N, NF, PIX, PIXF, LBOX
from ._util import compact

Q_EDGES = np.logspace(np.log10(5.0), np.log10(40.0), 5)
Z_EDGES = np.linspace(0.01, 1.5, 6)


def _wrap(d, period):
    """Periodic difference into [-period/2, period/2) (exact for integers)."""
    half = period // 2 if isinstance(period, int) else period / 2
    return (d + half) % period - half


def merge_local_max(peaks, r_merge_arcmin: float):
    """Boolean keep mask over the peak buffer."""
    dy = _wrap(peaks.iy[:, None] - peaks.iy[None, :], N)
    dx = _wrap(peaks.ix[:, None] - peaks.ix[None, :], N)
    d2 = dy * dy + dx * dx                                       # pixels², exact int32
    r2 = (r_merge_arcmin / PIX) ** 2
    both = peaks.valid[:, None] & peaks.valid[None, :]
    near = both & (d2.astype(jnp.float32) <= r2)
    k = jnp.arange(peaks.q.shape[0])
    better = ((peaks.q[None, :] > peaks.q[:, None])
              | ((peaks.q[None, :] == peaks.q[:, None]) & (k[None, :] < k[:, None])))
    return peaks.valid & ~jnp.any(near & better, axis=1)


class Confirmed(NamedTuple):
    ok: jax.Array          # (K,) bool
    z: jax.Array           # (K,) float32 (0 where not confirmed)
    overflow: jax.Array    # () bool: more massive halos than the buffer holds


def confirm(peaks, keep, halos, M_conf: float, r_conf_arcmin: float, capacity: int):
    massive = halos.valid & (halos.lnM >= np.float32(np.log(M_conf)))
    cap_n = halos.y0.shape[0]
    idx, n_massive = compact(massive, capacity, cap_n)

    def take(x, fill):
        return jnp.take(x, idx, mode="fill", fill_value=fill)
    hy = (take(halos.iy, 0).astype(jnp.float32) + take(halos.oy, 0.0)) * PIXF
    hx = (take(halos.ix, 0).astype(jnp.float32) + take(halos.ox, 0.0)) * PIXF
    hv = take(massive, False)
    hz = take(halos.z, 0.0)
    hy0 = take(halos.y0, 0.0)
    py = peaks.iy.astype(jnp.float32) * PIX
    px = peaks.ix.astype(jnp.float32) * PIX
    dy = _wrap(py[:, None] - hy[None, :], LBOX)
    dx = _wrap(px[:, None] - hx[None, :], LBOX)
    match = (dy * dy + dx * dx <= r_conf_arcmin ** 2) & hv[None, :] & keep[:, None]
    score = jnp.where(match, hy0[None, :], -1.0)
    best = jnp.argmax(score, axis=1)
    ok = jnp.any(match, axis=1)
    return Confirmed(ok=ok, z=jnp.where(ok, hz[best], 0.0),
                     overflow=n_massive > capacity)


def histogram(q, z, ok):
    """(4, 5) int32 counts in the paper's q and z bins."""
    qe = jnp.asarray(Q_EDGES, dtype=jnp.float32)
    ze = jnp.asarray(Z_EDGES, dtype=jnp.float32)
    inside = ok & (q > qe[0]) & (q < qe[-1]) & (z > ze[0]) & (z < ze[-1])
    iq = jnp.sum(q[:, None] > qe[None, 1:-1], axis=1)
    iz = jnp.sum(z[:, None] > ze[None, 1:-1], axis=1)
    return jnp.zeros((4, 5), dtype=jnp.int32).at[iq, iz].add(inside.astype(jnp.int32))
