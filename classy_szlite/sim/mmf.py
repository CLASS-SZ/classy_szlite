"""Multi-frequency matched filter on one tile (PLAN §5, mmf.py).

* Covariance from the maps: 6×6 cross-spectra averaged over the in-band
  modes of each |k| bin (rfft multiplicities), accumulated in float64. The
  per-bin sums use a static padded layout (no atomics), so the reduction
  order is fixed.
* Per-bin inverse (Cholesky, float64); shared weights w(k) = C_b⁻¹ a(k) for
  all scales, a_i(k) = f_i B_i(|k|); one combined map cx(k) = Σ_i w_i X_i.
* Per scale θ_j: ŷ0_j = irfft2(τ_j cx) N_px / norm_j with
  τ_j(k) = θ_j² T(|k| θ_j) / Ω_pix and norm_j = Σ_k W τ_j² aᵀC⁻¹a;
  σ_j = norm_j^(-1/2); q_j = ŷ0_j / σ_j.
* Peaks: q > q_th, strictly above the four map neighbours (periodic),
  q_j > q_{j-1} and q_j ≥ q_{j+1} along the scale axis (SZiFi's rules, as
  szsim). Compacted in catalogue order (scale, then row-major pixel) into a
  fixed-capacity buffer, with an overflow flag.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .. import profiles
from . import instrument as ins
from .flatsky import N, ARCMIN, grid
from ._util import compact

THETAS15 = np.geomspace(0.5, 15.0, 15)     # arcmin
Q_TH = 4.0


class FilterStatic(NamedTuple):
    """Static arrays of the filter (pass as jit arguments)."""
    m_idx: jax.Array        # (n_modes,) int32, flat rfft index, sorted by bin
    m_W: jax.Array          # (n_modes,) float64
    m_binpos: jax.Array     # (n_modes,) int32
    bin_pad: jax.Array      # (n_bins, maxc) int32, position in the mode list or -1
    bin_count: jax.Array    # (n_bins,) float64
    a: jax.Array            # (n_modes, 6) float64, f_i B_i(|k|)
    tau_m: jax.Array        # (n_th, n_modes) float64, τ_j on the mode list
    tau_full: jax.Array     # (n_th, N, N/2+1) float32, τ_j on the rfft grid (0 out of band)


def filter_static(thetas=THETAS15) -> FilterStatic:
    G = grid()
    f = ins.tsz_f_uK(ins.SO_LAT["freqs"])
    a = (ins.beams(G.m_L) * f[:, None]).T                              # (n_modes, 6)
    T_m = np.stack([profiles.T_interp(G.m_L * t * ARCMIN) for t in thetas])
    tau_m = (np.asarray(thetas)[:, None] * ARCMIN) ** 2 * T_m / G.Om
    tau_full = np.zeros((len(thetas),) + G.L.shape, dtype=np.float32)
    flat = tau_full.reshape(len(thetas), -1)
    flat[:, G.m_idx] = tau_m
    return FilterStatic(
        m_idx=jnp.asarray(G.m_idx, dtype=jnp.int32), m_W=jnp.asarray(G.m_W),
        m_binpos=jnp.asarray(G.m_binpos, dtype=jnp.int32),
        bin_pad=jnp.asarray(G.bin_pad, dtype=jnp.int32),
        bin_count=jnp.asarray(G.bin_count), a=jnp.asarray(a), tau_m=jnp.asarray(tau_m),
        tau_full=jnp.asarray(tau_full))


_PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]


def binned_cov(Xm, fs: FilterStatic):
    """Binned 6×6 covariance (float64) from the in-band modes Xm (6, n_modes)."""
    X = Xm.astype(jnp.complex128)
    W = fs.m_W
    pad = fs.bin_pad
    ok = pad >= 0
    pidx = jnp.where(ok, pad, 0)
    C = jnp.zeros((pad.shape[0], 6, 6), dtype=jnp.float64)
    for (i, j) in _PAIRS:
        prod = W * (X[i].real * X[j].real + X[i].imag * X[j].imag)   # (n_modes,)
        v = jnp.sum(jnp.where(ok, prod[pidx], 0.0), axis=1) / fs.bin_count
        C = C.at[:, i, j].set(v)
        if i != j:
            C = C.at[:, j, i].set(v)
    return C


def weights(C, fs: FilterStatic):
    """w(k) = C_b⁻¹ a(k) (n_modes, 6) and aᵀC⁻¹a (n_modes,), float64."""
    L = jnp.linalg.cholesky(C)
    eye = jnp.broadcast_to(jnp.eye(6, dtype=C.dtype), C.shape)
    Ci = jax.scipy.linalg.cho_solve((L, True), eye)
    Ci = 0.5 * (Ci + jnp.swapaxes(Ci, 1, 2))
    b = fs.m_binpos
    cols = []
    for i in range(6):
        wi = sum(Ci[b, i, j] * fs.a[:, j] for j in range(6))
        cols.append(wi)
    w = jnp.stack(cols, axis=1)
    aCa = jnp.sum(w * fs.a, axis=1)
    return w, aCa


def q_cube(X, fs: FilterStatic, C=None, dtype=jnp.float32):
    """S/N cube (n_th, N, N), σ_j (n_th,) and the covariance used.

    ``X``: channel maps (6, N, N/2+1) in rfft units (complex). ``C``: binned
    covariance; estimated from X if None.
    """
    G = grid()
    Xm = X.reshape(6, -1)[:, fs.m_idx]
    if C is None:
        C = binned_cov(Xm, fs)
    w, aCa = weights(C, fs)
    norm = jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)   # (n_th,)
    cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    # elementwise product and sum (not a dot): no TF32 on Ampere and later
    cx_m = jnp.sum(w.T.astype(dtype) * Xm.astype(cdtype), axis=0)
    cx = jnp.zeros((G.L.size,), dtype=cdtype).at[fs.m_idx].set(cx_m).reshape(G.L.shape)
    scale = (G.Npx / jnp.sqrt(norm)).astype(dtype)                           # (n_th,)
    q = jnp.fft.irfft2(fs.tau_full.astype(dtype) * cx[None], s=(N, N)) * scale[:, None, None]
    return q, 1.0 / jnp.sqrt(norm), C


class Peaks(NamedTuple):
    j: jax.Array          # (K,) int32 scale index
    iy: jax.Array         # (K,) int32
    ix: jax.Array         # (K,) int32
    q: jax.Array          # (K,) float32
    valid: jax.Array      # (K,) bool
    count: jax.Array      # () int32
    overflow: jax.Array   # () bool


def find_peaks(q, capacity: int, qth=Q_TH):
    """Peaks of the S/N cube (see the module docstring)."""
    one = jnp.ones((1,) + q.shape[1:], dtype=bool)
    above_prev = jnp.concatenate([one, q[1:] > q[:-1]], axis=0)     # q_j > q_{j-1}
    above_next = jnp.concatenate([q[:-1] >= q[1:], one], axis=0)    # q_j >= q_{j+1}
    m = ((q > qth)
         & (q > jnp.roll(q, 1, axis=1)) & (q > jnp.roll(q, -1, axis=1))
         & (q > jnp.roll(q, 1, axis=2)) & (q > jnp.roll(q, -1, axis=2))
         & above_prev & above_next)
    idx, count = compact(m.ravel(), capacity, 0)
    valid = jnp.arange(capacity) < count
    n2 = q.shape[1] * q.shape[2]
    j = idx // n2
    rem = idx % n2
    iy, ix = rem // q.shape[2], rem % q.shape[2]
    qv = jnp.where(valid, q.ravel()[idx], -jnp.inf)
    return Peaks(j=j, iy=iy, ix=ix, q=qv, valid=valid, count=count, overflow=count > capacity)
