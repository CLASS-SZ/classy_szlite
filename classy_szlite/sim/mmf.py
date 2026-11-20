"""Multi-frequency matched filter on one tile (PLAN §5, mmf.py).

* Covariance from the maps: n×n cross-spectra (n channels: the six SO
  channels by default, or a subset, ``filter_static(channels=...)``) averaged over the in-band
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
* Masking iteration (SZiFi's, as szsim's step-3 finder): every peak with
  q > 4 is masked with a binary disc of radius 3 θ_j (pixel centres at
  distance ≤ r, periodic). The discs are built as FFTs of per-scale peak
  maps times integer-disc kernels, which reproduces "distance ≤ r" exactly.
  The covariance is re-estimated outside the discs and the filter is run
  again on the unmasked maps. Estimator (szsim's 'whitened'): each in-band
  mode is whitened with the first-pass covariance of its bin,
  Z = C0^(-1/2) X; the whitened maps are masked; their binned spectra are
  divided by the kept fraction f and un-whitened, C0^(1/2) C_w C0^(1/2).
  The whitened field is nearly white, so the disc edges do not leak the steep
  low-ℓ CMB and atmosphere power to high ℓ (the literal binary-mask estimate
  inflates σ by 16-47% on noise-only tiles).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .. import profiles
from . import instrument as ins
from .flatsky import N, PIX, ARCMIN, grid
from ._util import compact

THETAS15 = np.geomspace(0.5, 15.0, 15)     # arcmin
Q_TH = 4.0
Q_MASK = 4.0          # peaks above this are masked (SZiFi's q_th_noise)
MASK_RADIUS = 3.0     # disc radius in units of the peak's scale θ_j (SZiFi's mask_radius)


class FilterStatic(NamedTuple):
    """Static arrays of the filter (pass as jit arguments)."""
    m_idx: jax.Array        # (n_modes,) int32, flat rfft index, sorted by bin
    m_W: jax.Array          # (n_modes,) float64
    m_binpos: jax.Array     # (n_modes,) int32
    bin_pad: jax.Array      # (n_bins, maxc) int32, position in the mode list or -1
    bin_count: jax.Array    # (n_bins,) float64
    a: jax.Array            # (n_modes, n_ch) float64, f_i B_i(|k|); n_ch channels (6 by default)
    tau_m: jax.Array        # (n_th, n_modes) float64, τ_j on the mode list
    tau_full: jax.Array     # (n_th, N, N/2+1) float32, τ_j on the rfft grid (0 out of band)
    disc_k: jax.Array       # (n_th, N, N/2+1) float32, rfft of the masking disc of scale j (real)


def disc_kernels(thetas=THETAS15, radius=MASK_RADIUS):
    """rfft2 of the binary discs (1 at pixel offsets with dy² + dx² ≤ R_j²,
    R_j = radius θ_j / PIX, periodic, centred on pixel 0), (n_th, N, N/2+1)
    float64. The discs are symmetric, so their transforms are real."""
    d = np.fft.fftfreq(N, d=1.0 / N).round().astype(np.int64)
    d2 = d[:, None] ** 2 + d[None, :] ** 2
    out = np.empty((len(thetas), N, N // 2 + 1))
    for j, t in enumerate(thetas):
        R = radius * float(t) / PIX
        assert R < N // 2
        out[j] = np.fft.rfft2((d2 <= R * R).astype(np.float64)).real
    return out


def filter_static(thetas=THETAS15, mask_radius=MASK_RADIUS, channels=None) -> FilterStatic:
    """``channels``: indices of the SO channels the filter uses, in the maps' order (default all
    six). The maps passed to q_cube and masked_cov must then hold exactly these channels."""
    G = grid()
    n_all = len(ins.SO_LAT["freqs"])
    ch = list(range(n_all)) if channels is None else [int(c) for c in channels]
    if not ch or len(set(ch)) != len(ch) or min(ch) < 0 or max(ch) >= n_all:
        raise ValueError(f"channels must be distinct indices in [0, {n_all}): {channels}")
    f = ins.tsz_f_uK(ins.SO_LAT["freqs"])
    a = (ins.beams(G.m_L) * f[:, None]).T[:, ch]                       # (n_modes, n_ch)
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
        tau_full=jnp.asarray(tau_full),
        disc_k=jnp.asarray(disc_kernels(thetas, mask_radius).astype(np.float32)))


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def _n_channels(X, fs: FilterStatic):
    """The filter's channel count, checked against the maps' (static shapes)."""
    n = fs.a.shape[1]
    if X.shape[0] != n:
        raise ValueError(f"the maps have {X.shape[0]} channels, the filter {n}")
    return n


def binned_cov(Xm, fs: FilterStatic):
    """Binned n×n covariance (float64) from the in-band modes Xm (n, n_modes)."""
    n = _n_channels(Xm, fs)
    X = Xm.astype(jnp.complex128)
    W = fs.m_W
    pad = fs.bin_pad
    ok = pad >= 0
    pidx = jnp.where(ok, pad, 0)
    C = jnp.zeros((pad.shape[0], n, n), dtype=jnp.float64)
    for (i, j) in _pairs(n):
        prod = W * (X[i].real * X[j].real + X[i].imag * X[j].imag)   # (n_modes,)
        v = jnp.sum(jnp.where(ok, prod[pidx], 0.0), axis=1) / fs.bin_count
        C = C.at[:, i, j].set(v)
        if i != j:
            C = C.at[:, j, i].set(v)
    return C


def weights(C, fs: FilterStatic):
    """w(k) = C_b⁻¹ a(k) (n_modes, n) and aᵀC⁻¹a (n_modes,), float64."""
    n = fs.a.shape[1]
    L = jnp.linalg.cholesky(C)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=C.dtype), C.shape)
    Ci = jax.scipy.linalg.cho_solve((L, True), eye)
    Ci = 0.5 * (Ci + jnp.swapaxes(Ci, 1, 2))
    b = fs.m_binpos
    cols = []
    for i in range(n):
        wi = sum(Ci[b, i, j] * fs.a[:, j] for j in range(n))
        cols.append(wi)
    w = jnp.stack(cols, axis=1)
    aCa = jnp.sum(w * fs.a, axis=1)
    return w, aCa


def q_cube(X, fs: FilterStatic, C=None, dtype=jnp.float32):
    """S/N cube (n_th, N, N), σ_j (n_th,) and the covariance used.

    ``X``: channel maps (n, N, N/2+1) in rfft units (complex), the filter's channels.
    ``C``: binned covariance; estimated from X if None.
    """
    G = grid()
    n = _n_channels(X, fs)
    Xm = X.reshape(n, -1)[:, fs.m_idx]
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


# ---- masking iteration ------------------------------------------------------------------

def cluster_mask(peaks: Peaks, fs: FilterStatic, q_mask=Q_MASK):
    """Binary mask (N, N) float32, 1 = kept, and the number of discs: 0 within
    MASK_RADIUS θ_j of every valid peak with q > q_mask (see the module
    docstring). Peaks beyond the buffer (overflow) are not masked."""
    n_th = fs.disc_k.shape[0]
    sel = peaks.valid & (peaks.q > q_mask)
    j = jnp.where(sel, peaks.j, n_th)                          # out of range: dropped
    pts = jnp.zeros((n_th, N, N), jnp.float32).at[j, peaks.iy, peaks.ix].add(
        jnp.float32(1.0), mode="drop")
    cover = jnp.fft.irfft2(jnp.sum(fs.disc_k * jnp.fft.rfft2(pts), axis=0), s=(N, N))
    return (cover < 0.5).astype(jnp.float32), jnp.sum(sel.astype(jnp.int32))


def whitening(C):
    """C^(-1/2) and C^(1/2) per bin (symmetric square roots, float64)."""
    ev, U = jnp.linalg.eigh(C)
    Wm = jnp.einsum("bij,bj,bkj->bik", U, ev ** -0.5, U)
    Wi = jnp.einsum("bij,bj,bkj->bik", U, ev ** 0.5, U)
    return Wm, Wi


def masked_cov(X, mask, C0, fs: FilterStatic):
    """Binned covariance of the tile outside the mask (szsim's 'whitened'
    estimate, see the module docstring) and the kept fraction f.

    ``X``: channel maps (n, N, N/2+1) in rfft units, the filter's channels; ``mask``: (N, N),
    1 = kept; ``C0``: the unmasked binned covariance of the same maps (whitening)."""
    G = grid()
    n = _n_channels(X, fs)
    Wm, Wi = whitening(C0)
    Xm = X.reshape(n, -1)[:, fs.m_idx]
    b = fs.m_binpos
    Wm32 = Wm.astype(jnp.float32)
    Zm = jnp.stack([sum(Wm32[b, i, j] * Xm[j] for j in range(n)) for i in range(n)])
    Z = jnp.zeros((n, G.L.size), Zm.dtype).at[:, fs.m_idx].set(Zm).reshape((n,) + G.L.shape)
    z = jnp.fft.irfft2(Z, s=(N, N)) * mask[None]
    Zk = jnp.fft.rfft2(z).reshape(n, -1)[:, fs.m_idx]
    f = jnp.mean(mask.astype(jnp.float64))
    Cw = binned_cov(Zk, fs) / f
    return jnp.einsum("bij,bjk,bkl->bil", Wi, Cw, Wi), f
