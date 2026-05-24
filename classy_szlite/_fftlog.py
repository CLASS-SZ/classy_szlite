"""Pure-JAX FFTLog (Hamilton 2000) for the two transforms classy_szlite needs.

Drop-in replacement for mcfit.TophatVar(dim=3) and mcfit.SphericalBessel(nu=0)
that runs end-to-end inside JAX so the forward + grad path is JIT- and
TPU-compatible.

Why we don't just call mcfit
----------------------------
mcfit's runtime path (rfft → multiply → hfft) is already JAX-aware when built
with backend='jax', but the FFT runs in c128 when the input is f64, and the
TPU XLA compiler rejects the f64→c128 cast that appears in the precompute.
This module keeps mcfit's algorithm exactly (same Mellin coefficients, same
lowring condition, same N=2*Nin doubling, same power-law extrap, same
prefac/postfac) but lets the caller pick the runtime dtype: f64 for CPU
(matches mcfit bit-for-bit) or f32 for TPU (~6 digits, fine for halo-model
physics).

All setup (xy, y-grid, Mellin coefficients) is host-side numpy; only the
per-call transform runs in jax. Setup uses scipy.special.loggamma which is
host-only — that's fine because we only build FFTLog instances at import time.
"""
from __future__ import annotations

from math import ceil, exp, log, log2, pi
from cmath import phase
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.special import loggamma


# ----------------------------------------------------------------------
# Runtime dtype selection
# ----------------------------------------------------------------------
# On TPU the XLA compiler cannot lower the f64→c128 cast that FFTLog
# precompute emits, so we default to f32/c64 there. On CPU we stay in
# f64/c128 to match mcfit exactly. Override with $CLASSY_SZLITE_FFTLOG_DTYPE.

def _default_runtime_dtype() -> np.dtype:
    env = os.environ.get("CLASSY_SZLITE_FFTLOG_DTYPE")
    if env:
        return np.dtype(env)
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "cpu"
    return np.dtype("float32") if backend == "tpu" else np.dtype("float64")


# ----------------------------------------------------------------------
# Mellin transforms of the kernels we use
# (verbatim from mcfit.kernels, just inlined here to drop the dep at runtime)
# ----------------------------------------------------------------------

def Mellin_SphericalBesselJ_nu0(z):
    """Mellin transform of j_0(x). Matches mcfit.kernels.Mellin_SphericalBesselJ(0)."""
    return np.exp(np.log(2.0) * (z - 1.5)
                  + loggamma(0.5 * z)
                  - loggamma(0.5 * (3.0 - z)))


def Mellin_TophatSq_dim3(z):
    """Mellin transform of W^2_top-hat in 3D. Matches mcfit.kernels.Mellin_TophatSq(3)."""
    return (2.25 * np.sqrt(np.pi) * (z - 2.0) / (z - 6.0)
            * np.exp(loggamma(0.5 * (z - 4.0)) - loggamma(0.5 * (5.0 - z))))


# ----------------------------------------------------------------------
# Core FFTLog class
# ----------------------------------------------------------------------

class FFTLog:
    """Minimal FFTLog. Mirrors mcfit.mcfit's algorithm exactly.

    Parameters
    ----------
    x : (Nin,) ndarray
        Log-spaced input grid.
    MK : callable
        Mellin transform of the integral kernel (host-side, called once).
    q : float
        Tilt parameter (Hamilton 2000).
    prefac, postfac : array_like
        Multiplicative pre/post factors (functions of x / y respectively),
        e.g. x**3/(2π²) for TophatVar dim=3 or x**3 for SphericalBessel.
    N : int, optional
        Convolution size. Defaults to next power of two ≥ 2·Nin (matches
        mcfit's N=2j default).
    lowring : bool, optional
        Use the low-ringing y-grid alignment. Defaults True.
    dtype : np.dtype, optional
        Runtime float dtype. Default = f64 on CPU, f32 on TPU.

    The full transform G(y) = ∫₀^∞ F(x) · K(xy) · dx/x is then evaluated by
    calling the instance: ``y, G = fftlog(F)``.
    """

    def __init__(self, x, MK, q, prefac=1.0, postfac=1.0,
                 N=None, lowring=True, dtype=None):
        x = np.asarray(x, dtype=np.float64)
        self.x_host = x
        self.Nin = len(x)
        self.q = float(q)
        self.lowring = bool(lowring)
        self.dtype = np.dtype(dtype) if dtype is not None else _default_runtime_dtype()
        self.cdtype = np.complex64 if self.dtype == np.float32 else np.complex128

        # log spacing
        self.Delta = log(x[-1] / x[0]) / (self.Nin - 1)

        # convolution size: default = next power of two ≥ 2·Nin
        if N is None:
            N = 2 ** ceil(log2(2 * self.Nin))
        if N < self.Nin:
            raise ValueError(f"N={N} must be ≥ Nin={self.Nin}")
        self.N = int(N)

        # low-ring xy and output y-grid (reversed input)
        if self.lowring and self.N % 2 == 0:
            ln_xy = self.Delta / pi * phase(MK(self.q + 1j * pi / self.Delta))
        else:
            ln_xy = 0.0
        self.xy = exp(ln_xy)
        self.y_host = exp(ln_xy - self.Delta) / x[::-1]

        # Mellin coefficients on the half-frequency grid
        m = np.arange(0, self.N // 2 + 1)
        u = MK(self.q + 2j * pi / self.N / self.Delta * m)
        u *= np.exp(-2j * pi * ln_xy / self.N / self.Delta * m)
        self._u = jnp.asarray(u, dtype=self.cdtype)

        # xfac / yfac (broadcast into dtype-typed jax arrays)
        prefac_arr = np.broadcast_to(np.asarray(prefac, dtype=np.float64), x.shape)
        postfac_arr = np.broadcast_to(np.asarray(postfac, dtype=np.float64), self.y_host.shape)
        self.xfac = jnp.asarray(prefac_arr * x ** (-self.q), dtype=self.dtype)
        self.yfac = jnp.asarray(postfac_arr * self.y_host ** (-self.q), dtype=self.dtype)

        # padded ratios used by the power-law extrap (only need the first two)
        # — we cache the slice indices for unpad
        self._pad_lo = (self.N - self.Nin) - (self.N - self.Nin) // 2
        self._pad_hi = (self.N - self.Nin) // 2

        # expose y in canonical jax dtype
        self.y = jnp.asarray(self.y_host, dtype=self.dtype)

    def _extrap_pad(self, F, axis):
        """Power-law symmetric padding (matches mcfit's extrap=True)."""
        Npad = self.N - self.Nin
        if Npad == 0:
            return F
        # Left pad uses (F[0], F[1]); right pad uses (F[-2], F[-1]).
        f0 = jnp.take(F, jnp.array(0), axis=axis)
        f1 = jnp.take(F, jnp.array(1), axis=axis)
        fNm2 = jnp.take(F, jnp.array(self.Nin - 2), axis=axis)
        fNm1 = jnp.take(F, jnp.array(self.Nin - 1), axis=axis)

        # power-law ratios (per the mcfit convention)
        rL = f1 / f0                                     # x_n+1 / x_n
        rR = fNm1 / fNm2

        # broadcast shape for the pad blocks
        shape = list(F.shape)
        left_shape  = shape[:axis] + [self._pad_lo] + shape[axis + 1:] if axis >= 0 \
                      else shape  # axis<0 handled separately
        # simpler: assume axis is last (we always call with axis=-1)
        assert axis == -1 or axis == F.ndim - 1, "this _extrap_pad assumes axis=-1"

        # Exponents per mcfit: arange(-_Npad, 0) for left, arange(1, Npad_+1) for right
        eL = jnp.arange(-self._pad_lo, 0, dtype=F.dtype)
        eR = jnp.arange(1, self._pad_hi + 1, dtype=F.dtype)

        # broadcast: f0[..., None] * rL[..., None]**eL[None, ...]
        left  = f0[..., None] * rL[..., None] ** eL
        right = fNm1[..., None] * rR[..., None] ** eR

        return jnp.concatenate([left, F, right], axis=-1)

    def __call__(self, F, axis=-1):
        """Evaluate G(y) = ∫₀^∞ F(x) K(xy) dx/x for F sampled on self.x_host.

        Always uses extrap=True (power-law) padding. F may be batched along
        leading axes; axis must be the last axis (sufficient for our use).
        """
        F = jnp.asarray(F, dtype=self.dtype)
        f = self._extrap_pad(F, axis=-1)                           # (..., N)
        xfac_padded = self._extrap_pad(self.xfac, axis=-1)          # (N,)
        f = xfac_padded * f

        f_k = jnp.fft.rfft(f, axis=-1)                              # (..., N//2+1)
        g_k = f_k * self._u
        g = jnp.fft.hfft(g_k, n=self.N, axis=-1) / self.N           # (..., N)

        # mcfit: _unpad for output uses Npad_=Npad//2 on left
        out_lo = (self.N - self.Nin) // 2
        G = jax.lax.dynamic_slice_in_dim(g, out_lo, self.Nin, axis=-1)
        G = self.yfac * G
        return self.y, G


# ----------------------------------------------------------------------
# Convenience constructors matching mcfit's API
# ----------------------------------------------------------------------

def TophatVar(k, q=1.5, lowring=True, dtype=None) -> FFTLog:
    """σ²(R) = ∫ dk/(2π²) · k² · P(k) · W²(kR).

    Matches mcfit.cosmology.TophatVar(k, q=1.5, lowring=True) — same
    prefac (k³ / (2π²)), same MK, same q.
    """
    k = np.asarray(k, dtype=np.float64)
    prefac = k ** 3 / (2.0 * np.pi ** 2)
    return FFTLog(k, Mellin_TophatSq_dim3, q,
                  prefac=prefac, postfac=1.0,
                  lowring=lowring, dtype=dtype)


def SphericalBessel(x, nu=0, q=1.5, lowring=True, dtype=None) -> FFTLog:
    """Spherical Bessel transform of order nu (only nu=0 implemented here).

    G(y) = ∫ dx · x² · j_nu(xy) · F(x), matching mcfit.transforms.SphericalBessel(x, nu=0).
    """
    if nu != 0:
        raise NotImplementedError(f"_fftlog.SphericalBessel only nu=0 (got nu={nu})")
    x = np.asarray(x, dtype=np.float64)
    prefac = x ** 3
    return FFTLog(x, Mellin_SphericalBesselJ_nu0, q,
                  prefac=prefac, postfac=1.0,
                  lowring=lowring, dtype=dtype)
