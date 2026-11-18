"""Flat periodic tile geometry and its Fourier grid (host-side numpy).

Conventions (as szsim's finder, for the identity tests):

* Tile 600′ × 600′, N = 1200 pixels of 0.5′; pixel (iy, ix) sits at
  (y, x) = (iy, ix) × 0.5′; array axis 0 is y. Periodic.
* Painting grid: NF = 2400 pixels of 0.25′.
* rfft2 layout: axis 0 full (``fftfreq``), axis 1 the half plane 0..N/2.
  Integer wavenumbers (i0, i1); |k| = DK √(i0² + i1²) with DK = 2π/600′ = 36
  exactly.
* "rfft units": X(k) = Σ_p x(p) e^{-ik·p}. A field of power C_ℓ has
  E|X|² = Npx C_ℓ / Ω_pix.
* Band: 100 ≤ |k| ≤ 20000, a sharp cut (n² from 8 to 308641).
* Covariance bins of width 72 = 2 DK (SZiFi's bin_fac = 4): bin = ⌊√n² / 2⌋,
  with an exact integer square root.
* rfft multiplicities W = 2, except the i1 = 0 and i1 = N/2 columns (W = 1):
  sums over the half plane weighted by W are full-plane sums.
"""
from __future__ import annotations

import numpy as np

ARCMIN = np.pi / 180.0 / 60.0          # rad

N, PIX = 1200, 0.5                      # coarse grid [pixels, arcmin]
NF, PIXF = 2400, 0.25                   # painting grid
LBOX = N * PIX                          # 600 arcmin
DK = 2.0 * np.pi / (LBOX * ARCMIN)      # 36.0
LMIN, LMAX = 100.0, 20000.0
BIN_DK = 2                              # bin width 72 = 2 DK


def isqrt(n):
    """Exact integer square root of a non-negative int64 array."""
    n = np.asarray(n, dtype=np.int64)
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r -= (r * r > n)
    r += ((r + 1) * (r + 1) <= n)
    return r


class FlatGrid:
    """Fourier grid of an n × n periodic tile of pixel ``pix`` arcmin.

    Attributes (numpy): ``i0`` (n,), ``i1`` (n//2+1,), ``n2``, ``L`` [rad⁻¹],
    ``W`` (rfft multiplicities) and ``sel`` (in-band) of shape (n, n//2+1);
    ``ib`` (bin index); the in-band mode list ``m_idx`` (flat indices, sorted
    by bin) with ``m_L``, ``m_W``, ``m_bin``; the padded per-bin layout
    ``bin_pad`` (n_bins_used, max_count) of positions in the mode list, with
    -1 for padding, and ``bin_count`` (W-weighted counts).
    """

    def __init__(self, n=N, pix=PIX, lmin=LMIN, lmax=LMAX, bin_dk=BIN_DK):
        self.n, self.pix = n, pix
        self.lbox = n * pix
        self.dk = 2.0 * np.pi / (self.lbox * ARCMIN)
        self.Om = (pix * ARCMIN) ** 2                     # sr
        self.Npx = n * n
        self.i0 = np.fft.fftfreq(n, d=1.0 / n).round().astype(np.int64)
        self.i1 = np.arange(n // 2 + 1, dtype=np.int64)
        self.n2 = self.i0[:, None] ** 2 + self.i1[None, :] ** 2
        self.L = self.dk * np.sqrt(self.n2.astype(np.float64))
        self.k0 = (self.i0 * self.dk)[:, None] * np.ones((1, n // 2 + 1))
        self.k1 = np.ones((n, 1)) * (self.i1 * self.dk)[None, :]
        w = np.full(n // 2 + 1, 2.0)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        self.W = np.broadcast_to(w[None, :], self.L.shape).copy()
        n2min = int(np.ceil((lmin / self.dk) ** 2 - 1e-9))
        n2max = int(np.floor((lmax / self.dk) ** 2 + 1e-9))
        self.sel = (self.n2 >= n2min) & (self.n2 <= n2max)
        self.ib = isqrt(self.n2) // bin_dk

        # in-band modes, sorted by bin (stable: row-major within a bin)
        flat = np.flatnonzero(self.sel.ravel())
        order = np.argsort(self.ib.ravel()[flat], kind="stable")
        self.m_idx = flat[order]
        self.m_L = self.L.ravel()[self.m_idx]
        self.m_W = self.W.ravel()[self.m_idx]
        self.m_bin = self.ib.ravel()[self.m_idx]
        self.bins_used, starts, counts = np.unique(self.m_bin, return_index=True,
                                                   return_counts=True)
        self.n_bins = len(self.bins_used)
        self.m_binpos = np.searchsorted(self.bins_used, self.m_bin)   # 0..n_bins-1
        maxc = int(counts.max())
        pad = -np.ones((self.n_bins, maxc), dtype=np.int64)
        for b, (s0, c) in enumerate(zip(starts, counts)):
            pad[b, :c] = np.arange(s0, s0 + c)
        self.bin_pad = pad
        self.bin_count = np.bincount(self.m_binpos, weights=self.m_W,
                                     minlength=self.n_bins)
        self.bin_L = (self.bins_used + 0.5) * bin_dk * self.dk        # bin centres

    @property
    def n_modes(self):
        return self.m_idx.size


_GRID = {}


def grid(n=N, pix=PIX):
    """Cached FlatGrid."""
    key = (n, pix)
    if key not in _GRID:
        _GRID[key] = FlatGrid(n, pix)
    return _GRID[key]


def fine_to_coarse_rows(n=N, nf=NF):
    """Rows of the fine rfft2 array that hold the coarse grid's modes
    (|i0| < n/2), in the coarse ``fftfreq`` order; columns are 0..n/2."""
    return np.r_[0:n // 2, nf - n // 2:nf]
