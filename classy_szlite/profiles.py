"""Cluster pressure profile for the map-level simulator (mapsim).

Arnaud et al. (2010) eq. 12 GNFW profile, the set SZiFi and szsim use:

    P(x) = P0 / [(c500 x)^γ (1 + (c500 x)^α)^((β-γ)/α)],   x = r / R500,
    P0 = 8.403 h70^(-3/2), c500 = 1.177, γ = 0.3081, α = 1.0510, β = 5.4905,

truncated at x = 5 in 3D and projected along the line of sight. The template
is normalised to unit central value before any beam: y(θ) = y0 τ(θ/θ500)
with τ(0) = 1. These constants are kept separate from ``cl_yy``'s defaults
(``power_spectrum._A10_*``), which are a different set.

Fourier convention (flat sky): the 2D transform of τ(|θ|/θ500) is
θ500² T(ℓ θ500), with

    T(u) = (4π / p0) ∫_0^5 x² P(x) j0(u x) dx,     p0 = 2 ∫_0^5 P(x) dx,

because the 2D transform of a projected 3D profile is its 3D transform at
k_z = 0. p0 is the central projected value. All of this is host-side numpy,
evaluated once; the simulator consumes static tables built from it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Arnaud et al. 2010, eq. 12 (the set used by SZiFi and szsim)
A10_EQ12 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
X_TRUNC = 5.0   # truncation radius in units of R500 (3D)


def gnfw_shape(x, c500=A10_EQ12["c500"], gamma=A10_EQ12["gamma"],
               alpha=A10_EQ12["alpha"], beta=A10_EQ12["beta"]):
    """Dimensionless GNFW shape P(x)/P0, x = r/R500."""
    cx = c500 * np.asarray(x, dtype=np.float64)
    return 1.0 / (cx ** gamma * (1.0 + cx ** alpha) ** ((beta - gamma) / alpha))


def _gl_panels(a, b, n_panels, n_nodes=16):
    """Composite Gauss-Legendre nodes and weights on [a, b]."""
    xg, wg = np.polynomial.legendre.leggauss(n_nodes)
    edges = np.linspace(a, b, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    x = (0.5 * (hi - lo) * xg + 0.5 * (hi + lo)).ravel()
    w = (0.5 * (hi - lo) * wg).ravel()
    return x, w


def _radial_nodes(n_panels=4000, n_nodes=16):
    """Nodes x in (0, 5] and weights for ∫_0^5 f(x) dx, with x = 5 t^q.

    q = 1/(1-γ) makes x^(-γ) dx smooth in t (P has an x^(-γ) cusp), so both
    p0 and T(u) converge fast. Returns (x, w) with w including dx/dt.
    """
    q = 1.0 / (1.0 - A10_EQ12["gamma"])
    t, wt = _gl_panels(0.0, 1.0, n_panels, n_nodes)
    x = X_TRUNC * t ** q
    w = wt * X_TRUNC * q * t ** (q - 1.0)
    return x, w


@lru_cache(maxsize=None)
def central_projection():
    """p0 = 2 ∫_0^5 P(x)/P0 dx: the projected shape at the centre."""
    x, w = _radial_nodes()
    return float(2.0 * np.sum(w * gnfw_shape(x)))


def hankel_T(u, n_panels=4000, chunk=256):
    """T(u) of the unit-centre template (see the module docstring).

    Direct quadrature of the 3D sinc transform: 64,000 Gauss-Legendre nodes,
    accurate to ~1e-12 of T(0) for u ≤ 3000.
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    x, w = _radial_nodes(n_panels)
    f = w * x * x * gnfw_shape(x) * (4.0 * np.pi / central_projection())
    out = np.empty(u.shape)
    flat_u, flat_out = u.ravel(), out.reshape(-1)
    for a in range(0, flat_u.size, chunk):
        ux = np.outer(flat_u[a:a + chunk], x)
        flat_out[a:a + chunk] = np.sinc(ux / np.pi) @ f        # np.sinc(z) = sin(πz)/(πz)
    return out


def projected_shape(s, n_nodes=2000):
    """τ(s): projected shape normalised to τ(0) = 1, s = θ/θ500 (s ≥ 5 → 0).

    Independent of :func:`hankel_T` (used to cross-check it): the line-of-sight
    integral ∫ P(√(s²+l²)) dl over |l| ≤ √(25 - s²), by Gauss-Legendre in
    l = L t^q (cusp-adapted at s = 0).
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.zeros(s.shape)
    q = 1.0 / (1.0 - A10_EQ12["gamma"])
    t, wt = _gl_panels(0.0, 1.0, max(1, n_nodes // 16), 16)
    for i, si in np.ndenumerate(s):
        if si >= X_TRUNC:
            continue
        L = np.sqrt(X_TRUNC ** 2 - si ** 2)
        lt = L * t ** q
        wl = wt * L * q * t ** (q - 1.0)
        out[i] = 2.0 * np.sum(wl * gnfw_shape(np.sqrt(si * si + lt * lt)))
    return out / central_projection()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# u nodes: log-spaced up to 20, then linear (Δu = 0.1) up to 3000 so that the
# oscillations from the 5 R500 edge (period Δu = 2π/5) are resolved.
U_SPLIT, U_MAX = 20.0, 3000.0


_T_TABLE_VERSION = "a10eq12_x5_v1"


def _cache_dir():
    import os
    d = os.environ.get("CLASSY_SZLITE_CACHE", os.path.expanduser("~/.cache/classy_szlite"))
    os.makedirs(d, exist_ok=True)
    return d


@lru_cache(maxsize=None)
def T_table():
    """(u, T(u)) on the default nodes, float64.

    Cached in memory and on disk ($CLASSY_SZLITE_CACHE, default
    ~/.cache/classy_szlite): the quadrature takes ~50 s. The disk copy is used
    only if its nodes and profile parameters match exactly.
    """
    import os
    u = np.concatenate([[0.0], np.geomspace(1e-4, U_SPLIT, 2400)[:-1],
                        np.arange(U_SPLIT, U_MAX + 0.05, 0.1)])
    pars = np.array([A10_EQ12[k] for k in ("c500", "gamma", "alpha", "beta")] + [X_TRUNC])
    path = os.path.join(_cache_dir(), f"T_table_{_T_TABLE_VERSION}.npz")
    try:
        d = np.load(path)
        if np.array_equal(d["u"], u) and np.array_equal(d["pars"], pars):
            return u, d["T"]
    except (OSError, KeyError, ValueError):
        pass
    T = hankel_T(u)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, u=u, T=T, pars=pars)
    os.replace(tmp, path)
    return u, T


@lru_cache(maxsize=None)
def _T_splines():
    from scipy.interpolate import CubicSpline
    uu, TT = T_table()
    lo = (uu > 0) & (uu <= U_SPLIT)
    hi = uu >= U_SPLIT
    # near u = 0, T(u) = T(0) [1 - u² <s²>_T / 4 + O(u⁴)]: quadratic through T(0), T(u1)
    c2 = (TT[0] - TT[1]) / uu[1] ** 2
    return CubicSpline(np.log(uu[lo]), TT[lo]), CubicSpline(uu[hi], TT[hi]), c2, uu[1]


def T_interp(u):
    """T(u) by cubic-spline interpolation of :func:`T_table` (host-side).

    A cubic spline in ln u on the log nodes (u ≤ 20), a cubic spline in u on
    the linear nodes (20 < u ≤ 3000), and T(0) - c u² below the first node.
    u > 3000 raises: T is not assumed to vanish there.
    """
    u = np.asarray(u, dtype=np.float64)
    if np.any(u > U_MAX * (1 + 1e-12)):
        raise ValueError(f"T_interp: u > {U_MAX} requested")
    cs_lo, cs_hi, c2, u1 = _T_splines()
    T0 = T_table()[1][0]
    out = np.empty(u.shape)
    small = u < u1
    mid = (~small) & (u <= U_SPLIT)
    big = u > U_SPLIT
    out[small] = T0 - c2 * u[small] ** 2
    out[mid] = cs_lo(np.log(u[mid]))
    out[big] = cs_hi(u[big])
    return out
