"""Linearised JAX foreground model for the ACT DR6 multifrequency likelihood.

The model anchors on the foreground total computed by cobaya at the chain
best-fit (read from ``mflike_fg_bestfit_{tt,te,ee}`` in the main npz) and
adds the deviation from each amplitude as
``(a_c / a_c_bf − 1) × per-component template``, where the per-component
templates were extracted from a fully-initialised ``mflike.BandpowerForeground``
with the bandpass shifts applied (see ``extract_data.py``).

This makes the foreground exact at the best-fit point and exact in every
amplitude parameter that enters linearly. The ``tSZ_and_CIB`` cross-spectrum
is handled separately so the non-linear ``−ξ √(a_tSZ a_c)`` dependence is
also captured exactly. SED tilts (``alpha_*, beta_*``) and bandpass shifts
are held fixed at their best-fit values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._data import jnp_table, load_fg_components

# Best-fit foreground amplitudes from the published ACT-DR6 + Planck EDE chain.
_BF = dict(
    a_kSZ=1.2858443, a_p=7.6669211, a_s=2.9096742,
    a_tSZ=3.4983345, a_c=4.0215689,
    a_gtt=7.9189003, a_gte=0.41908216, a_gee=0.16687108,
    a_psee=0.0096106134, a_pste=-0.025321767, xi=0.08599627,
)

_TABLES = None


def _load():
    """Build a cached dict of per-component templates as jnp arrays."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    d = load_fg_components()
    if d is None:
        raise FileNotFoundError(
            "mflike_fg_components.npz not found; "
            "fg_totals_jax requires it. Run classy_szlite.likelihoods.extract_data."
        )

    def get(pol, comp):
        return jnp_table(d[f"mflike_fg_{pol}_{comp}_bf"])

    tt = {c: get("tt", c) for c in
          ("kSZ", "cibp", "radio", "tSZ", "cibc", "dust", "tSZxCIB")}
    te = {c: get("te", c) for c in ("radio", "dust")}
    ee = {c: get("ee", c) for c in ("radio", "dust")}

    # The "fg_bestfit" total from the main likelihood_data.npz: anchor point.
    from . import _data as _d
    main = _d.load()
    fg_bf = {p: jnp_table(main[f"mflike_fg_bestfit_{p}"])
             for p in ("tt", "te", "ee")}

    bf = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _BF.items()}
    bf["amp_tSZxCIB"] = -bf["xi"] * jnp.sqrt(bf["a_tSZ"] * bf["a_c"])
    _TABLES = (tt, te, ee, bf, fg_bf)
    return _TABLES


def fg_totals_jax(params: dict) -> dict:
    """Total foreground D_ℓ μK² per ``(pol, exp_i, exp_j, ℓ)`` given amplitudes.

    Returns a dict ``{"tt": (5,5,n_ell), "te": ..., "ee": ...}`` consistent
    with the layout expected by :func:`chi2_mflike_v2`.
    """
    tt, te, ee, bf, fg_bf = _load()

    # Linear deviation factors (r − 1) so we exactly reproduce fg_bf at BF.
    d_kSZ = jnp.asarray(params["a_kSZ"]) / bf["a_kSZ"] - 1.0
    d_p   = jnp.asarray(params["a_p"])   / bf["a_p"]   - 1.0
    d_s   = jnp.asarray(params["a_s"])   / bf["a_s"]   - 1.0
    d_tSZ = jnp.asarray(params["a_tSZ"]) / bf["a_tSZ"] - 1.0
    d_c   = jnp.asarray(params["a_c"])   / bf["a_c"]   - 1.0
    d_gtt = jnp.asarray(params["a_gtt"]) / bf["a_gtt"] - 1.0
    d_pste = jnp.asarray(params["a_pste"]) / bf["a_pste"] - 1.0
    d_gte  = jnp.asarray(params["a_gte"])  / bf["a_gte"]  - 1.0
    d_psee = jnp.asarray(params["a_psee"]) / bf["a_psee"] - 1.0
    d_gee  = jnp.asarray(params["a_gee"])  / bf["a_gee"]  - 1.0

    # Non-linear amplitude of the tSZxCIB cross-spectrum.
    amp_x = -jnp.asarray(params["xi"]) * jnp.sqrt(
        jnp.asarray(params["a_tSZ"]) * jnp.asarray(params["a_c"]))
    d_cross = amp_x / bf["amp_tSZxCIB"] - 1.0

    return {
        "tt": fg_bf["tt"]
              + d_kSZ * tt["kSZ"] + d_p * tt["cibp"] + d_s * tt["radio"]
              + d_tSZ * tt["tSZ"] + d_c * tt["cibc"] + d_cross * tt["tSZxCIB"]
              + d_gtt * tt["dust"],
        "te": fg_bf["te"] + d_pste * te["radio"] + d_gte * te["dust"],
        "ee": fg_bf["ee"] + d_psee * ee["radio"] + d_gee * ee["dust"],
    }
