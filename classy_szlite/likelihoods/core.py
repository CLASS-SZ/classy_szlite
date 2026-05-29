"""Pure-JAX implementations of four cobaya CMB likelihoods.

See :mod:`classy_szlite.likelihoods` for the public API. All functions accept a
``cls_dict`` of D_ell = ℓ(ℓ+1)Cℓ/(2π) in μK², indexed by integer ℓ from 0
(so ``cls_dict["tt"][30]`` is D_ℓ at ℓ = 30).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._data import jnp_table, load, load_fg_components

_TCMB_UK2 = (2.7255e6) ** 2

# ─────────────────────────────────────────────────────────────────────────
# Lazy table builders
# ─────────────────────────────────────────────────────────────────────────

_TABLES: dict[str, object] = {}


def _lowTT_tables():
    if "lowTT" not in _TABLES:
        d = load()
        _TABLES["lowTT"] = (
            int(d["lowTT_lmin"]), int(d["lowTT_lmax"]),
            jnp_table(d["lowTT_mu"]), jnp_table(d["lowTT_covinv"]),
            float(d["lowTT_offset"]),
            jnp_table(d["lowTT_spline_cl"]),
            jnp_table(d["lowTT_spline_val"]),
            jnp_table(d["lowTT_spline_dval"]),
        )
    return _TABLES["lowTT"]


def _sroll2_tables():
    if "sroll2" not in _TABLES:
        d = load()
        _TABLES["sroll2"] = (
            int(d["sroll2_lmin"]), int(d["sroll2_lmax"]),
            float(d["sroll2_step"]),
            jnp_table(d["sroll2_probEE"]),
        )
    return _TABLES["sroll2"]


def _plac_tables():
    if "plac" not in _TABLES:
        d = load()
        _TABLES["plac"] = (
            int(d["plac_lmax"]),
            jnp_table(d["plac_win_TT"]),
            jnp_table(d["plac_win_TE"]),
            jnp_table(d["plac_win_EE"]),
            jnp_table(d["plac_X_data"]),
            jnp_table(d["plac_invcov"]),
        )
    return _TABLES["plac"]


def _mflike_tables():
    if "mflike" not in _TABLES:
        d = load()
        experiments = [str(e) for e in d["mflike_experiments"]]
        exp_to_idx = {e: i for i, e in enumerate(experiments)}
        meta_list = []
        for i in range(int(d["mflike_n_meta"])):
            pol = str(d[f"meta_{i}_pol"])
            hasYX = bool(d[f"meta_{i}_hasYX"])
            t1 = str(d[f"meta_{i}_t1"])
            t2 = str(d[f"meta_{i}_t2"])
            meta_list.append({
                "pol": pol, "hasYX": hasYX,
                "t1_idx": exp_to_idx[t1], "t2_idx": exp_to_idx[t2],
                "ids": np.asarray(d[f"meta_{i}_ids"]),
                "bpw_w": jnp_table(d[f"meta_{i}_bpw_weight"]),
            })
        _TABLES["mflike"] = (
            jnp_table(d["mflike_data_vec"]),
            jnp_table(d["mflike_inv_cov"]),
            jnp_table(d["mflike_l_bpws"]),
            float(d["mflike_logp_const"]),
            {"tt": jnp_table(d["mflike_fg_bestfit_tt"]),
             "te": jnp_table(d["mflike_fg_bestfit_te"]),
             "ee": jnp_table(d["mflike_fg_bestfit_ee"])},
            meta_list, experiments,
        )
    return _TABLES["mflike"]


# ─────────────────────────────────────────────────────────────────────────
# 1. Planck 2018 low-ℓ TT (Commander)
# ─────────────────────────────────────────────────────────────────────────


def _interp1d(x_query, x_tab, y_tab):
    """Linear interpolation of scalar ``x_query`` into table (x_tab, y_tab)."""
    idx = jnp.searchsorted(x_tab, x_query) - 1
    idx = jnp.clip(idx, 0, x_tab.shape[0] - 2)
    x0, x1 = x_tab[idx], x_tab[idx + 1]
    y0, y1 = y_tab[idx], y_tab[idx + 1]
    return y0 + (x_query - x0) / (x1 - x0) * (y1 - y0)


def chi2_lowTT(cls_dict, A_planck: float = 1.0):
    """Planck 2018 Commander low-ℓ TT χ² (ℓ = 2..29, spline-based, non-Gaussian)."""
    lmin, lmax, mu, covinv, offset, spline_cl, spline_val, spline_dval = _lowTT_tables()
    theory = cls_dict["tt"][lmin: lmax + 1] / (A_planck ** 2)
    n = theory.shape[0]

    def _per_ell(i):
        x_i = _interp1d(theory[i], spline_cl[i], spline_val[i])
        dxdcl_i = _interp1d(theory[i], spline_cl[i], spline_dval[i])
        return x_i, jnp.log(jnp.where(dxdcl_i > 0, dxdcl_i, 1e-300))

    xs, log_dxdcl = jax.vmap(_per_ell)(jnp.arange(n))
    delta = xs - mu
    logl = jnp.sum(log_dxdcl) - 0.5 * delta @ covinv @ delta - offset
    return -2.0 * logl


# ─────────────────────────────────────────────────────────────────────────
# 2. Planck 2018 low-ℓ EE (sroll2)
# ─────────────────────────────────────────────────────────────────────────


def chi2_sroll2(cls_dict, A_planck: float = 1.0):
    """Planck 2018 sroll2 low-ℓ EE χ² (ℓ = 2..29, table lookup, smoothly interpolated)."""
    lmin, lmax, step, probEE = _sroll2_tables()
    theory = cls_dict["ee"][lmin: lmax + 1] / (A_planck ** 2)
    pos = jnp.clip(theory / step, 0.0, probEE.shape[0] - 1.0 - 1e-9)
    i0 = pos.astype(jnp.int32)
    frac = pos - i0.astype(pos.dtype)
    col = jnp.arange(theory.shape[0])
    log_probs = probEE[i0, col] + frac * (probEE[i0 + 1, col] - probEE[i0, col])
    return -2.0 * jnp.sum(log_probs)


# ─────────────────────────────────────────────────────────────────────────
# 3. Planck plik_lite v22 with per-spectrum ℓ-cuts (PlanckActCut)
# ─────────────────────────────────────────────────────────────────────────


def chi2_plac(cls_dict, A_planck: float = 1.0):
    """Planck plik_lite v22 χ² with custom ℓ-cuts (ACT-DR6 + Planck EDE config)."""
    lmax, win_TT, win_TE, win_EE, X_data, invcov = _plac_tables()
    cl_TT = win_TT @ cls_dict["tt"][:lmax + 1]
    cl_TE = win_TE @ cls_dict["te"][:lmax + 1]
    cl_EE = win_EE @ cls_dict["ee"][:lmax + 1]
    cl_theory = jnp.concatenate([cl_TT, cl_TE, cl_EE]) / (A_planck ** 2)
    diff = X_data - cl_theory
    return diff @ invcov @ diff


# ─────────────────────────────────────────────────────────────────────────
# 4. ACT DR6 multifrequency — fixed-foreground variant
# ─────────────────────────────────────────────────────────────────────────


# Per-experiment-channel best-fit calibration used as default. Sampling
# calibrations is fine: just override the relevant entries in ``params``.
_MFLIKE_DEFAULT_NUIS = dict(
    calG_all=1.0013228,
    calE_dr6_pa4_f220=1.0,
    calE_dr6_pa5_f090=0.98883908, calE_dr6_pa5_f150=0.99891013,
    calE_dr6_pa6_f090=0.99875697, calE_dr6_pa6_f150=0.99889845,
    cal_dr6_pa4_f220=0.97804664, cal_dr6_pa5_f090=1.0004643,
    cal_dr6_pa5_f150=0.99900579, cal_dr6_pa6_f090=1.0001087,
    cal_dr6_pa6_f150=1.0011954,
)


def _build_cal_arrays(nuis, experiments):
    calG = jnp.asarray(nuis["calG_all"], dtype=jnp.float64)

    def cal_t(exp):
        return (1.0 / calG) / jnp.asarray(nuis.get(f"cal_{exp}", 1.0), dtype=jnp.float64)

    def cal_e(exp):
        return ((1.0 / calG)
                / jnp.asarray(nuis.get(f"cal_{exp}", 1.0), dtype=jnp.float64)
                / jnp.asarray(nuis.get(f"calE_{exp}", 1.0), dtype=jnp.float64))

    cal_T = jnp.array([cal_t(e) for e in experiments])
    cal_E = jnp.array([cal_e(e) for e in experiments])
    return cal_T, cal_E


def _mflike_kernel(cls_dict, fg_dict, params: dict | None):
    """Inner kernel shared between :func:`chi2_mflike` and :func:`chi2_mflike_v2`."""
    data_vec, inv_cov, l_bpws, logp_const, _, meta_list, experiments = _mflike_tables()
    nuis = dict(_MFLIKE_DEFAULT_NUIS)
    if params:
        nuis.update(params)
    cal_T, cal_E = _build_cal_arrays(nuis, experiments)
    l_bpws_int = l_bpws.astype(jnp.int32)
    dls_cmb = {p: cls_dict[p][l_bpws_int] for p in ("tt", "te", "ee")}
    ps_vec = jnp.zeros(data_vec.shape[0])
    for m in meta_list:
        pol = m["pol"]
        i1, i2 = m["t1_idx"], m["t2_idx"]
        hasYX = m["hasYX"]
        # The data may be stored as ET, in which case (t2, t1) is (T, E).
        fg_i, fg_j = (i2, i1) if hasYX else (i1, i2)
        total_dl = dls_cmb[pol] + fg_dict[pol][fg_i, fg_j]
        if pol == "tt":
            cal = cal_T[i1] * cal_T[i2]
        elif pol == "te":
            cal = (cal_T[i2] * cal_E[i1]) if hasYX else (cal_T[i1] * cal_E[i2])
        elif pol == "ee":
            cal = cal_E[i1] * cal_E[i2]
        else:  # pragma: no cover
            cal = jnp.ones(())
        binned = m["bpw_w"].T @ (total_dl * cal)
        ps_vec = ps_vec.at[m["ids"]].set(binned)
    delta = data_vec - ps_vec
    return delta @ inv_cov @ delta - 2.0 * logp_const


def chi2_mflike(cls_dict, nuis_dict: dict | None = None):
    """ACT DR6 multifrequency χ² with foregrounds *fixed at best-fit*.

    Use this when sampling cosmology + calibrations only (the foreground
    parameters then have no effect on the likelihood). Foregrounds are read
    from ``mflike_fg_bestfit_{tt,te,ee}`` arrays in the data npz.
    """
    *_, fg_fixed, _, _ = _mflike_tables()
    return _mflike_kernel(cls_dict, fg_fixed, nuis_dict)


def chi2_mflike_v2(cls_dict, params: dict):
    """ACT DR6 multifrequency χ² with sampled foreground amplitudes.

    The foreground model is the linearised template combination from
    :func:`classy_szlite.likelihoods.foreground.fg_totals_jax` — exact in the
    11 amplitudes (``a_kSZ, a_p, a_s, a_tSZ, a_c, a_gtt, a_gee, a_psee,
    a_pste, a_gte, xi``), with SED tilts (``alpha_*, beta_*, T_*``) and
    bandpass shifts fixed at the chain best-fit. This is the version used by
    the EDE NUTS demo.
    """
    from .foreground import fg_totals_jax  # local import keeps base import light
    fg = fg_totals_jax(params)
    return _mflike_kernel(cls_dict, fg, params)


# ─────────────────────────────────────────────────────────────────────────
# Convenience: cosmology → Cls → total χ²
# ─────────────────────────────────────────────────────────────────────────


def _cosmo_to_cls(cosmo_dict, ell_convention: str = "classy_szfast"):
    """Build the D_ℓ-from-ℓ=0 dict that the χ² functions consume.

    ``cosmo_dict`` accepts either the emulator-native key ``ln10^{10}A_s`` or
    the convenience alias ``ln10_10_As``.
    """
    from classy_szlite.api import cl_TTTEEE_jax
    out = cl_TTTEEE_jax(cosmo_dict, ell_factor=False, ell_convention=ell_convention)
    ell = out["ell"]
    ell_min = 2 if ell_convention == "classy_szfast" else 1
    dl = {}
    for s in ("tt", "te", "ee"):
        dl_s = ell * (ell + 1) * out[s] / (2.0 * jnp.pi) * _TCMB_UK2
        prefix = jnp.zeros(ell_min, dtype=dl_s.dtype)
        dl[s] = jnp.concatenate([prefix, dl_s])
    return dl


def total_chi2(cosmo, params: dict | None = None,
               use_v2_foregrounds: bool = True,
               ell_convention: str = "classy_szfast"):
    """End-to-end JAX χ² for the four bundled cobaya likelihoods.

    Parameters
    ----------
    cosmo
        Either a :class:`classy_szlite.CosmoParams` or a dict with keys
        ``H0, omega_b, omega_cdm, ln10_10_As, n_s, tau_reio, fEDE, log10z_c,
        thetai_scf``.
    params
        Nuisance parameters (calibrations, foreground amplitudes if
        ``use_v2_foregrounds=True``, plus ``A_planck`` for low-ℓ Planck).
    use_v2_foregrounds
        If ``True`` (default), use :func:`chi2_mflike_v2` so foreground
        amplitudes are sampled. If ``False``, foregrounds are fixed at the
        chain best-fit (faster, but doesn't marginalise over them).
    ell_convention
        Forwarded to :func:`cl_TTTEEE_jax`; the default matches the published
        chains.
    """
    from classy_szlite.api import cosmo_to_dict
    if isinstance(cosmo, dict):
        cosmo_dict = dict(cosmo)
    elif hasattr(cosmo, "__class__") and cosmo.__class__.__name__ == "CosmoParams":
        cosmo_dict = cosmo_to_dict(cosmo)
    else:
        cosmo_dict = dict(cosmo)
    cls = _cosmo_to_cls(cosmo_dict, ell_convention=ell_convention)
    p = dict(params or {})
    A_planck = p.get("A_planck", p.get("calG_all", 1.0))
    chi2 = (chi2_lowTT(cls, A_planck=A_planck)
            + chi2_sroll2(cls, A_planck=A_planck)
            + chi2_plac(cls, A_planck=A_planck))
    if use_v2_foregrounds and load_fg_components() is not None:
        chi2 = chi2 + chi2_mflike_v2(cls, p)
    else:
        chi2 = chi2 + chi2_mflike(cls, nuis_dict=p)
    return chi2
