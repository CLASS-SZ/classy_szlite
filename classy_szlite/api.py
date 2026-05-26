"""High-level public API: derived params, CMB Cls, matter Pk, distances,
tSZ Cl^yy. All functions take a :class:`classy_szlite.params.CosmoParams`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ._registry import get_emulator, DEFAULT_COSMO
from .params import CosmoParams, ProfileParamsA10
from .cosmology import build as build_cosmo_grids, get_pk, get_pknl, get_distances
from .hmf import build_halo_grids
from .power_spectrum import cl_yy_1h_2h, cl_yy_1h_trispectrum

jax.config.update("jax_enable_x64", True)


def cosmo_to_dict(cosmo: CosmoParams) -> dict:
    """Convert ``CosmoParams`` to the emulator-style dict (with curly-brace key).

    Values pass through as-is (float or jax.Array) so the returned dict is
    JAX-traceable — pass a ``CosmoParams`` of tracers to ``jax.grad``.
    """
    return {
        "omega_b":       cosmo.omega_b,
        "omega_cdm":     cosmo.omega_cdm,
        "H0":            cosmo.H0,
        "tau_reio":      cosmo.tau_reio,
        "ln10^{10}A_s":  cosmo.ln10_10_As,
        "n_s":           cosmo.n_s,
        "m_ncdm":        cosmo.m_ncdm,
        "N_ur":          cosmo.N_ur,
        "fEDE":          cosmo.fEDE,
        "log10z_c":      cosmo.log10z_c,
        "thetai_scf":    cosmo.thetai_scf,
        "r":             cosmo.r,
    }


# ---------------------------------------------------------------------------
# Derived parameters (σ8, Ω_m, S8)
# ---------------------------------------------------------------------------

def derived(cosmo: CosmoParams) -> dict:
    """Derived parameters: ``sigma_8``, ``Omega_m``, ``S8``.

    Also returns the full 17-element DER emulator output as ``'der_full'``
    (σ8 is at index 1; consult the CosmoPower DER training script for the
    full list).
    """
    em = get_emulator('der')
    full = dict(DEFAULT_COSMO); full.update(cosmo_to_dict(cosmo))
    p_in = {k: [full[k]] for k in em.parameters}
    der = 10.0 ** np.asarray(em.predict(p_in)).flatten()

    sigma_8 = float(der[1])
    h = float(cosmo.H0) / 100.0
    # Σmν = 3 × m_ncdm for ede-v2 ν convention (3 degenerate ν)
    sum_mnu = 3.0 * float(cosmo.m_ncdm)
    Omega_m = (float(cosmo.omega_b) + float(cosmo.omega_cdm)
               + sum_mnu / 93.14) / h ** 2
    S8 = sigma_8 * (Omega_m / 0.3) ** 0.5
    return {"sigma_8": sigma_8, "Omega_m": Omega_m, "S8": S8, "der_full": der}


# ---------------------------------------------------------------------------
# CMB angular power spectra
# ---------------------------------------------------------------------------

# TT, EE, PP emulators output log10(prefactored Cl); TE outputs Cl directly
# (it can be negative). Recovery factor: ede-v2 uses 1/ell² to get raw Cl.
_CMB_LOG_CONVENTION = {"tt": True, "ee": True, "pp": True, "te": False}


def cl_TTTEEE(cosmo: CosmoParams,
              spectra: tuple[str, ...] = ("tt", "te", "ee"),
              ell_factor: bool = True) -> dict:
    """CMB angular power spectra.

    Returns a dict with keys ``'ell'`` and the requested spectra
    (``'tt','te','ee'``). Values are **dimensionless** — multiply by
    ``Tcmb_uK² = (2.7255e6)²`` to convert to μK².

    ``ell_factor`` (default ``True``) — return ``D_ell = ell(ell+1) Cl / (2π)``;
    ``False`` returns raw Cl.
    """
    full = dict(DEFAULT_COSMO); full.update(cosmo_to_dict(cosmo))

    out = {}
    ell = None
    for spec in spectra:
        if spec not in _CMB_LOG_CONVENTION:
            raise ValueError(f"Unknown spectrum {spec!r}. Pick from {tuple(_CMB_LOG_CONVENTION)}.")
        em = get_emulator(spec)
        p_in = {k: [full[k]] for k in em.parameters}
        pred = np.asarray(em.predict(p_in)).flatten()
        if _CMB_LOG_CONVENTION[spec]:
            pred = 10.0 ** pred
        out[spec] = pred
        if ell is None:
            ell = np.asarray(em.modes)

    # ede-v2 recovery factor: raw → Cl
    factor_to_Cl = 1.0 / (ell ** 2)
    for s in spectra:
        out[s] = out[s] * factor_to_Cl

    out["ell"] = ell
    if ell_factor:
        fac_dl = ell * (ell + 1) / (2.0 * np.pi)
        for s in spectra:
            out[s] = out[s] * fac_dl
    return out


# ---------------------------------------------------------------------------
# Pk, Pnl, distances
# ---------------------------------------------------------------------------

def Pk(cosmo: CosmoParams, z_arr):
    """Linear P(k, z) — returns ``(k, pk(z, k))``."""
    return get_pk(cosmo_to_dict(cosmo), z_arr)


def Pnl(cosmo: CosmoParams, z_arr):
    """Non-linear P(k, z) (HMcode) — returns ``(k, pk(z, k))``."""
    return get_pknl(cosmo_to_dict(cosmo), z_arr)


def distances(cosmo: CosmoParams, z_arr):
    """Returns ``(Hz, chi, Da)``. ``Hz`` is H(z)/c in 1/Mpc; distances in Mpc."""
    return get_distances(cosmo_to_dict(cosmo), z_arr)


# ---------------------------------------------------------------------------
# tSZ Cl^yy (halo-model, Arnaud 2010 profile)
# ---------------------------------------------------------------------------

def cl_yy(cosmo: CosmoParams, profile: ProfileParamsA10, ell,
          z_grid: jax.Array | None = None,
          n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
          n_m: int = 200, delta_crit: float = 500.0):
    """Halo-model tSZ angular power spectrum (full pipeline per call).

    Returns ``(cl_1h, cl_2h)`` — dimensionless C_ell. Multiply by
    ``ell*(ell+1)/(2π)*1e12`` to get ``D_ell × 1e12``.

    For MCMC sampling only profile parameters at fixed cosmology, use
    :func:`cl_yy_factory` instead — ~3× faster.
    """
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    pp_dict = profile._asdict()
    cl_1h, cl_2h = cl_yy_1h_2h(jnp.asarray(ell), cg, hg, cosmo_dict,
                                profile='arnaud10', profile_params=pp_dict)
    return cl_1h, cl_2h


def cl_yy_factory(cosmo: CosmoParams, ell,
                  z_grid: jax.Array | None = None,
                  n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
                  n_m: int = 200, delta_crit: float = 500.0,
                  x_outSZ: float = 4.0, c500_fiducial: float = 1.156):
    """Fixed-cosmology fast-path: precompute the heavy bits, get a closure.

    Builds ``CosmoGrids`` (emulators → P_lin, distances, σ(R)) and
    ``HaloGrids`` (Tinker 08 HMF, bias) **once**, then returns:

        ev(profile) -> (cl_1h, cl_2h)

    A subsequent ``ev(profile)`` call only runs the ``cl_yy_1h_2h``
    halo-model integration — typically ~5 ms per call. Intended for MCMC
    over profile / nuisance parameters with fixed cosmology.

    Parameters
    ----------
    x_outSZ : float, optional
        Outer truncation radius of the GNFW pressure profile in units of
        r_500c (i.e. x = r / r_500c). The FT look-up table u-grid runs
        from 1e-5 to ``c500_fiducial * x_outSZ`` (in u = c500 * x units),
        matching the classy_sz ``x_outSZ`` convention. Default 4.0
        (literature / ACT-DR6 may26 convention).
    c500_fiducial : float, optional
        c500 used to convert x_outSZ → u_max at table-build time.
        Should match the c500 you will pass in ProfileParamsA10.
        Default 1.156 (Arnaud et al. 2010).
    """
    import numpy as _np
    import warnings as _warnings
    import mcfit as _mcfit

    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    ell_jax = jnp.asarray(ell)

    # Build a FT table with sharp profile truncation at x_outSZ.
    #
    # classy_sz sets the GNFW profile to exactly zero for r/r_500c > x_outSZ
    # before FFT-transforming.  We replicate this by building a u-grid that
    # runs exactly from 1e-5 to u_max = c500_fiducial * x_outSZ (256 points)
    # and transforming with extrap=False (no power-law extrapolation beyond the
    # grid edges).  This avoids two failure modes:
    #
    #   - extrap=True on the truncated grid: power-law extrapolation beyond
    #     u_max incorrectly adds contribution from the non-zero profile tail
    #     for slowly decaying (low-beta) kernels.
    #   - extrap=True / extrap=False on the full grid (1e-5…100) with trailing
    #     zeros: mcfit's ratio-based extrapolation computes 0/0 → NaN, and
    #     extrap=False with a sharp zero edge introduces Gibbs ringing.
    #
    # Truncated grid + extrap=False replicates classy_sz's x_outSZ truncation
    # to within <1% (beta>5), <3% (beta~3), <8% (beta~1.7).
    from .power_spectrum import (_A10_GAMMA as _G0, _A10_ALPHA as _AL0)

    _u_max         = float(c500_fiducial) * float(x_outSZ)
    _u_grid_trunc  = _np.geomspace(1e-5, _u_max, 256)   # ends exactly at u_max
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="use backend='jax' if desired")
        _sb_trunc = _mcfit.SphericalBessel(_u_grid_trunc, backend='jax')
    _local_log_s   = jnp.asarray(_np.log(_np.asarray(_sb_trunc.y)))

    def _build_local_g(gamma, alpha, beta):
        """Build sharp-truncated g(s) for given shape params (extrap=False)."""
        u = jnp.asarray(_u_grid_trunc)
        kernel = (u ** (-gamma)
                  * (1.0 + u ** alpha) ** ((gamma - beta) / alpha))
        _, g = _sb_trunc(kernel, extrap=False)
        return g * jnp.sqrt(jnp.pi / 2.0)

    @jax.jit
    def evaluate(profile: ProfileParamsA10):
        pp = profile._asdict()
        gamma = pp.get('gamma', _G0)
        alpha = pp.get('alpha', _AL0)
        beta  = pp.get('beta',  5.4807)
        # Build truncated g-table for this (gamma, alpha, beta) combo.
        # JAX traces through _sb_trunc since mcfit backend='jax' uses
        # rfft / multiply / hfft — all JAX primitives.
        g_tab = _build_local_g(gamma, alpha, beta)
        # Inject truncated tables into profile_params so _y_ell_grid uses them.
        pp_ext = dict(pp, _g_table=g_tab, _log_s_grid=_local_log_s)
        cl_1h, cl_2h = cl_yy_1h_2h(
            ell_jax, cg, hg, cosmo_dict,
            profile='arnaud10', profile_params=pp_ext,
        )
        return cl_1h, cl_2h

    return evaluate


def cl_yy_trispectrum(cosmo: CosmoParams, profile: ProfileParamsA10, ell,
                       z_grid: jax.Array | None = None,
                       n_z: int = 100, m_min: float = 1e10, m_max: float = 3.5e15,
                       n_m: int = 200, delta_crit: float = 500.0) -> jax.Array:
    """1-halo connected tSZ trispectrum :math:`T^{1h}(\\ell, \\ell')`.

    Symmetric ``(n_ell, n_ell)`` matrix.  Used to construct the
    non-Gaussian part of the bandpower covariance; see
    :func:`cl_yy_covariance`.

    Same per-call cost as :func:`cl_yy` for the cosmology grids, plus an
    extra ``O(n_ell² n_z n_m)`` integral for the trispectrum contraction.
    """
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    return cl_yy_1h_trispectrum(jnp.asarray(ell), cg, hg, cosmo_dict,
                                 profile='arnaud10',
                                 profile_params=profile._asdict())


def cl_yy_covariance(cosmo: CosmoParams, profile: ProfileParamsA10, ell,
                      delta_ell, fsky: float = 1.0,
                      include_trispectrum: bool = True,
                      z_grid: jax.Array | None = None,
                      n_z: int = 100, m_min: float = 1e10,
                      m_max: float = 3.5e15, n_m: int = 200,
                      delta_crit: float = 500.0) -> jax.Array:
    """Bandpower covariance for tSZ :math:`C_\\ell^{yy}`.

    .. math::
       \\mathrm{Cov}(C_\\ell, C_{\\ell'}) =
           \\frac{2\\,C_\\ell^2}{(2\\ell + 1)\\,\\Delta\\ell\\,f_\\mathrm{sky}}
           \\,\\delta_{\\ell\\ell'}
           \\;+\\; \\frac{T^{1h}(\\ell, \\ell')}{4\\pi\\,f_\\mathrm{sky}}

    Returns the full :math:`(n_\\ell, n_\\ell)` covariance matrix
    suitable for a Cholesky decomposition to generate synthetic
    bandpower realisations:

    >>> L = jnp.linalg.cholesky(cov)
    >>> y_synth = y_fid + L @ jax.random.normal(key, (len(ell),))

    The covariance is on the dimensionless :math:`C_\\ell`. If the data
    vector is in :math:`D_\\ell\\times 10^{12}` units, rescale by the
    outer product of ``ell(ell+1)/(2π) × 1e12`` before Cholesky.

    Parameters
    ----------
    cosmo, profile, ell : as for :func:`cl_yy`.
    delta_ell : float or array_like
        Bandpower width(s) :math:`\\Delta\\ell`.  Scalar broadcasts to all
        bins.
    fsky : float
        Observed sky fraction; the Gaussian variance scales as
        :math:`1/f_\\mathrm{sky}` and the trispectrum term as
        :math:`1/(4\\pi f_\\mathrm{sky})`.
    include_trispectrum : bool
        If False, return Gaussian variance only (diagonal).
    z_grid, n_z, m_min, m_max, n_m, delta_crit
        Forwarded to the cosmology / halo-model grid builders.
    """
    ell_arr = jnp.asarray(ell)
    delta_ell_arr = jnp.broadcast_to(jnp.asarray(delta_ell, dtype=ell_arr.dtype),
                                      ell_arr.shape)
    # Build the cosmology and halo grids ONCE and reuse for both cl_yy
    # and the trispectrum.  Avoids the 2× emulator + HMF build cost the
    # naive composition cl_yy(...) + cl_yy_trispectrum(...) would pay.
    if z_grid is None:
        z_grid = jnp.geomspace(0.005, 3.0, n_z)
    cosmo_dict = cosmo_to_dict(cosmo)
    cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
    hg = build_halo_grids(cg, cosmo_dict, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)
    pp_dict = profile._asdict()

    cl_1h, cl_2h = cl_yy_1h_2h(ell_arr, cg, hg, cosmo_dict,
                                profile='arnaud10', profile_params=pp_dict)
    cl_tot = cl_1h + cl_2h
    gaussian_diag = 2.0 * cl_tot ** 2 / ((2.0 * ell_arr + 1.0)
                                          * delta_ell_arr * fsky)
    cov = jnp.diag(gaussian_diag)
    if include_trispectrum:
        T = cl_yy_1h_trispectrum(ell_arr, cg, hg, cosmo_dict,
                                  profile='arnaud10', profile_params=pp_dict)
        cov = cov + T / (4.0 * jnp.pi * fsky)
    return cov
