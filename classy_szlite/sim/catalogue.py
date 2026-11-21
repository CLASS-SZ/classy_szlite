"""Catalogue-level simulator of the SO-like cluster survey (the paper's model; PLAN M7).

Zubeldia, Bolliet, Challinor & Handley (2025, arXiv:2504.10230) train neural posterior
estimation on catalogues from cosmocnc's generator. This module is our own JAX version of that
model on classy_szlite's cosmology, mass function, scaling relations and lensing model (no
cosmocnc code is used). For parameters ``theta`` (simulator.Theta):

1. Halos. Expected counts per (z, ln M500) cell over the survey area f_sky 4 pi (the map-level
   tile rates of simulator.cosmology_stage, scaled from one tile to the survey), Poisson counts
   per cell, and (z, ln M) drawn exactly from the bilinear interpolant of the corner densities
   inside each cell (as halos.sample_halos; no sky positions).
2. Observables, for each halo:
     ln q_true = ln[ybar0(M, z) / sigma_y0(theta500)] + sigma_lnq g1
     ln p_true = ln pbar(M, z)                       + sigma_lnp (r g1 + sqrt(1 - r^2) g2)
     q_obs = sqrt(q_true^2 + dof) + n1,    p_obs = p_true + n2
   with g1, g2, n1, n2 unit Gaussians, ybar0 and theta500 from sz_scaling (bias 1 - b),
   pbar = a_lens kappa(r_s) / sigma_kappa(theta500,lens) from lensing.py (mass beta_lens M500).
3. Selection q_obs > q_min (strict), the paper's 4 x 5 counts in (q, z) (summaries.histogram,
   open intervals), and the mean p_obs over every selected cluster (any z, q above 40 included):
   the paper's data vector of 21 numbers.

Noise tables:
  "paper"  the paper's sigma_y0 (SZiFi on SO-like Websky maps) and sigma_kappa tables, as
           cosmocnc's SO-like survey uses them: cubic fits of ln sigma in ln theta[arcmin],
           extrapolated outside the tables (or held at the table ends with clamp=True). The
           tables are read from cosmocnc_jax's data folder (COSMOCNC_DATA); the fit is
           numpy.polyfit.
  sigma_y0 arrays (theta, sigma) from our own filter (variant (i)) are passed the same way.

Random numbers: one key per simulation. Each halo's draws come from a key folded with its cell
and its index in the cell, so a count that changes in one cell changes only that cell's halos.
Halo buffers have a static capacity; Catalogue.overflow flags a draw that does not fit (the
caller reruns with a larger capacity).
"""
from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ..sz_scaling import y0_median, theta500_arcmin
from . import halos as hl, lensing, simulator as simu, summaries as sm

SR_FULL = 4.0 * np.pi
COSMOCNC_DATA = os.environ.get("COSMOCNC_DATA", os.path.expanduser("~/GitHub/cosmocnc_jax/cosmocnc_jax/data"))
Z_BIN_WIDTH = float(sm.Z_EDGES[1] - sm.Z_EDGES[0])        # 0.298


def survey_cells(z_per_bin=30, n_bins_z=10, Mmin=5e13, Mmax=1e16, n_m=150):
    """(z, ln M) cells aligned with the data vector's z edges: z from 0.01 in steps of
    Z_BIN_WIDTH / z_per_bin up to 0.01 + n_bins_z Z_BIN_WIDTH (2.99), M from M_min = 5e13."""
    z0 = float(sm.Z_EDGES[0])
    nz = z_per_bin * n_bins_z
    return hl.CellGrid(z0 + Z_BIN_WIDTH * np.arange(nz + 1) / z_per_bin,
                       np.linspace(np.log(Mmin), np.log(Mmax), n_m + 1))


class NoiseFit(NamedTuple):
    """ln sigma = polyval(coef, ln theta[arcmin]); within [lnth_min, lnth_max] unless extrapolated."""
    coef: jax.Array          # (4,) highest power first
    lnth_min: float
    lnth_max: float


def fit_table(theta_arcmin, sigma, deg=3):
    th, s = np.asarray(theta_arcmin, float), np.asarray(sigma, float)
    return NoiseFit(coef=jnp.asarray(np.polyfit(np.log(th), np.log(s), deg)),
                    lnth_min=float(np.log(th.min())), lnth_max=float(np.log(th.max())))


def paper_tables(data_dir=None):
    """The paper's sigma_y0 and sigma_kappa fits from cosmocnc_jax's data files (theta in radians)."""
    d = data_dir or COSMOCNC_DATA
    to_arcmin = 180.0 * 60.0 / np.pi
    sz = np.load(os.path.join(d, "so_sim_sz_mf_noise.npy"))
    le = np.load(os.path.join(d, "so_sim_lensing_mf_noise.npy"))
    return fit_table(sz[0] * to_arcmin, sz[1]), fit_table(le[0] * to_arcmin, le[1])


def eval_fit(fit: NoiseFit, theta_arcmin, clamp=False):
    lnth = jnp.log(theta_arcmin)
    if clamp:
        lnth = jnp.clip(lnth, fit.lnth_min, fit.lnth_max)
    return jnp.exp(jnp.polyval(fit.coef, lnth))


class PatchTable(NamedTuple):
    """Noise tables per sky patch: ln sigma linear in ln theta[arcmin], held at the table ends
    (variant (i): our own filter's sigma_y0 per survey tile; or one row for sigma_kappa)."""
    lnth: jax.Array          # (n_th,)
    lnsig: jax.Array         # (n_patch, n_th)


def eval_table(tab: PatchTable, theta_arcmin, patch=None):
    x = jnp.clip(jnp.log(theta_arcmin), tab.lnth[0], tab.lnth[-1])
    i = jnp.clip(jnp.searchsorted(tab.lnth, x, side="right") - 1, 0, tab.lnth.shape[0] - 2)
    w = (x - tab.lnth[i]) / (tab.lnth[i + 1] - tab.lnth[i])
    row = jnp.zeros_like(i) if patch is None or tab.lnsig.shape[0] == 1 else patch
    return jnp.exp((1.0 - w) * tab.lnsig[row, i] + w * tab.lnsig[row, i + 1])


def noise_sigma(tab, theta_arcmin, clamp=False, patch=None):
    """sigma at theta from a NoiseFit (cubic, the paper's) or a PatchTable (per patch)."""
    if isinstance(tab, NoiseFit):
        return eval_fit(tab, theta_arcmin, clamp)
    return eval_table(tab, theta_arcmin, patch)


def lens_table(mode="goal"):
    """Our sigma_kappa table (lens_tables.py) as a one-row PatchTable."""
    lnth, lnsig = lensing.table(mode)
    return PatchTable(lnth=jnp.asarray(lnth), lnsig=jnp.asarray(lnsig)[None])


def analytic_sigma_y0_forecast(cmb_cl, mode="goal", fs=None):
    """sigma_y0 of our filter at its 15 scales for the 2019 SO forecast noise (instrument.noise_cov:
    white plus atmospheric, correlated within a tube; the simulator's default) plus the lensed CMB:
    (1, 15) and the ln theta grid."""
    from . import mmf, instrument as ins
    from .flatsky import grid
    G = grid()
    fs = fs or mmf.filter_static()
    B = ins.beams(G.m_L)
    C = (ins.noise_cov(G.m_L, mode) + B[:, None, :] * B[None, :, :] * cmb_cl(G.m_L)[None, None, :]) * G.Npx / G.Om
    Cb = np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[i, j], minlength=G.n_bins)
                             / G.bin_count for j in range(6)], -1) for i in range(6)], 1)
    _, aCa = mmf.weights(jnp.asarray(Cb), fs)
    norm = jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)
    return np.asarray(1.0 / jnp.sqrt(norm))[None], np.log(np.asarray(mmf.THETAS15))


def analytic_sigma_y0(noise_tables, specs, cmb_cl, fs=None):
    """sigma_y0 of our matched filter at its 15 scales (mmf.THETAS15) for the noise of the MSS2
    spectra `specs` (noise.NoiseTables rows) plus the lensed CMB, without tSZ: the covariance is
    each mode's noise power (F F^T, per-mode units of noise.draw: E|X|^2 = N^2 C) plus
    B_i B_j C_l N^2 / Omega_pix, averaged over the filter's bins as mmf's estimate is.
    Returns (len(specs), 15) and the ln theta grid."""
    from . import mmf, instrument as ins
    from .flatsky import grid
    G = grid()
    fs = fs or mmf.filter_static()
    b = np.asarray(noise_tables.bin_idx).ravel()[np.asarray(G.m_idx)]
    B = ins.beams(G.m_L)
    cmb = B[:, None, :] * B[None, :, :] * cmb_cl(G.m_L)[None, None, :] * G.Npx / G.Om      # (6, 6, n_modes)
    F = np.asarray(noise_tables.F, dtype=np.float64)
    out = []
    for s in np.atleast_1d(specs):
        N = np.einsum("bik,bjk->bij", F[s], F[s])[b] * G.Npx                                 # (n_modes, 6, 6)
        C = np.moveaxis(cmb, -1, 0) + N
        Cb = np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[:, i, j], minlength=G.n_bins)
                                 / G.bin_count for j in range(6)], -1) for i in range(6)], 1)
        _, aCa = mmf.weights(jnp.asarray(Cb), fs)
        norm = jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)
        out.append(np.asarray(1.0 / jnp.sqrt(norm)))
    return np.array(out), np.log(np.asarray(mmf.THETAS15))


class CatalogueConfig(NamedTuple):
    """Static settings (hashable)."""
    capacity: int = 2 ** 21          # halo buffer slots
    f_sky: float = 0.4
    q_min: float = 5.0
    dof: float = 0.0
    corr_lnq_lnp: float = 0.0
    a_lens: float = 1.0
    clamp: bool = False              # hold the noise fits at the table ends
    # Calibrated observation (variant (ii), from the finder on map-level simulations, qcal.py):
    # None, or per redshift bin a tuple (z_upper, s, k, w, f): a halo is lost with probability f,
    # otherwise q_obs = sqrt((s q_true)^2 + k) + w N(0, 1). The last bin extends to any z. dof is
    # then unused.
    qcal: tuple | None = None


class Catalogue(NamedTuple):
    counts: jax.Array        # (4, 5) int32: q bins x z bins (summaries.histogram)
    p_sum: jax.Array         # () sum of p_obs over the selected clusters
    n_sel: jax.Array         # () number of selected clusters (q_obs > q_min)
    n_halos: jax.Array       # () halos drawn
    overflow: jax.Array      # () bool: more halos than the buffer holds


def data_vector(cat: Catalogue):
    """The paper's 21 numbers: the 4 x 5 counts (q-major) and the mean p over all selected."""
    return jnp.concatenate([cat.counts.ravel().astype(jnp.float64),
                            (cat.p_sum / jnp.maximum(cat.n_sel, 1))[None]])


def survey_rates(theta, cells):
    """Cosmology stage scaled from one map tile to the whole sky (multiply by f_sky):
    (lam per cell, corner densities, cosmology grids, chi_star)."""
    cs = simu.cosmology_stage(theta, cells)
    s = SR_FULL / hl.OMEGA_TILE
    return cs.lam * s, cs.d2N * s, cs.cg, cs.chi_star


def _qbar_pbar(M, z, theta, cg, chi_s, sz_fit, lens_fit, cfg, patch=None):
    """Median q and median p of halos (M500 [M_sun], z)."""
    H0_over_c = theta.h * 100.0 / lensing.C_KMS
    Ez = jnp.interp(z, cg.z, cg.Hz) / H0_over_c
    DA = jnp.interp(z, cg.z, cg.Da)
    y0 = y0_median(M, Ez, theta.h, theta.A_SZ, theta.alpha_SZ, theta.one_minus_b)
    th = theta500_arcmin(M, Ez, DA, theta.h, theta.one_minus_b)
    qbar = y0 / noise_sigma(sz_fit, th, cfg.clamp, patch)
    kap = lensing.kappa_rs(M, z, cg, theta.h, theta.beta_lens, chi_s)
    th_l = lensing.theta500_lens(M, z, cg, theta.h, theta.beta_lens)
    pbar = cfg.a_lens * kap / noise_sigma(lens_fit, th_l, cfg.clamp, patch)
    return qbar, pbar


class HaloDraw(NamedTuple):
    """Every slot of the halo buffer (valid marks the drawn halos)."""
    z: jax.Array
    M: jax.Array             # M500 [M_sun]
    qbar: jax.Array
    pbar: jax.Array
    q_true: jax.Array
    p_true: jax.Array
    q_obs: jax.Array
    p_obs: jax.Array
    valid: jax.Array
    total: jax.Array


def draw(key, theta, cells, sz_fit, lens_fit, cfg: CatalogueConfig = CatalogueConfig()) -> HaloDraw:
    """The halos of one catalogue and their observables (traceable in theta)."""
    lam_all, d2N_all, cg, chi_s = survey_rates(theta, cells)
    lam, d2N = lam_all * cfg.f_sky, d2N_all * cfg.f_sky
    k_pois, k_attr = jax.random.split(key)
    n_z, n_m = lam.shape
    counts = jax.random.poisson(k_pois, lam.ravel(), dtype=jnp.int32)
    csum = jnp.cumsum(counts)
    total = csum[-1]
    slot = jnp.arange(cfg.capacity, dtype=jnp.int32)
    valid = slot < total
    cell = jnp.minimum(jnp.searchsorted(csum, slot, side="right").astype(jnp.int32), n_z * n_m - 1)
    iz, im = cell // n_m, cell % n_m
    j_in_cell = slot - (csum[cell] - counts[cell])

    def draws(c, j):
        k = jax.random.fold_in(jax.random.fold_in(k_attr, c), j)
        ku, kn = jax.random.split(k)
        return jax.random.uniform(ku, (2,), dtype=jnp.float64), jax.random.normal(kn, (4,), dtype=jnp.float64)
    U, G = jax.vmap(draws)(cell, j_in_cell)                      # (C, 2), (C, 4)

    f00, f10 = d2N[iz, im], d2N[iz + 1, im]
    f01, f11 = d2N[iz, im + 1], d2N[iz + 1, im + 1]
    tz = jnp.clip(hl._inv_linear_cdf(U[:, 0], f00 + f01, f10 + f11), 0.0, 1.0)
    tm = jnp.clip(hl._inv_linear_cdf(U[:, 1], (1 - tz) * f00 + tz * f10, (1 - tz) * f01 + tz * f11), 0.0, 1.0)
    ze, me = jnp.asarray(cells.z_edges), jnp.asarray(cells.lnM_edges)
    z = ze[iz] + tz * (ze[iz + 1] - ze[iz])
    M = jnp.exp(me[im] + tm * (me[im + 1] - me[im]))

    patch = None
    if isinstance(sz_fit, PatchTable) and sz_fit.lnsig.shape[0] > 1:     # equal-area patches, own key
        k_patch = jax.random.fold_in(key, 0x5A7C)
        patch = jax.vmap(lambda c, j: jax.random.randint(
            jax.random.fold_in(jax.random.fold_in(k_patch, c), j), (), 0, sz_fit.lnsig.shape[0]))(cell, j_in_cell)
    qbar, pbar = _qbar_pbar(M, z, theta, cg, chi_s, sz_fit, lens_fit, cfg, patch)
    r = cfg.corr_lnq_lnp
    q_true = qbar * jnp.exp(theta.sigma_lnq * G[:, 0])
    p_true = pbar * jnp.exp(theta.sigma_lnp * (r * G[:, 0] + np.sqrt(1.0 - r * r) * G[:, 1]))
    if cfg.qcal is None:
        q_obs = jnp.sqrt(q_true ** 2 + cfg.dof) + G[:, 2]
    else:                                                     # calibrated; the loss from its own key
        s_, k_, w_, f_ = qcal_params(cfg.qcal, z)
        q_obs = jnp.sqrt(jnp.maximum((s_ * q_true) ** 2 + k_, 0.0)) + w_ * G[:, 2]
        k_loss = jax.random.fold_in(key, 0x1055)
        u = jax.vmap(lambda c, j: jax.random.uniform(jax.random.fold_in(jax.random.fold_in(k_loss, c), j), (),
                                                     dtype=jnp.float64))(cell, j_in_cell)
        q_obs = jnp.where(u < f_, -jnp.inf, q_obs)
    p_obs = p_true + G[:, 3]
    return HaloDraw(z=z, M=M, qbar=qbar, pbar=pbar, q_true=q_true, p_true=p_true, q_obs=q_obs,
                    p_obs=p_obs, valid=valid, total=total)


def qcal_params(qcal, z):
    """(s, k, w, f) of each z from the calibration table (see CatalogueConfig.qcal)."""
    t = np.asarray(qcal, dtype=np.float64)
    i = jnp.clip(jnp.searchsorted(jnp.asarray(t[:, 0]), z, side="left"), 0, t.shape[0] - 1)
    return tuple(jnp.asarray(t[:, c])[i] for c in (1, 2, 3, 4))


def load_qcal(path):
    """Calibration from qcal.py (calibration.npz): the CatalogueConfig.qcal tuple and the PatchTable
    of the finder's noise (per tile, mean over the mocks)."""
    d = np.load(path)
    qc = tuple(tuple(float(v) for v in row) for row in np.stack([d["z_upper"], d["s"], d["k"], d["w"], d["f"]], 1))
    return qc, PatchTable(lnth=jnp.asarray(d["lnth"]), lnsig=jnp.asarray(np.log(d["sigma"])))


def summarise(h: HaloDraw, cfg: CatalogueConfig = CatalogueConfig()) -> Catalogue:
    sel = h.valid & (h.q_obs > cfg.q_min)
    return Catalogue(counts=sm.histogram(h.q_obs, h.z, sel),
                     p_sum=jnp.sum(jnp.where(sel, h.p_obs, 0.0)),
                     n_sel=jnp.sum(sel.astype(jnp.int32)), n_halos=h.total, overflow=h.total > cfg.capacity)


def simulate(key, theta, cells, sz_fit, lens_fit, cfg: CatalogueConfig = CatalogueConfig()) -> Catalogue:
    """One catalogue (traceable in theta; jit with cells and cfg static)."""
    return summarise(draw(key, theta, cells, sz_fit, lens_fit, cfg), cfg)


# ---- expectations (for tests and checks) --------------------------------------------------

def expected(theta, cells, sz_fit, lens_fit, cfg: CatalogueConfig = CatalogueConfig(), n_gh=24, n_sub=3):
    """Expected 4 x 5 counts, expected number selected and expected sum of p over the selected
    clusters. Each cell is integrated with n_sub x n_sub Gauss-Legendre points weighted by the
    bilinear density (the sampler's), the scatter with Gauss-Hermite in (g1, g2), and the unit
    noise of q_obs exactly (normal CDF). E[p_obs | selected] uses E[n2] = 0."""
    from jax.scipy.stats import norm
    lam_all, d2N_all, cg, chi_s = survey_rates(theta, cells)
    d2N = d2N_all * cfg.f_sky
    x, w = np.polynomial.legendre.leggauss(n_sub)
    x, w = 0.5 * (x + 1.0), 0.5 * w                                  # on [0, 1]
    ze, me = np.asarray(cells.z_edges), np.asarray(cells.lnM_edges)
    dz, dm = np.diff(ze), np.diff(me)
    Z = ze[:-1, None] + dz[:, None] * x[None, :]                     # (n_z, n_sub)
    LM = me[:-1, None] + dm[:, None] * x[None, :]                    # (n_m, n_sub)
    tz, tm = jnp.asarray(x), jnp.asarray(x)
    f = (d2N[:-1, :-1][:, :, None, None] * (1 - tz)[None, None, :, None] * (1 - tm)[None, None, None, :]
         + d2N[1:, :-1][:, :, None, None] * tz[None, None, :, None] * (1 - tm)[None, None, None, :]
         + d2N[:-1, 1:][:, :, None, None] * (1 - tz)[None, None, :, None] * tm[None, None, None, :]
         + d2N[1:, 1:][:, :, None, None] * tz[None, None, :, None] * tm[None, None, None, :])
    wt = f * (jnp.asarray(dz)[:, None, None, None] * jnp.asarray(dm)[None, :, None, None]
              * jnp.asarray(w)[None, None, :, None] * jnp.asarray(w)[None, None, None, :])   # (n_z, n_m, s, s)
    zz = jnp.broadcast_to(jnp.asarray(Z)[:, None, :, None], wt.shape)
    MM = jnp.exp(jnp.broadcast_to(jnp.asarray(LM)[None, :, None, :], wt.shape))
    n_patch = sz_fit.lnsig.shape[0] if isinstance(sz_fit, PatchTable) else 1
    gh, gw = np.polynomial.hermite_e.hermegauss(n_gh)
    gw = gw / gw.sum()
    g, gwj = jnp.asarray(gh), jnp.asarray(gw)
    r = cfg.corr_lnq_lnp
    qe = jnp.asarray(sm.Q_EDGES)
    counts = jnp.zeros((4, 5))
    n_sel = 0.0
    p_sum = 0.0
    zb = jnp.searchsorted(jnp.asarray(sm.Z_EDGES), zz, side="right") - 1      # z bin of each point
    inz = [(zb == j).astype(wt.dtype) for j in range(5)]
    for pa in range(n_patch):
        patch = None if n_patch == 1 else jnp.full(zz.shape, pa)
        qbar, pbar = _qbar_pbar(MM, zz, theta, cg, chi_s, sz_fit, lens_fit, cfg, patch)
        if cfg.qcal is None:
            s_, k_, w_, f_ = 1.0, cfg.dof, 1.0, 0.0
        else:
            s_, k_, w_, f_ = qcal_params(cfg.qcal, zz)
        for a in range(n_gh):
            qt = jnp.sqrt(jnp.maximum((s_ * qbar * jnp.exp(theta.sigma_lnq * g[a])) ** 2 + k_, 0.0))
            wq = w_ if cfg.qcal is None else w_[..., None]
            cdf = norm.cdf((qe[None, None, None, None, :] - qt[..., None]) / wq)   # P(q_obs < edge)
            P_bin = (1.0 - (f_ if cfg.qcal is None else f_[..., None])) * (cdf[..., 1:] - cdf[..., :-1])
            P_sel = (1.0 - f_) * (1.0 - norm.cdf((cfg.q_min - qt) / w_))
            # E[exp(sigma_lnp (r g1 + sqrt(1-r^2) g2))] over g2 at fixed g1
            e_p = jnp.exp(theta.sigma_lnp * r * g[a] + 0.5 * (theta.sigma_lnp ** 2) * (1.0 - r * r))
            ww = wt * gwj[a] / n_patch
            for j in range(5):
                counts = counts.at[:, j].add(jnp.sum((ww * inz[j])[..., None] * P_bin, axis=(0, 1, 2, 3)))
            n_sel = n_sel + jnp.sum(ww * P_sel)
            p_sum = p_sum + jnp.sum(ww * P_sel * pbar * e_p)
    return counts, n_sel, p_sum
