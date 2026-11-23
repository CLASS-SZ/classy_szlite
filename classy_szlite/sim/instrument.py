"""SO LAT instrument model for mapsim (temperature only), as szsim's.

Numbers from The Simons Observatory Collaboration (Ade et al. 2019, JCAP 02,
056, arXiv:1808.07445):

* Table 1 (LAT, f_sky = 0.4): frequencies, beam FWHM, white noise
  (baseline and goal, μK-arcmin).
* Sec. 2.2, eq. 1: N_ℓ = N_red (ℓ/ℓ_knee)^α_knee + N_white, ℓ_knee = 1000,
  α_knee = -3.5; Table 2 LAT temperature N_red [μK² s]; atmospheric noise
  correlated with coefficient 0.9 between the two bands of an optics tube.
  The red level converts to μK² sr with A_sky / t_eff, A_sky = 4π f_sky,
  t_eff = 5 yr × 0.2 × 0.85.

tSZ SED: non-relativistic, at the band centres: ΔT/T_CMB = g(x) y,
g(x) = x coth(x/2) - 4. Gaussian beams B_ℓ = exp(-ℓ² σ_b² / 2).
"""
from __future__ import annotations

import numpy as np

from .flatsky import ARCMIN

T_CMB = 2.7255                     # K
H_PLANCK = 6.62607015e-34          # J s
K_BOLTZ = 1.380649e-23             # J / K

SO_LAT = dict(
    freqs=np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0]),          # GHz
    fwhm=np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),                    # arcmin
    white={"baseline": np.array([71.0, 36.0, 8.0, 10.0, 22.0, 54.0]),  # μK-arcmin
           "goal": np.array([52.0, 27.0, 5.8, 6.3, 15.0, 37.0])},
    N_red=np.array([100.0, 39.0, 230.0, 1500.0, 17000.0, 31000.0]),   # μK² s
    ell_knee=1000.0, alpha_knee=-3.5,
    tubes=((0, 1), (2, 3), (4, 5)),
    atm_corr=0.9,
    fsky=0.4, years=5.0, eff=0.2 * 0.85,
)


def t_eff():
    return SO_LAT["years"] * 365.25 * 86400.0 * SO_LAT["eff"]


def tsz_g(nu_GHz):
    x = H_PLANCK * np.asarray(nu_GHz, float) * 1e9 / (K_BOLTZ * T_CMB)
    return x / np.tanh(x / 2.0) - 4.0


def tsz_f_uK(nu_GHz):
    """ΔT [μK_CMB] per unit y."""
    return T_CMB * 1e6 * tsz_g(nu_GHz)


def sigma_beam_rad(fwhm_arcmin):
    return np.asarray(fwhm_arcmin, float) * ARCMIN / np.sqrt(8.0 * np.log(2.0))


def beams(L, fwhm=SO_LAT["fwhm"]):
    """B_i(ℓ), shape (n_ch,) + L.shape."""
    sb = sigma_beam_rad(fwhm)
    L = np.asarray(L, float)
    return np.exp(-0.5 * (L[None, ...] * sb.reshape((-1,) + (1,) * L.ndim)) ** 2)


def red_levels():
    """Atmospheric amplitude at ℓ_knee [μK² sr] per channel."""
    return SO_LAT["N_red"] * 4.0 * np.pi * SO_LAT["fsky"] / t_eff()


def atm_shape(L):
    L = np.asarray(L, float)
    with np.errstate(divide="ignore"):
        return np.where(L > 0, (L / SO_LAT["ell_knee"]) ** SO_LAT["alpha_knee"], 0.0)


def noise_cov(L, mode="goal", atm=True, atm_corr=None):
    """Noise covariance N_ij(ℓ) [μK² sr], not beam-deconvolved, (6, 6) + L.shape."""
    L = np.asarray(L, float)
    r = SO_LAT["atm_corr"] if atm_corr is None else atm_corr
    w = SO_LAT["white"][mode] * ARCMIN
    Nn = np.zeros((6, 6) + L.shape)
    for i in range(6):
        Nn[i, i] += w[i] ** 2
    if atm:
        red, shp = red_levels(), atm_shape(L)
        for i in range(6):
            Nn[i, i] += red[i] * shp
        for (i, j) in SO_LAT["tubes"]:
            c = r * np.sqrt(red[i] * red[j]) * shp
            Nn[i, j] += c
            Nn[j, i] += c
    return Nn


def cmb_cl_from_table(path):
    """C_ℓ^TT [μK²] from szsim's CAMB table (``data_real/cmb_lensed_TT_camb.npz``):
    returns a function of ℓ (power law above the stored range, 0 below ℓ = 2)."""
    d = np.load(path)
    le, cl = d["ell"].astype(float), d["cl_uK2"]
    sl, amp, lt = float(d["ext_slope"]), float(d["ext_lnamp"]), int(d["l_trust"])

    def f(ell):
        ell = np.asarray(ell, float)
        out = np.interp(ell, le, cl)
        hi = ell > lt
        out[hi] = np.exp(amp + sl * np.log(ell[hi]))
        out[ell < 2] = 0.0
        return out
    return f


# Lensed CMB from the classy_szlite TT emulator (production)
EMU_LMAX_USED = 9000.0                 # the emulator's last decade (9000-9500) is not used
EMU_FIT = (7000.0, 9000.0)             # power-law fit range for the extension


def cmb_cl_emulator(cosmo=None):
    """C_ℓ^TT [μK²] of the lensed CMB from the TT emulator at ``cosmo``
    (a CosmoParams; default: the emulator's default point). Returns a function
    of ℓ: the emulator for 2 ≤ ℓ ≤ 9000, a power law in ln C_ℓ vs ln ℓ fitted
    on 7000-9000 beyond, 0 below ℓ = 2. Host-side numpy.
    """
    from ..api import cl_TTTEEE
    from ..params import CosmoParams
    out = cl_TTTEEE(cosmo or CosmoParams(), spectra=("tt",), ell_factor=False)
    le = np.asarray(out["ell"], float)
    cl = np.asarray(out["tt"], float) * (T_CMB * 1e6) ** 2
    keep = le <= EMU_LMAX_USED
    le, cl = le[keep], cl[keep]
    fit = (le >= EMU_FIT[0]) & (le <= EMU_FIT[1])
    slope, lnamp = np.polyfit(np.log(le[fit]), np.log(cl[fit]), 1)

    def f(ell):
        ell = np.asarray(ell, float)
        res = np.interp(ell, le, cl)
        hi = ell > le[-1]
        res[hi] = np.exp(lnamp + slope * np.log(ell[hi]))
        res[ell < 2] = 0.0
        return res
    f.slope, f.lnamp = slope, lnamp
    return f
