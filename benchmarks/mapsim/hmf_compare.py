"""Halo mass function: classy_szlite against cosmocnc_jax and colossus.

A physics comparison: each code runs with its own P(k), sigma(R) and conventions,
and classy_szlite is the reference. Percent-level agreement is expected. The
cross-feeds below attribute each difference to its cause.

Codes
-----
classy_szlite (reference)
    Production conventions, as in ``sim/halos.corner_density``:
    - ede-v3 emulator P_tot(k), with neutrinos as 3 x 0.02 eV;
    - A_s from sigma8 with ``as_reference=3.0``;
    - FFTLog sigma(R);
    - Tinker08 interpolated linearly in Delta;
    - rho_m and Omega_m(z) without neutrinos.
    Also its defaults: interpolation in log10 Delta, and neutrinos in rho_m and Omega_m(z).
cosmocnc_jax
    The CAMB backend (``cosmology_tool='camb'``, ``cosmo_model='mnu'``): one massive
    state of 0.06 eV, with P_tot rescaled exactly to sigma8. It runs through the
    module functions of its fast HMF path (``cnc.get_hmf``):
    - mcfit FFTLog sigma(R), with a numerical dsigma/dR;
    - Tinker08 linear in Delta;
    - rho_m of cb, and Delta_mean = 500 / Omega_m,nonu(z).
    It is imported from a separate reference copy, without running its package
    ``__init__`` (which needs cobaya). Reference use only: no cosmocnc_jax code is
    copied here.
colossus 1.4
    - Eisenstein & Hu (1998) P(k), normalised to sigma8;
    - no massive neutrinos (Om0 = Omega_cb);
    - Tinker08 with Delta_m rounded to an integer and interpolated linearly.

Cross-feeds, at the evaluation redshifts
----------------------------------------
xclass   classy_szlite's sigma(R) and assembly, fed CLASS P(k) with 3 x 0.02 eV (the
         emulators' training set-up): isolates the emulator error.
xclass1  The same with CLASS 1 x 0.06 eV (cosmocnc's set-up): isolates the neutrino set-up.
xcamb    The same with cosmocnc's CAMB P(k): isolates CAMB against CLASS. The rest of
         cosmocnc / classy_szlite is conventions and numerics.
cllog    classy_szlite with only the Tinker interpolation switched to log10 Delta.
xasm     cosmocnc_jax's sigma(R) and assembly, fed classy_szlite's P(k), rho_m and Delta_mean.
direct   sigma(R) and its slope by direct quadrature at the evaluation masses: an
         independent check of both assemblies.
f(sigma) Tinker08 at identical sigma, z and integer Delta_m in all three codes.

Steps, in order, each in its own environment
--------------------------------------------
class     sz-poisson-blending venv (classy)                        -> $HMF_OUT/class_pk.npz
jax       classy_szlite venv; PYTHONPATH adds the cosmocnc_jax copy and camb -> $HMF_OUT/jax.npz
colossus  sz-poisson-blending venv (colossus)                      -> $HMF_OUT/colossus.npz
report    any venv with numpy and scipy                            -> tables on stdout
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

OUT = os.environ.get("HMF_OUT", "/mnt/ceph/users/bbolliet/mapsim/hmf")
CNC_DIR = os.environ.get("COSMOCNC_DIR", "/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax")

OMEGA_B, N_S = 0.0222, 0.96
POINTS = {                      # (sigma8, omega_cdm, h): fiducial point and two prior corners
    "fid": (0.811, 0.121, 0.674),
    "low": (0.7, 0.0804, 0.704),
    "high": (0.9, 0.204, 0.644),
}
Z_EVAL = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
M_EVAL = np.array([1e13, 3e13, 1e14, 3e14, 1e15, 3e15, 1e16])
Z_COUNT = np.linspace(0.01, 3.0, 601)          # classy_szlite and colossus count grids
Z_COUNT_CNC = np.linspace(0.01, 3.0, 151)      # cosmocnc's CAMB table (CAMB takes <= 256 z)
Z_BINS = np.linspace(0.01, 1.5, 6)             # the paper's redshift bins
M_LO, M_HI = (1e13, 5e13), 1e16
OMEGA_TILE = np.deg2rad(10.0) ** 2             # one 10 x 10 deg tile [sr]
LNM_MIN, LNM_MAX, N_M = np.log(1e12), np.log(3e16), 1201
N_CNC = 4096                                   # cosmocnc's default n_points
SIG_TEST = np.geomspace(0.25, 4.0, 121)
Z_TEST = np.array([0.1, 0.5, 1.0, 2.0])
C_KMS = 299792.458


# ----------------------------------------------------------------------------- helpers

def k_grid_classy_szlite():
    """classy_szlite's K_GRID (cosmology._k_grids), rebuilt without JAX; checked in `jax`."""
    kmin, kmax, nk, extrap = 5e-4, 10.0, 1000, 1e-4
    k_emu = np.geomspace(kmin, kmax, nk)
    step = (np.log10(kmax) - np.log10(kmin)) / (nk - 1)
    n_extra = int(round((np.log10(kmin) - np.log10(extrap)) / step))
    k_extra = 10.0 ** (np.log10(kmin) - np.arange(n_extra, 0, -1) * step)
    return np.concatenate([k_extra, k_emu])


def simpson_w(x):
    n = len(x)
    assert n % 2 == 1 and np.allclose(np.diff(x), x[1] - x[0], rtol=1e-9, atol=0)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return w * (x[-1] - x[0]) / (n - 1) / 3.0


def mass_integral(lnM, dndlnm, lo, hi, n=801):
    """Integral of dn/dlnM over [ln lo, ln hi] for each row; ln(dn/dlnM) linear in ln M."""
    x = np.linspace(np.log(lo), np.log(hi), n)
    w = simpson_w(x)
    ln = np.log(np.maximum(np.asarray(dndlnm, dtype=float), 1e-300))
    return np.array([np.sum(w * np.exp(np.interp(x, lnM, row))) for row in ln])


def at_masses(lnM, dndlnm, M):
    """dn/dlnM at the masses M (ln dn/dlnM interpolated linearly in ln M)."""
    ln = np.log(np.maximum(np.asarray(dndlnm, dtype=float), 1e-300))
    return np.exp(np.array([np.interp(np.log(M), lnM, row) for row in ln]))


def sigma_R_quad(k, pk, R, n=20001):
    """Top-hat sigma(R) by Simpson quadrature in ln k (log-log interpolated P)."""
    lk = np.linspace(np.log(k[0]), np.log(k[-1]), n)
    kk = np.exp(lk)
    P = np.exp(np.interp(lk, np.log(k), np.log(pk)))
    x = kk * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return float(np.sqrt(np.sum(simpson_w(lk) * kk ** 3 * P * W ** 2) / (2.0 * np.pi ** 2)))


# ----------------------------------------------------------------------------- CLASS

NU_SETUPS = {   # key suffix: CLASS neutrino set-up (N_eff = 3.044 / 3.046)
    "": {"N_ur": 0.00441, "N_ncdm": 1, "deg_ncdm": 3, "m_ncdm": 0.02},   # the emulators' training
    "1": {"N_ur": 2.0328, "N_ncdm": 1, "deg_ncdm": 1, "m_ncdm": 0.06},   # cosmocnc's CAMB set-up
}


def step_class():
    from classy import Class
    k = k_grid_classy_szlite()
    out = {"k": k, "z_eval": Z_EVAL}
    for name, (s8, oc, h) in POINTS.items():
        for nu, nu_par in NU_SETUPS.items():
            t0 = time.time()
            c = Class()
            c.set({"output": "mPk", "P_k_max_1/Mpc": 20.0, "z_max_pk": 3.0,
                   "omega_b": OMEGA_B, "omega_cdm": oc, "h": h, "n_s": N_S, "sigma8": s8,
                   "tau_reio": 0.054, **nu_par})
            c.compute()
            out[f"pk{nu}_{name}"] = np.array([[c.pk_lin(kk, z) for kk in k] for z in Z_EVAL])
            pk0 = np.array([c.pk_lin(kk, 0.0) for kk in k])
            der = c.get_current_derived_parameters(["sigma8", "ln10^{10}A_s"])
            out[f"s8{nu}_{name}"] = der["sigma8"]
            out[f"lnAs{nu}_{name}"] = der["ln10^{10}A_s"]
            out[f"s8pk{nu}_{name}"] = sigma_R_quad(k, pk0, 8.0 / h)
            print(f"CLASS {name} nu '{nu}': sigma8 {der['sigma8']:.6f} (quadrature "
                  f"{out[f's8pk{nu}_{name}']:.6f}), ln10As {der['ln10^{10}A_s']:.5f}, "
                  f"{time.time() - t0:.1f} s", flush=True)
            c.struct_cleanup()
            c.empty()
    np.savez(f"{OUT}/class_pk.npz", **out)


def direct_sigma(k, pk, R):
    """Independent check of sigma(R) and dln(sigma)/dlnR at the radii R: Simpson
    quadrature in ln k of the log-log interpolated P(k), with the analytic
    derivative of the top-hat window (series below x = 0.01). pk: (n_z, n_k)."""
    lk = np.linspace(np.log(k[0]), np.log(k[-1]), 40001)
    kk = np.exp(lk)
    x = kk[None, :] * np.asarray(R)[:, None]
    xs = np.where(x < 1e-2, 1.0, x)
    W = np.where(x < 1e-2, 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0,
                 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3)
    dW = np.where(x < 1e-2, -x / 5.0 + x ** 3 / 70.0,
                  3.0 * ((xs * xs - 3.0) * np.sin(xs) + 3.0 * xs * np.cos(xs)) / xs ** 4)
    a = kk ** 3 * simpson_w(lk) / (2.0 * np.pi ** 2)
    sig, dls = [], []
    for row in np.asarray(pk):
        P = np.exp(np.interp(lk, np.log(k), np.log(row)))
        s2 = (W * W) @ (a * P)
        rds2 = (2.0 * W * dW * x) @ (a * P)          # R dsigma^2/dR
        sig.append(np.sqrt(s2))
        dls.append(0.5 * rds2 / s2)
    return np.array(sig), np.array(dls)


# ----------------------------------------------------------------------------- JAX codes

def _import_cosmocnc():
    """cosmocnc_jax's hmf, cosmo and params modules, without the package __init__."""
    import importlib
    import types
    os.environ.setdefault("PATH_TO_COSMOCNC", CNC_DIR.rstrip("/") + "/")
    os.environ.setdefault("PATH_TO_COSMOPOWER_ORGANIZATION", "/nonexistent")
    pkg = types.ModuleType("cosmocnc_jax")
    pkg.__path__ = [CNC_DIR]
    sys.modules["cosmocnc_jax"] = pkg
    return tuple(importlib.import_module(f"cosmocnc_jax.{m}") for m in ("hmf", "cosmo", "params"))


def step_jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from mcfit import TophatVar
    from classy_szlite._registry import DEFAULT_COSMO
    from classy_szlite.cosmology import (build, ln10As_from_sigma8, LN10AS_REFERENCE,
                                         _predict_pk, _compute_sigma, K_GRID)
    from classy_szlite.hmf import build_halo_grids, sigma_dndlnm_tinker08, tinker08_hmf
    import classy_szlite
    hc, cc, pc = _import_cosmocnc()
    print("classy_szlite from", os.path.dirname(classy_szlite.__file__))
    print("cosmocnc_jax from", CNC_DIR)
    assert np.allclose(np.asarray(K_GRID), k_grid_classy_szlite(), rtol=1e-12, atol=0)
    cls = np.load(f"{OUT}/class_pk.npz")

    lnM = np.linspace(LNM_MIN, LNM_MAX, N_M)
    M_vec = jnp.exp(jnp.linspace(LNM_MIN, LNM_MAX, N_CNC))
    out = {"lnM": lnM, "lnM_cnc": np.log(np.asarray(M_vec)), "z_eval": Z_EVAL,
           "z_count": Z_COUNT, "z_count_cnc": Z_COUNT_CNC}
    T08 = (hc.TINKER08_DELTA_LIN, hc.TINKER08_A, hc.TINKER08_a, hc.TINKER08_b, hc.TINKER08_c)

    def cnc_assembly(tv0, tv1, fns, k_arr, pk_batch, rho_m, z, delta):
        """cosmocnc_jax's own sigma(M) and Tinker08 functions (fast path of cnc.get_hmf).
        Returns dn/dlnM and sigma on M_vec."""
        sig, dsig, Rm = hc.batch_sigma_R_from_tophat(tv0, tv1, pk_batch, k_arr, M_vec, rho_m,
                                                     type_deriv="numerical", _cached_fns=fns)
        return (np.asarray(hc.compute_hmf_matrix_jit(sig, dsig, Rm, M_vec, rho_m, jnp.asarray(z), delta,
                                                     jnp.ones(len(z)), *T08, -1.0, False)),
                np.asarray(sig))

    def tophat(k):
        k = np.asarray(k)
        tv0 = TophatVar(k, lowring=True, deriv=0, backend="jax")
        tv1 = TophatVar(k, lowring=True, deriv=1, backend="jax")
        return tv0, tv1, hc.build_batch_sigma_fns(tv0, tv1, jnp.asarray(k), type_deriv="numerical")

    tv_K = tophat(K_GRID)
    for name, (s8, oc, h) in POINTS.items():
        t0 = time.time()
        # ---- classy_szlite (reference)
        p = {"omega_cdm": oc, "H0": 100.0 * h, "omega_b": OMEGA_B, "n_s": N_S}
        p["ln10^{10}A_s"] = float(ln10As_from_sigma8(p, s8, as_reference=LN10AS_REFERENCE))
        out[f"lnAs_cl_{name}"] = p["ln10^{10}A_s"]
        mm = dict(delta_crit=500.0, m_min=float(np.exp(LNM_MIN)), m_max=float(np.exp(LNM_MAX)), n_m=N_M)
        for tag, zg in (("eval", Z_EVAL), ("count", Z_COUNT)):
            cg = build(p, z_grid=jnp.asarray(zg), as_reference=LN10AS_REFERENCE)
            prod = build_halo_grids(cg, p, tinker_interp="linear", nu_in_rho_m=False, **mm)
            dflt = build_halo_grids(cg, p, **mm)
            lg = build_halo_grids(cg, p, tinker_interp="log10", nu_in_rho_m=False, **mm)
            out[f"cl_{tag}_{name}"] = np.asarray(prod.dndlnm)
            out[f"cldef_{tag}_{name}"] = np.asarray(dflt.dndlnm)
            out[f"cllog_{tag}_{name}"] = np.asarray(lg.dndlnm)
            out[f"vol_cl_{tag}_{name}"] = np.asarray(cg.chi) ** 2 / np.asarray(cg.Hz)
            if tag == "eval":
                cg_e, prod_e = cg, prod
        out[f"rhom_cl_{name}"] = float(prod_e.rho_m0)
        out[f"delta_cl_{name}"] = np.asarray(500.0 / prod_e.Omega_m_z)
        out[f"sig_cl_{name}"] = np.asarray(prod_e.sigma_m)
        # independent check of classy_szlite's sigma(M) and slope: direct quadrature at M_EVAL
        R_ev = (3.0 * M_EVAL / (4.0 * np.pi * float(prod_e.rho_m0))) ** (1.0 / 3.0)
        s_d, dls_d = direct_sigma(np.asarray(K_GRID), np.asarray(cg_e.pk), R_ev)
        f_d = 2.0 * np.asarray(tinker08_hmf(jnp.asarray(s_d), jnp.asarray(Z_EVAL), 500.0 / prod_e.Omega_m_z,
                                            interp="linear"))
        out[f"direct_{name}"] = float(prod_e.rho_m0) / M_EVAL[None, :] * f_d * np.abs(dls_d) / 3.0
        k0, pk0 = _predict_pk(dict(DEFAULT_COSMO, **p), jnp.array([0.0]), as_reference=LN10AS_REFERENCE)
        out[f"s8pk_cl_{name}"] = sigma_R_quad(np.asarray(k0), np.asarray(pk0)[0], 8.0 / h)

        # ---- cosmocnc_jax, CAMB backend
        cp = dict(pc.cosmo_params_default)
        cp.update(Ob0h2=OMEGA_B, Oc0h2=oc, h=h, sigma_8=s8, n_s=N_S, m_nu=0.06, A_s=2.1e-9)
        cn = dict(pc.cnc_params_default)
        cn.update(cosmology_tool="camb", cosmo_model="mnu", z_min=float(Z_COUNT_CNC[0]),
                  z_max=float(Z_COUNT_CNC[-1]), n_z=len(Z_COUNT_CNC), cosmo_param_density="physical",
                  interp_tinker="linear", hmf_type_deriv="numerical")
        cm = cc.cosmology_model(cosmo_params=cp, cosmology_tool="camb", amplitude_parameter="sigma_8",
                                cnc_params=cn)
        hobj = cc.halo_mass_function(cosmology=cm, hmf_type="Tinker08", mass_definition="500c",
                                     M_min=float(np.exp(LNM_MIN)), M_max=float(np.exp(LNM_MAX)),
                                     n_points=N_CNC, type_deriv="numerical", hmf_calc="cnc",
                                     interp_tinker="linear")
        rho_m = hobj.rho_c_0 * cm.cosmo_params["Om0"]
        k_c = cm._k_arr
        tv_c = tophat(k_c)
        for tag, zv in (("eval", Z_EVAL), ("count", Z_COUNT_CNC)):
            zj = jnp.asarray(zv)
            pkb = cm._predict_pk_batch(None, zj)
            delta = 500.0 / cm._Omega_m_z_nonu(zj)
            out[f"cnc_{tag}_{name}"], sig_c = cnc_assembly(*tv_c, k_c, pkb, rho_m, zv, delta)
            Hc, DA = np.asarray(cm._predict_H(None, zj)), np.asarray(cm._predict_DA(None, zj))
            out[f"vol_cnc_{tag}_{name}"] = (DA * (1.0 + zv)) ** 2 / Hc
            if tag == "eval":
                pk_camb = np.asarray(pkb)
                out[f"delta_cnc_{name}"] = np.asarray(delta)
                out[f"sig_cnc_{name}"] = sig_c
        out[f"rhom_cnc_{name}"] = float(rho_m)
        out[f"s8_cnc_{name}"] = float(cm.sigma8)
        out[f"s8camb_{name}"] = float(cm._camb_sigma8_internal)
        out[f"lnAs_cnc_{name}"] = float(cm._pvd["ln10^{10}A_s"])

        # ---- cross-feeds at the evaluation redshifts
        def classy_from_pk(pk_K):
            R, sig, _ = _compute_sigma(jnp.asarray(pk_K))
            return np.asarray(sigma_dndlnm_tinker08(jnp.asarray(lnM), jnp.asarray(Z_EVAL), R, sig,
                                                    prod_e.rho_m0, 500.0 / prod_e.Omega_m_z, "linear")[1])
        # CAMB's 500-point table onto K_GRID: a cubic spline in log-log (linear interpolation
        # would bias P low by ~1e-4 where ln P is concave, i.e. sigma by ~-5e-5)
        from scipy.interpolate import CubicSpline
        lk, lK = np.log(np.asarray(k_c)), np.log(np.asarray(K_GRID))
        pk_camb_K = np.exp(CubicSpline(lk, np.log(pk_camb), axis=1)(lK))
        out[f"xcamb_{name}"] = classy_from_pk(pk_camb_K)
        out[f"xclass_{name}"] = classy_from_pk(cls[f"pk_{name}"])
        out[f"xclass1_{name}"] = classy_from_pk(cls[f"pk1_{name}"])
        out[f"xasm_{name}"], _ = cnc_assembly(*tv_K, jnp.asarray(K_GRID), cg_e.pk, prod_e.rho_m0, Z_EVAL,
                                              500.0 / prod_e.Omega_m_z)
        out[f"pkratio_camb_{name}"] = pk_camb_K / np.asarray(cg_e.pk)
        out[f"pkratio_class_{name}"] = cls[f"pk_{name}"] / np.asarray(cg_e.pk)
        print(f"{name}: lnAs classy_szlite {p['ln10^{10}A_s']:.5f}, cosmocnc/CAMB "
              f"{out[f'lnAs_cnc_{name}']:.5f}, CLASS {float(cls[f'lnAs_{name}']):.5f}; "
              f"{time.time() - t0:.1f} s", flush=True)

    # ---- f(sigma) at identical inputs: integer Delta_m from the fiducial production Delta_mean
    d_fid = np.interp(Z_TEST, Z_EVAL, out["delta_cl_fid"])
    d_test = np.round(d_fid)
    sig = np.tile(SIG_TEST, (len(Z_TEST), 1))
    out["z_test"], out["d_test"], out["sig_test"] = Z_TEST, d_test, SIG_TEST
    out["f_cl"] = 2.0 * np.asarray(tinker08_hmf(jnp.asarray(sig), jnp.asarray(Z_TEST), jnp.asarray(d_test),
                                                interp="linear"))
    out["f_cnc"] = np.array([np.asarray(hc.f_sigma_jit(jnp.asarray(SIG_TEST), z, d, *T08, interp_log=False))
                             for z, d in zip(Z_TEST, d_test)])
    out["versions"] = np.array([f"jax {jax.__version__}", f"camb {sys.modules['camb'].__version__}"])
    np.savez(f"{OUT}/jax.npz", **out)


# ----------------------------------------------------------------------------- colossus

def step_colossus():
    from importlib.metadata import version
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function, peaks
    jx = np.load(f"{OUT}/jax.npz")
    lnM = np.linspace(LNM_MIN, LNM_MAX, N_M)
    out = {"lnM": lnM, "version": version("colossus")}
    cosmos = {}
    for name, (s8, oc, h) in POINTS.items():
        t0 = time.time()
        par = {"flat": True, "H0": 100.0 * h, "Om0": (OMEGA_B + oc) / h ** 2, "Ob0": OMEGA_B / h ** 2,
               "sigma8": s8, "ns": N_S, "Tcmb0": 2.7255, "Neff": 3.046}
        cosmo = cosmology.setCosmology(f"mapsim_{name}", params=par)
        cosmos[name] = cosmo
        Mh = np.exp(lnM) * h                                        # M_sun/h
        for tag, zg in (("eval", Z_EVAL), ("count", Z_COUNT)):
            dn = np.array([mass_function.massFunction(Mh, z, mdef="500c", model="tinker08", q_out="dndlnM")
                           for z in zg]) * h ** 3                   # (Mpc/h)^-3 -> Mpc^-3
            chi = np.array([cosmo.comovingDistance(0.0, z, transverse=True) for z in zg]) / h
            out[f"col_{tag}_{name}"] = dn
            out[f"vol_col_{tag}_{name}"] = chi ** 2 * C_KMS / cosmo.Hz(zg)
        out[f"s8_col_{name}"] = cosmo.sigma(8.0, 0.0)
        R_h = peaks.lagrangianR(Mh)                                 # Mpc/h, Om0 = Omega_cb
        out[f"sig_col_{name}"] = np.array([cosmo.sigma(R_h, z) for z in Z_EVAL])
        out[f"delta_col_{name}"] = np.array([round(500.0 * cosmo.rho_c(z) / cosmo.rho_m(z)) for z in Z_EVAL])
        print(f"colossus {name}: {time.time() - t0:.1f} s", flush=True)
    cosmology.setCurrent(cosmos["fid"])
    out["f_col"] = np.array([mass_function.modelTinker08(jx["sig_test"], z, f"{int(d)}m")
                             for z, d in zip(jx["z_test"], jx["d_test"])])
    np.savez(f"{OUT}/colossus.npz", **out)


# ----------------------------------------------------------------------------- report

def _dNdz(lnM, dn, vol, lo):
    return OMEGA_TILE * vol * mass_integral(lnM, dn, lo, M_HI)


def _integrate(z, f, a, b, n=401):
    from scipy.interpolate import CubicSpline
    x = np.linspace(a, b, n)
    return float(np.sum(simpson_w(x) * CubicSpline(z, f)(x)))


def step_report():
    jx = np.load(f"{OUT}/jax.npz")
    co = np.load(f"{OUT}/colossus.npz")
    cl_ = np.load(f"{OUT}/class_pk.npz")
    lnM, lnMc = jx["lnM"], jx["lnM_cnc"]
    Ms = M_EVAL
    hdr = "   z  " + "".join(f"{m:9.0e}" for m in Ms)

    def show(title, r, fmt="9.4f"):
        print(f"-- {title}")
        print(hdr)
        for z, row in zip(Z_EVAL, r):
            print(f"{z:5.1f} " + "".join(f"{x:{fmt}}" for x in row))
        return r

    def table(title, num, num_lnM, den=None, den_lnM=None, name="fid"):
        den = jx[f"cl_eval_{name}"] if den is None else den
        den_lnM = lnM if den_lnM is None else den_lnM
        return show(title, at_masses(num_lnM, num, Ms) / at_masses(den_lnM, den, Ms))

    print("sigma8 of each code's z=0 P(k) (same quadrature), and ln10^10 A_s:")
    for n, (s8, oc, h) in POINTS.items():
        print(f"  {n:4s} target {s8}: classy_szlite PKL {float(jx[f's8pk_cl_{n}']):.5f} "
              f"(lnAs {float(jx[f'lnAs_cl_{n}']):.4f}); cosmocnc/CAMB {float(jx[f's8_cnc_{n}']):.5f} "
              f"(CAMB's own {float(jx[f's8camb_{n}']):.5f}, lnAs {float(jx[f'lnAs_cnc_{n}']):.4f}); "
              f"CLASS 3x0.02 {float(cl_[f's8_{n}']):.5f} (lnAs {float(cl_[f'lnAs_{n}']):.4f}); "
              f"CLASS 1x0.06 lnAs {float(cl_[f'lnAs1_{n}']):.4f}; colossus {float(co[f's8_col_{n}']):.5f}")
    for n in POINTS:
        print(f"  Delta_mean(z_eval) {n}: classy_szlite {np.round(jx[f'delta_cl_{n}'], 1).tolist()}; "
              f"cosmocnc {np.round(jx[f'delta_cnc_{n}'], 1).tolist()}; colossus {co[f'delta_col_{n}'].tolist()}")
        print(f"  rho_m {n}: cosmocnc/classy_szlite - 1 = {float(jx[f'rhom_cnc_{n}']) / float(jx[f'rhom_cl_{n}']) - 1:.2e}")

    for n in POINTS:
        print(f"\n==================== {n}: dn/dlnM ratios to classy_szlite (production conventions)")
        table("cosmocnc_jax (CAMB, 1 x 0.06 eV)", jx[f"cnc_eval_{n}"], lnMc, name=n)
        table("colossus (EH98, no massive nu)", co[f"col_eval_{n}"], lnM, name=n)
        table("classy_szlite defaults (log10 Delta, nu in rho_m and Delta)", jx[f"cldef_eval_{n}"], lnM, name=n)
        print("   cosmocnc / classy_szlite = (1) x (2) x (3) x (4):")
        table("(1) emulator error: CLASS P(k) 3 x 0.02 eV / emulator P(k), same assembly",
              jx[f"xclass_{n}"], lnM, name=n)
        table("(2) neutrino set-up: CLASS 1 x 0.06 eV / CLASS 3 x 0.02 eV, same assembly",
              jx[f"xclass1_{n}"], lnM, den=jx[f"xclass_{n}"], name=n)
        table("(3) Boltzmann code: CAMB 1 x 0.06 eV / CLASS 1 x 0.06 eV, same assembly",
              jx[f"xcamb_{n}"], lnM, den=jx[f"xclass1_{n}"], name=n)
        table("(4) conventions and numerics: cosmocnc / classy_szlite assembly, both on CAMB P(k)",
              jx[f"cnc_eval_{n}"], lnMc, den=jx[f"xcamb_{n}"], name=n)
        print("   classy_szlite defaults / production = (5) x (6):")
        table("(5) Tinker parameters interpolated in log10 Delta instead of Delta", jx[f"cllog_eval_{n}"], lnM, name=n)
        table("(6) neutrinos in rho_m and Omega_m(z)", jx[f"cldef_eval_{n}"], lnM, den=jx[f"cllog_eval_{n}"], name=n)
        show("sigma(M): colossus / classy_szlite - 1",
             at_masses(lnM, co[f"sig_col_{n}"], Ms) / at_masses(lnM, jx[f"sig_cl_{n}"], Ms) - 1, "9.5f")
        show("sigma(M): cosmocnc / classy_szlite - 1",
             at_masses(lnMc, jx[f"sig_cnc_{n}"], Ms) / at_masses(lnM, jx[f"sig_cl_{n}"], Ms) - 1, "9.5f")
        show("code test vs direct quadrature (same P(k), rho_m, Delta): classy_szlite / direct - 1",
             at_masses(lnM, jx[f"cl_eval_{n}"], Ms) / jx[f"direct_{n}"] - 1, "9.5f")
        show("code test vs direct quadrature: cosmocnc sigma(R) + assembly / direct - 1",
             at_masses(lnMc, jx[f"xasm_{n}"], Ms) / jx[f"direct_{n}"] - 1, "9.5f")
        kk = cl_["k"]
        sel = (kk > 0.01) & (kk < 2.0)
        print(f"-- P(k) ratios over 0.01 < k < 2 /Mpc at z_eval (max |r - 1|): "
              f"CAMB/emulator {np.abs(jx[f'pkratio_camb_{n}'][:, sel] - 1).max(axis=1).round(4).tolist()}; "
              f"CLASS/emulator {np.abs(jx[f'pkratio_class_{n}'][:, sel] - 1).max(axis=1).round(4).tolist()}")

    print("\n==================== expected halos per 10 x 10 deg tile (M <= 1e16)")
    edges = [(0.01, 3.0)] + list(zip(Z_BINS[:-1], Z_BINS[1:]))
    for n in POINTS:
        for lo in M_LO:
            codes = {
                "classy_szlite": (jx["z_count"], _dNdz(lnM, jx[f"cl_count_{n}"], jx[f"vol_cl_count_{n}"], lo)),
                "cl-defaults": (jx["z_count"], _dNdz(lnM, jx[f"cldef_count_{n}"], jx[f"vol_cl_count_{n}"], lo)),
                "cosmocnc": (jx["z_count_cnc"], _dNdz(lnMc, jx[f"cnc_count_{n}"], jx[f"vol_cnc_count_{n}"], lo)),
                "colossus": (jx["z_count"], _dNdz(lnM, co[f"col_count_{n}"], co[f"vol_col_count_{n}"], lo)),
            }
            N = {c: [_integrate(z, f, a, b) for a, b in edges] for c, (z, f) in codes.items()}
            ref = np.array(N["classy_szlite"])
            print(f"-- {n}, M >= {lo:.0e}: z 0.01-3 | paper z bins (0.01-0.308-0.606-0.904-1.202-1.5)")
            print(f"   classy_szlite " + " ".join(f"{x:9.1f}" for x in ref))
            for c in ("cl-defaults", "cosmocnc", "colossus"):
                print(f"   {c:13s} " + " ".join(f"{x:9.4f}" for x in np.array(N[c]) / ref) + "   (ratio)")

    print("\n==================== f(sigma) at identical sigma, z and integer Delta_m")
    print(f"   z {jx['z_test'].tolist()}, Delta_m {jx['d_test'].tolist()}, sigma 0.25-4")
    print(f"   max |f_cosmocnc / f_classy_szlite - 1| = {np.abs(jx['f_cnc'] / jx['f_cl'] - 1).max():.2e}")
    print(f"   max |f_colossus / f_classy_szlite - 1| = {np.abs(co['f_col'] / jx['f_cl'] - 1).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["class", "jax", "colossus", "report"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"class": step_class, "jax": step_jax, "colossus": step_colossus, "report": step_report}[a.step]()
