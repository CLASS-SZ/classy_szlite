"""B12 pressure-profile validation: classy_szlite vs classy_sz.

Both codes run the same Battaglia-2012 GNFW pressure profile, Tinker-08
M200c HMF, and matched physical mass range / cosmology. Output is the
1-halo Cl^yy plus timing for both code paths. The figure is saved to
docs/_static/b12_validation.png.

Requires classy_sz + classy_szlite, both with the same cosmology.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static" / "b12_validation.png"

# ── Cosmology (Planck-18-ish; matched between codes) ─────────────────────
COSMO = dict(omega_b=0.0226, omega_cdm=0.118, H0=68.22, tau_reio=0.0561,
             n_s=0.9743)
LOGA = 3.06
h = COSMO["H0"] / 100.0

# Multipoles: log grid 100 → 5000 (SZ-relevant range)
ell_grid = np.geomspace(100.0, 5000.0, 30)

# Matched physical mass range (classy_sz expects M_sun/h, csl uses M_sun)
M_PHYS_MIN, M_PHYS_MAX = 1e10, 3.5e15
M_H_MIN, M_H_MAX = M_PHYS_MIN * h, M_PHYS_MAX * h

# ── 1. classy_sz B12 ─────────────────────────────────────────────────────
from classy_sz import Class as Class_sz

csz = Class_sz()
csz.set({"omega_b": COSMO["omega_b"], "omega_cdm": COSMO["omega_cdm"],
         "H0": COSMO["H0"], "tau_reio": COSMO["tau_reio"],
         "n_s": COSMO["n_s"], "ln10^{10}A_s": LOGA})
csz.set({
    "output": "tSZ_1h",
    "skip_input": 0, "cosmo_model": 0,
    "ell_min": float(ell_grid[0]), "ell_max": float(ell_grid[-1]),
    "dell": 0.0, "dlogell": 0.10,
    "mass_function": "T08M200c", "pressure_profile": "B12",
    "z_min": 0.005, "z_max": 3.0,
    "M_min": M_H_MIN, "M_max": M_H_MAX,
    "ndim_redshifts": 500, "x_outSZ": 100.0,
    "use_fft_for_profiles_transform": 1, "N_samp_fftw": 1024,
    "x_min_gas_pressure_fftw": 1e-4, "x_max_gas_pressure_fftw": 1e6,
    "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-4,
    "mass_epsabs":     1e-40, "mass_epsrel":     1e-4,
    # Converged precision for the B12 FFT tabulation (matches the official
    # classy_sz test test_classy_sz_clyy_b12_fast.py). Without these the
    # default n_l_pressure_profile=300 leaves the FFTLog under-resolved at
    # low ell and produces a spurious ~30% offset at ell < 200.
    "n_z_pressure_profile": 500, "n_m_pressure_profile": 500,
    "n_l_pressure_profile": 500,
    "l_min_gas_pressure_profile": 1e-2, "l_max_gas_pressure_profile": 5e4,
    "pressure_profile_epsrel": 1e-4, "pressure_profile_epsabs": 1e-100,
    # B12 defaults from Battaglia+2012 fits
    "P0_B12": 18.1,    "alpha_m_P0_B12": 0.154,    "alpha_z_P0_B12": -0.758,
    "xc_B12": 0.497,   "alpha_m_xc_B12": -0.00865, "alpha_z_xc_B12": 0.731,
    "beta_B12": 4.35,  "alpha_m_beta_B12": 0.0393, "alpha_z_beta_B12": 0.415,
    "alpha_B12": 1.0, "gamma_B12": -0.3,
})
t0 = time.time(); csz.initialize_classy_szfast(); t_csz_init = time.time() - t0
o = csz.cl_sz()                                  # dict lookup, ~µs
dl_csz_at_grid = np.interp(ell_grid, np.asarray(o["ell"]), np.asarray(o["1h"]))
# In an MCMC, every new parameter point needs init+cl_sz, so init is the
# real per-call cost.
t_csz_per_eval = t_csz_init

# ── 2. classy_szlite B12 ────────────────────────────────────────────────
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import classy_szlite as csl
from classy_szlite.api import cosmo_to_dict
from classy_szlite.cosmology import build as build_cosmo_grids
from classy_szlite.hmf import build_halo_grids
from classy_szlite.power_spectrum import cl_yy_1h_2h

cosmo = csl.CosmoParams(
    omega_b=COSMO["omega_b"], omega_cdm=COSMO["omega_cdm"],
    H0=COSMO["H0"], tau_reio=COSMO["tau_reio"],
    ln10_10_As=LOGA, n_s=COSMO["n_s"],
)
cd = cosmo_to_dict(cosmo)
t0 = time.time()
cg = build_cosmo_grids(cd, z_grid=jnp.geomspace(0.005, 3.0, 100))
hg = build_halo_grids(cg, cd, delta_crit=200.0,
                       m_min=M_PHYS_MIN, m_max=M_PHYS_MAX, n_m=200)
ell_jnp = jnp.asarray(ell_grid)
t_csl_init = time.time() - t0


@jax.jit
def evaluate_b12():
    return cl_yy_1h_2h(ell_jnp, cg, hg, cd, profile='battaglia12',
                       profile_params=None)


t0 = time.time(); c1, c2 = evaluate_b12(); c1.block_until_ready()
t_csl_first = time.time() - t0
warm = []
for _ in range(20):
    t0 = time.time(); c1, c2 = evaluate_b12(); c1.block_until_ready()
    warm.append((time.time() - t0) * 1000)
t_csl_warm = float(np.median(warm))

DL_FAC = ell_grid * (ell_grid + 1) / (2.0 * np.pi) * 1e12
dl_csl_1h, dl_csl_2h = np.asarray(DL_FAC * c1), np.asarray(DL_FAC * c2)
ratio = dl_csl_1h / dl_csz_at_grid

# ── 3. Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
ax.plot(ell_grid, dl_csz_at_grid, "C0-", lw=2.2,
        label=f"classy_sz B12   ({t_csz_per_eval*1000:.0f} ms / eval)")
ax.plot(ell_grid, dl_csl_1h, "C3--", lw=2.2,
        label=f"classy_szlite B12 1h   ({t_csl_warm:.1f} ms warm, "
              f"{t_csz_per_eval / (t_csl_warm/1000):.0f}× faster)")
ax.plot(ell_grid, dl_csl_1h + dl_csl_2h, "C2:", lw=1.5,
        label="classy_szlite 1h + 2h")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12}$")
ax.set_title("B12 pressure profile: classy_sz vs classy_szlite (1-halo)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, which="both")

ax = axes[1]
ax.plot(ell_grid, ratio, "C3o-", lw=2)
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.axhspan(0.99, 1.01, alpha=0.15, color="gray", label="±1% band")
ax.axhspan(0.95, 1.05, alpha=0.08, color="gray", label="±5% band")
ax.set_xscale("log")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("csl 1h / classy_sz 1h")
ax.set_title("ratio: <1% for ℓ ≥ 1500 (SZ-relevant range)")
ax.set_ylim(0.85, 1.15)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130)
print(f"Saved: {OUT}")
print(f"\nAgreement summary:")
print(f"  ℓ ≥ 1500:  max |ratio - 1| = {np.max(np.abs(ratio[ell_grid >= 1500] - 1)) * 100:.2f}%")
print(f"  ℓ ≥ 1000:  max |ratio - 1| = {np.max(np.abs(ratio[ell_grid >= 1000] - 1)) * 100:.2f}%")
print(f"  ℓ ≥ 500:   max |ratio - 1| = {np.max(np.abs(ratio[ell_grid >= 500]  - 1)) * 100:.2f}%")
print(f"  full grid: max |ratio - 1| = {np.max(np.abs(ratio - 1)) * 100:.2f}% (at ℓ = {ell_grid[np.argmax(np.abs(ratio - 1))]:.0f})")
print(f"\nTiming (per-eval cost for an MCMC):")
print(f"  classy_sz:     {t_csz_per_eval*1000:.0f} ms / eval")
print(f"  classy_szlite: {t_csl_warm:.1f} ms / eval (after a {t_csl_init:.1f} s factory build)")
print(f"  speedup at fixed cosmology: {t_csz_per_eval / (t_csl_warm/1000):.0f}×")
