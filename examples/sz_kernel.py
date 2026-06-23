r"""tSZ "y-kernel": where the Compton-y power spectrum comes from.

Reproduces the classic class_sz SZ-kernel heat-map
(https://github.com/CLASS-SZ/notebooks/blob/main/class_sz_szkernel.ipynb)
with **classy_szlite**.

The 1-halo tSZ power spectrum is

.. math::

    C_\ell^{yy,\,1h} = \int dz\, \frac{dV}{dz\,d\Omega}
        \int d\ln M\, \frac{dn}{d\ln M}\, |y_\ell(M,z)|^2 ,

so the quantity under the double integral,

.. math::

    \frac{d^2 C_\ell^{yy}}{dz\, d\ln M}
        = \frac{dV}{dz\,d\Omega}\,\frac{dn}{d\ln M}\,|y_\ell(M,z)|^2 ,

is the **tSZ kernel**: it shows, at a fixed multipole, which halo masses and
redshifts source the signal. As ``ell`` grows the bright region migrates to
lower mass / lower redshift (smaller, nearer haloes resolve higher ell).

Run::

    python examples/sz_kernel.py          # writes docs/_static/sz_kernel.png
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import classy_szlite as csl
from classy_szlite.api import cosmo_to_dict
from classy_szlite.cosmology import build as build_cosmo_grids
from classy_szlite.hmf import build_halo_grids
from classy_szlite.power_spectrum import _y_ell_grid

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static" / "sz_kernel.png"

# ── Cosmology (Planck-2018, matching the class_sz notebook) ──────────────
cosmo = csl.CosmoParams(
    omega_b=0.02242, omega_cdm=0.11933, H0=67.66,
    ln10_10_As=3.047, n_s=0.9665, tau_reio=0.0561,
)
cosmo_dict = cosmo_to_dict(cosmo)

# ── (z, M) grids ─────────────────────────────────────────────────────────
N_Z, N_M = 300, 300
Z_MIN, Z_MAX = 0.01, 5.0
M_MIN, M_MAX = 1e10, 1e15                  # M_sun
z_grid = jnp.linspace(Z_MIN, Z_MAX, N_Z)

# delta_crit=200 → Tinker-08 M200c HMF, matching the notebook's 'T08M200c'.
cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
hg = build_halo_grids(cg, cosmo_dict, delta_crit=200.0,
                      m_min=M_MIN, m_max=M_MAX, n_m=N_M)

log10M = np.asarray(hg.lnM) / np.log(10.0)   # mass-integration axis → log10 M

# ── tSZ kernel at several multipoles ─────────────────────────────────────
ells = jnp.asarray([100.0, 500.0, 2000.0, 10000.0])

# Battaglia-2012 electron-pressure profile (the notebook's 'B12'), x_out=4.
y_ell, dVdzdOmega = _y_ell_grid(ells, cg, hg, cosmo_dict,
                                profile="battaglia12", profile_params=None)
# kernel(ell, z, M) = dV/dzdΩ · dn/dlnM · y_ell²
kernel = np.asarray(dVdzdOmega[None, :, None] * hg.dndlnm[None, :, :] * y_ell ** 2)

# ── Plot 1 — single 'hot' heat-map (the classic SZ-kernel image) ─────────
extent = [log10M[0], log10M[-1], float(z_grid[-1]), float(z_grid[0])]
HERO = 3                                     # index of ells → ell = 10000
fig, ax = plt.subplots(figsize=(5.2, 5.0))
im = ax.imshow(kernel[HERO], cmap="hot", interpolation="nearest",
               extent=extent, aspect="auto")
ax.grid(visible=True, which="both", alpha=0.2, linestyle="--")
ax.tick_params(which="both", length=5, direction="in")
ax.xaxis.set_ticks_position("both"); ax.yaxis.set_ticks_position("both")
ax.set_xlabel(r"$\mathrm{Mass}\ \ \log_{10}(M/M_\odot)$", size=15)
ax.set_ylabel(r"$\mathrm{Redshift}\ \ z$", size=15)
ax.set_title(rf"tSZ $y$-kernel,  $\ell = {int(ells[HERO])}$", size=15)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$d^2 C_\ell^{yy}/dz\,d\ln M$", size=13)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"Saved: {OUT}")

# ── Plot 2 — multipole migration strip (bonus) ───────────────────────────
OUT2 = OUT.parent / "sz_kernel_migration.png"
fig2, axes = plt.subplots(1, len(ells), figsize=(4.2 * len(ells), 4.4))
for ax, il in zip(np.atleast_1d(axes), range(len(ells))):
    ax.imshow(kernel[il], cmap="hot", interpolation="nearest",
              extent=extent, aspect="auto")
    ax.grid(visible=True, which="both", alpha=0.2, linestyle="--")
    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$", size=13)
    if il == 0:
        ax.set_ylabel(r"redshift $z$", size=13)
    ax.set_title(rf"$\ell = {int(ells[il])}$", size=14)
fig2.suptitle(r"tSZ $y$-kernel migration: $d^2 C_\ell^{yy}/dz\,d\ln M$ "
              r"(classy_szlite, Battaglia-12)", size=14)
fig2.tight_layout()
fig2.savefig(OUT2, dpi=130)
print(f"Saved: {OUT2}")

# Quick sanity: peak (z, log10M) of each kernel
for il in range(len(ells)):
    iz, im_ = np.unravel_index(np.argmax(kernel[il]), kernel[il].shape)
    print(f"  ell={int(ells[il]):5d}: peak at z={float(z_grid[iz]):.2f}, "
          f"log10M={log10M[im_]:.2f}")
