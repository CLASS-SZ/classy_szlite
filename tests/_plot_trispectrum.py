"""4-panel diagnostic for the 1-halo tSZ trispectrum + bandpower covariance.

Outputs tests/trispectrum_diagnostic.png:
  (a) log |T(ell, ell')| heatmap
  (b) correlation matrix of the full Gauss+trispectrum covariance
  (c) D_ell with Gaussian-only vs Gauss+trispectrum 1-sigma error bars
  (d) per-bin ratio: trispectrum diagonal / Gaussian diagonal
"""
from __future__ import annotations
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import classy_szlite as csl


def main():
    # baseline cosmology + fiducial Arnaud-10 profile
    cosmo    = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                               tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    fiducial = csl.ProfileParamsA10(           # Arnaud 2010 universal profile
        P0=8.130, c500=1.156, gamma=0.3292, alpha=1.062, beta=5.4807, B=1.25,
    )
    fsky     = 0.6

    # use a slightly finer ell-binning so the heatmap has structure
    ell       = jnp.geomspace(100.0, 5000.0, 16)
    delta_ell = ell * jnp.log(ell[1] / ell[0])
    ell_np    = np.asarray(ell)

    # Trispectrum and Gaussian / Gauss+Trispectrum covariances
    T       = np.asarray(csl.cl_yy_trispectrum(cosmo, fiducial, ell))
    cov_g   = np.asarray(csl.cl_yy_covariance(cosmo, fiducial, ell, delta_ell,
                                                fsky=fsky,
                                                include_trispectrum=False))
    cov_gt  = np.asarray(csl.cl_yy_covariance(cosmo, fiducial, ell, delta_ell,
                                                fsky=fsky,
                                                include_trispectrum=True))

    # Fiducial D_ell × 10^12
    c1, c2 = csl.cl_yy(cosmo, fiducial, ell)
    Dell_fid = np.asarray(ell * (ell + 1) / (2 * np.pi) * (c1 + c2) * 1e12)
    pref = np.asarray(ell * (ell + 1) / (2 * np.pi)) * 1e12

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # ---- (a) Trispectrum heatmap (log scale) ----
    ax = axs[0, 0]
    im = ax.imshow(T, norm=LogNorm(vmin=T.max()*1e-6, vmax=T.max()),
                    cmap='viridis', origin='lower',
                    extent=[ell_np[0], ell_np[-1], ell_np[0], ell_np[-1]],
                    aspect='auto')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\ell'$"); ax.set_ylabel(r"$\ell$")
    ax.set_title(r"1-halo trispectrum $T^{1h}(\ell, \ell')$")
    plt.colorbar(im, ax=ax, label=r"$T^{1h}$  [sr]")

    # ---- (b) Correlation matrix of full Gauss+trispectrum covariance ----
    ax = axs[0, 1]
    d = np.sqrt(np.diag(cov_gt))
    corr = cov_gt / np.outer(d, d)
    im = ax.imshow(corr, vmin=0, vmax=1, cmap='magma_r', origin='lower',
                    extent=[ell_np[0], ell_np[-1], ell_np[0], ell_np[-1]],
                    aspect='auto')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\ell'$"); ax.set_ylabel(r"$\ell$")
    ax.set_title("correlation matrix (Gaussian + trispectrum)")
    plt.colorbar(im, ax=ax, label=r"$\mathrm{Corr}(C_\ell, C_{\ell'})$")

    # ---- (c) D_ell with two error bands ----
    ax = axs[1, 0]
    sig_g  = pref * np.sqrt(np.diag(cov_g))
    sig_gt = pref * np.sqrt(np.diag(cov_gt))
    ax.errorbar(ell_np, Dell_fid, yerr=sig_g, fmt='o',
                color='C0', label="Gaussian only",
                capsize=2, ms=4, lw=1)
    ax.errorbar(ell_np * 1.06, Dell_fid, yerr=sig_gt, fmt='s',
                color='C3', label="Gaussian + 1h trispectrum",
                capsize=2, ms=4, lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12}$")
    ax.set_title(r"$D_\ell^{yy}$ with $\pm 1\sigma$ bandpower errors")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ---- (d) Diagonal ratio: full cov / Gaussian only ----
    ax = axs[1, 1]
    ratio = np.diag(cov_gt) / np.diag(cov_g)
    ax.semilogx(ell_np, ratio, 'C2o-', lw=1.5, ms=5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma^2_\mathrm{Gauss+T} / \sigma^2_\mathrm{Gauss}$")
    ax.set_title("trispectrum boost to the bandpower variance")
    ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')

    plt.suptitle(
        rf"tSZ bandpower covariance with 1-halo trispectrum "
        rf"($f_\mathrm{{sky}}={fsky}$, fiducial $P_0={fiducial.P0}$, "
        rf"$\beta={fiducial.beta}$)",
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "trispectrum_diagnostic.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")

    # also print quick numerical summary
    print(f"\nT.shape = {T.shape}, symmetric: "
          f"{np.allclose(T, T.T)}")
    print(f"max corr off-diagonal: {corr[np.triu_indices_from(corr, k=1)].max():.3f}")
    print(f"trispectrum boost per bin: min {ratio.min():.1f}× max {ratio.max():.1f}×")


if __name__ == "__main__":
    main()
