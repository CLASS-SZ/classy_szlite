"""Regenerate docs figures at the A10 fiducial profile, baseline cosmology,
with synthetic Cholesky-generated bandpowers + inflated covariance
(noise_factor = 9) consistent with the docs/examples.md inference example.

Produces:
  docs/_static/profile_bands.png   — GNFW profile median/68% band recovered
                                       by NUTS, overlaid on the A10 fiducial.
  docs/_static/fisher_overlay.png  — 68%/95% Fisher ellipses + L-BFGS bestfit
                                       overlaid on the NUTS posterior.
"""
from __future__ import annotations
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.optimize as so
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import classy_szlite as csl


# ----- shared setup -----------------------------------------------------
COSMO    = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                           tau_reio=0.0561, ln10_10_As=3.060, n_s=0.9743)
FIDUCIAL = csl.ProfileParamsA10(P0=8.130, c500=1.156, gamma=0.3292,
                                 alpha=1.062, beta=5.4807, B=1.25)
ELL          = jnp.geomspace(100.0, 5000.0, 8)
DELTA_ELL    = ELL * jnp.log(ELL[1] / ELL[0])
FSKY         = 0.6
NOISE_FACTOR = 9

EV         = csl.cl_yy_factory(COSMO, ELL)
DL_FACTOR  = ELL * (ELL + 1) / (2 * jnp.pi) * 1e12
COV_CL     = csl.cl_yy_covariance(COSMO, FIDUCIAL, ELL, DELTA_ELL, fsky=FSKY)
COV        = COV_CL * (DL_FACTOR[:, None] * DL_FACTOR[None, :]) * NOISE_FACTOR
INV_COV    = jnp.linalg.inv(COV)
L_CHOL     = jnp.linalg.cholesky(COV)
C1, C2     = EV(FIDUCIAL)
DELL_FID   = DL_FACTOR * (C1 + C2)
DELL_DATA  = DELL_FID + L_CHOL @ jax.random.normal(jax.random.PRNGKey(42), ELL.shape)


def forward(P0, beta):
    prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                 alpha=1.062, beta=beta, B=1.25)
    c1, c2 = EV(prof)
    return DL_FACTOR * (c1 + c2)


def find_bestfit():
    def nll(x):
        r = DELL_DATA - forward(x[0], x[1])
        return 0.5 * r @ INV_COV @ r
    nllj = jax.jit(nll); gj = jax.jit(jax.grad(nll))
    return so.minimize(lambda x: float(nllj(x)),
                       np.asarray([8.13, 5.48], dtype=np.float64),
                       jac=lambda x: np.asarray(gj(x)),
                       method="L-BFGS-B", bounds=[(0.1, 20), (0.5, 10)])


def run_nuts(bf):
    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = DELL_DATA - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ INV_COV @ r)
    mcmc = MCMC(NUTS(model, dense_mass=True), num_warmup=300, num_samples=1000,
                num_chains=2, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0),
             init_params={"P0":   jnp.full(2, float(bf.x[0])),
                          "beta": jnp.full(2, float(bf.x[1]))})
    s = mcmc.get_samples()
    return np.asarray(s["P0"]), np.asarray(s["beta"])


def main():
    bf = find_bestfit()
    print(f"bestfit: P0={bf.x[0]:.3f}  β={bf.x[1]:.3f}  χ²={2*bf.fun:.2f}/6  evals={bf.nfev}")
    P0, beta = run_nuts(bf)
    print(f"NUTS:    P0={P0.mean():.2f}±{P0.std():.2f}  β={beta.mean():.2f}±{beta.std():.2f}")

    # ----- profile_bands.png ---------------------------------------------
    x_arr = np.geomspace(1e-2, 5.0, 200)

    def gnfw_dimless(P0_, beta_, x):
        c500 = 1.156; gamma = 0.3292; alpha = 1.062
        return (P0_ * (c500 * x) ** (-gamma)
                * (1.0 + (c500 * x) ** alpha) ** (-(beta_ - gamma) / alpha))

    n_draw = 500
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P0), n_draw, replace=False)
    samp = np.stack(
        [gnfw_dimless(P0[i], beta[i], x_arr) for i in idx], axis=0
    )  # (n_draw, n_x)
    med = np.median(samp, axis=0)
    lo, hi = np.percentile(samp, [16, 84], axis=0)
    fid_curve = gnfw_dimless(8.130, 5.4807, x_arr)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x_arr, fid_curve * x_arr ** 2, "k-", lw=2,
            label=r"A10 fiducial ($P_0\!=\!8.13$, $\beta\!=\!5.48$)")
    ax.plot(x_arr, med * x_arr ** 2, "C0-", lw=1.6,
            label=fr"NUTS median ($P_0\!=\!{P0.mean():.2f}$, $\beta\!=\!{beta.mean():.2f}$)")
    ax.fill_between(x_arr, lo * x_arr ** 2, hi * x_arr ** 2,
                     color="C0", alpha=0.25, label="68% band")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$x = r/r_{500}$")
    ax.set_ylabel(r"$\mathbb{P}(x)\,x^2$  (dimensionless)")
    ax.set_title("GNFW profile recovered from synthetic $C_\\ell^{yy}$ bandpowers")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_p = os.path.expanduser("~/classy_szlite/docs/_static/profile_bands.png")
    plt.savefig(out_p, dpi=120, bbox_inches="tight")
    print(f"Saved {out_p}")
    plt.close()

    # ----- fisher_overlay.png --------------------------------------------
    # Fisher F = J.T @ inv_cov @ J at the bestfit
    def mu(x):
        return forward(x[0], x[1])
    J = np.asarray(jax.jacfwd(mu)(jnp.asarray([bf.x[0], bf.x[1]])))
    F = J.T @ np.asarray(INV_COV) @ J
    F_cov = np.linalg.inv(F)
    # Fisher ellipses
    eigvals, eigvecs = np.linalg.eigh(F_cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width_68 = 2.0 * np.sqrt(2.30 * eigvals.max())   # χ²₂ at 68% = 2.30
    height_68 = 2.0 * np.sqrt(2.30 * eigvals.min())
    width_95 = 2.0 * np.sqrt(6.18 * eigvals.max())   # χ²₂ at 95% = 6.18
    height_95 = 2.0 * np.sqrt(6.18 * eigvals.min())

    from getdist import MCSamples, plots
    samples = MCSamples(samples=np.column_stack([P0, beta]),
                         names=["P0", "beta"], labels=["P_0", r"\beta"],
                         label="NumPyro NUTS")
    g = plots.get_subplot_plotter(width_inch=6)
    g.settings.alpha_filled_add = 0.55
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 10
    g.settings.legend_fontsize = 9
    g.triangle_plot(
        [samples], params=["P0", "beta"], filled=True,
        contour_colors=["C0"],
        legend_labels=["NumPyro NUTS (synthetic data, A10 fiducial)"],
        markers={"P0": 8.130, "beta": 5.4807},
    )
    ax2d = g.subplots[1, 0]
    for w, h, ls, lbl in [(width_68, height_68, "--", "68% Fisher"),
                            (width_95, height_95, ":",  "95% Fisher")]:
        e = Ellipse(xy=(bf.x[0], bf.x[1]), width=w, height=h, angle=angle,
                    fill=False, edgecolor="C3", lw=1.5, ls=ls, label=lbl)
        ax2d.add_patch(e)
    ax2d.plot(bf.x[0], bf.x[1], "x", color="C3", ms=10, mew=2,
              label="L-BFGS bestfit")
    ax2d.legend(loc="upper left", fontsize=8, framealpha=0.95)
    out_f = os.path.expanduser("~/classy_szlite/docs/_static/fisher_overlay.png")
    g.export(out_f)
    print(f"Saved {out_f}")
    print(f"σ_Fisher(P0)={np.sqrt(F_cov[0,0]):.3f}  σ_NUTS(P0)={P0.std():.3f}")
    print(f"σ_Fisher(β)={np.sqrt(F_cov[1,1]):.3f}   σ_NUTS(β)={beta.std():.3f}")


if __name__ == "__main__":
    main()
