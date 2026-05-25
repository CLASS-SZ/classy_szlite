"""Regenerate docs/_static/posterior_compare.png from synthetic
Cholesky-generated bandpowers at A10 fiducial, with an inflated
covariance (noise_factor = 9) so the (P0, β) degeneracy is visible."""
from __future__ import annotations
import os
import time
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
from getdist import MCSamples, plots
import classy_szlite as csl


def main():
    cosmo    = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                               tau_reio=0.0561, ln10_10_As=3.060, n_s=0.9743)
    fiducial = csl.ProfileParamsA10(P0=8.130, c500=1.156, gamma=0.3292,
                                     alpha=1.062, beta=5.4807, B=1.25)
    ell       = jnp.geomspace(100.0, 5000.0, 8)
    delta_ell = ell * jnp.log(ell[1] / ell[0])
    fsky      = 0.6
    noise_factor = 9     # inflate cov so the degeneracy direction is visible

    ev          = csl.cl_yy_factory(cosmo, ell)
    dl_factor   = ell * (ell + 1) / (2 * jnp.pi) * 1e12
    cov_cl      = csl.cl_yy_covariance(cosmo, fiducial, ell, delta_ell, fsky=fsky)
    cov_dell    = cov_cl * (dl_factor[:, None] * dl_factor[None, :])
    cov         = cov_dell * noise_factor
    inv_cov     = jnp.linalg.inv(cov)
    L_chol      = jnp.linalg.cholesky(cov)

    c1, c2    = ev(fiducial)
    Dell_fid  = dl_factor * (c1 + c2)
    Dell_data = Dell_fid + L_chol @ jax.random.normal(jax.random.PRNGKey(42), ell.shape)

    def forward(P0, beta):
        prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                     alpha=1.062, beta=beta, B=1.25)
        c1, c2 = ev(prof)
        return dl_factor * (c1 + c2)

    def nll(x):
        r = Dell_data - forward(x[0], x[1])
        return 0.5 * r @ inv_cov @ r
    nllj = jax.jit(nll); g_j = jax.jit(jax.grad(nll))
    bf = so.minimize(lambda x: float(nllj(x)), [8.13, 5.48],
                     jac=lambda x: np.asarray(g_j(x)),
                     method="L-BFGS-B", bounds=[(0.1, 20), (0.5, 10)])
    print(f"bestfit: P0={bf.x[0]:.3f}  β={bf.x[1]:.3f}  chi2={2*bf.fun:.2f}/6  evals={bf.nfev}")

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = Dell_data - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ inv_cov @ r)

    # Pull a slightly tighter chain so the degeneracy direction is
    # well-traced (R-hat <= 1.01 with 2 chains × 1000 samples).
    t0 = time.perf_counter()
    mcmc = MCMC(NUTS(model, dense_mass=True), num_warmup=300, num_samples=1000,
                num_chains=2, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0),
             init_params={"P0":   jnp.full(2, float(bf.x[0])),
                          "beta": jnp.full(2, float(bf.x[1]))})
    wall = time.perf_counter() - t0
    s = mcmc.get_samples()
    sm = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    P0 = np.asarray(s["P0"]); beta = np.asarray(s["beta"])
    corr = float(np.corrcoef(P0, beta)[0, 1])
    print(f"NUTS:  P0={P0.mean():.2f}±{P0.std():.2f}  β={beta.mean():.2f}±{beta.std():.2f}  "
          f"corr={corr:+.2f}  wall={wall:.1f}s")
    print(f"       R-hat: P0={float(sm['P0']['r_hat']):.3f}  β={float(sm['beta']['r_hat']):.3f}; "
          f"ESS: P0={int(sm['P0']['n_eff'])}  β={int(sm['beta']['n_eff'])}")

    # Triangle plot
    samples = MCSamples(
        samples=np.column_stack([P0, beta]),
        names=["P0", "beta"],
        labels=["P_0", r"\beta"],
        label="NumPyro NUTS",
    )
    g = plots.get_subplot_plotter(width_inch=6)
    g.settings.alpha_filled_add = 0.65
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 10
    g.settings.legend_fontsize = 10
    g.triangle_plot(
        [samples], params=["P0", "beta"], filled=True,
        contour_colors=["C0"],
        legend_labels=[
            f"NumPyro NUTS  ({wall:.0f}\\,s, ESS$\\,\\sim\\,${int(sm['P0']['n_eff'])})"
        ],
        markers={"P0": 8.130, "beta": 5.4807},
    )
    g.fig.suptitle(
        f"Synthetic-data posterior on (P$_0$, $\\beta$) at A10 fiducial\n"
        f"covariance inflated $\\times {noise_factor}$ so the degeneracy "
        f"(corr = {corr:+.2f}) is visible",
        y=1.05, fontsize=10, fontweight="bold",
    )
    out = os.path.expanduser("~/classy_szlite/docs/_static/posterior_compare.png")
    g.export(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
