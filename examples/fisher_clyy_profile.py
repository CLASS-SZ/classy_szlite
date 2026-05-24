"""Fisher matrix for (P0, β) via jax.jacfwd, overlaid with the
NUTS posterior on the same Cl^yy bandpower likelihood.

The Fisher information matrix for a Gaussian likelihood with
fixed covariance Σ is

    F_ij(θ) = (∂_i mu)(θ)ᵀ Σ⁻¹ (∂_j mu)(θ),

where mu(θ) = forward(P0, β) is the model bandpower vector.
Computing the Jacobian J = ∂mu/∂θ takes a single forward-mode
autodiff sweep via jax.jacfwd — no parameter-by-parameter
finite-difference loop required.  The 1σ Gaussian approximation
to the posterior is then F⁻¹ centred at the L-BFGS-B bestfit.

Compares to:
  * NumPyro NUTS posterior (full Bayesian sampling)
  * finite-difference Fisher with ε = 10⁻³ (∼same answer but
    requires 2 × N forward evaluations and a noisy ε choice)

Reports timing: autodiff Fisher in tens of ms vs the
finite-difference version, then plots the Fisher ellipse over
the NUTS contour for visual consistency.
"""
from __future__ import annotations
import os, time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.optimize as so

import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl

DATA_DIR  = os.environ.get(
    "CLYY_DATA_DIR",
    os.path.expanduser("~/Desktop/class-sz-plugin-tests/data"),
)
ELL_FILE  = "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"
DATA_FILE = "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"
COV_FILE  = "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"

COSMO = csl.CosmoParams(
    omega_b=0.0226, omega_cdm=0.118,
    H0=68.22, tau_reio=0.0561,
    ln10_10_As=3.06, n_s=0.9743,
)

C500, GAMMA, ALPHA, B_FIX = 1.156, 0.3292, 1.062, 1.25


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def load_bandpowers():
    ell  = np.loadtxt(os.path.join(DATA_DIR, ELL_FILE))
    data = np.loadtxt(os.path.join(DATA_DIR, DATA_FILE))
    cov  = np.loadtxt(os.path.join(DATA_DIR, COV_FILE))
    return ell, data[:, 1], cov


def build_forward():
    ell_np, _, _ = load_bandpowers()
    ell = jnp.asarray(ell_np)
    ev  = csl.cl_yy_factory(COSMO, ell)
    dl_factor = jnp.asarray(ell_np * (ell_np + 1) / (2 * np.pi) * 1e12)

    def forward(P0, beta):
        prof = csl.ProfileParamsA10(
            P0=P0, c500=C500, gamma=GAMMA, alpha=ALPHA,
            beta=beta, B=B_FIX,
        )
        cl1, cl2 = ev(prof)
        return dl_factor * (cl1 + cl2)
    return forward, ell_np


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48),
                 bounds=((0.1, 20.), (0.5, 10.))):
    y_j, ic = jnp.asarray(y), jnp.asarray(inv_cov)

    def nll(x):
        r = y_j - forward(x[0], x[1])
        return 0.5 * r @ ic @ r

    nll_j, ng = jax.jit(nll), jax.jit(jax.grad(nll))
    return so.minimize(lambda x: float(nll_j(x)), np.asarray(x0),
                        jac=lambda x: np.asarray(ng(x)),
                        method="L-BFGS-B", bounds=bounds)


# ---------------------------------------------------------------------------
# Fisher
# ---------------------------------------------------------------------------
def fisher_autodiff(forward, theta_bf, inv_cov):
    """One-shot Fisher matrix via jax.jacfwd.

    Returns the 2x2 Fisher matrix at theta_bf, plus the wall-time
    used (excluding JAX warmup if any).
    """
    inv_j = jnp.asarray(inv_cov)

    def mu(x):                          # x = (P0, β)
        return forward(x[0], x[1])

    # forward-mode Jacobian — fast when len(output) >> len(input)
    jac_fn = jax.jit(jax.jacfwd(mu))

    # Warm
    J0 = np.asarray(jac_fn(jnp.asarray(theta_bf)))
    J0.shape

    t0 = time.perf_counter()
    n_iter = 10
    for _ in range(n_iter):
        J = jax.jit(jax.jacfwd(mu))(jnp.asarray(theta_bf))
        J.block_until_ready()
    t_per = (time.perf_counter() - t0) / n_iter
    J = np.asarray(J)                              # (n_bp, 2)
    F = J.T @ inv_cov @ J                          # (2, 2)
    return F, J, t_per


def fisher_finite_diff(forward, theta_bf, inv_cov, eps=1e-3):
    """Reference Fisher via central finite differences (2 × N evals)."""
    n_par = len(theta_bf)
    J = np.zeros((forward(*theta_bf).shape[0], n_par))
    t0 = time.perf_counter()
    for i in range(n_par):
        th_p = list(theta_bf); th_p[i] += eps
        th_m = list(theta_bf); th_m[i] -= eps
        f_p = np.asarray(forward(*th_p))
        f_m = np.asarray(forward(*th_m))
        J[:, i] = (f_p - f_m) / (2 * eps)
    t_fd = time.perf_counter() - t0
    F = J.T @ inv_cov @ J
    return F, J, t_fd


# ---------------------------------------------------------------------------
# NUTS (matches nuts_clyy_profile.py at the same cosmology)
# ---------------------------------------------------------------------------
def run_nuts(forward, y, inv_cov, P0_init, beta_init,
             num_warmup=500, num_samples=2000, num_chains=4, seed=0):
    y_j, ic = jnp.asarray(y), jnp.asarray(inv_cov)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = y_j - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ ic @ r)

    mcmc = MCMC(NUTS(model, dense_mass=True),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=False)
    init = {"P0":   jnp.full((num_chains,), P0_init),
            "beta": jnp.full((num_chains,), beta_init)}
    mcmc.run(jax.random.PRNGKey(seed), init_params=init)
    return mcmc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ell_np, y, cov = load_bandpowers()
    inv_cov = np.linalg.inv(cov)
    print(f"Bandpowers: {len(ell_np)} bins")

    forward, _ = build_forward()
    print(f"Cosmology σ8 = {csl.derived(COSMO)['sigma_8']:.4f}")

    # ---- L-BFGS bestfit ---------------------------------------------------
    t0 = time.perf_counter()
    bf = find_bestfit(forward, y, inv_cov)
    print(f"L-BFGS-B bestfit in {time.perf_counter()-t0:.2f}s "
          f"→ P0={bf.x[0]:.3f}, β={bf.x[1]:.3f}")

    # ---- Fisher: autodiff (jax.jacfwd) -----------------------------------
    F_ad, J_ad, t_ad = fisher_autodiff(forward, bf.x, inv_cov)
    cov_ad = np.linalg.inv(F_ad)
    sigma_ad = np.sqrt(np.diag(cov_ad))
    print(f"Fisher (jax.jacfwd): {t_ad*1e3:.1f} ms / call (10-run avg)")
    print(f"  σ(P0) = {sigma_ad[0]:.3f}, σ(β) = {sigma_ad[1]:.3f}, "
          f"ρ = {cov_ad[0,1]/np.prod(sigma_ad):.3f}")

    # ---- Fisher: finite difference (reference) ---------------------------
    F_fd, J_fd, t_fd = fisher_finite_diff(forward, bf.x, inv_cov, eps=1e-3)
    sigma_fd = np.sqrt(np.diag(np.linalg.inv(F_fd)))
    print(f"Fisher (2-pt FD, eps=1e-3): {t_fd*1e3:.1f} ms")
    print(f"  σ(P0) = {sigma_fd[0]:.3f}, σ(β) = {sigma_fd[1]:.3f}")
    rel = np.max(np.abs(F_ad - F_fd) / np.abs(F_ad))
    print(f"  autodiff vs FD: max |ΔF|/|F| = {rel:.2e}")

    # ---- NUTS -----------------------------------------------------------
    t0 = time.perf_counter()
    mcmc = run_nuts(forward, y, inv_cov,
                    P0_init=float(bf.x[0]), beta_init=float(bf.x[1]))
    print(f"NUTS in {time.perf_counter()-t0:.1f}s")
    nuts = mcmc.get_samples()
    nuts_arr = np.column_stack([np.asarray(nuts["P0"]),
                                 np.asarray(nuts["beta"])])
    print(f"  NUTS posterior P0={nuts_arr[:,0].mean():.3f}±{nuts_arr[:,0].std():.3f}  "
          f"β={nuts_arr[:,1].mean():.3f}±{nuts_arr[:,1].std():.3f}")

    # ---- Plot Fisher ellipse over NUTS posterior -------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples
    from getdist import plots as gdplots
    from matplotlib.patches import Ellipse

    out_dir = os.path.dirname(__file__) or "."

    # Use getdist for the NUTS posterior (2D + 1D marginals)
    gd_nuts = MCSamples(
        samples=nuts_arr, names=["P0", "beta"],
        labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
        label="NUTS posterior",
    )
    g = gdplots.get_subplot_plotter(width_inch=5.5)
    g.settings.alpha_filled_add = 0.5
    g.settings.legend_fontsize  = 10
    g.triangle_plot([gd_nuts], params=["P0", "beta"],
                    filled=True, contour_colors=["C0"],
                    legend_labels=["NUTS posterior"])

    # Overlay Fisher ellipses in the 2D panel
    ax_2d = g.subplots[1, 0]
    # ellipse params from 2x2 cov
    eig_vals, eig_vecs = np.linalg.eigh(cov_ad)
    theta_deg = np.degrees(np.arctan2(eig_vecs[1, 1], eig_vecs[0, 1]))
    for n_sigma, label in [(np.sqrt(2.30), "68% Fisher"),
                            (np.sqrt(6.18), "95% Fisher")]:
        w = 2 * n_sigma * np.sqrt(eig_vals[1])
        h = 2 * n_sigma * np.sqrt(eig_vals[0])
        e = Ellipse(xy=bf.x, width=w, height=h, angle=theta_deg,
                    facecolor="none", edgecolor="C3", lw=1.8,
                    ls="--" if "68" in label else ":", label=label)
        ax_2d.add_patch(e)
    ax_2d.plot([bf.x[0]], [bf.x[1]], "x", color="C3", ms=8, mew=2,
               label="L-BFGS bestfit")
    ax_2d.legend(loc="upper left", fontsize=8)

    # Overlay 1D Gaussian on the diagonals
    for i, ax_diag in enumerate([g.subplots[0, 0], g.subplots[1, 1]]):
        x_lo, x_hi = ax_diag.get_xlim()
        xs = np.linspace(x_lo, x_hi, 400)
        gauss = np.exp(-0.5 * ((xs - bf.x[i]) / sigma_ad[i]) ** 2)
        # Match peak height to NUTS density max
        ymax = max([line.get_ydata().max()
                    for line in ax_diag.get_lines() if len(line.get_ydata())])
        ax_diag.plot(xs, gauss * ymax, "--", color="C3", lw=1.8,
                     label="Fisher" if i == 0 else None)
        if i == 0:
            ax_diag.legend(loc="upper right", fontsize=8)

    out = os.path.join(out_dir, "fisher_overlay.png")
    g.export(out, dpi=300)
    print(f"\n  -> wrote {out}")

    print("\nSummary:")
    print(f"  NUTS:                   "
          f"σ(P0) = {nuts_arr[:,0].std():.3f}   σ(β) = {nuts_arr[:,1].std():.3f}")
    print(f"  Fisher (jax.jacfwd):    "
          f"σ(P0) = {sigma_ad[0]:.3f}   σ(β) = {sigma_ad[1]:.3f}")
    print(f"  Wall: NUTS ~40 s, Fisher autodiff {t_ad*1e3:.1f} ms, "
          f"Fisher FD {t_fd*1e3:.1f} ms")


if __name__ == "__main__":
    main()
