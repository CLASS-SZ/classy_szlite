"""Self-similar GNFW profile P(x) × x² — fiducial A10 vs posterior bands
from the Cl^yy bandpower fit (baseline + lows8 cosmologies).

The GNFW pressure profile in dimensionless form is

    p(x) = P0 * (c500 * x)^(-γ) * [1 + (c500 * x)^α] ^ (-(β-γ)/α)

where x = r / r_500.  Of the five GNFW parameters, the Cl^yy
bandpower fit in ``nuts_clyy_profile.py`` holds (c500, γ, α) fixed
at the A10 best-fit values and only samples (P0, β).  So the
posterior on p(x) is just the family of curves parameterised by
the marginal (P0, β) draws.

This script:

  1. Runs the same NUTS setup as ``nuts_clyy_profile.py`` for both
     the baseline (σ8 ≈ 0.81) and lows8 (σ8 ≈ 0.75) cosmologies.
  2. Draws 500 random posterior samples of (P0, β) from each chain
     and evaluates p(x) on a log-spaced x grid.
  3. Plots the **fiducial A10 profile** (single black line) plus a
     **median curve + 1σ shaded band** for each cosmology.

The y-axis is ``p(x) × x²`` rather than plain p(x) — the x²
weighting flattens the inner power-law fall-off and makes the
outer-slope (β) differences much easier to read.

Run from the classy_szlite repo root:

    python examples/profile_bands.py

Requires (beyond classy_szlite): numpyro, scipy, matplotlib.
Emulator data at $CLASSY_SZLITE_DATA_DIR or ~/class_sz_data/.
"""
from __future__ import annotations
import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.optimize as so

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl

# ---------------------------------------------------------------------------
# Setup (mirrors examples/nuts_clyy_profile.py)
# ---------------------------------------------------------------------------
DATA_DIR  = os.environ.get(
    "CLYY_DATA_DIR",
    os.path.expanduser("~/Desktop/class-sz-plugin-tests/data"),
)
ELL_FILE  = "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"
DATA_FILE = "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"
COV_FILE  = "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"

FIT_COSMOS = [
    dict(label="baseline (σ8≈0.81)", color="C0",
         cosmo_kwargs=dict(omega_b=0.0226, omega_cdm=0.118,
                           H0=68.22, tau_reio=0.0561,
                           ln10_10_As=3.06, n_s=0.9743)),
    dict(label="lows8 (σ8≈0.75)",    color="C3",
         cosmo_kwargs=dict(omega_b=0.0226, omega_cdm=0.118,
                           H0=67.14116850291264, tau_reio=0.0561,
                           ln10_10_As=2.910, n_s=0.9743)),
]

# Fixed Arnaud-10 shape (matches the fit in nuts_clyy_profile.py)
C500_FIX, GAMMA_FIX, ALPHA_FIX, B_FIX = 1.156, 0.3292, 1.062, 1.25
# Reference A10 values for P0 and β (the fiducial profile in the plot)
P0_A10, BETA_A10 = 8.130, 5.4807


# ---------------------------------------------------------------------------
# Bandpowers + factory closure + NUTS  (lightly trimmed copy of the
# helpers in nuts_clyy_profile.py — kept self-contained on purpose)
# ---------------------------------------------------------------------------
def load_bandpowers():
    ell  = np.loadtxt(os.path.join(DATA_DIR, ELL_FILE))
    data = np.loadtxt(os.path.join(DATA_DIR, DATA_FILE))
    cov  = np.loadtxt(os.path.join(DATA_DIR, COV_FILE))
    return ell, data[:, 1], cov


def build_forward(cosmo, ell_np):
    ell = jnp.asarray(ell_np)
    ev  = csl.cl_yy_factory(cosmo, ell)
    dl_factor = jnp.asarray(ell * (ell + 1) / (2 * np.pi) * 1e12)

    def forward(P0, beta):
        prof = csl.ProfileParamsA10(
            P0=P0, c500=C500_FIX, gamma=GAMMA_FIX, alpha=ALPHA_FIX,
            beta=beta, B=B_FIX,
        )
        cl_1h, cl_2h = ev(prof)
        return dl_factor * (cl_1h + cl_2h)
    return forward


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48), bounds=((0.1, 20.), (0.5, 10.))):
    y_jnp   = jnp.asarray(y)
    inv_jnp = jnp.asarray(inv_cov)

    def nll(x):
        r = y_jnp - forward(x[0], x[1])
        return 0.5 * r @ inv_jnp @ r

    nll_j  = jax.jit(nll)
    nll_g  = jax.jit(jax.grad(nll))
    return so.minimize(
        lambda x: float(nll_j(x)),
        np.asarray(x0, dtype=np.float64),
        jac=lambda x: np.asarray(nll_g(x)),
        method="L-BFGS-B",
        bounds=bounds,
    )


def run_nuts(forward, y, inv_cov, P0_init, beta_init,
             num_warmup=500, num_samples=2000, num_chains=4, seed=0):
    y_jnp   = jnp.asarray(y)
    inv_jnp = jnp.asarray(inv_cov)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = y_jnp - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ inv_jnp @ r)

    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.85, dense_mass=True),
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, chain_method="sequential",
        progress_bar=False,
    )
    init = {"P0":   jnp.full((num_chains,), P0_init),
            "beta": jnp.full((num_chains,), beta_init)}
    mcmc.run(jax.random.PRNGKey(seed), init_params=init)
    return mcmc


# ---------------------------------------------------------------------------
# GNFW dimensionless profile p(x)
# ---------------------------------------------------------------------------
def gnfw_pofx(x, P0, beta, c500=C500_FIX, gamma=GAMMA_FIX, alpha=ALPHA_FIX):
    """Dimensionless GNFW pressure profile p(x) = P / P_500 (Arnaud-10).

    p(x) = P0 * (c500 x)^(-γ) * [1 + (c500 x)^α] ^ (-(β-γ)/α)
    """
    cx = c500 * x
    return P0 * cx ** (-gamma) * (1.0 + cx ** alpha) ** (-(beta - gamma) / alpha)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ell_np, y, cov = load_bandpowers()
    inv_cov = np.linalg.inv(cov)
    print(f"Bandpowers: {len(ell_np)} bins, ell ∈ [{ell_np.min():.0f}, {ell_np.max():.0f}]")

    results = []
    for cfg in FIT_COSMOS:
        cosmo = csl.CosmoParams(**cfg["cosmo_kwargs"])
        s8 = csl.derived(cosmo)["sigma_8"]
        print(f"\n=== {cfg['label']}  [σ8={s8:.4f}] ===")
        forward = build_forward(cosmo, ell_np)

        t0 = time.perf_counter()
        bf = find_bestfit(forward, y, inv_cov)
        print(f"  L-BFGS bestfit in {time.perf_counter()-t0:.1f}s "
              f"→ P0={bf.x[0]:.3f}, β={bf.x[1]:.3f}")

        t0 = time.perf_counter()
        mcmc = run_nuts(forward, y, inv_cov,
                        P0_init=float(bf.x[0]), beta_init=float(bf.x[1]))
        print(f"  NUTS in {time.perf_counter()-t0:.1f}s")
        samples = mcmc.get_samples()
        results.append({
            "cfg":     cfg,
            "sigma_8": s8,
            "P0":      np.asarray(samples["P0"]),
            "beta":    np.asarray(samples["beta"]),
        })

    # -------------------------------------------------------------------
    # Build the p(x) bands
    # -------------------------------------------------------------------
    x = np.geomspace(1e-2, 5.0, 200)
    n_draws = 500
    rng = np.random.default_rng(0)
    for r in results:
        idx = rng.choice(len(r["P0"]), size=n_draws, replace=False)
        # Vectorised: outer-product over (sample, x)
        P0_s   = r["P0"][idx][:, None]                     # (N, 1)
        beta_s = r["beta"][idx][:, None]                   # (N, 1)
        p_s = gnfw_pofx(x[None, :], P0_s, beta_s)          # (N, n_x)
        r["p_med"] = np.median(p_s, axis=0)
        r["p_lo"]  = np.percentile(p_s, 16, axis=0)
        r["p_hi"]  = np.percentile(p_s, 84, axis=0)

    # Fiducial A10
    p_a10 = gnfw_pofx(x, P0_A10, BETA_A10)

    # -------------------------------------------------------------------
    # Plot — p(x) × x² in log-log
    # -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.7), dpi=300)
    x2 = x ** 2

    ax.plot(x, p_a10 * x2, "k-", lw=2,
            label=f"fiducial A10  (P0={P0_A10}, β={BETA_A10})")

    for r in results:
        c = r["cfg"]["color"]
        ax.plot(x, r["p_med"] * x2, "-", color=c, lw=1.8,
                label=f"{r['cfg']['label']} median  "
                      f"(P0={np.mean(r['P0']):.2f}±{np.std(r['P0']):.2f}, "
                      f"β={np.mean(r['beta']):.2f}±{np.std(r['beta']):.2f})")
        ax.fill_between(x, r["p_lo"] * x2, r["p_hi"] * x2,
                        color=c, alpha=0.25,
                        label=f"{r['cfg']['label']} 68% (NUTS)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$x = r / r_{500}$")
    ax.set_ylabel(r"$\mathbb{P}(x)\,x^2$  (dimensionless, A10 self-similar)")
    ax.set_title("GNFW pressure profile from $C_\\ell^{yy}$ NUTS posteriors")
    ax.set_xlim(x.min(), x.max())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower center")

    out = os.path.join(os.path.dirname(__file__) or ".", "profile_bands.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> wrote {out}")


if __name__ == "__main__":
    main()
