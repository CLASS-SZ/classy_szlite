"""Profile-only NUTS + L-BFGS bestfit on a Cl^yy bandpower dataset,
with a low-σ8 sweep.

What this script does:

1.  Loads a tSZ Cl^yy bandpower dataset (bandpowers + covariance + ell
    grid).  The bandpowers are treated as a generic synthetic dataset
    for the purposes of this example.

2.  For each of three test cosmologies (high, medium, low σ8 — tuned
    via ``ln10_10_As``), assumed *fixed* at fit time, it
        (a) finds the bestfit ``(P0, beta)`` via L-BFGS-B with exact
            ``jax.grad`` gradients (no finite differences),
        (b) draws the full posterior with NumPyro NUTS using
            ``classy_szlite.cl_yy_factory`` as the gradient-friendly
            forward model (~5 ms / leapfrog step).

3.  Writes two figures:
       * ``synthetic_bestfit.png`` — bandpowers ± σ + bestfit curves
         for all three cosmologies overlaid
       * ``synthetic_corner.png``  — getdist triangle plot of the
         three NUTS posteriors

This is a clean demonstration of the well-known σ8 ↔ P0 degeneracy:
lower σ8 in the *fitting* cosmology forces the bestfit ``P0`` upward
to match the observed bandpower amplitude.

Run from the classy_szlite repo root:

    python examples/nuts_clyy_profile.py

Requires (beyond classy_szlite): numpyro, scipy, getdist, matplotlib.
Emulator data at $CLASSY_SZLITE_DATA_DIR or ~/class_sz_data/.

The bandpower dataset path is taken from $CLYY_DATA_DIR
(default: ~/Desktop/class-sz-plugin-tests/data).
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
# Setup
# ---------------------------------------------------------------------------
DATA_DIR  = os.environ.get(
    "CLYY_DATA_DIR",
    os.path.expanduser("~/Desktop/class-sz-plugin-tests/data"),
)
ELL_FILE  = "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"
DATA_FILE = "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"
COV_FILE  = "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"

# Reference Planck-18 LCDM-equivalent base cosmology; the σ8 sweep just
# varies ln10_10_As around this.
BASE_COSMO = csl.CosmoParams(
    omega_b=0.0226, omega_cdm=0.118, H0=68.22,
    tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743,
)

# Three fit cosmologies — only ln10_10_As varies, generating a σ8 ladder.
FIT_COSMOS = [
    dict(label="high σ8 (≈0.81)",  ln10_10_As=3.060, color="C0"),
    dict(label="medium σ8 (≈0.77)", ln10_10_As=2.950, color="C2"),
    dict(label="low σ8 (≈0.74)",    ln10_10_As=2.850, color="C3"),
]


def load_bandpowers():
    """Load the Cl^yy bandpower dataset: ell, y (D_ell × 1e12), full cov."""
    ell  = np.loadtxt(os.path.join(DATA_DIR, ELL_FILE))
    data = np.loadtxt(os.path.join(DATA_DIR, DATA_FILE))      # 3 cols: ell, y, σ
    cov  = np.loadtxt(os.path.join(DATA_DIR, COV_FILE))
    assert cov.shape == (len(ell), len(ell))
    return ell, data[:, 1], cov                               # y in D_ell × 1e12


# ---------------------------------------------------------------------------
# Forward model + bestfit / NUTS for a single fit cosmology
# ---------------------------------------------------------------------------
def build_forward(cosmo, ell_np):
    ell = jnp.asarray(ell_np)
    ev  = csl.cl_yy_factory(cosmo, ell)
    dl_factor = jnp.asarray(ell * (ell + 1) / (2 * np.pi) * 1e12)
    C500, GAMMA, ALPHA, B_FIX = 1.156, 0.3292, 1.062, 1.25     # cobaya may26 baseline

    def forward(P0, beta):
        prof = csl.ProfileParamsA10(
            P0=P0, c500=C500, gamma=GAMMA, alpha=ALPHA, beta=beta, B=B_FIX,
        )
        cl_1h, cl_2h = ev(prof)
        return dl_factor * (cl_1h + cl_2h)
    return forward


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48), bounds=((0.1, 20.), (0.5, 10.))):
    """L-BFGS-B bestfit using exact jax.grad gradients."""
    y_jnp   = jnp.asarray(y)
    inv_jnp = jnp.asarray(inv_cov)

    def neg_log_like(x):
        mu = forward(x[0], x[1])
        r  = y_jnp - mu
        return 0.5 * r @ inv_jnp @ r

    nll      = jax.jit(neg_log_like)
    nll_grad = jax.jit(jax.grad(neg_log_like))

    res = so.minimize(
        lambda x: float(nll(x)),
        np.asarray(x0, dtype=np.float64),
        jac=lambda x: np.asarray(nll_grad(x)),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return res                                                # res.x = (P0_bf, beta_bf)


def run_nuts(forward, y, inv_cov, P0_init=8.13, beta_init=5.48,
             num_warmup=500, num_samples=2000, num_chains=4, seed=0):
    y_jnp   = jnp.asarray(y)
    inv_jnp = jnp.asarray(inv_cov)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        mu   = forward(P0, beta)
        r    = y_jnp - mu
        numpyro.factor("loglike", -0.5 * r @ inv_jnp @ r)

    kernel = NUTS(model, target_accept_prob=0.85, dense_mass=True)
    mcmc = MCMC(kernel,
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=False)
    # init_params must broadcast over chains
    init = {
        "P0":   jnp.full((num_chains,), P0_init),
        "beta": jnp.full((num_chains,), beta_init),
    }
    mcmc.run(jax.random.PRNGKey(seed), init_params=init)
    return mcmc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ell_np, y, cov = load_bandpowers()
    inv_cov = np.linalg.inv(cov)
    print(f"Bandpowers: {len(ell_np)} bins, ell ∈ [{ell_np.min():.0f}, {ell_np.max():.0f}]")

    # Per-cosmology bestfit + NUTS
    results = []
    for cfg in FIT_COSMOS:
        cosmo = csl.CosmoParams(
            omega_b=BASE_COSMO.omega_b, omega_cdm=BASE_COSMO.omega_cdm,
            H0=BASE_COSMO.H0, tau_reio=BASE_COSMO.tau_reio,
            ln10_10_As=cfg["ln10_10_As"], n_s=BASE_COSMO.n_s,
        )
        s8 = csl.derived(cosmo)["sigma_8"]
        print()
        print(f"=== {cfg['label']}  [σ8={s8:.4f}] ===")
        forward = build_forward(cosmo, ell_np)

        # Bestfit
        t0 = time.perf_counter()
        bf = find_bestfit(forward, y, inv_cov)
        chi2_bf = 2 * bf.fun
        print(f"  L-BFGS-B bestfit in {time.perf_counter()-t0:.2f} s, "
              f"{bf.nfev} fn evals → P0={bf.x[0]:.3f}, β={bf.x[1]:.3f}, "
              f"χ²={chi2_bf:.2f}  ({len(y)-2} dof)")

        # NUTS, init at bestfit for fast warmup
        t0 = time.perf_counter()
        mcmc = run_nuts(forward, y, inv_cov,
                        P0_init=float(bf.x[0]), beta_init=float(bf.x[1]))
        print(f"  NUTS in {time.perf_counter()-t0:.1f} s")
        samples = mcmc.get_samples()
        print(f"  posterior P0={np.mean(samples['P0']):.3f}±{np.std(samples['P0']):.3f}  "
              f"β={np.mean(samples['beta']):.3f}±{np.std(samples['beta']):.3f}")

        results.append({
            "cfg":     cfg,
            "cosmo":   cosmo,
            "sigma_8": s8,
            "forward": forward,
            "bf":      bf,
            "chi2_bf": chi2_bf,
            "samples": samples,
        })

    # ---------- plots --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples
    from getdist import plots as gdplots

    out_dir = os.path.dirname(__file__) or "."

    # 1) Bestfit vs bandpowers
    ell_dense = np.geomspace(50, 9000, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.7), dpi=300)
    sigma = np.sqrt(np.diag(cov))
    ax.errorbar(ell_np, y, yerr=sigma, fmt="ko", capsize=3, ms=4,
                label="bandpower data", zorder=5)
    for r in results:
        forward_dense = build_forward(r["cosmo"], ell_dense)
        dl_bf = np.asarray(forward_dense(jnp.asarray(r["bf"].x[0]),
                                          jnp.asarray(r["bf"].x[1])))
        ax.plot(ell_dense, dl_bf, "-", color=r["cfg"]["color"], lw=2,
                label=f"bestfit @ σ8={r['sigma_8']:.3f}  "
                      f"(P0={r['bf'].x[0]:.2f}, β={r['bf'].x[1]:.2f}, "
                      f"χ²={r['chi2_bf']:.1f}/{len(y)-2})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$10^{12}\, D_\ell^{yy}$")
    ax.set_title("L-BFGS bestfit on $C_\\ell^{yy}$ bandpowers "
                 "across a σ8 sweep")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8.5, loc="lower center")
    out_bf = os.path.join(out_dir, "synthetic_bestfit.png")
    fig.savefig(out_bf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> wrote {out_bf}")

    # 2) Triangle plot of the three NUTS posteriors
    gd_samples = []
    for r in results:
        arr = np.column_stack([np.asarray(r["samples"]["P0"]),
                                np.asarray(r["samples"]["beta"])])
        gd = MCSamples(
            samples=arr,
            names=["P0", "beta"],
            labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
            label=f"σ8 = {r['sigma_8']:.3f}",
        )
        gd_samples.append(gd)

    g = gdplots.get_subplot_plotter(width_inch=5.5)
    g.settings.alpha_filled_add = 0.4
    g.settings.legend_fontsize  = 11
    g.triangle_plot(
        gd_samples,
        params=["P0", "beta"],
        filled=True,
        legend_labels=[s.label for s in gd_samples],
        contour_colors=[r["cfg"]["color"] for r in results],
    )
    out_corner = os.path.join(out_dir, "synthetic_corner.png")
    g.export(out_corner, dpi=300)
    print(f"  -> wrote {out_corner}")


if __name__ == "__main__":
    main()
