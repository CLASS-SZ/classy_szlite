"""Profile-only NUTS + L-BFGS bestfit on Cl^yy bandpowers, for two
fitting cosmologies (baseline and lows8), with cobaya RW-MH overlays.

What this script does, for each of two fitting cosmologies that share
the same baseline ω_b, ω_cdm, n_s but differ in σ8 via (ln10_10_As, H0):

  1. L-BFGS-B bestfit of (P0, β) using scipy.minimize with exact
     jax.grad gradients on the classy_szlite.cl_yy_factory closure.
  2. NumPyro NUTS for the full posterior, initialised at the bestfit.
  3. (If a matching cobaya chain is present on disk) load it via
     getdist for an MH overlay.

Then writes:

  * ``synthetic_bestfit.png``  — bandpowers ± σ + 2 bestfit curves +
    2 NUTS 68% bands.
  * ``synthetic_corner.png``   — getdist triangle plot with 4
    contours: NUTS (this work) + cobaya RW-MH (baseline), for each
    cosmology.

Two cosmologies:

  * **baseline**: standard 6-param fixed cosmology used in the may26
    cobaya run (σ8 ≈ 0.81).
  * **lows8**: Flamingo low-S8 (ln10_10_As=2.910, H0=67.14) — σ8 ≈
    0.74. This shifts P0 upward to compensate for the reduced
    cluster abundance.

Cobaya chains (optional overlays) are expected at:

  * baseline → ``$CLYY_COBAYA_BASE`` / ``~/Desktop/class-sz-plugin-tests/chains/clyy_v2``
  * lows8    → ``$CLYY_COBAYA_LOWS8`` / ``~/Desktop/class-sz-plugin-tests/chains/clyy_v2_lows8``

Run from the classy_szlite repo root:

    python examples/nuts_clyy_profile.py

Requires (beyond classy_szlite): numpyro, scipy, getdist, matplotlib.
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
# Setup
# ---------------------------------------------------------------------------
DATA_DIR  = os.environ.get(
    "CLYY_DATA_DIR",
    os.path.expanduser("~/Desktop/class-sz-plugin-tests/data"),
)
ELL_FILE  = "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"
DATA_FILE = "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"
COV_FILE  = "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"

COBAYA_BASE_DEFAULT  = "~/Desktop/class-sz-plugin-tests/chains/clyy_v2"
COBAYA_LOWS8_DEFAULT = "~/Desktop/class-sz-plugin-tests/chains/clyy_v2_lows8"

# Two fitting cosmologies.
#   baseline: the may26 fixed-cosmology cobaya YAML
#   lows8:    the Flamingo low-S8 setup
#             (ln10_10_As=2.910, H0=67.14, omega_b/omega_cdm/n_s same)
FIT_COSMOS = [
    dict(
        key   = "baseline",
        label = "baseline (σ8≈0.81)",
        color = "C0",
        cobaya_root = os.environ.get("CLYY_COBAYA_BASE",  COBAYA_BASE_DEFAULT),
        cosmo_kwargs = dict(
            omega_b=0.0226, omega_cdm=0.118,
            H0=68.22, tau_reio=0.0561,
            ln10_10_As=3.06, n_s=0.9743,
        ),
    ),
    dict(
        key   = "lows8",
        label = "lows8 (σ8≈0.74)",
        color = "C3",
        cobaya_root = os.environ.get("CLYY_COBAYA_LOWS8", COBAYA_LOWS8_DEFAULT),
        cosmo_kwargs = dict(
            omega_b=0.0226, omega_cdm=0.118,
            H0=67.14116850291264, tau_reio=0.0561,
            ln10_10_As=2.910, n_s=0.9743,
        ),
    ),
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


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48),
                 bounds=((0.1, 20.), (0.5, 10.))):
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

    from getdist import MCSamples, loadMCSamples

    results = []
    for cfg in FIT_COSMOS:
        cosmo = csl.CosmoParams(**cfg["cosmo_kwargs"])
        s8 = csl.derived(cosmo)["sigma_8"]
        print()
        print(f"=== {cfg['label']}  [σ8={s8:.4f}] ===")
        forward = build_forward(cosmo, ell_np)

        # L-BFGS bestfit
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
        nuts_arr = np.column_stack([np.asarray(samples["P0"]),
                                     np.asarray(samples["beta"])])
        print(f"  NUTS posterior P0={np.mean(samples['P0']):.3f}±{np.std(samples['P0']):.3f}  "
              f"β={np.mean(samples['beta']):.3f}±{np.std(samples['beta']):.3f}")

        # cobaya RW-MH overlay (if chain present)
        cobaya_root = os.path.expanduser(cfg["cobaya_root"])
        if os.path.isfile(cobaya_root + ".1.txt"):
            cobaya_gd = loadMCSamples(cobaya_root, settings={"ignore_rows": 0.3})
            print(f"  cobaya MH overlay loaded from {cobaya_root}*.txt  "
                  f"(n={cobaya_gd.numrows})")
        else:
            cobaya_gd = None
            print(f"  no cobaya chain at {cobaya_root}*.txt — skipping MH overlay")

        results.append({
            "cfg":       cfg,
            "cosmo":     cosmo,
            "sigma_8":   s8,
            "forward":   forward,
            "bf":        bf,
            "chi2_bf":   chi2_bf,
            "samples":   samples,
            "nuts_arr":  nuts_arr,
            "cobaya_gd": cobaya_gd,
        })

    # ---------- plots --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import plots as gdplots

    out_dir = os.path.dirname(__file__) or "."

    # 1) Cl^yy: bandpowers + bestfit curve + NUTS 68% band per cosmology
    ell_dense = np.geomspace(50, 9000, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.7), dpi=300)
    sigma_bp = np.sqrt(np.diag(cov))
    ax.errorbar(ell_np, y, yerr=sigma_bp, fmt="ko", capsize=3, ms=4,
                label="bandpower data", zorder=10)
    for r in results:
        forward_dense = build_forward(r["cosmo"], ell_dense)
        # Bestfit
        dl_bf = np.asarray(forward_dense(jnp.asarray(r["bf"].x[0]),
                                          jnp.asarray(r["bf"].x[1])))
        # 68% band from 500 NUTS samples
        idx = np.random.default_rng(0).choice(len(r["nuts_arr"]), 500, replace=False)
        mus = np.stack([np.asarray(forward_dense(jnp.asarray(r["nuts_arr"][i, 0]),
                                                  jnp.asarray(r["nuts_arr"][i, 1])))
                        for i in idx])
        lo, hi = np.percentile(mus, [16, 84], axis=0)
        c = r["cfg"]["color"]
        ax.fill_between(ell_dense, lo, hi, color=c, alpha=0.25,
                        label=f"{r['cfg']['label']} 68% (NUTS)")
        ax.plot(ell_dense, dl_bf, "-", color=c, lw=2,
                label=f"{r['cfg']['label']} bestfit "
                      f"(P0={r['bf'].x[0]:.2f}, β={r['bf'].x[1]:.2f}, "
                      f"χ²={r['chi2_bf']:.1f}/{len(y)-2})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$10^{12}\, D_\ell^{yy}$")
    ax.set_title("Bestfit + NUTS 68% band on $C_\\ell^{yy}$ bandpowers")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8.5, loc="lower center")
    out_bf = os.path.join(out_dir, "synthetic_bestfit.png")
    fig.savefig(out_bf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> wrote {out_bf}")

    # 2) Triangle plot: 4 contours (2 NUTS + 2 MH) per cosmology
    gd_samples = []
    labels     = []
    colors     = []
    contour_ls = []
    filled     = []
    for r in results:
        c = r["cfg"]["color"]
        # NUTS (solid filled)
        nuts_gd = MCSamples(
            samples=r["nuts_arr"],
            names=["P0", "beta"],
            labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
            label=f"NUTS — {r['cfg']['label']}",
        )
        gd_samples.append(nuts_gd)
        labels.append(nuts_gd.label)
        colors.append(c)
        contour_ls.append("-"); filled.append(True)
        # cobaya MH (dashed, unfilled). Extract the (P0GNFW, betaGNFW)
        # columns into a fresh MCSamples that aligns with the NUTS one.
        if r["cobaya_gd"] is not None:
            src = r["cobaya_gd"]
            p0  = src.samples[:, src.paramNames.numberOfName("P0GNFW")]
            bt  = src.samples[:, src.paramNames.numberOfName("betaGNFW")]
            mh = MCSamples(
                samples=np.column_stack([p0, bt]),
                weights=src.weights,
                names=["P0", "beta"],
                labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
                label=f"cobaya MH — {r['cfg']['label']}",
            )
            gd_samples.append(mh)
            labels.append(mh.label)
            colors.append(c)
            contour_ls.append("--"); filled.append(False)

    g = gdplots.get_subplot_plotter(width_inch=5.5)
    g.settings.alpha_filled_add = 0.4
    g.settings.legend_fontsize  = 10
    g.triangle_plot(
        gd_samples,
        params=["P0", "beta"],
        filled=filled,
        legend_labels=labels,
        contour_colors=colors,
        contour_ls=contour_ls,
    )
    out_corner = os.path.join(out_dir, "synthetic_corner.png")
    g.export(out_corner, dpi=300)
    print(f"  -> wrote {out_corner}")


if __name__ == "__main__":
    main()
