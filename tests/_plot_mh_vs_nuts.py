"""Overlay cobaya RW-MH and NumPyro NUTS posteriors on the same
Cholesky-synthetic-data (A10 fiducial, noise_factor=9) bandpower
likelihood. Output: docs/_static/posterior_compare.png
"""
from __future__ import annotations
import os, time, shutil, tempfile
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.optimize as so
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots, loadMCSamples
import classy_szlite as csl


# ---- shared setup (matches docs/examples.md exactly) -------------------
COSMO    = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                           tau_reio=0.0561, ln10_10_As=3.060, n_s=0.9743)
FIDUCIAL = csl.ProfileParamsA10(P0=8.130, c500=1.156, gamma=0.3292,
                                 alpha=1.062, beta=5.4807, B=1.25)
ELL          = jnp.geomspace(100.0, 5000.0, 8)
DELTA_ELL    = ELL * jnp.log(ELL[1] / ELL[0])
FSKY         = 0.6
NOISE_FACTOR = 9
EV           = csl.cl_yy_factory(COSMO, ELL)
DL_FACTOR    = ELL * (ELL + 1) / (2 * jnp.pi) * 1e12
_COV_CL      = csl.cl_yy_covariance(COSMO, FIDUCIAL, ELL, DELTA_ELL, fsky=FSKY)
COV          = _COV_CL * (DL_FACTOR[:, None] * DL_FACTOR[None, :]) * NOISE_FACTOR
INV_COV      = np.asarray(jnp.linalg.inv(COV))    # numpy → cobaya-friendly
L_CHOL       = jnp.linalg.cholesky(COV)
_C1, _C2     = EV(FIDUCIAL)
DELL_FID     = DL_FACTOR * (_C1 + _C2)
DELL_DATA_J  = DELL_FID + L_CHOL @ jax.random.normal(jax.random.PRNGKey(42), ELL.shape)
DELL_DATA    = np.asarray(DELL_DATA_J)


def forward_jax(P0, beta):
    prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                 alpha=1.062, beta=beta, B=1.25)
    c1, c2 = EV(prof)
    return DL_FACTOR * (c1 + c2)


def forward_np(P0, beta):
    """Numpy-returning wrapper for cobaya (which calls into native Python)."""
    return np.asarray(forward_jax(P0, beta))


# ---- L-BFGS bestfit (for NUTS init + MH proposal scale) ----------------
def find_bestfit():
    def nll(x):
        r = DELL_DATA - forward_jax(x[0], x[1])
        return 0.5 * r @ INV_COV @ r
    nllj = jax.jit(nll); gj = jax.jit(jax.grad(nll))
    return so.minimize(lambda x: float(nllj(x)),
                       np.asarray([8.13, 5.48], dtype=np.float64),
                       jac=lambda x: np.asarray(gj(x)),
                       method="L-BFGS-B", bounds=[(0.1, 20), (0.5, 10)])


# ---- NUTS (same as docs example) ---------------------------------------
def run_nuts(bf):
    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = DELL_DATA - forward_jax(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ INV_COV @ r)
    mcmc = MCMC(NUTS(model, dense_mass=True), num_warmup=300, num_samples=1000,
                num_chains=2, chain_method="sequential", progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0),
             init_params={"P0":   jnp.full(2, float(bf.x[0])),
                          "beta": jnp.full(2, float(bf.x[1]))})
    wall = time.perf_counter() - t0
    s = mcmc.get_samples()
    return np.asarray(s["P0"]), np.asarray(s["beta"]), wall


# ---- cobaya RW-MH on the same likelihood -------------------------------
def cobaya_loglike(P0, beta):
    """cobaya likelihood callable. Returns log L."""
    r = DELL_DATA - forward_np(P0, beta)
    return -0.5 * float(r @ INV_COV @ r)


def run_mh(bf, tmpdir):
    from cobaya.run import run as cobaya_run
    # Tight reference covariance: bestfit ± 3σ_Fisher
    info = {
        "likelihood": {"clyy_synth": cobaya_loglike},
        "params": {
            "P0":   {"prior": {"min": 0.0, "max": 20.0},
                     "ref":   {"dist": "norm",
                                "loc":   float(bf.x[0]), "scale": 0.5},
                     "proposal": 0.4,
                     "latex": "P_0"},
            "beta": {"prior": {"min": 0.0, "max": 10.0},
                     "ref":   {"dist": "norm",
                                "loc":   float(bf.x[1]), "scale": 0.2},
                     "proposal": 0.15,
                     "latex": r"\beta"},
        },
        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.05,
                "max_tries": 50000,
                "burn_in": 200,
                "learn_proposal": True,
                "proposal_scale": 1.9,
            },
        },
        "output": os.path.join(tmpdir, "mh_chain"),
        "debug": False,
    }
    t0 = time.perf_counter()
    cobaya_run(info, force=True)
    wall = time.perf_counter() - t0
    sm = loadMCSamples(os.path.join(tmpdir, "mh_chain"),
                        settings={"ignore_rows": 0.2})
    return sm, wall


def main():
    bf = find_bestfit()
    print(f"bestfit: P0={bf.x[0]:.3f}  β={bf.x[1]:.3f}  χ²={2*bf.fun:.2f}/6")

    print("\n--- NUTS ---")
    P0_n, be_n, wall_n = run_nuts(bf)
    print(f"  P0={P0_n.mean():.2f}±{P0_n.std():.2f}  β={be_n.mean():.2f}±{be_n.std():.2f}  "
          f"corr={np.corrcoef(P0_n, be_n)[0,1]:+.2f}  wall={wall_n:.1f}s")

    print("\n--- cobaya RW-MH ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        mh_samples, wall_mh = run_mh(bf, tmpdir)
    P0_m = mh_samples.samples[:, mh_samples.paramNames.numberOfName("P0")]
    be_m = mh_samples.samples[:, mh_samples.paramNames.numberOfName("beta")]
    w_m  = mh_samples.weights
    print(f"  P0={np.average(P0_m, weights=w_m):.2f}±"
          f"{np.sqrt(np.average((P0_m - np.average(P0_m, weights=w_m))**2, weights=w_m)):.2f}  "
          f"β={np.average(be_m, weights=w_m):.2f}±"
          f"{np.sqrt(np.average((be_m - np.average(be_m, weights=w_m))**2, weights=w_m)):.2f}  "
          f"n_eff(Kish)≈{(w_m.sum()**2/(w_m**2).sum()):.0f}  wall={wall_mh:.1f}s")

    # ---- Triangle overlay ----
    nuts = MCSamples(samples=np.column_stack([P0_n, be_n]),
                      names=["P0", "beta"], labels=["P_0", r"\beta"],
                      label="NumPyro NUTS")
    g = plots.get_subplot_plotter(width_inch=6)
    g.settings.alpha_filled_add = 0.55
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 10
    g.settings.legend_fontsize = 9
    g.triangle_plot(
        [mh_samples, nuts], params=["P0", "beta"], filled=True,
        contour_colors=["C3", "C0"],
        legend_labels=[
            f"cobaya RW-MH (t={wall_mh:.0f}s, n_eff={(w_m.sum()**2/(w_m**2).sum()):.0f})",
            f"NumPyro NUTS (t={wall_n:.0f}s, ESS={len(P0_n)//2})",
        ],
        markers={"P0": 8.130, "beta": 5.4807},
    )
    out = os.path.expanduser("~/classy_szlite/docs/_static/posterior_compare.png")
    g.export(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
