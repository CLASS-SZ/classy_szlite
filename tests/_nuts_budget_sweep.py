"""Sweep NUTS budget vs wall time / ESS / R-hat for the baseline posterior.

Output: tests/_nuts_sweep.csv + tests/nuts_sweep.png
       (wall vs ESS / R-hat trade-off).
"""
from __future__ import annotations
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import classy_szlite as csl

DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")


def load_data():
    ell  = np.loadtxt(os.path.join(DATA_DIR, "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    bp   = np.loadtxt(os.path.join(DATA_DIR, "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"))
    cov  = np.loadtxt(os.path.join(DATA_DIR, "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    return jnp.asarray(ell), jnp.asarray(bp[:, 1]), jnp.asarray(cov)


def run_nuts(model, n_warm, n_samp, chains, seed):
    kernel = NUTS(model, dense_mass=True, max_tree_depth=5, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=n_samp,
                num_chains=chains, chain_method='sequential', progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed))
    wall = time.perf_counter() - t0
    samples = mcmc.get_samples(group_by_chain=True)
    summary = numpyro.diagnostics.summary(samples)
    return wall, summary, samples


def main():
    print(f"Backend: {jax.default_backend()}")
    ell, Dell_data, cov = load_data()
    cov_inv = jnp.linalg.inv(cov)
    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    ev = csl.cl_yy_factory(cosmo, ell)
    dl_fac = ell * (ell + 1.0) / (2.0 * jnp.pi) * 1e12
    # warm up the closure
    _ = ev(csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)); jax.block_until_ready(_)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
        cl_1h, cl_2h = ev(prof)
        Dell = dl_fac * (cl_1h + cl_2h)
        r = Dell - Dell_data
        numpyro.factor("loglik", -0.5 * (r @ cov_inv @ r))

    # Budget sweep — explore the trade-off
    sweep = [
        # (warmup, samples, chains)
        (50,  100, 1),
        (50,  100, 2),
        (50,  200, 2),
        (100, 200, 2),
        (100, 400, 2),
        (200, 500, 2),
        (200, 1000, 4),
    ]

    rows = []
    header = f"{'warm':>5} {'samp':>5} {'ch':>3} {'wall (s)':>9} {'ESS_P0':>7} {'ESS_b':>7} {'R-hat P0':>9} {'R-hat b':>9}"
    print(header); print("-" * len(header))
    for n_warm, n_samp, chains in sweep:
        wall, sm, samples = run_nuts(model, n_warm, n_samp, chains, seed=0)
        P0_ess, P0_rh = float(sm['P0']['n_eff']),   float(sm['P0']['r_hat'])
        be_ess, be_rh = float(sm['beta']['n_eff']), float(sm['beta']['r_hat'])
        print(f"{n_warm:>5} {n_samp:>5} {chains:>3} {wall:>9.2f} {P0_ess:>7.0f} {be_ess:>7.0f} {P0_rh:>9.3f} {be_rh:>9.3f}")
        rows.append((n_warm, n_samp, chains, wall, P0_ess, be_ess, P0_rh, be_rh))

    out = os.path.join(os.path.dirname(__file__), "_nuts_sweep.npz")
    np.savez(out, sweep=np.asarray(rows, dtype=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
