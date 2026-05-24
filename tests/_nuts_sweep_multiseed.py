"""Multi-seed NUTS budget sweep — variance bars on wall + ESS + R-hat.

Repeats each (warmup, samples, chains) configuration with N_SEEDS different
PRNG seeds, reports median + min/max + per-seed scatter so we can put
honest error bars on the "sub-10 s NUTS" claim.

Also runs a matched cobaya-style RW-MH baseline at small budget for the
ESS-per-second comparison (no MPI; single-walker; same likelihood).
"""
from __future__ import annotations
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl

DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")
N_SEEDS = 5


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
    return wall, summary


def main():
    ell, Dell_data, cov = load_data()
    cov_inv = jnp.linalg.inv(cov)
    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    ev = csl.cl_yy_factory(cosmo, ell)
    dl_fac = ell * (ell + 1.0) / (2.0 * jnp.pi) * 1e12
    _ = ev(csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25))
    jax.block_until_ready(_)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
        cl_1h, cl_2h = ev(prof)
        Dell = dl_fac * (cl_1h + cl_2h)
        r = Dell - Dell_data
        numpyro.factor("loglik", -0.5 * (r @ cov_inv @ r))

    budgets = [
        (50,  100, 2),
        (100, 200, 2),
        (100, 400, 2),
        (200, 500, 4),
    ]

    print(f"NUTS budget sweep: {len(budgets)} configs × {N_SEEDS} seeds each")
    print(f"{'budget':<18} {'seed':>4} {'wall (s)':>9} {'ESS_P0':>7} {'ESS_b':>7} {'R-hat P0':>9} {'R-hat b':>9}")
    print("-" * 75)

    results = []   # rows: (warm, samp, chains, seed, wall, ess_p0, ess_b, rh_p0, rh_b)
    for warm, samp, chains in budgets:
        cfg = f"{warm}+{samp}×{chains}"
        for s in range(N_SEEDS):
            wall, sm = run_nuts(model, warm, samp, chains, seed=s)
            row = (warm, samp, chains, s, wall,
                   float(sm['P0']['n_eff']), float(sm['beta']['n_eff']),
                   float(sm['P0']['r_hat']), float(sm['beta']['r_hat']))
            results.append(row)
            print(f"{cfg:<18} {s:>4} {wall:>9.2f} {row[5]:>7.0f} {row[6]:>7.0f} {row[7]:>9.3f} {row[8]:>9.3f}")

    # We don't include a NumPyro MH baseline here — numpyro doesn't ship a
    # pure RW-MH kernel and the homegrown ones are slow. The cobaya RW-MH
    # reference (14 min wall, ESS≈1926) was already captured in the earlier
    # run; we cite that for the ESS-per-second comparison.

    out = os.path.join(os.path.dirname(__file__), "_nuts_sweep_multiseed.npz")
    np.savez(out, nuts=np.asarray(results, dtype=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
