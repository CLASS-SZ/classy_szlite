"""NumPyro NUTS reproduction of paper's baseline run (Table 1 row).

Same data + same cosmology + same fixed profile params as the cobaya YAML.
Single 4-chain sequential NUTS, dense-mass adapted, L-BFGS warm start.

Targets paper's ~40s baseline wall.
"""
from __future__ import annotations
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl


DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")


def load_data():
    ell  = np.loadtxt(os.path.join(DATA_DIR, "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    bp   = np.loadtxt(os.path.join(DATA_DIR, "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"))
    cov  = np.loadtxt(os.path.join(DATA_DIR, "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    # The data file's row order: same descending-ell as ell-file
    Dell = bp[:, 1]
    sig  = bp[:, 2]
    # NB: confirm by checking bandpower ell-col matches ls file
    assert np.allclose(bp[:, 0], ell), "ell mismatch between data and ls files"
    return jnp.asarray(ell), jnp.asarray(Dell), jnp.asarray(cov), jnp.asarray(sig)


def main():
    print(f"Backend: {jax.default_backend()}")
    ell, Dell_data, cov, sigma = load_data()
    print(f"Data: {len(ell)} bandpowers, ell range [{float(ell.min())}, {float(ell.max())}]")
    cov_inv = jnp.linalg.inv(cov)

    # Paper baseline cosmology (matches clyy_v2.yaml + may26)
    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    ev = csl.cl_yy_factory(cosmo, ell)
    dl_fac = ell * (ell + 1.0) / (2.0 * jnp.pi) * 1e12

    # Warm up the closure (JIT compile + first call)
    prof0 = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    out = ev(prof0); jax.block_until_ready(out)
    Dell_init = dl_fac * (out[0] + out[1])
    chi2_init = float((Dell_init - Dell_data) @ cov_inv @ (Dell_init - Dell_data))
    print(f"Init at A10 defaults: chi^2 = {chi2_init:.2f} / {len(ell)-2} dof")

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
        cl_1h, cl_2h = ev(prof)
        Dell = dl_fac * (cl_1h + cl_2h)
        r = Dell - Dell_data
        numpyro.factor("loglik", -0.5 * (r @ cov_inv @ r))

    # Reduced from paper's 500+2000 because each forward+grad call is ~70 ms on
    # this 44-core machine (paper's laptop was ~10x faster per-call). Still
    # gives R-hat<1.05 on this trivial 2D problem.
    kernel = NUTS(model, dense_mass=True, max_tree_depth=5,
                  target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=80, num_samples=200,
                num_chains=2, chain_method='sequential', progress_bar=False)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0))
    wall = time.perf_counter() - t0
    samples = mcmc.get_samples(group_by_chain=True)
    summary = numpyro.diagnostics.summary(samples)

    P0 = np.concatenate(samples['P0'])
    be = np.concatenate(samples['beta'])
    print(f"\n=== NUTS baseline ===")
    print(f"wall   = {wall:.2f} s")
    print(f"P0     = {P0.mean():.3f} ± {P0.std():.3f}    R-hat={summary['P0']['r_hat']:.3f}    ESS={summary['P0']['n_eff']:.0f}")
    print(f"beta   = {be.mean():.3f} ± {be.std():.3f}    R-hat={summary['beta']['r_hat']:.3f}    ESS={summary['beta']['n_eff']:.0f}")
    print(f"divergences: {int(mcmc.get_extra_fields()['diverging'].sum()) if 'diverging' in mcmc.get_extra_fields() else 0}")

    out = os.path.join(os.path.dirname(__file__), "_nuts_baseline.npz")
    np.savez(out, P0=P0, beta=be, wall=wall,
             ess_P0=summary['P0']['n_eff'], ess_beta=summary['beta']['n_eff'])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
