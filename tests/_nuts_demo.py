"""Parallel-chain NUTS demo: cobaya/sequential ~60s baseline vs vmap'd CPU NUTS.

Reuses the may26 / fionapaper setup: ACT-DR4 Cl^yy bandpowers (8 bins) with
the 8x8 covariance, fit at fixed cosmology (baseline cosmo from the paper)
with profile parameters (P0, beta). Cosmology and (c500, gamma, alpha) all
fixed at A10 defaults; B = 1.25.

Variants timed:
  1. Sequential 4-chain NUTS (paper's ~40s point).
  2. Vectorized many-chain NUTS via chain_method='vectorized' (vmap'd).
  3. (TPU run if requested, just for the contrast.)

Reports wall time and per-parameter R-hat + effective sample size, plus
posterior mean ± std for (P0, beta). Saves posteriors to .npz for plotting.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl


DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")
ELL_FILE = "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"
DAT_FILE = "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"
COV_FILE = "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"


def load_data():
    ell  = np.loadtxt(os.path.join(DATA_DIR, ELL_FILE))
    bp   = np.loadtxt(os.path.join(DATA_DIR, DAT_FILE))  # cols: ell, D_ell*1e12, sigma
    cov  = np.loadtxt(os.path.join(DATA_DIR, COV_FILE))
    order = np.argsort(ell)
    ell   = ell[order]
    Dell  = bp[order, 1]   # already in 1e12 units
    # The covariance file's ordering matches the original ell ordering (descending).
    # Sort cov rows+cols to match ascending ell:
    cov_sorted = cov[order][:, order]
    return jnp.asarray(ell), jnp.asarray(Dell), jnp.asarray(cov_sorted)


def make_model(ev, ell, Dell_data, cov_inv):
    """numpyro model with broad uniform prior on (P0, beta)."""
    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.5, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(2.0, 8.0))
        prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
        cl_1h, cl_2h = ev(prof)
        Dell = ell * (ell + 1) / (2 * jnp.pi) * (cl_1h + cl_2h) * 1e12
        resid = Dell - Dell_data
        # Standard Gaussian likelihood with fixed covariance
        chi2 = resid @ cov_inv @ resid
        numpyro.factor("loglik", -0.5 * chi2)
    return model


def run_nuts(model, n_warm=200, n_samp=500, chains=4, chain_method='sequential',
             seed=0):
    """Run NUTS with the given chain method and return (samples_dict, wall_s)."""
    kernel = NUTS(model, dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=n_samp,
                num_chains=chains, chain_method=chain_method,
                progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed))
    wall = time.perf_counter() - t0
    samples = mcmc.get_samples(group_by_chain=True)
    # ESS + R-hat
    summary = numpyro.diagnostics.summary(samples)
    return samples, wall, summary


def main():
    backend = jax.default_backend()
    print(f"Backend: {backend}  Devices: {jax.devices()}")
    numpyro.set_host_device_count(int(os.environ.get("NUTS_HOST_DEVICES", "4")))
    print(f"Host devices for parallel-chain: {jax.local_device_count()}")

    ell, Dell_data, cov = load_data()
    print(f"Data: {len(ell)} bandpowers, ell range [{float(ell.min())}, {float(ell.max())}]")
    cov_inv = jnp.linalg.inv(cov)

    cosmo = csl.CosmoParams()  # paper baseline
    ev = csl.cl_yy_factory(cosmo, ell)

    # warm the closure (compile + first call)
    prof0 = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    _ = ev(prof0); jax.block_until_ready(_)

    model = make_model(ev, ell, Dell_data, cov_inv)

    print("\n--- 1. Sequential 4-chain NUTS ---")
    s1, w1, sum1 = run_nuts(model, n_warm=200, n_samp=500, chains=4,
                             chain_method='sequential', seed=0)
    P0 = np.concatenate(s1['P0']); be = np.concatenate(s1['beta'])
    print(f"  wall = {w1:.2f} s")
    print(f"  P0   = {P0.mean():.3f} ± {P0.std():.3f}   R-hat = {sum1['P0']['r_hat']:.3f}  ESS = {sum1['P0']['n_eff']:.0f}")
    print(f"  beta = {be.mean():.3f} ± {be.std():.3f}   R-hat = {sum1['beta']['r_hat']:.3f}  ESS = {sum1['beta']['n_eff']:.0f}")

    print("\n--- 2. Vectorized 32-chain NUTS (vmap across chains) ---")
    s2, w2, sum2 = run_nuts(model, n_warm=200, n_samp=500, chains=32,
                             chain_method='vectorized', seed=1)
    P0 = np.concatenate(s2['P0']); be = np.concatenate(s2['beta'])
    print(f"  wall = {w2:.2f} s")
    print(f"  P0   = {P0.mean():.3f} ± {P0.std():.3f}   R-hat = {sum2['P0']['r_hat']:.3f}  ESS = {sum2['P0']['n_eff']:.0f}")
    print(f"  beta = {be.mean():.3f} ± {be.std():.3f}   R-hat = {sum2['beta']['r_hat']:.3f}  ESS = {sum2['beta']['n_eff']:.0f}")

    print("\n--- Summary ---")
    print(f"  sequential 4-chain : {w1:6.2f} s  (ESS_P0={sum1['P0']['n_eff']:5.0f}, ESS_beta={sum1['beta']['n_eff']:5.0f})")
    print(f"  vectorized 32-chain: {w2:6.2f} s  (ESS_P0={sum2['P0']['n_eff']:5.0f}, ESS_beta={sum2['beta']['n_eff']:5.0f})")
    print(f"  per-ESS speedup    : {(w1*sum2['P0']['n_eff'])/(w2*sum1['P0']['n_eff']):.1f}x")

    # Save the larger of the two for plotting
    out = os.path.join(os.path.dirname(__file__), f"_nuts_{backend}.npz")
    np.savez(out,
             P0_seq=np.concatenate(s1['P0']), beta_seq=np.concatenate(s1['beta']),
             P0_vec=np.concatenate(s2['P0']), beta_vec=np.concatenate(s2['beta']),
             w_seq=w1, w_vec=w2,
             ess_P0_seq=sum1['P0']['n_eff'], ess_P0_vec=sum2['P0']['n_eff'],
             ess_beta_seq=sum1['beta']['n_eff'], ess_beta_vec=sum2['beta']['n_eff'])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
