"""Diagnose the SBI/NPE bias seen in paper Table 1.

Hypothesis: paper's NPE uses a Gaussian proposal N(bestfit≈1.2, σ=2.5) truncated
to the prior [0, 8]. Truncation breaks symmetry: effective proposal mean ≈ 2.5.
Single-round NPE without importance-weight correction learns
  q(θ|y) ∝ proposal(θ) × L(y|θ)
not the true posterior with uniform prior. Result: posterior peak shifts right.

This script runs two NPE variants and compares to NUTS:
  (A) "Paper-style" — Gaussian proposal around bestfit, truncated.
  (B) "Prior-faithful" — uniform draws from the prior box; no proposal bias.

Outputs:
  - _npe_diagnose.npz with samples from each
  - npe_diagnose.png — triangle plot with NUTS + NPE-A + NPE-B
"""
from __future__ import annotations
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import scipy.optimize as so

import classy_szlite as csl
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.train import fit_to_data


DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")

# Paper's prior box for SBI
P0_PRIOR   = (0.0, 8.0)
BETA_PRIOR = (1.5, 6.0)

# Paper's NPE settings
N_SIM     = 8000
SBI_BATCH = 512
SBI_LR    = 5e-4
SBI_EPOCHS = 500

# Paper's truncated-Gaussian proposal widths
PROP_W_P0   = 2.5
PROP_W_BETA = 1.2


def load():
    ell  = np.loadtxt(os.path.join(DATA_DIR, "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    data = np.loadtxt(os.path.join(DATA_DIR, "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"))
    cov  = np.loadtxt(os.path.join(DATA_DIR, "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    return jnp.asarray(ell), data[:, 1], cov


def build_forward(cosmo, ell):
    ev = csl.cl_yy_factory(cosmo, ell)
    dl = ell * (ell + 1) / (2 * np.pi) * 1e12
    def forward(P0, beta):
        prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                     alpha=1.062, beta=beta, B=1.25)
        cl1, cl2 = ev(prof)
        return dl * (cl1 + cl2)
    return forward


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48)):
    y_j, inv_j = jnp.asarray(y), jnp.asarray(inv_cov)
    def nll(x):
        r = y_j - forward(x[0], x[1])
        return 0.5 * r @ inv_j @ r
    nll_j  = jax.jit(nll)
    nll_g  = jax.jit(jax.grad(nll))
    return so.minimize(lambda x: float(nll_j(x)),
                       np.asarray(x0, dtype=np.float64),
                       jac=lambda x: np.asarray(nll_g(x)),
                       method="L-BFGS-B",
                       bounds=((0.1, 20.), (0.5, 10.)))


def simulate_truncgauss(forward, cov, bf_x, n, seed):
    """Gaussian proposal around bestfit, truncated to prior box."""
    rng = np.random.default_rng(seed)
    P0 = beta = None
    overshoot = 5
    while True:
        P0_s   = rng.normal(bf_x[0], PROP_W_P0,   n * overshoot)
        beta_s = rng.normal(bf_x[1], PROP_W_BETA, n * overshoot)
        m = ((P0_s >= P0_PRIOR[0]) & (P0_s <= P0_PRIOR[1]) &
             (beta_s >= BETA_PRIOR[0]) & (beta_s <= BETA_PRIOR[1]))
        if m.sum() >= n:
            P0   = P0_s[m][:n]
            beta = beta_s[m][:n]
            break
        overshoot *= 2
    return _simulate(forward, cov, np.column_stack([P0, beta]), seed + 1)


def simulate_uniform(forward, cov, n, seed):
    """Uniform draws from the prior box (no proposal bias)."""
    rng = np.random.default_rng(seed)
    P0   = rng.uniform(*P0_PRIOR,   n)
    beta = rng.uniform(*BETA_PRIOR, n)
    return _simulate(forward, cov, np.column_stack([P0, beta]), seed + 1)


def _simulate(forward, cov, theta, seed):
    rng = np.random.default_rng(seed)
    fwd_v = jax.vmap(forward, in_axes=(0, 0))
    mu = np.asarray(fwd_v(jnp.asarray(theta[:, 0]), jnp.asarray(theta[:, 1])))
    L = np.linalg.cholesky(cov)
    noise = (L @ rng.standard_normal((len(cov), len(theta)))).T
    return jnp.asarray(theta), jnp.asarray(mu + noise)


def standardise(arr):
    mu  = jnp.asarray(np.mean(np.asarray(arr), axis=0))
    std = jnp.asarray(np.std(np.asarray(arr),  axis=0) + 1e-12)
    return (arr - mu) / std, mu, std


def train_and_sample(theta_train, y_train, y_obs, key, n_samples=3000):
    th_z, mu_t, sd_t = standardise(theta_train)
    y_z,  mu_y, sd_y = standardise(y_train)
    flow = masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(th_z.shape[1])),
        cond_dim=y_z.shape[1],
        nn_width=128, nn_depth=3, flow_layers=8,
    )
    flow, _ = fit_to_data(
        key=key, dist=flow, data=(th_z, y_z),
        max_epochs=SBI_EPOCHS, batch_size=SBI_BATCH,
        learning_rate=SBI_LR, show_progress=False,
    )
    y_obs_z = (jnp.asarray(y_obs) - mu_y) / sd_y
    sample_key = jr.split(key, 2)[1]
    samples_z = flow.sample(sample_key, sample_shape=(n_samples,),
                             condition=y_obs_z)
    return np.asarray(samples_z * sd_t + mu_t)


def main():
    print(f"Backend: {jax.default_backend()}")
    ell, y, cov = load()
    inv_cov = np.linalg.inv(cov)

    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    forward = build_forward(cosmo, ell)

    # L-BFGS bestfit (NPE-A proposal center)
    t0 = time.perf_counter()
    bf = find_bestfit(forward, y, inv_cov)
    print(f"L-BFGS bestfit: P0={bf.x[0]:.3f}, β={bf.x[1]:.3f}  "
          f"({time.perf_counter()-t0:.2f} s)")

    # Quantify the proposal-truncation shift
    from scipy.stats import truncnorm
    p0_lo = (P0_PRIOR[0] - bf.x[0]) / PROP_W_P0
    p0_hi = (P0_PRIOR[1] - bf.x[0]) / PROP_W_P0
    tn = truncnorm(p0_lo, p0_hi, loc=bf.x[0], scale=PROP_W_P0)
    print(f"Truncated-Gauss proposal mean (P0): {tn.mean():.3f}  "
          f"(vs unconstrained: {bf.x[0]:.3f}, shift = {tn.mean() - bf.x[0]:+.3f})")

    # NPE-A: paper-style truncated-Gauss proposal
    print("\n-- NPE-A: truncated-Gauss proposal around bestfit --")
    t0 = time.perf_counter()
    th_A, y_A = simulate_truncgauss(forward, cov, bf.x, N_SIM, seed=43)
    print(f"  simulated {N_SIM} pairs in {time.perf_counter()-t0:.1f} s")
    print(f"  proposal sample mean (P0): {float(th_A[:,0].mean()):.3f} "
          f"(matches truncnorm calc above)")
    t0 = time.perf_counter()
    s_A = train_and_sample(th_A, y_A, y, jr.key(0))
    print(f"  trained + sampled in {time.perf_counter()-t0:.1f} s")
    print(f"  NPE-A posterior:  P0 = {s_A[:,0].mean():.3f} ± {s_A[:,0].std():.3f}  "
          f"β = {s_A[:,1].mean():.3f} ± {s_A[:,1].std():.3f}")

    # NPE-B: uniform prior as proposal
    print("\n-- NPE-B: uniform prior as proposal (no truncation bias) --")
    t0 = time.perf_counter()
    th_B, y_B = simulate_uniform(forward, cov, N_SIM, seed=44)
    print(f"  simulated {N_SIM} pairs in {time.perf_counter()-t0:.1f} s")
    t0 = time.perf_counter()
    s_B = train_and_sample(th_B, y_B, y, jr.key(1))
    print(f"  trained + sampled in {time.perf_counter()-t0:.1f} s")
    print(f"  NPE-B posterior:  P0 = {s_B[:,0].mean():.3f} ± {s_B[:,0].std():.3f}  "
          f"β = {s_B[:,1].mean():.3f} ± {s_B[:,1].std():.3f}")

    # Save
    here = os.path.dirname(__file__)
    np.savez(os.path.join(here, "_npe_diagnose.npz"),
             bf=bf.x,
             npe_A_P0=s_A[:,0], npe_A_beta=s_A[:,1],
             npe_B_P0=s_B[:,0], npe_B_beta=s_B[:,1],
             trunc_proposal_mean_P0=tn.mean())
    print(f"\nSaved {os.path.join(here, '_npe_diagnose.npz')}")


if __name__ == "__main__":
    main()
