"""Extended NUTS diagnostics: traces + ACF + step-size adaptation + tree depth.

Targeting paper figure showing every NUTS diagnostic in one place.
"""
from __future__ import annotations
import os, time
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import classy_szlite as csl

DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")


def load_data():
    ell  = np.loadtxt(os.path.join(DATA_DIR, "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    bp   = np.loadtxt(os.path.join(DATA_DIR, "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"))
    cov  = np.loadtxt(os.path.join(DATA_DIR, "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    return jnp.asarray(ell), jnp.asarray(bp[:, 1]), jnp.asarray(cov)


def autocorr(x, max_lag=None):
    """Normalised autocorrelation function of a 1-D time series."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 200)
    var = np.var(x)
    if var == 0:
        return np.ones(max_lag)
    rho = np.empty(max_lag)
    for k in range(max_lag):
        rho[k] = np.mean(x[: n - k] * x[k:]) / var
    return rho


def integrated_autocorr_time(rho, c=5.0):
    """Sokal estimator: τ = 1 + 2 Σ_k=1^{M} ρ_k, M = first k where c·τ_init < k."""
    tau = 1.0
    for M in range(1, len(rho)):
        cum = 1.0 + 2.0 * np.sum(rho[1:M+1])
        if M >= c * cum:
            return cum
        tau = cum
    return tau


def main():
    print(f"Backend: {jax.default_backend()}")
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

    n_warm, n_samp, n_chains = 200, 1000, 4
    kernel = NUTS(model, dense_mass=True, max_tree_depth=5, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=n_samp,
                num_chains=n_chains, chain_method='sequential', progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0), extra_fields=(
        "energy", "diverging", "num_steps", "accept_prob",
        "adapt_state.step_size",
    ))
    wall = time.perf_counter() - t0

    s  = mcmc.get_samples(group_by_chain=True)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    summary = numpyro.diagnostics.summary(s)

    P0_ess, P0_rh = float(summary['P0']['n_eff']),   float(summary['P0']['r_hat'])
    be_ess, be_rh = float(summary['beta']['n_eff']), float(summary['beta']['r_hat'])
    diverg = int(np.sum(ex['diverging']))
    acc    = float(np.mean(ex['accept_prob']))
    leap   = float(np.mean(ex['num_steps']))
    energies = ex['energy']     # (chains, samples)
    ebfmi = []
    for c in range(energies.shape[0]):
        e = energies[c]
        de2 = np.mean(np.diff(e) ** 2)
        var_e = np.var(e)
        ebfmi.append(de2 / var_e if var_e > 0 else np.nan)
    ebfmi = np.asarray(ebfmi)

    # Numpyro stores num_steps per draw — for NUTS this is # leapfrog steps,
    # NOT tree depth. Tree depth = log2(num_steps + 1).  Convert:
    num_steps = np.asarray(ex['num_steps'])        # (chains, samples)
    tree_depths = np.log2(num_steps + 1).astype(int)

    # Step-size adapter (warmup adaptation history)
    step_size_adapt = np.asarray(ex['adapt_state.step_size'])    # (chains, samples)

    print(f"\n=== NUTS diagnostics (extended) ===")
    print(f"wall                       : {wall:6.2f} s")
    print(f"divergences                : {diverg}")
    print(f"mean accept-prob           : {acc:.3f}  (target 0.85)")
    print(f"mean leapfrog steps / draw : {leap:.1f}")
    print(f"tree depth histogram       : "
          + "  ".join(f"d{d}:{np.sum(tree_depths==d)}" for d in range(7) if np.sum(tree_depths==d) > 0))
    print(f"E-BFMI per chain           : {' '.join(f'{x:.2f}' for x in ebfmi)}  (good > 0.3)")
    print(f"R-hat (P0)                 : {P0_rh:.4f}   ESS (P0)   = {P0_ess:.0f}")
    print(f"R-hat (beta)               : {be_rh:.4f}   ESS (beta) = {be_ess:.0f}")

    # ACF
    rho_P0 = autocorr(np.concatenate(np.asarray(s['P0'])), max_lag=80)
    rho_be = autocorr(np.concatenate(np.asarray(s['beta'])), max_lag=80)
    tau_P0 = integrated_autocorr_time(rho_P0)
    tau_be = integrated_autocorr_time(rho_be)
    print(f"integrated autocorr τ(P0)  : {tau_P0:.2f}   (ESS_naive = N/τ = {n_samp*n_chains/tau_P0:.0f})")
    print(f"integrated autocorr τ(beta): {tau_be:.2f}   (ESS_naive = N/τ = {n_samp*n_chains/tau_be:.0f})")

    # ---- 6-panel diagnostic figure ----
    fig, axs = plt.subplots(3, 2, figsize=(11, 9))

    ax = axs[0, 0]
    for c in range(s['P0'].shape[0]):
        ax.plot(s['P0'][c], lw=0.6, alpha=0.7, label=f"chain {c+1}")
    ax.set_ylabel(r"$P_0$"); ax.set_xlabel("sample idx")
    ax.set_title(f"trace: $P_0$   (R-hat = {P0_rh:.3f}, ESS = {P0_ess:.0f})")
    ax.legend(fontsize=7, ncol=2)

    ax = axs[0, 1]
    for c in range(s['beta'].shape[0]):
        ax.plot(s['beta'][c], lw=0.6, alpha=0.7)
    ax.set_ylabel(r"$\beta$"); ax.set_xlabel("sample idx")
    ax.set_title(rf"trace: $\beta$  (R-hat = {be_rh:.3f}, ESS = {be_ess:.0f})")

    ax = axs[1, 0]
    ax.plot(rho_P0, color='C0', label=rf"$P_0$   τ$_\mathrm{{int}}$ = {tau_P0:.1f}")
    ax.plot(rho_be, color='C3', label=rf"$\beta$  τ$_\mathrm{{int}}$ = {tau_be:.1f}")
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("lag (samples)")
    ax.set_ylabel("autocorrelation $\\rho_k$")
    ax.set_title("autocorrelation function (per-param)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axs[1, 1]
    e = energies.flatten()
    de = np.diff(energies, axis=1).flatten()
    ax.hist(e - e.mean(), bins=40, alpha=0.6, density=True, color='C0',
            label=f"E − ⟨E⟩  (σ$_E$ = {np.std(e):.2f})")
    ax.hist(de, bins=40, alpha=0.6, density=True, color='C3',
            label=f"ΔE       (σ$_{{ΔE}}$ = {np.std(de):.2f})")
    ax.set_xlabel("energy")
    ax.set_yticks([])
    ax.set_title(f"energy diagnostic — E-BFMI = {ebfmi.mean():.2f} (good > 0.3)")
    ax.legend(fontsize=9)

    ax = axs[2, 0]
    # Warmup + sampling step-size adaptation; numpyro records during warmup
    # only — what we see here is the final step size held during sampling.
    for c in range(step_size_adapt.shape[0]):
        ax.plot(step_size_adapt[c], color=f'C{c}', lw=0.5, alpha=0.6,
                label=f"chain {c+1}")
    ax.set_xlabel("sample idx")
    ax.set_ylabel("adapted step size")
    ax.set_yscale("log")
    ax.set_title("warmup-adapted step size")

    ax = axs[2, 1]
    # tree depth distribution
    max_d = int(tree_depths.max())
    ax.hist(tree_depths.flatten(), bins=np.arange(max_d + 2) - 0.5,
            color='C2', alpha=0.8, edgecolor='k')
    ax.set_xlabel("tree depth")
    ax.set_ylabel("count")
    ax.set_title(f"NUTS tree depth (mean leapfrog/draw = {leap:.1f})")
    ax.set_xticks(np.arange(max_d + 1))

    plt.suptitle(
        f"NUTS diagnostics — baseline cosmology, $C_\\ell^{{yy}}$, "
        f"{n_chains} chains × {n_samp} samples ({wall:.0f}s wall)\n"
        f"R-hat ≤ {max(P0_rh, be_rh):.3f},  "
        f"ESS = {min(P0_ess, be_ess):.0f},  "
        f"divergences = {diverg},  "
        f"E-BFMI = {ebfmi.mean():.2f},  accept = {acc:.2f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "nuts_diagnostics_v2.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
