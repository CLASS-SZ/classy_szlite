"""Run NUTS and extract a full diagnostic set.

Outputs:
  - tests/_nuts_diagnostics.npz: traces, energies, accept-rates, etc.
  - tests/nuts_diagnostics.png: 4-panel figure (trace P0, trace beta,
    energy histogram, step-size by chain).
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

    kernel = NUTS(model, dense_mass=True, max_tree_depth=5, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=1000,
                num_chains=4, chain_method='sequential', progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("energy", "diverging",
                                                     "num_steps", "accept_prob"))
    wall = time.perf_counter() - t0

    s = mcmc.get_samples(group_by_chain=True)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    summary = numpyro.diagnostics.summary(s)

    P0_ess, P0_rh = summary['P0']['n_eff'],   summary['P0']['r_hat']
    be_ess, be_rh = summary['beta']['n_eff'], summary['beta']['r_hat']
    diverg = int(np.sum(ex['diverging']))
    acc    = float(np.mean(ex['accept_prob']))
    leap   = float(np.mean(ex['num_steps']))

    # E-BFMI per chain
    energies = ex['energy']                              # (chains, samples)
    ebfmi = []
    for c in range(energies.shape[0]):
        e = energies[c]
        de2 = np.mean(np.diff(e) ** 2)
        var_e = np.var(e)
        ebfmi.append(de2 / var_e if var_e > 0 else np.nan)
    ebfmi = np.asarray(ebfmi)

    print(f"\n=== NUTS diagnostics ===")
    print(f"wall                       : {wall:6.2f} s")
    print(f"divergences                : {diverg}")
    print(f"mean accept-prob           : {acc:.3f}  (target 0.85)")
    print(f"mean leapfrog steps / draw : {leap:.1f}")
    print(f"E-BFMI per chain           : {' '.join(f'{x:.2f}' for x in ebfmi)}  (good > 0.3)")
    print(f"R-hat (P0)                 : {P0_rh:.4f}   ESS (P0)   = {P0_ess:.0f}")
    print(f"R-hat (beta)               : {be_rh:.4f}   ESS (beta) = {be_ess:.0f}")

    # ---- Diagnostic figure (4 panels) ----
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # P0 trace
    ax = axs[0, 0]
    for c in range(s['P0'].shape[0]):
        ax.plot(s['P0'][c], lw=0.7, alpha=0.7, label=f"chain {c+1}")
    ax.set_ylabel(r"$P_0$"); ax.set_xlabel("sample idx")
    ax.set_title("trace: $P_0$")
    ax.legend(fontsize=8, ncol=2)

    # beta trace
    ax = axs[0, 1]
    for c in range(s['beta'].shape[0]):
        ax.plot(s['beta'][c], lw=0.7, alpha=0.7)
    ax.set_ylabel(r"$\beta$"); ax.set_xlabel("sample idx")
    ax.set_title(r"trace: $\beta$")

    # Energy histogram (overlay marginal & transition-energy stds; if these
    # are widely separated, E-BFMI is bad.)
    ax = axs[1, 0]
    e = energies.flatten()
    de = np.diff(energies, axis=1).flatten()
    ax.hist(e - e.mean(), bins=40, alpha=0.6, density=True,
            label=f"E − ⟨E⟩  (σ = {np.std(e):.2f})")
    ax.hist(de, bins=40, alpha=0.6, density=True,
            label=f"ΔE  (σ = {np.std(de):.2f})")
    ax.set_xlabel("energy"); ax.set_yticks([])
    ax.set_title(f"energy diagnostic (E-BFMI={ebfmi.mean():.2f})")
    ax.legend(fontsize=8)

    # Accept-prob distribution
    ax = axs[1, 1]
    ax.hist(np.asarray(ex['accept_prob']).flatten(), bins=40, alpha=0.7,
            color='C2')
    ax.axvline(0.85, color='k', ls='--', label='target 0.85')
    ax.set_xlabel("accept probability per draw")
    ax.set_yticks([])
    ax.set_title(f"accept rate: mean = {acc:.3f}")
    ax.legend(fontsize=8)

    plt.suptitle(
        f"NUTS diagnostics — baseline cosmology, $C_\\ell^{{yy}}$, "
        f"4 chains × 1000 samples ({wall:.1f}s wall)\n"
        f"R-hat ≤ {max(P0_rh, be_rh):.3f},  "
        f"ESS = {min(P0_ess, be_ess):.0f},  divergences = {diverg},  "
        f"E-BFMI = {ebfmi.mean():.2f}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "nuts_diagnostics.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved {out}")

    out_npz = os.path.join(os.path.dirname(__file__), "_nuts_diagnostics.npz")
    np.savez(out_npz,
             trace_P0=np.asarray(s['P0']), trace_beta=np.asarray(s['beta']),
             energy=np.asarray(energies), accept=np.asarray(ex['accept_prob']),
             diverging=np.asarray(ex['diverging']),
             num_steps=np.asarray(ex['num_steps']),
             wall=wall, ebfmi=ebfmi, P0_ess=float(P0_ess), be_ess=float(be_ess),
             P0_rh=float(P0_rh), be_rh=float(be_rh))
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
