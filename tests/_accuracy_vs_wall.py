"""Gold-standard accuracy comparison between NUTS and cobaya RW-MH.

Method:
  1. Long NUTS chain → "gold-standard" posterior (treat as truth).
  2. For NUTS at each (warmup, samples, chains, seed) point in the multi-seed
     sweep, compute |mean_estimate - mean_gold| / sigma_gold (Z-score).
  3. For cobaya RW-MH, use the existing chain at clyy_v2.1.txt, but compute
     the same metric on PREFIX SUBSETS of the chain (first N weighted samples)
     to get an "accuracy vs wall" curve without re-running.

Output: tests/accuracy_vs_wall.png + tests/_accuracy_vs_wall.npz
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
COBAYA_CHAIN = os.path.expanduser("~/class-sz-plugin-tests/chains/clyy_v2.1.txt")
# From the actual cobaya run: 838s wall, 35416 evals, so 0.0237 s/eval.
# The chain *file* records accepted moves with weights; the wall time is
# proportional to number of EVALS, which is sum(weights).
COBAYA_S_PER_EVAL = 838.0 / 35416.0


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


def _clear_jit_caches():
    """Avoid LLVM compilation OOM when re-running NUTS over many seeds
    (each (warm, samp, chains) shape triggers a fresh JIT trace + cache
    entry that keeps memory until the process exits)."""
    try:
        jax.clear_caches()
    except Exception:
        pass
    import gc
    gc.collect()


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

    # ---- 1. Gold standard NUTS (long run) ----------------------------
    print("\n[1] Gold-standard NUTS (500 warm + 4000 samp × 4 chains)")
    t0 = time.perf_counter()
    wall_gold, sm_gold, s_gold = run_nuts(model, 500, 4000, 4, seed=999)
    P0g = np.concatenate(s_gold['P0']); beg = np.concatenate(s_gold['beta'])
    P0_mu_g, P0_std_g = P0g.mean(), P0g.std()
    be_mu_g, be_std_g = beg.mean(), beg.std()
    print(f"  wall = {wall_gold:.1f} s")
    print(f"  GOLD P0   = {P0_mu_g:.4f} ± {P0_std_g:.4f}  ESS={sm_gold['P0']['n_eff']:.0f}  R-hat={sm_gold['P0']['r_hat']:.3f}")
    print(f"  GOLD beta = {be_mu_g:.4f} ± {be_std_g:.4f}  ESS={sm_gold['beta']['n_eff']:.0f}  R-hat={sm_gold['beta']['r_hat']:.3f}")

    # ---- 2. NUTS accuracy at each budget (load multi-seed sweep) -----
    nuts_sweep_file = os.path.join(os.path.dirname(__file__),
                                     "_nuts_sweep_multiseed.npz")
    if os.path.exists(nuts_sweep_file):
        ms = np.load(nuts_sweep_file)['nuts']
        # cols: warm, samp, chains, seed, wall, ess_P0, ess_b, rh_P0, rh_b
        # but we need the actual samples to compute mean-Z — re-run the sweep
        # is too expensive, so we cleverly construct "wall vs ESS" and use the
        # relationship SE = sigma_gold / sqrt(ESS) ⇒ |mean - gold|/sigma_gold
        # has expected magnitude 1/sqrt(ESS).  i.e. for each (wall, ESS) point
        # the expected Z is 1/sqrt(ESS).  We can also do an empirical version:
        # actually re-run the small budgets only and measure Z directly.
        print(f"\n[2] Multi-seed NUTS sweep loaded: {len(ms)} rows")
    else:
        ms = None
        print("\n[2] No multi-seed sweep file found, will skip NUTS Z-curve")

    # Empirical NUTS Z-score curve: re-run a sweep of small budgets and
    # extract actual posterior means (not just ESS) to compute Z directly.
    # This is the most honest measure of "did the chain hit the right mean".
    # N_SEEDS controls how well-resolved the IQR ribbon is; default 25 so
    # the 25/75 percentiles are computed from ~6 samples on each side.
    N_SEEDS = int(os.environ.get("NUTS_N_SEEDS", "25"))
    print(f"\n[3] Empirical NUTS Z-scores at small budgets ({N_SEEDS} seeds)")
    nuts_z_rows = []
    # The 200+500×4 config has tripped LLVM "Cannot allocate memory" on the
    # JIT compile cache after the 100+400×2 chain; dropped for now since the
    # 4 remaining budgets already span wall ∈ [1, 14] s — the relevant range
    # for the "sub-10 s NUTS" story.
    for (warm, samp, chains) in [(30, 50, 1), (50, 100, 2), (100, 200, 2),
                                   (100, 400, 2)]:
        for seed in range(N_SEEDS):
            wall, sm, samples = run_nuts(model, warm, samp, chains, seed=seed)
            P0_s = np.concatenate(samples['P0']); be_s = np.concatenate(samples['beta'])
            Z_P0 = abs(P0_s.mean() - P0_mu_g) / P0_std_g
            Z_be = abs(be_s.mean() - be_mu_g) / be_std_g
            ess_P0 = float(sm['P0']['n_eff']); ess_be = float(sm['beta']['n_eff'])
            nuts_z_rows.append((warm, samp, chains, seed, wall, Z_P0, Z_be,
                                ess_P0, ess_be))
        cfg = f"{warm}+{samp}×{chains}"
        walls = [r[4] for r in nuts_z_rows[-N_SEEDS:]]
        zs    = [r[5] for r in nuts_z_rows[-N_SEEDS:]]
        print(f"  {cfg:<14}: wall = {np.median(walls):.2f}s, |Z_P0| median = {np.median(zs):.3f}  (n={N_SEEDS})", flush=True)
        # Flush JIT cache between budget points to avoid LLVM-side OOM
        # when many distinct (warm, samp, chains) trace shapes accumulate.
        _clear_jit_caches()

    nuts_z = np.asarray(nuts_z_rows, dtype=float)

    # ---- 4. cobaya RW-MH accuracy via prefix sub-sampling ------------
    print(f"\n[4] cobaya RW-MH accuracy via prefix subsetting of {COBAYA_CHAIN}")
    co = np.loadtxt(COBAYA_CHAIN, comments='#')
    # cols: weight, minus_log_post, P0, beta, ...
    weights = co[:, 0]
    P0_co = co[:, 2]; be_co = co[:, 3]
    # cum-evals = cum-weights
    cumw = np.cumsum(weights)
    # For each prefix endpoint, compute mean and Z (with weighted average)
    Ns = np.unique(np.round(np.logspace(np.log10(50),
                                          np.log10(len(weights)), 30)).astype(int))
    rwmh_rows = []
    for n in Ns:
        w = weights[:n]; p = P0_co[:n]; b = be_co[:n]
        if w.sum() < 1: continue
        mu_p = np.average(p, weights=w)
        mu_b = np.average(b, weights=w)
        Z_P0 = abs(mu_p - P0_mu_g) / P0_std_g
        Z_be = abs(mu_b - be_mu_g) / be_std_g
        # Wall time: sum of weights × 0.0237 s/eval
        wall = cumw[n - 1] * COBAYA_S_PER_EVAL
        # ESS (Kish) for this prefix
        ess = (w.sum() ** 2) / (w ** 2).sum()
        rwmh_rows.append((n, wall, ess, Z_P0, Z_be))
    rwmh = np.asarray(rwmh_rows)
    print(f"  {len(rwmh)} prefix points constructed")
    print(f"  RW-MH at full chain ({rwmh[-1, 1]:.0f}s): Z_P0 = {rwmh[-1, 3]:.3f}, ESS = {rwmh[-1, 2]:.0f}")

    # ---- 5. Plot: accuracy vs wall -----------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: |Z| vs wall (lower is more accurate)
    ax = axs[0]
    # NUTS scatter (per seed) + median line
    walls_n = nuts_z[:, 4]; z_n = nuts_z[:, 5]
    # group by budget
    cfg_id = (nuts_z[:, 0]*1e6 + nuts_z[:, 1]*1e3 + nuts_z[:, 2]).astype(int)
    uniq = np.unique(cfg_id)
    nuts_med_x, nuts_med_y = [], []
    for u in uniq:
        m = cfg_id == u
        nuts_med_x.append(np.median(walls_n[m]))
        nuts_med_y.append(np.median(z_n[m]))
    nuts_med_x = np.asarray(nuts_med_x); nuts_med_y = np.asarray(nuts_med_y)
    order = np.argsort(nuts_med_x)
    nuts_med_x, nuts_med_y = nuts_med_x[order], nuts_med_y[order]
    ax.scatter(walls_n, z_n, c='C0', s=18, alpha=0.45, label='NUTS (per seed)')
    ax.plot(nuts_med_x, nuts_med_y, 'C0-', lw=2, label='NUTS (median)')

    ax.plot(rwmh[:, 1], rwmh[:, 3], 'C3o-', ms=4, lw=1.4,
            label="cobaya RW-MH (prefix subsetting of converged chain)")

    # 1/sqrt(N) reference for visual scale
    wref = np.geomspace(rwmh[:, 1].min(), rwmh[:, 1].max(), 50)
    # Visual fit: at one wall point, see where curves land
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$|\hat\mu_{P_0} - \mu_{P_0}^\mathrm{gold}|$ / $\sigma_{P_0}^\mathrm{gold}$")
    ax.set_title(r"Accuracy of posterior mean estimate")
    ax.axhline(0.1, color='k', ls=':', alpha=0.6, label=r"0.1$\sigma$ (publication-grade)")
    ax.axhline(0.01, color='k', ls='--', alpha=0.6, label=r"0.01$\sigma$ (overkill)")
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')

    # Panel B: ESS vs wall (rate)
    ax = axs[1]
    ax.scatter(walls_n, nuts_z[:, 7], c='C0', s=18, alpha=0.5, label='NUTS')
    ax.plot(rwmh[:, 1], rwmh[:, 2], 'C3o-', ms=4, lw=1.4,
            label='cobaya RW-MH')
    # ESS/s reference lines
    for ess_per_s, color in [(10, 'C0'), (2.3, 'C3')]:
        w = np.geomspace(walls_n.min(), rwmh[:, 1].max(), 50)
        ax.plot(w, ess_per_s * w, color=color, ls=':', alpha=0.7, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("effective sample size")
    ax.set_title("ESS accumulation rate — NUTS ≈ 10 ESS/s, RW-MH ≈ 2.3 ESS/s")
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle("Which sampler reaches a converged posterior faster?",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "accuracy_vs_wall.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved {out}")

    out_npz = os.path.join(os.path.dirname(__file__), "_accuracy_vs_wall.npz")
    np.savez(out_npz, nuts_z=nuts_z, rwmh=rwmh, P0_gold=P0g, be_gold=beg,
             P0_mu=P0_mu_g, P0_std=P0_std_g, be_mu=be_mu_g, be_std=be_std_g,
             wall_gold=wall_gold)
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
