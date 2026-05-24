"""Simulation-Based Inference of (P0, β) on Cl^yy bandpowers — NPE via flowjax.

Trains a conditional Masked Autoregressive Flow q_phi(θ | y) on simulations
drawn from classy_szlite at two fitting cosmologies (baseline + Flamingo
lows8), then evaluates the trained flow at the observed bandpower vector.

Key points the script demonstrates:

  1. **No gradients of the forward model are needed.**  SBI only requires
     the ability to simulate (θ, y) pairs.  classy_szlite's
     ``cl_yy_factory(cosmo, ell)`` closure makes this fast: each (θ → y)
     pair takes ~5 ms (or ~3-4 ms when batched with jax.vmap).
  2. **Amortisation.**  Once trained, a single flow gives a posterior
     for ANY new bandpower realisation in O(ms).  We demonstrate this by
     drawing 5 fresh synthetic realisations at the same truth and
     comparing the resulting amortised posteriors.

Also runs cobaya MH (loaded from disk) and NumPyro NUTS (in-memory) for
each cosmology so the figures can show all three samplers overlaid.

Produces two figures next to this script:
  * sbi_corner_6contours.png  — 6 contours (NUTS + MH + SBI × 2 cosmologies)
  * sbi_amortised.png         — amortised SBI posterior over 5 synthetic
                                bandpower realisations at the baseline

Wall time on a single-thread laptop CPU:
  * ~50 s simulation (5000 samples × 2 cosmologies)
  * ~30 s flow training (per cosmology)
  * ~80 s NUTS (40 s × 2 cosmologies) — already in nuts_clyy_profile.py
  Total: ~5 min cold; subsequent re-evaluations are O(ms) via the flow.
"""
from __future__ import annotations
import os, time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.optimize as so

import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import classy_szlite as csl

# flowjax — JAX-native normalizing flows for NPE
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.train import fit_to_data


# ---------------------------------------------------------------------------
# Setup (matches examples/nuts_clyy_profile.py)
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

FIT_COSMOS = [
    dict(
        key="baseline", label="baseline (σ8≈0.81)", color="C0",
        cobaya_root=os.environ.get("CLYY_COBAYA_BASE", COBAYA_BASE_DEFAULT),
        cosmo_kwargs=dict(omega_b=0.0226, omega_cdm=0.118,
                          H0=68.22, tau_reio=0.0561,
                          ln10_10_As=3.06, n_s=0.9743),
    ),
    dict(
        key="lows8", label="lows8 (σ8≈0.75)", color="C3",
        cobaya_root=os.environ.get("CLYY_COBAYA_LOWS8", COBAYA_LOWS8_DEFAULT),
        cosmo_kwargs=dict(omega_b=0.0226, omega_cdm=0.118,
                          H0=67.14116850291264, tau_reio=0.0561,
                          ln10_10_As=2.910, n_s=0.9743),
    ),
]

C500_FIX, GAMMA_FIX, ALPHA_FIX, B_FIX = 1.156, 0.3292, 1.062, 1.25
# Physically motivated prior box — wide enough not to truncate the
# bandpower posterior (NUTS posterior at the baseline cosmology is
# P0 ≈ 2 ± 1.5, β ≈ 3.2 ± 0.7) but narrow enough for single-round
# NPE to be sample-efficient.
P0_PRIOR  = (0.0,  8.0)
BETA_PRIOR = (1.5, 6.0)

N_SIM_R1        = 8000          # Sequential NPE — round 1, uniform prior
N_SIM_R2        = 8000          # Sequential NPE — round 2, proposal near bestfit
SBI_BATCH       = 512
SBI_LR          = 5e-4
SBI_EPOCHS      = 500           # early-stops on val-loss patience
N_AMORTISED     = 5             # bandpower realisations for the amortisation demo


# ---------------------------------------------------------------------------
# Bandpowers + forward model + samplers
# ---------------------------------------------------------------------------
def load_bandpowers():
    ell  = np.loadtxt(os.path.join(DATA_DIR, ELL_FILE))
    data = np.loadtxt(os.path.join(DATA_DIR, DATA_FILE))
    cov  = np.loadtxt(os.path.join(DATA_DIR, COV_FILE))
    return ell, data[:, 1], cov


def build_forward(cosmo, ell_np):
    ell = jnp.asarray(ell_np)
    ev  = csl.cl_yy_factory(cosmo, ell)
    dl_factor = jnp.asarray(ell * (ell + 1) / (2 * np.pi) * 1e12)

    def forward(P0, beta):
        prof = csl.ProfileParamsA10(
            P0=P0, c500=C500_FIX, gamma=GAMMA_FIX, alpha=ALPHA_FIX,
            beta=beta, B=B_FIX,
        )
        cl_1h, cl_2h = ev(prof)
        return dl_factor * (cl_1h + cl_2h)
    return forward


def find_bestfit(forward, y, inv_cov, x0=(8.13, 5.48),
                 bounds=((0.1, 20.), (0.5, 10.))):
    y_jnp, inv_jnp = jnp.asarray(y), jnp.asarray(inv_cov)

    def nll(x):
        r = y_jnp - forward(x[0], x[1])
        return 0.5 * r @ inv_jnp @ r

    nll_j = jax.jit(nll); nll_g = jax.jit(jax.grad(nll))
    return so.minimize(lambda x: float(nll_j(x)),
                       np.asarray(x0, dtype=np.float64),
                       jac=lambda x: np.asarray(nll_g(x)),
                       method="L-BFGS-B", bounds=bounds)


def run_nuts(forward, y, inv_cov, P0_init, beta_init,
             num_warmup=500, num_samples=2000, num_chains=4, seed=0):
    y_jnp, inv_jnp = jnp.asarray(y), jnp.asarray(inv_cov)

    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(*P0_PRIOR))
        beta = numpyro.sample("beta", dist.Uniform(*BETA_PRIOR))
        r    = y_jnp - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ inv_jnp @ r)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.85, dense_mass=True),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=False)
    init = {"P0":   jnp.full((num_chains,), P0_init),
            "beta": jnp.full((num_chains,), beta_init)}
    mcmc.run(jax.random.PRNGKey(seed), init_params=init)
    return mcmc


# ---------------------------------------------------------------------------
# SBI: simulator + NPE training
# ---------------------------------------------------------------------------
def _simulate_pairs(forward, cov, theta, seed):
    """Given a (n_sim, 2) theta array, simulate y = model(theta) + noise."""
    rng = np.random.default_rng(seed)
    forward_v = jax.vmap(forward, in_axes=(0, 0))
    mu_clean = np.asarray(forward_v(jnp.asarray(theta[:, 0]),
                                     jnp.asarray(theta[:, 1])))
    L     = np.linalg.cholesky(cov)
    noise = (L @ rng.standard_normal((len(cov), len(theta)))).T
    return jnp.asarray(theta), jnp.asarray(mu_clean + noise)


def simulate_round1(forward, cov, n_sim, seed):
    """Round-1 sims: uniform draws from the prior box."""
    rng = np.random.default_rng(seed)
    theta = np.column_stack([
        rng.uniform(*P0_PRIOR,  n_sim),
        rng.uniform(*BETA_PRIOR, n_sim),
    ])
    return _simulate_pairs(forward, cov, theta, seed + 1)


def simulate_round2(forward, cov, bf_x, n_sim, seed,
                    width_P0=2.0, width_beta=1.0):
    """Round-2 sims: Gaussian proposal centered on L-BFGS bestfit, truncated
    to the prior box.  Mimics sequential NPE without the importance-weight
    correction (APT-style)."""
    rng = np.random.default_rng(seed)
    # Draw 5x overshoot, then reject outside the prior box
    over = 5
    P0_s   = rng.normal(bf_x[0], width_P0,   n_sim * over)
    beta_s = rng.normal(bf_x[1], width_beta, n_sim * over)
    mask = ((P0_s >= P0_PRIOR[0]) & (P0_s <= P0_PRIOR[1]) &
            (beta_s >= BETA_PRIOR[0]) & (beta_s <= BETA_PRIOR[1]))
    P0_s, beta_s = P0_s[mask][:n_sim], beta_s[mask][:n_sim]
    if len(P0_s) < n_sim:
        # Refill via uniform tail (rare; large overshoot above should cover this)
        rng2 = np.random.default_rng(seed + 99)
        extra_P0   = rng2.uniform(*P0_PRIOR,  n_sim - len(P0_s))
        extra_beta = rng2.uniform(*BETA_PRIOR, n_sim - len(beta_s))
        P0_s   = np.concatenate([P0_s,   extra_P0])
        beta_s = np.concatenate([beta_s, extra_beta])
    theta = np.column_stack([P0_s, beta_s])
    return _simulate_pairs(forward, cov, theta, seed + 1)


def standardise(arr):
    """Return ((arr - mu) / std, mu, std) along axis 0."""
    mu  = jnp.asarray(np.mean(np.asarray(arr), axis=0))
    std = jnp.asarray(np.std(np.asarray(arr),  axis=0) + 1e-12)
    return (arr - mu) / std, mu, std


def train_npe(theta_train, y_train, key):
    """Train a conditional MAF q_phi(theta | y) via maximum likelihood.

    Internally standardises both theta and y to (mu=0, std=1) before
    training the flow.  Returns (flow, losses, theta_stats, y_stats)
    so the caller can map samples back to physical units.
    """
    theta_z, mu_th, std_th = standardise(theta_train)
    y_z,     mu_y,  std_y  = standardise(y_train)

    n_theta, n_y = theta_z.shape[1], y_z.shape[1]
    flow = masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(n_theta)),
        cond_dim=n_y,
        nn_width=128, nn_depth=3,
        flow_layers=8,
    )
    # In flowjax 19, conditional fits are signalled by passing
    # data as a tuple (x, condition); the default MaximumLikelihoodLoss
    # accepts (x, condition) automatically.
    flow, losses = fit_to_data(
        key=key,
        dist=flow,
        data=(theta_z, y_z),
        max_epochs=SBI_EPOCHS,
        batch_size=SBI_BATCH,
        learning_rate=SBI_LR,
        show_progress=False,
    )
    return flow, losses, (mu_th, std_th), (mu_y, std_y)


def sample_npe(flow, theta_stats, y_stats, y_obs, n=3000, seed=1):
    """Draw n samples from q_phi(theta | y_obs), in physical units."""
    mu_th, std_th = theta_stats
    mu_y,  std_y  = y_stats
    y_z = (jnp.asarray(y_obs) - mu_y) / std_y
    key = jr.key(seed)                                              # new-style typed PRNG key
    samp_z = flow.sample(key, sample_shape=(n,), condition=y_z)
    return np.asarray(samp_z * std_th + mu_th)


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

        # --- 1. Bestfit + NUTS (same as nuts_clyy_profile.py) -------------
        t0 = time.perf_counter()
        bf = find_bestfit(forward, y, inv_cov)
        chi2_bf = 2 * bf.fun
        print(f"  L-BFGS bestfit in {time.perf_counter()-t0:.2f} s "
              f"→ P0={bf.x[0]:.3f}, β={bf.x[1]:.3f}, χ²={chi2_bf:.2f}")

        t0 = time.perf_counter()
        mcmc = run_nuts(forward, y, inv_cov,
                        P0_init=float(bf.x[0]), beta_init=float(bf.x[1]))
        nuts_samples = mcmc.get_samples()
        nuts_arr = np.column_stack([np.asarray(nuts_samples["P0"]),
                                     np.asarray(nuts_samples["beta"])])
        print(f"  NUTS in {time.perf_counter()-t0:.1f} s  "
              f"→ P0={nuts_arr[:,0].mean():.3f}±{nuts_arr[:,0].std():.3f}  "
              f"β={nuts_arr[:,1].mean():.3f}±{nuts_arr[:,1].std():.3f}")

        # --- 2. cobaya MH overlay (if chain present) ----------------------
        cobaya_root = os.path.expanduser(cfg["cobaya_root"])
        cobaya_arr = None
        if os.path.isfile(cobaya_root + ".1.txt"):
            src = loadMCSamples(cobaya_root, settings={"ignore_rows": 0.3})
            p0  = src.samples[:, src.paramNames.numberOfName("P0GNFW")]
            bt  = src.samples[:, src.paramNames.numberOfName("betaGNFW")]
            cobaya_arr = np.column_stack([p0, bt])
            cobaya_w   = src.weights
            print(f"  cobaya MH loaded from {cobaya_root}*.txt  (n={src.numrows})")
        else:
            cobaya_w = None
            print(f"  no cobaya chain at {cobaya_root}*.txt — skipping MH")

        # --- 3. Sequential NPE: train on round-2 proposal around bestfit --
        # Round-1 (uniform) sims are included only to provide context for
        # the proposal kernel — for NPE training we use round-2 only,
        # which is dense around the bestfit (APT-without-importance-weights).
        t0 = time.perf_counter()
        theta_train, y_train = simulate_round2(
            forward, cov, bf.x, N_SIM_R2, seed=43,
            width_P0=2.5, width_beta=1.2,
        )
        t_sim = time.perf_counter() - t0
        print(f"  SBI: {N_SIM_R2} sims around bestfit "
              f"(P0={bf.x[0]:.2f}±2.5, β={bf.x[1]:.2f}±1.2) in {t_sim:.1f}s")

        t0 = time.perf_counter()
        flow, losses, th_stats, y_stats = train_npe(
            theta_train, y_train, jr.key(0),
        )
        t_train = time.perf_counter() - t0
        print(f"    train ({SBI_EPOCHS} epoch max) in {t_train:.1f} s; "
              f"final val-loss {losses['val'][-1]:.3f}")

        t0 = time.perf_counter()
        sbi_arr = sample_npe(flow, th_stats, y_stats,
                              jnp.asarray(y), n=3000, seed=1)
        t_eval = time.perf_counter() - t0
        # Reject outside prior
        mask = ((sbi_arr[:, 0] > P0_PRIOR[0]) & (sbi_arr[:, 0] < P0_PRIOR[1]) &
                (sbi_arr[:, 1] > BETA_PRIOR[0]) & (sbi_arr[:, 1] < BETA_PRIOR[1]))
        sbi_arr = sbi_arr[mask]
        print(f"    eval at observed in {t_eval*1e3:.0f} ms  →  "
              f"P0={sbi_arr[:,0].mean():.3f}±{sbi_arr[:,0].std():.3f}  "
              f"β={sbi_arr[:,1].mean():.3f}±{sbi_arr[:,1].std():.3f}  "
              f"(n_post={len(sbi_arr)})")

        results.append({
            "cfg":          cfg,
            "cosmo":        cosmo,
            "sigma_8":      s8,
            "bf":           bf,
            "chi2_bf":      chi2_bf,
            "nuts_arr":     nuts_arr,
            "cobaya_arr":   cobaya_arr,
            "cobaya_w":     cobaya_w,
            "flow":         flow,
            "theta_stats":  th_stats,
            "y_stats":      y_stats,
            "sbi_arr":      sbi_arr,
            "t_sim":        t_sim,
            "t_train":      t_train,
            "t_eval_ms":    t_eval * 1e3,
        })

    # ---------- plots --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import plots as gdplots

    out_dir = os.path.dirname(__file__) or "."

    # 1) Corner with 6 contours
    print("\nDrawing 6-contour corner plot ...")
    gd_samples, labels, colors, ls_list, fill = [], [], [], [], []
    for r in results:
        c = r["cfg"]["color"]
        # NUTS — solid filled
        nuts_gd = MCSamples(
            samples=r["nuts_arr"],
            names=["P0", "beta"],
            labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
            label=f"NUTS — {r['cfg']['label']}",
        )
        gd_samples.append(nuts_gd); labels.append(nuts_gd.label)
        colors.append(c); ls_list.append("-"); fill.append(True)
        # cobaya MH — dashed unfilled
        if r["cobaya_arr"] is not None:
            mh = MCSamples(samples=r["cobaya_arr"], weights=r["cobaya_w"],
                names=["P0", "beta"],
                labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
                label=f"cobaya MH — {r['cfg']['label']}")
            gd_samples.append(mh); labels.append(mh.label)
            colors.append(c); ls_list.append("--"); fill.append(False)
        # SBI — dotted unfilled
        sbi_gd = MCSamples(
            samples=r["sbi_arr"],
            names=["P0", "beta"],
            labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"],
            label=f"SBI — {r['cfg']['label']}",
        )
        gd_samples.append(sbi_gd); labels.append(sbi_gd.label)
        colors.append(c); ls_list.append(":"); fill.append(False)

    g = gdplots.get_subplot_plotter(width_inch=6.0)
    g.settings.alpha_filled_add = 0.35
    g.settings.legend_fontsize  = 9
    g.triangle_plot(gd_samples, params=["P0", "beta"],
                    filled=fill, legend_labels=labels,
                    contour_colors=colors, contour_ls=ls_list)
    out_corner = os.path.join(out_dir, "sbi_corner_6contours.png")
    g.export(out_corner, dpi=300)
    print(f"  -> wrote {out_corner}")

    # 2) Amortisation demo — multiple realisations at the baseline truth
    print("\nDrawing SBI amortisation demo ...")
    base = results[0]
    forward_base = build_forward(base["cosmo"], ell_np)
    # Draw fresh noise realisations at the bestfit
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(99)
    truth_mu = np.asarray(forward_base(jnp.asarray(base["bf"].x[0]),
                                        jnp.asarray(base["bf"].x[1])))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), dpi=300)
    for r_ix in range(N_AMORTISED):
        noise = L @ rng.standard_normal(len(cov))
        y_new = truth_mu + noise
        sbi_post = sample_npe(base["flow"], base["theta_stats"],
                              base["y_stats"], jnp.asarray(y_new),
                              n=2000, seed=1000+r_ix)
        mask = ((sbi_post[:, 0] > P0_PRIOR[0]) & (sbi_post[:, 0] < P0_PRIOR[1]) &
                (sbi_post[:, 1] > BETA_PRIOR[0]) & (sbi_post[:, 1] < BETA_PRIOR[1]))
        sbi_post = sbi_post[mask]
        gd_a = MCSamples(samples=sbi_post, names=["P0", "beta"],
            labels=[r"P_0^{\rm GNFW}", r"\beta^{\rm GNFW}"])
        # 1-D KDE per parameter
        for ax, name in zip(axes, ["P0", "beta"]):
            d = gd_a.get1DDensity(name)
            ax.plot(d.x, d.P / d.P.max(),
                    label=f"realisation #{r_ix+1}", alpha=0.8)
    axes[0].axvline(base["bf"].x[0], color="k", ls="--", lw=1, label="truth")
    axes[1].axvline(base["bf"].x[1], color="k", ls="--", lw=1, label="truth")
    axes[0].set_xlabel(r"$P_0^{\rm GNFW}$"); axes[0].set_ylabel("posterior (1D)")
    axes[1].set_xlabel(r"$\beta^{\rm GNFW}$")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Amortised SBI posterior over fresh noise realisations "
                 "(baseline cosmology, single trained flow)", fontsize=10)
    fig.tight_layout()
    out_amor = os.path.join(out_dir, "sbi_amortised.png")
    fig.savefig(out_amor, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_amor}")

    print("\nSummary:")
    print(f"{'cosmology':24s}  {'L-BFGS P0/β':14s}  {'NUTS P0/β':18s}  "
          f"{'SBI P0/β':18s}  {'SBI sim/train/eval':18s}")
    for r in results:
        print(f"  {r['cfg']['label']:22s}  "
              f"{r['bf'].x[0]:5.2f}/{r['bf'].x[1]:.2f}     "
              f"{r['nuts_arr'][:,0].mean():.2f}±{r['nuts_arr'][:,0].std():.2f}/"
              f"{r['nuts_arr'][:,1].mean():.2f}±{r['nuts_arr'][:,1].std():.2f}  "
              f"{r['sbi_arr'][:,0].mean():.2f}±{r['sbi_arr'][:,0].std():.2f}/"
              f"{r['sbi_arr'][:,1].mean():.2f}±{r['sbi_arr'][:,1].std():.2f}  "
              f"{r['t_sim']:.0f}+{r['t_train']:.0f}+{r['t_eval_ms']:.0f}ms")


if __name__ == "__main__":
    main()
