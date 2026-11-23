"""Catalogue-level neural posterior estimation, as in the paper, and its validation.

    $SBI_PY benchmarks/mapsim/catalogue_npe.py [--n-train 10000] [--seed 0] [--samples 10000]

Inputs: catalogue_trainset.py's $MAPSIM_OUT/catalogue_npe/{train,heldout,truth}.npz.
Training (the paper's set-up, Sec. II.4): sbi NPE (NPE_C) with a masked autoregressive flow
('maf', sbi's default size), one round, learning rate 5e-4, stop after 50 epochs without
improvement in the validation loss; sbi's defaults for the rest (batch, validation fraction,
z-scoring of theta and x). Prior: the paper's Table 1 (sbi MultipleIndependent of Uniform and
Normal), with omega_cdm up to 0.200 as in the simulations.

Validation:
  truth     the paper's: 100 catalogues at the fiducial point; per parameter the mean over
            them of (posterior mean - truth) / posterior std, with its standard error, beside the
            paper's Table 1; also the spread of the posterior means in units of the posterior std
            (about 1 for a calibrated posterior)
  coverage  500 held-out prior draws: per-parameter ranks of the truth among posterior samples
            (KS test of their uniformity) and expected coverage (the fraction of truths inside the
            central credible interval at levels 10%-90%); TARP if sbi provides it
  widths    the median posterior std at the fiducial point against the paper's likelihood widths
Writes $MAPSIM_OUT/catalogue_npe/npe_{tag}.{pt,json,txt}.

Options for the bias test (PLAN T1, T2): --data DIR (the train/heldout/truth sets, default
catalogue_npe/), --prior {paper, maps} (the map-level box of maps_trainset.py), and --apply NAME=FILE
(repeatable): the trained posterior on another set's data vectors, e.g. the catalogue-level
posterior on the map-level mocks (maps_npe/truth.npz): bias of the posterior means against that
set's true parameters, also for S8 = sigma8 sqrt(Omega_m / 0.3) and Omega_m. Posterior samples of
the first catalogue of the truth set and of each applied set are saved for plots.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "catalogue_npe")
NAMES = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ", "h", "omega_b", "n_s", "sigma_lnq", "beta_lens", "sigma_lnp")
UNIFORMS = {"paper": {"sigma8": (0.7, 0.9), "omega_cdm": (0.0804, 0.200), "A_SZ": (-4.41, -4.21), "alpha_SZ": (1.0, 1.24)},
            "maps": {"sigma8": (0.770, 0.852), "omega_cdm": (0.0804, 0.169), "A_SZ": (-4.53, -4.08), "alpha_SZ": (1.0, 1.24)}}
UNIFORM = dict(UNIFORMS["paper"])
OMEGA_NU_H2 = 3 * 0.02 / 93.14


def derived(s):
    """(Omega_m, S8) from samples (n, 10) in NAMES order."""
    om = (s[:, 1] + s[:, 5] + OMEGA_NU_H2) / s[:, 4] ** 2
    return om, s[:, 0] * np.sqrt(om / 0.3)
GAUSS = {"h": (0.674, 0.01), "omega_b": (0.0222, 0.00015), "n_s": (0.96, 0.0042), "sigma_lnq": (0.173, 0.05),
         "beta_lens": (0.92, 0.01), "sigma_lnp": (0.22, 0.05)}
PAPER_BIAS = [-0.107, 0.031, 0.052, 0.10, 0.0211, -0.1039, -0.0395, -0.0585, -0.0122, -0.0417]   # Table 1
PAPER_BIAS_ERR = [0.086, 0.063, 0.076, 0.10, 0.0033, 0.0023, 0.0015, 0.0063, 0.0048, 0.0048]
PAPER_WIDTH = [0.0051, 0.0060, 0.028, 0.016, 0.010, 0.00015, 0.0042, 0.045, 0.010, 0.050]           # sigma_like


def make_prior():
    from sbi.utils import MultipleIndependent
    from torch.distributions import Uniform, Normal
    d = []
    for n in NAMES:
        if n in UNIFORM:
            lo, hi = UNIFORM[n]
            d.append(Uniform(torch.tensor([lo], dtype=torch.float32), torch.tensor([hi], dtype=torch.float32)))
        else:
            mu, sd = GAUSS[n]
            d.append(Normal(torch.tensor([mu], dtype=torch.float32), torch.tensor([sd], dtype=torch.float32)))
    return MultipleIndependent(d, validate_args=False)


def sbi_api():
    import sbi.inference as si
    NPE = getattr(si, "NPE", None) or getattr(si, "NPE_C", None) or getattr(si, "SNPE_C")
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils import posterior_nn
    return NPE, posterior_nn


DATA = OUT


def load(name):
    d = np.load(name if name.endswith(".npz") else os.path.join(DATA, f"{name}.npz"))
    ok = d["done"] & np.all(np.isfinite(d["x"]), axis=1)
    return d["theta"][ok].astype(np.float32), d["x"][ok].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=OUT)
    ap.add_argument("--prior", choices=list(UNIFORMS), default="paper")
    ap.add_argument("--apply", action="append", default=[], help="NAME=FILE.npz")
    args = ap.parse_args()
    global DATA
    DATA = args.data
    UNIFORM.clear()
    UNIFORM.update(UNIFORMS[args.prior])
    tag = args.tag or f"n{args.n_train}_s{args.seed}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import sbi
    NPE, posterior_nn = sbi_api()
    th, x = load("train")
    th, x = th[:args.n_train], x[:args.n_train]
    prior = make_prior()
    lines = []
    P = lambda s: (print(s, flush=True), lines.append(s))
    P(f"sbi {sbi.__version__}, torch {torch.__version__}; {len(th)} training simulations; tag {tag}")

    t0 = time.time()
    inf = NPE(prior=prior, density_estimator=posterior_nn(model="maf"))
    inf.append_simulations(torch.from_numpy(th), torch.from_numpy(x))
    de = inf.train(learning_rate=5e-4, stop_after_epochs=50, show_train_summary=False)
    post = inf.build_posterior(de)
    summ = getattr(inf, "summary", {}) or {}
    n_ep = len(summ.get("validation_loss", summ.get("validation_log_probs", [])))
    P(f"trained in {time.time() - t0:.0f} s, {n_ep} epochs")
    torch.save(de.state_dict() if hasattr(de, "state_dict") else de, os.path.join(DATA, f"npe_{tag}.pt"))
    res = dict(tag=tag, n_train=len(th), sbi=sbi.__version__, epochs=n_ep)

    def samples(xo, n):
        return post.sample((n,), x=torch.from_numpy(xo[None]), show_progress_bars=False).numpy()

    # 1. the paper's validation: 100 catalogues at the fiducial point
    tt, xt = load("truth")
    truth = tt[0]
    mu, sd = [], []
    t0 = time.time()
    for xo in xt:
        s = samples(xo, args.samples)
        mu.append(s.mean(0)); sd.append(s.std(0))
    mu, sd = np.array(mu), np.array(sd)
    b = (mu - truth) / sd
    bias, bias_err = b.mean(0), b.std(0, ddof=1) / np.sqrt(len(b))
    spread = mu.std(0, ddof=1) / sd.mean(0)
    P(f"\n1. {len(xt)} catalogues at the fiducial point ({time.time() - t0:.0f} s): bias of the posterior means "
      "<(mu - truth)/sigma>, ours and the paper's; spread of the means / mean sigma; median sigma / paper's sigma_like")
    for j, n in enumerate(NAMES):
        P(f"   {n:10s} {bias[j]:+7.3f} +- {bias_err[j]:.3f}   paper {PAPER_BIAS[j]:+7.3f} +- {PAPER_BIAS_ERR[j]:.3f}"
          f"   spread {spread[j]:.2f}   width {np.median(sd[:, j]) / PAPER_WIDTH[j]:.2f}")
    res["truth"] = dict(bias=bias.tolist(), bias_err=bias_err.tolist(), spread=spread.tolist(),
                        median_sigma=np.median(sd, 0).tolist())

    # 2. coverage on held-out prior draws
    th_h, x_h = load("heldout")
    levels = np.linspace(0.1, 0.9, 9)
    ranks, inside = [], []
    t0 = time.time()
    n_cov = min(args.samples, 2000)
    for t_i, x_i in zip(th_h, x_h):
        s = samples(x_i, n_cov)
        ranks.append((s < t_i[None]).mean(0))
        lo = np.quantile(s, 0.5 - levels / 2, axis=0)                     # (9, 10)
        hi = np.quantile(s, 0.5 + levels / 2, axis=0)
        inside.append((t_i[None] >= lo) & (t_i[None] <= hi))
    ranks, inside = np.array(ranks), np.array(inside)                   # (n, 10), (n, 9, 10)
    from scipy import stats
    ks = [stats.kstest(ranks[:, j], "uniform").pvalue for j in range(len(NAMES))]
    cov = inside.mean(0)                                               # (9, 10)
    P(f"\n2. coverage, {len(th_h)} held-out prior draws ({time.time() - t0:.0f} s, {n_cov} samples each): "
      "KS p of the ranks; coverage of the central 30%, 50%, 70%, 90% intervals")
    for j, n in enumerate(NAMES):
        P(f"   {n:10s} KS p {ks[j]:.3f}   coverage " + "  ".join(f"{cov[k, j]:.2f}" for k in (2, 4, 6, 8)))
    res["coverage"] = dict(levels=levels.tolist(), coverage=cov.tolist(), ks_p=ks)
    try:
        from sbi.diagnostics import run_tarp, check_tarp
        ecp, alpha = run_tarp(torch.from_numpy(th_h), torch.from_numpy(x_h), post, num_posterior_samples=1000)
        atc, ks_p = check_tarp(ecp, alpha)
        P(f"   TARP: area to diagonal {float(atc):+.3f}, KS p {float(ks_p):.3f}")
        res["tarp"] = dict(atc=float(atc), ks_p=float(ks_p))
    except Exception as exc:                                           # older sbi: no TARP
        P(f"   TARP not run: {exc!r}")

    # 3. the posterior on other sets (e.g. the catalogue-level posterior on map-level mocks)
    saved = {"truth": samples(xt[0], args.samples)}
    for item in args.apply:
        name, f = item.split("=", 1)
        ta, xa = load(f)
        mu, sd, mu_d, sd_d, rk = [], [], [], [], []
        for xo, to in zip(xa, ta):
            sm_ = samples(xo, args.samples)
            mu.append(sm_.mean(0)); sd.append(sm_.std(0))
            rk.append(np.mean(sm_ < to[None, :], axis=0))          # posterior rank of the truth
            d = np.stack(derived(sm_), 1)
            mu_d.append(d.mean(0)); sd_d.append(d.std(0))
        mu, sd, mu_d, sd_d, rk = map(np.array, (mu, sd, mu_d, sd_d, rk))
        np.savez(os.path.join(DATA, f"npe_{tag}_applied_{name}.npz"), theta=ta, mu=mu, sd=sd, mu_derived=mu_d,
                 sd_derived=sd_d, rank=rk, names=np.array(NAMES))
        b = (mu - ta) / sd
        td = np.stack(derived(ta), 1)
        bd = (mu_d - td) / sd_d
        P(f"\n3. applied to {name} ({len(xa)} data vectors, {f}): bias <(mu - truth)/sigma> +- error of the mean")
        for j, n in enumerate(NAMES):
            P(f"   {n:10s} {b[:, j].mean():+7.3f} +- {b[:, j].std(ddof=1) / np.sqrt(len(b)):.3f}")
        for j, n in enumerate(("Omega_m", "S8")):
            P(f"   {n:10s} {bd[:, j].mean():+7.3f} +- {bd[:, j].std(ddof=1) / np.sqrt(len(bd)):.3f}"
              f"   (median sigma {np.median(sd_d[:, j]):.4f})")
        if np.ptp(ta[:, 0]) > 0:                                # prior draws: coverage, and bias against sigma8
            cov = {lv: np.mean(np.abs(rk - 0.5) < lv / 2, axis=0) for lv in (0.5, 0.9)}
            P("   coverage of the central 50% and 90% intervals (true parameters of the set):")
            for j, n in enumerate(NAMES[:4]):
                P(f"   {n:10s} {cov[0.5][j]:.2f}  {cov[0.9][j]:.2f}")
            edges = np.quantile(ta[:, 0], [0, 1 / 3, 2 / 3, 1])
            P("   bias (sigma8, omega_cdm, A_SZ, alpha_SZ) in thirds of the true sigma8:")
            for a_, b_ in zip(edges[:-1], edges[1:]):
                m = (ta[:, 0] >= a_) & (ta[:, 0] <= b_)
                P(f"   sigma8 {a_:.3f}-{b_:.3f} ({m.sum()}): " + "  ".join(
                    f"{b[m, j].mean():+.2f}+-{b[m, j].std(ddof=1) / np.sqrt(m.sum()):.2f}" for j in range(4)))
        res.setdefault("applied", {})[name] = dict(file=f, bias=b.mean(0).tolist(),
                                                   bias_err=(b.std(0, ddof=1) / np.sqrt(len(b))).tolist(),
                                                   bias_derived=bd.mean(0).tolist(),
                                                   bias_derived_err=(bd.std(0, ddof=1) / np.sqrt(len(bd))).tolist(),
                                                   median_sigma=np.median(sd, 0).tolist(),
                                                   median_sigma_derived=np.median(sd_d, 0).tolist())
        saved[name] = samples(xa[0], args.samples)
    np.savez(os.path.join(DATA, f"npe_{tag}_samples.npz"), **saved, names=np.array(NAMES))
    open(os.path.join(DATA, f"npe_{tag}.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(DATA, f"npe_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
