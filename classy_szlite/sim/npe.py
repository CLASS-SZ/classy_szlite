"""Neural posterior estimation of the ten parameters from the 21 numbers, and its checks.

Needs the optional ``sbi`` extra (sbi, torch): ``pip install "classy_szlite[sbi]"``.

Training follows Zubeldia et al. (2025, Sec. II.4): sbi's NPE (NPE_C) with a masked autoregressive flow
of sbi's default size, one round, learning rate 5e-4, stop after 50 epochs without improvement of the
validation loss. Typical cost: 10,000 training simulations train in 2-7 min on 2-16 CPU cores.

Checks, on simulations whose true parameters are known:

* ``bias``: the mean over data vectors of (posterior mean - truth) / posterior sd, with its error of the
  mean (on mocks at one parameter point; 0 for a calibrated posterior);
* ``coverage``: the fraction of data vectors whose truth lies in the central credible interval of each
  level (on prior draws; equal to the level for a calibrated posterior), and TARP;
* ``ensemble``: an equal mixture of posteriors trained with different seeds.
"""
from __future__ import annotations

import time

import numpy as np

from . import priors


def load_set(path):
    """(theta, x) of a simulated set (.npz from survey.simulate_*_set), finished and finite rows only."""
    d = np.load(path)
    ok = d["done"] & np.all(np.isfinite(d["x"]), axis=1)
    return d["theta"][ok].astype(np.float32), d["x"][ok].astype(np.float32)


def train(theta, x, box="map", seed=0, log=print):
    """A posterior q(theta | x) trained on (theta, x); returns (posterior, info)."""
    import torch
    import sbi
    import sbi.inference as si
    NPE = getattr(si, "NPE", None) or getattr(si, "NPE_C", None) or getattr(si, "SNPE_C")
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils import posterior_nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    inf = NPE(prior=priors.sbi_prior(box), density_estimator=posterior_nn(model="maf"))
    inf.append_simulations(torch.from_numpy(np.asarray(theta, np.float32)), torch.from_numpy(np.asarray(x, np.float32)))
    de = inf.train(learning_rate=5e-4, stop_after_epochs=50, show_train_summary=False)
    post = inf.build_posterior(de)
    summ = getattr(inf, "summary", {}) or {}
    n_ep = len(summ.get("validation_loss", summ.get("validation_log_probs", [])))
    info = dict(n_train=len(theta), epochs=n_ep, seconds=time.time() - t0, sbi=sbi.__version__, seed=seed)
    log(f"trained on {len(theta)} simulations in {info['seconds']:.0f} s, {n_ep} epochs")
    return post, info


def sample(post, x, n):
    """n posterior samples (n, 10) for one data vector."""
    import torch
    return post.sample((n,), x=torch.from_numpy(np.asarray(x, np.float32)[None]), show_progress_bars=False).numpy()


def summarise(post, theta, x, n_samples=4000):
    """Per data vector: posterior mean, sd, the rank of the truth among the samples, and the same for
    (Omega_m, S8). Arrays of shape (n, 10) and (n, 2)."""
    mu, sd, rk, mu_d, sd_d = [], [], [], [], []
    for t, xo in zip(theta, x):
        s = sample(post, xo, n_samples)
        mu.append(s.mean(0)); sd.append(s.std(0)); rk.append(np.mean(s < t[None], axis=0))
        d = np.stack(priors.derived(s), 1)
        mu_d.append(d.mean(0)); sd_d.append(d.std(0))
    return dict(theta=np.asarray(theta), mu=np.array(mu), sd=np.array(sd), rank=np.array(rk),
                mu_derived=np.array(mu_d), sd_derived=np.array(sd_d))


def bias(summary):
    """Mean of (mu - truth) / sd and its error of the mean, per parameter; also for (Omega_m, S8)."""
    b = (summary["mu"] - summary["theta"]) / summary["sd"]
    td = np.stack(priors.derived(summary["theta"]), 1)
    bd = (summary["mu_derived"] - td) / summary["sd_derived"]
    err = lambda a: a.std(0, ddof=1) / np.sqrt(len(a))
    return dict(bias=b.mean(0), err=err(b), bias_derived=bd.mean(0), err_derived=err(bd))


LEVELS = np.linspace(0.1, 0.9, 9)


def coverage(summary, levels=LEVELS):
    """(len(levels), 10): fraction of data vectors whose truth is in the central interval of each level
    (from the rank of the truth among the posterior samples)."""
    rk = summary["rank"]
    return np.array([np.mean(np.abs(rk - 0.5) < lv / 2, axis=0) for lv in levels])


def tarp(post, theta, x, n_samples=1000):
    """TARP (Lemos et al. 2023) of the joint posterior: (area to the diagonal, KS p-value)."""
    import torch
    from sbi.diagnostics import run_tarp, check_tarp
    ecp, alpha = run_tarp(torch.from_numpy(np.asarray(theta, np.float32)), torch.from_numpy(np.asarray(x, np.float32)),
                          post, num_posterior_samples=n_samples)
    atc, ks_p = check_tarp(ecp, alpha)
    return float(atc), float(ks_p)


def ensemble(summaries):
    """An equal mixture of posteriors (summaries of the same data vectors): mean of the means, variance =
    mean variance + variance of the means, rank = mean rank (exact for equal sample numbers)."""
    mus = np.stack([s["mu"] for s in summaries]); sds = np.stack([s["sd"] for s in summaries])
    mud = np.stack([s["mu_derived"] for s in summaries]); sdd = np.stack([s["sd_derived"] for s in summaries])
    return dict(theta=summaries[0]["theta"], mu=mus.mean(0), sd=np.sqrt((sds ** 2).mean(0) + mus.var(0)),
                rank=np.stack([s["rank"] for s in summaries]).mean(0), mu_derived=mud.mean(0),
                sd_derived=np.sqrt((sdd ** 2).mean(0) + mud.var(0)))
