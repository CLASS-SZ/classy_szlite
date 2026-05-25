"""Re-render accuracy_vs_wall.png with cleaner statistics.

- Replace raw per-seed scatter with median ± inter-quartile-range ribbon.
- Add explicit legend entries for the |Z| convergence thresholds.
- Tighten axis ranges so the legend doesn't overlap the data.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__)
    data = np.load(os.path.join(here, "_accuracy_vs_wall.npz"))
    nuts = data["nuts_z"]  # warm, samp, chains, seed, wall, Z_P0, Z_b, ess_P0, ess_b
    rwmh = data["rwmh"]    # n, wall, ess, Z_P0, Z_b
    P0_mu_g = float(data["P0_mu"])
    wall_gold = float(data["wall_gold"])

    # group NUTS by budget
    cfg_id = (nuts[:, 0] * 1e6 + nuts[:, 1] * 1e3 + nuts[:, 2]).astype(int)
    uniq = np.unique(cfg_id)
    nuts_summary = []  # (wall_median, wall_q1, wall_q3, Z_median, Z_q1, Z_q3, ess_med)
    for u in uniq:
        m = cfg_id == u
        w = nuts[m, 4]; z = nuts[m, 5]; e = nuts[m, 7]
        nuts_summary.append((np.median(w), np.quantile(w, 0.25), np.quantile(w, 0.75),
                              np.median(z), np.quantile(z, 0.25), np.quantile(z, 0.75),
                              np.median(e)))
    nuts_summary = np.asarray(nuts_summary)
    order = np.argsort(nuts_summary[:, 0])
    nuts_summary = nuts_summary[order]

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- Panel A: |Z| vs wall ----
    ax = axs[0]
    # per-seed scatter as faint underlay (the raw data — for honesty)
    n_per_budget = int(len(nuts) / max(1, len(np.unique(
        (nuts[:, 0] * 1e6 + nuts[:, 1] * 1e3 + nuts[:, 2]).astype(int)))))
    ax.scatter(nuts[:, 4], nuts[:, 5], c='C0', s=14, alpha=0.25,
                edgecolors='none',
                label=f"NUTS — per seed (n = {n_per_budget} per budget)")
    # IQR ribbon + median line on top
    ax.fill_between(nuts_summary[:, 0], nuts_summary[:, 4], nuts_summary[:, 5],
                     color='C0', alpha=0.22, label="NUTS — 25–75% IQR")
    ax.plot(nuts_summary[:, 0], nuts_summary[:, 3], 'C0o-', lw=2, ms=6,
            label="NUTS — median across seeds")
    ax.plot(rwmh[:, 1], rwmh[:, 3], 'C3o-', ms=4, lw=1.4,
            label="cobaya RW-MH — prefix subsets of converged chain")

    ax.axhline(0.1, color='k', ls=':', alpha=0.7, lw=1, label=r"$0.1\,\sigma$ (publication-grade)")
    ax.axhline(1.0, color='k', ls='-.', alpha=0.5, lw=1, label=r"$1\,\sigma$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$|\hat\mu_{P_0} - \mu_{P_0}^\mathrm{gold}| \,/\, \sigma_{P_0}^\mathrm{gold}$")
    ax.set_title("posterior-mean accuracy vs wall")
    ax.set_ylim(0.01, 3)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')

    # ---- Panel B: ESS rate ----
    ax = axs[1]
    # per-seed scatter as faint underlay (raw ESS values, by seed)
    ax.scatter(nuts[:, 4], nuts[:, 7], c='C0', s=14, alpha=0.25,
                edgecolors='none', label="NUTS — per seed")
    ax.plot(nuts_summary[:, 0], nuts_summary[:, 6], 'C0o-', lw=2, ms=6,
            label="NUTS — median across seeds")
    ax.plot(rwmh[:, 1], rwmh[:, 2], 'C3o-', ms=4, lw=1.4,
            label="cobaya RW-MH — prefix subsets")
    # asymptotic rate references
    w_ref = np.geomspace(0.5, 1500, 50)
    ax.plot(w_ref, 10.0 * w_ref, 'C0--', lw=0.8, alpha=0.6, label="NUTS: ~10 ESS/s")
    ax.plot(w_ref, 2.3 * w_ref, 'C3--', lw=0.8, alpha=0.6, label="RW-MH: ~2.3 ESS/s")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("effective sample size  $N / (1 + 2\\tau_\\mathrm{int})$")
    ax.set_title("ESS accumulation rate")
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')

    # No suptitle — the LaTeX figure caption carries the framing.
    # If running standalone, uncomment the next line.
    # plt.suptitle(rf"Gold-standard NUTS reference: $\mu_{{P_0}} = {P0_mu_g:.2f}$,  $\sigma_{{P_0}} = {float(data['P0_std']):.2f}$  ({wall_gold:.0f} s)", fontsize=10, y=1.04)
    plt.tight_layout()
    out = os.path.join(here, "accuracy_vs_wall.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
