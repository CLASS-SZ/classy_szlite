"""Plot NUTS + cobaya RW-MH + NPE-A + NPE-B for the baseline posterior."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import MCSamples, plots, loadMCSamples


def main():
    here = os.path.dirname(__file__)

    # cobaya RW-MH
    cobaya = loadMCSamples(os.path.expanduser("~/class-sz-plugin-tests/chains/clyy_v2"),
                            settings={"ignore_rows": 0.1})
    cobaya.label = "cobaya RW-MH"

    # NUTS
    nuts = np.load(os.path.join(here, "_nuts_baseline.npz"))
    nuts_s = MCSamples(
        samples=np.column_stack([nuts['P0'], nuts['beta']]),
        names=["P0GNFW", "betaGNFW"],
        labels=["P_0", r"\beta"],
        label="NumPyro NUTS",
    )

    # NPE-A and NPE-B
    npe = np.load(os.path.join(here, "_npe_diagnose.npz"))
    npe_A = MCSamples(
        samples=np.column_stack([npe['npe_A_P0'], npe['npe_A_beta']]),
        names=["P0GNFW", "betaGNFW"],
        labels=["P_0", r"\beta"],
        label="NPE: truncated-Gauss proposal (paper)",
    )
    npe_B = MCSamples(
        samples=np.column_stack([npe['npe_B_P0'], npe['npe_B_beta']]),
        names=["P0GNFW", "betaGNFW"],
        labels=["P_0", r"\beta"],
        label="NPE: uniform-prior proposal",
    )

    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 10
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 10
    g.triangle_plot(
        [cobaya, nuts_s, npe_A, npe_B],
        params=["P0GNFW", "betaGNFW"],
        filled=True,
        legend_labels=[
            "cobaya RW-MH (truth, $n_\\mathrm{eff}\\!\\approx\\!1926$)",
            "NumPyro NUTS",
            f"NPE-A: truncated-Gauss proposal\n   $\\mu_\\mathrm{{prop}}\\!=\\!{float(npe['trunc_proposal_mean_P0']):.2f}$, posterior peak $\\!\\approx\\!2.7$",
            "NPE-B: uniform-prior proposal\n   (worse — sample-inefficient)",
        ],
        contour_colors=["k", "C0", "C3", "C2"],
        contour_lws=[1.5, 1.5, 1.5, 1.5],
    )
    g.fig.suptitle(
        "Why NPE drifts from the truth: proposal-truncation bias\n"
        "(see Table 1 of paper)",
        y=1.04, fontsize=11, fontweight="bold",
    )
    out = os.path.join(here, "npe_diagnose.png")
    g.export(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
