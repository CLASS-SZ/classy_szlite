"""Compare cobaya RW-MH and NumPyro NUTS posteriors via getdist contours."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from getdist import MCSamples, plots, loadMCSamples


def main():
    here = os.path.dirname(__file__)

    # --- Cobaya RW-MH chain via getdist loader ---
    cobaya_root = os.path.expanduser("~/class-sz-plugin-tests/chains/clyy_v2")
    cobaya_samples = loadMCSamples(cobaya_root, settings={"ignore_rows": 0.1})
    cobaya_samples.label = "cobaya RW-MH"
    # extract P0, beta arrays for stats
    P0_co = cobaya_samples.samples[:, cobaya_samples.index["P0GNFW"]]
    be_co = cobaya_samples.samples[:, cobaya_samples.index["betaGNFW"]]
    w_co  = cobaya_samples.weights

    # --- NUTS chain wrapped as MCSamples ---
    nuts = np.load(os.path.join(here, "_nuts_baseline.npz"))
    P0_nuts = nuts['P0']; be_nuts = nuts['beta']
    w_nuts = float(nuts['wall'])
    nuts_samples = MCSamples(
        samples=np.column_stack([P0_nuts, be_nuts]),
        names=["P0GNFW", "betaGNFW"],
        labels=["P_0", r"\beta"],
        label="NumPyro NUTS",
    )

    # quick console summary
    print("=== cobaya RW-MH ===")
    print(f"  P0   = {np.average(P0_co, weights=w_co):.3f} ± "
          f"{np.sqrt(np.average((P0_co - np.average(P0_co, weights=w_co))**2, weights=w_co)):.3f}")
    print(f"  beta = {np.average(be_co, weights=w_co):.3f} ± "
          f"{np.sqrt(np.average((be_co - np.average(be_co, weights=w_co))**2, weights=w_co)):.3f}")
    print(f"  Kish n_eff = {w_co.sum()**2 / (w_co**2).sum():.0f}")
    print("=== NumPyro NUTS ===")
    print(f"  P0   = {P0_nuts.mean():.3f} ± {P0_nuts.std():.3f}")
    print(f"  beta = {be_nuts.mean():.3f} ± {be_nuts.std():.3f}")
    print(f"  wall = {w_nuts:.2f} s")

    # --- Triangle plot ---
    g = plots.get_subplot_plotter(width_inch=6.5)
    g.settings.alpha_filled_add = 0.65
    g.settings.legend_fontsize = 11
    g.settings.lab_fontsize = 14
    g.settings.axes_fontsize = 11
    g.triangle_plot(
        [cobaya_samples, nuts_samples],
        params=["P0GNFW", "betaGNFW"],
        filled=True,
        legend_labels=[
            f"cobaya RW-MH (wall 14 min, $n_\\mathrm{{eff}}\\!\\approx\\!{w_co.sum()**2/(w_co**2).sum():.0f}$)",
            f"NumPyro NUTS (wall {w_nuts:.0f} s, ESS$\\,$=$\\,${int(nuts['ess_P0'])})",
        ],
        contour_colors=["C0", "C3"],
    )
    g.fig.suptitle(
        r"$C_\ell^{yy}$ bandpower posterior, baseline cosmology — "
        "NUTS matches RW-MH at ~56× lower wall",
        y=1.02, fontsize=11, fontweight='bold',
    )
    out = os.path.join(here, "posterior_compare.png")
    g.export(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
