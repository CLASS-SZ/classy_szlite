"""The paper's SO-like noise table against ours, and the counts it implies.

    python benchmarks/mapsim/paper_sigma.py --table PATH/so_sim_sz_mf_noise.npy

The table is (θ500 [rad], σ_y0) from the SZiFi runs on the SO-like simulated
maps behind the paper; cosmocnc_jax fits ln σ with a cubic in ln θ500
(arcmin), and so do we here. We compare it with our analytic σ (CMB + noise)
and with the σ our single-pass finder measures on simulated tiles
($MAPSIM_OUT/counts/counts_modes.json), and compute the catalogue-level
counts each σ implies at the fiducial point with our λ cells (M ≥ 5e13,
q_obs ~ N(q_true, 1)).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from counts_modes import catalogue_hist, TILES_40PC  # noqa: E402

from classy_szlite.sim import halos as hl, mmf, simulator as simu  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    args = ap.parse_args()
    th_rad, sig = np.load(args.table)
    th = th_rad * 180.0 * 60.0 / np.pi
    poly = np.polyfit(np.log(th), np.log(sig), deg=3)
    fit = np.exp(np.polyval(poly, np.log(th)))
    print(f"paper table: {th.size} points, theta500 {th.min():.3f}-{th.max():.2f}', "
          f"cubic fit max |rel err| {np.max(np.abs(fit / sig - 1)):.3f}")
    sig_paper = np.exp(np.polyval(poly, np.log(mmf.THETAS15)))
    with open(os.path.join(OUT, "counts_modes.json")) as fh:
        cm = json.load(fh)
    print("theta_j [arcmin]      " + " ".join(f"{t:6.2f}" for t in mmf.THETAS15))
    print("sigma_paper [1e-6]    " + " ".join(f"{s * 1e6:6.2f}" for s in sig_paper))
    for mode in ("goal", "baseline"):
        for src in ("analytic", "map"):
            s = np.asarray(cm["modes"][mode][f"sigma_{src}"])
            print(f"paper / {mode:8s} {src:8s} " + " ".join(f"{a / b:6.2f}" for a, b in zip(sig_paper, s)))
    cells = hl.default_cells()
    theta = simu.THETA_FID
    cs = jax.jit(lambda t: simu.cosmology_stage(t, cells))(theta)
    res = {}
    for k in (0.0, 2.0):
        H = catalogue_hist(cs, theta, cells, sig_paper, k_dof=k)
        Hall = catalogue_hist(cs, theta, cells, sig_paper, k_dof=k, M_min=0.0)
        res[f"k{k:g}"] = dict(binned=float(H.sum()), binned_noMcut=float(Hall.sum()))
        print(f"paper sigma, k={k:g}: {H.sum():.1f} per tile -> {H.sum() * TILES_40PC:,.0f} in 40% of the sky "
              f"(5 < q < 40, 0.01 < z < 1.5, M >= 5e13); without the mass cut {Hall.sum() * TILES_40PC:,.0f}")
    # all q > 5 and all z < 3: the paper's headline number
    Ht = catalogue_hist(cs, theta, cells, sig_paper, k_dof=0.0, q_edges=[5.0, 1e6], z_edges=[0.01, 3.0])
    print(f"paper sigma, k=0, all q > 5 and 0.01 < z < 3, M >= 5e13: {Ht.sum() * TILES_40PC:,.0f} in 40% of the sky")
    res["all_q5_z3"] = float(Ht.sum() * TILES_40PC)
    for mode in ("goal", "baseline"):
        s = np.asarray(cm["modes"][mode]["sigma_analytic"])
        Ha = catalogue_hist(cs, theta, cells, s, k_dof=0.0, q_edges=[5.0, 1e6], z_edges=[0.01, 3.0])
        print(f"our analytic {mode} sigma, same cuts: {Ha.sum() * TILES_40PC:,.0f}")
        res[f"all_q5_z3_analytic_{mode}"] = float(Ha.sum() * TILES_40PC)
    res["sigma_paper_at_thetas15"] = sig_paper.tolist()
    with open(os.path.join(OUT, "paper_sigma.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
