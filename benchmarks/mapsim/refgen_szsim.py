"""Reference values from szsim (sz-poisson-blending) for the mapsim M1 tests.

Run with the sz-poisson-blending environment, which imports szsim:

    $SZPB_VENV/bin/python -B benchmarks/mapsim/refgen_szsim.py $SZPB_ROOT $MAPSIM_REF

Writes $MAPSIM_REF/szsim_instrument.npz: SO LAT constants, the tSZ SED, beams,
N_ij(ℓ), the lensed CMB C_ℓ, the finder's Fourier grid and bins, and its
binned analytic covariance (goal and baseline).
"""
import os
import sys

import numpy as np


def main():
    root, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    from szsim import instrument as ins, finder as fd

    os.makedirs(out, exist_ok=True)
    ell = np.concatenate([np.arange(0, 200, 1.0), np.geomspace(200, 30000, 800)])
    G = fd.grid()
    ref = dict(
        ell=ell,
        freqs=ins.SO_LAT["freqs"], fwhm=ins.SO_LAT["fwhm"],
        white_goal=ins.SO_LAT["white"]["goal"], white_baseline=ins.SO_LAT["white"]["baseline"],
        N_red=ins.SO_LAT["N_red"], t_eff=ins.so_t_eff(),
        f_uK=ins.tsz_f_uK(ins.SO_LAT["freqs"]), g_x=ins.tsz_g(ins.SO_LAT["freqs"]),
        beams=ins.beam_l(ell, ins.SO_LAT["fwhm"]),
        noise_goal=ins.so_lat_noise(ell, "goal"), noise_baseline=ins.so_lat_noise(ell, "baseline"),
        noise_goal_noatm=ins.so_lat_noise(ell, "goal", atm=False),
        cmb_cl=ins.cmb_cl_TT(ell),
        grid_L=G.L, grid_sel=G.sel, grid_ib=G.ib, grid_W=G.W, grid_n2=G.n2,
        bins_used=G.bins_used, m_idx=G.m_idx, m_bin=G.m_bin, m_L=G.m_L,
        abc_goal=fd.analytic_bin_cov("goal", G), abc_baseline=fd.analytic_bin_cov("baseline", G),
        dk=fd.DK, dbin=fd.DBIN, n2min=fd.N2MIN, n2max=fd.N2MAX,
    )
    np.savez(os.path.join(out, "szsim_instrument.npz"), **ref)
    print("wrote", os.path.join(out, "szsim_instrument.npz"), len(ref), "arrays")


if __name__ == "__main__":
    main()
