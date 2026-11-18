"""szsim's masking step on one of our tiles, as the reference for
tests/test_sim_mask.py::test_masked_cov_matches_szsim.

Run with the sz-poisson-blending environment, which imports szsim:

    $SZPB_VENV/bin/python -B benchmarks/mapsim/refgen_szsim_mask.py $SZPB_ROOT \\
        $MAPSIM_OUT/mask/tile0_goal_first_pass.npz $MAPSIM_REF/szsim_mask.npz

Input: our six channel maps (rfft units) and first-pass peaks, written by
``benchmarks/mapsim/mask_check.py export``. szsim then builds its cluster mask
(disc_mask: radius MASK_RADIUS θ_j around the peaks with q > Q_TH_NOISE), the
unmasked binned covariance (binned_cov) and the whitened masked covariance
(masked_cov, method 'whitened', Cref = the unmasked one), all in float64.
"""
import os
import sys
import time

import numpy as np


def main():
    root, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    from szsim import finder as fd

    d = np.load(inp)
    X = d["X"].astype(np.complex128)
    j, iy, ix, q = d["j"], d["iy"], d["ix"], d["q"]
    G = fd.grid()
    sel = q > fd.Q_TH_NOISE
    t0 = time.time()
    mask = fd.disc_mask(iy[sel], ix[sel], fd.MASK_RADIUS * fd.THETAS15[j[sel]], fd.N, fd.PIX)
    t1 = time.time()
    C0 = fd.binned_cov(X, G)
    C1, f = fd.masked_cov(X, mask, G, Cref=C0, method="whitened")
    t2 = time.time()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=d["X"], j=j, iy=iy, ix=ix, q=q, thetas=fd.THETAS15, mask=mask.astype(np.float32),
             f=f, C0=C0, C1=C1, q_th_noise=fd.Q_TH_NOISE, mask_radius=fd.MASK_RADIUS)
    print(f"wrote {out}: {sel.sum()} discs, masked fraction {1 - f:.4f}; disc_mask {t1 - t0:.1f} s, "
          f"covariances {t2 - t1:.1f} s")


if __name__ == "__main__":
    main()
