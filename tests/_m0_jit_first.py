"""M0 fresh-process check: jit(vmap(pipeline)) FIRST, then eager; compare.

Run as a script (``python tests/_m0_jit_first.py [--save out.npz]``) in a new
process, so that the emulators are first loaded inside a jit trace. That
catches two hazards: a cache that stores tracers (the eager calls after the
jitted ones would then fail), and squeeze() on batch or redshift axes of size 1.

Pipeline (the simulator's cosmology stage): θ = (σ8, ω_cdm, h, ω_b, n_s)
→ ln(10¹⁰ A_s) from the DER-emulator σ8 → CosmoGrids → HaloGrids (M500c,
cosmocnc conventions). Prints one JSON line: max elementwise relative
differences per output field, jit vs eager at batch sizes 1 and 7, and batch
1 vs batch 7 for the first parameter point, on the domain the simulator uses
(see ``domain_mask``) and, for the record, on the full grids.
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.cosmology import build, ln10As_from_sigma8  # noqa: E402
from classy_szlite.hmf import build_halo_grids  # noqa: E402

Z = np.geomspace(0.01, 3.0, 40)
LO = np.array([0.70, 0.10, 0.62, 0.0210, 0.94])
HI = np.array([0.90, 0.14, 0.74, 0.0230, 0.99])


def pipeline(theta):
    s8, och2, h, obh2, ns = (theta[i] for i in range(5))
    p = {"omega_cdm": och2, "H0": 100.0 * h, "omega_b": obh2, "n_s": ns}
    p["ln10^{10}A_s"] = ln10As_from_sigma8(p, s8)
    cg = build(p, z_grid=Z)
    hg = build_halo_grids(cg, p, delta_crit=500.0, m_min=1e13, m_max=5e15,
                          n_m=120, tinker_interp="linear", nu_in_rho_m=False)
    out = {"ln10As": p["ln10^{10}A_s"]}
    out.update({f"cg.{k}": v for k, v in cg._asdict().items()})
    out.update({f"hg.{k}": v for k, v in hg._asdict().items()})
    return out


# Domain the simulator uses. σ(R) and dσ²/dR are compared on R = 0.5-60 Mpc
# (M ≈ 4e10-7e16 M_sun); at the FFTLog grid edges (R ~ 1e4 Mpc) σ² is ~1e-12 of
# its peak and the transform's round-off is amplified. dn/dlnM is compared
# where it is ≥ 1e-12 of its maximum; the far exponential tail amplifies
# round-off by ~2c/σ², and contributes nothing to counts.
R_USED = (0.5, 60.0)
HMF_FLOOR = 1e-12


def max_rel(a, b, mask=None):
    """Max elementwise relative difference (absolute where b == 0), over mask."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    if mask is None:
        mask = np.ones(b.shape, dtype=bool)
    d = np.abs(a - b)[mask]
    b = b[mask]
    nz = b != 0
    rel = np.max(d[nz] / np.abs(b[nz])) if nz.any() else 0.0
    return float(max(rel, np.max(d[~nz]) if (~nz).any() else 0.0))


def domain_mask(key, ref):
    """Mask of the entries of field ``key`` that are tested, given the
    reference outputs ``ref`` of one parameter point."""
    v = np.asarray(ref[key])
    if key in ("cg.sigma", "cg.dsigma2dR"):
        R = np.asarray(ref["cg.R"])
        return np.broadcast_to((R >= R_USED[0]) & (R <= R_USED[1]), v.shape)
    if key == "hg.dndlnm":
        return v >= HMF_FLOOR * v.max()
    return None


def compare(out_a, out_b):
    """Per-field max relative differences: on the used domain, and on the full grid."""
    used = {k: max_rel(out_a[k], out_b[k], domain_mask(k, out_b)) for k in out_b}
    full = {k: max_rel(out_a[k], out_b[k]) for k in out_b}
    return used, full


def thetas():
    rng = np.random.default_rng(20261118)
    return LO + (HI - LO) * rng.random((7, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="save the batch-7 jitted outputs (npz)")
    args = ap.parse_args()

    th = thetas()
    f = jax.jit(jax.vmap(pipeline))
    jitted = {}
    for B in (1, 7):                          # jitted calls come first
        jitted[B] = {k: np.asarray(v) for k, v in f(jnp.asarray(th[:B])).items()}
    eager = [{k: np.asarray(v) for k, v in pipeline(jnp.asarray(th[i])).items()}
             for i in range(7)]

    def row(out, i):
        return {k: v[i] for k, v in out.items()}

    res = {"backend": jax.default_backend(), "jit_vs_eager": {}, "jit_vs_eager_full_grid": {}}
    for B in (1, 7):
        per = [compare(row(jitted[B], i), eager[i]) for i in range(B)]
        res["jit_vs_eager"][str(B)] = {k: max(u[k] for u, _ in per) for k in eager[0]}
        res["jit_vs_eager_full_grid"][str(B)] = {k: max(f[k] for _, f in per) for k in eager[0]}
    res["batch1_vs_batch7"], res["batch1_vs_batch7_full_grid"] = compare(
        row(jitted[1], 0), row(jitted[7], 0))
    if args.save:
        np.savez(args.save, thetas=th, **jitted[7])
    print(json.dumps(res))


if __name__ == "__main__":
    main()
