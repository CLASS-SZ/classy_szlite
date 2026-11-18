"""The masking iteration on fiducial tiles, and the input for the szsim check.

    python benchmarks/mapsim/mask_check.py export
    python benchmarks/mapsim/mask_check.py tiles --tiles 16 --modes goal baseline

``export``: one fiducial SO-goal tile (tile key 0): its six channel maps and
first-pass peaks, for benchmarks/mapsim/refgen_szsim_mask.py, which runs
szsim's disc mask and whitened masked covariance on them (the reference of
tests/test_sim_mask.py::test_masked_cov_matches_szsim).

``tiles``: per tile, the finder's first pass and two masking iterations from
the same maps. Records σ_j per pass against the analytic σ_j (CMB + noise, no
tSZ), the masked fraction, the number of discs, the peaks, and the confirmed
counts per pass (data vector: 5 < q < 40, 0.01 < z < 1.5). The second
iteration checks convergence: SZiFi and production stop after one. Also times
the finder with and without the masking iteration on one tile (this backend).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike  # noqa: E402
from counts_modes import analytic_binned_cov, sigma_from_cov  # noqa: E402

from classy_szlite.sim import instrument as ins, mmf, simulator as simu, summaries as sm  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mask")


def log(*a):
    print(*a, flush=True)


def cmd_export(args):
    theta = simu.THETA_FID
    cells, tb, cs, cfg, st, info = spike.setup(theta)
    key = spike.tile_keys(1)[0]

    @jax.jit
    def first_pass(key):
        halos, X, _ = simu.tile_maps(key, cs, theta, st, tb, cells, cfg)
        q, sig, C0 = mmf.q_cube(X, st.fs)
        pk = mmf.find_peaks(q, cfg.peak_cap)
        return X, pk

    X, pk = jax.device_get(first_pass(key))
    n = int(pk.count)
    assert not bool(pk.overflow)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "tile0_goal_first_pass.npz")
    np.savez(path, X=np.asarray(X), j=np.asarray(pk.j)[:n], iy=np.asarray(pk.iy)[:n],
             ix=np.asarray(pk.ix)[:n], q=np.asarray(pk.q)[:n])
    log(f"saved {path}: {n} peaks, scales {np.bincount(np.asarray(pk.j)[:n], minlength=15).tolist()}")


def tile_passes(key, cs, theta, st, tb, cells, cfg, n_iter=2):
    """First pass and n_iter masking iterations on one tile's maps."""
    halos, X, _ = simu.tile_maps(key, cs, theta, st, tb, cells, cfg)
    q, sig, C0 = mmf.q_cube(X, st.fs)
    pk = mmf.find_peaks(q, cfg.peak_cap)
    out = []

    def summary(pk, sig, f, nm):
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        return dict(sigma=sig, f_masked=1.0 - f, n_masked=nm, n_peaks=pk.count,
                    n_kept=jnp.sum(keep), n_conf=jnp.sum(keep & cf.ok),
                    counts=sm.histogram(pk.q, cf.z, keep & cf.ok), overflow=pk.overflow | cf.overflow)

    out.append(summary(pk, sig, jnp.float64(1.0), jnp.int32(0)))
    for _ in range(n_iter):
        mask, nm = mmf.cluster_mask(pk, st.fs, cfg.q_mask)
        C, f = mmf.masked_cov(X, mask, C0, st.fs)
        q, sig, _ = mmf.q_cube(X, st.fs, C=C)
        pk = mmf.find_peaks(q, cfg.peak_cap)
        out.append(summary(pk, sig, f, nm))
    return {f"p{i}_{k}": v for i, d in enumerate(out) for k, v in d.items()}


def time_finder(X, st, cfg, reps=3):
    res = {}
    for n_iter in (0, 1):
        c = cfg._replace(n_iter=n_iter)
        f = jax.jit(lambda X: simu.find_clusters(X, st.fs, c)[0].count)
        jax.block_until_ready(f(X))
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(f(X))
        res[n_iter] = (time.perf_counter() - t0) / reps
    return res


def cmd_tiles(args):
    theta = simu.THETA_FID
    t0 = time.time()
    cells, tb, cs, cfg, st_goal, info = spike.setup(theta)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    statics = {"goal": st_goal, "baseline": simu.tile_static(cmb, tb, mode="baseline")}
    log(f"setup {time.time() - t0:.1f} s; backend {jax.default_backend()}")
    f = jax.jit(jax.vmap(lambda key, st: tile_passes(key, cs, theta, st, tb, cells, cfg),
                         in_axes=(0, None)))
    keys = spike.tile_keys(args.tiles)
    res = dict(tiles=args.tiles, modes={})
    for mode in args.modes:
        st = statics[mode]
        t1 = time.time()
        outs = [jax.device_get(f(keys[a:a + args.batch], st)) for a in range(0, args.tiles, args.batch)]
        o = {k: np.concatenate([np.asarray(r[k]) for r in outs]) for k in outs[0]}
        sig_an = sigma_from_cov(analytic_binned_cov(cmb, mode), st.fs)
        T = args.tiles
        r = dict(seconds=time.time() - t1, sigma_analytic=sig_an.tolist(), passes=[])
        log(f"\n=== {mode}: {T} tiles in {r['seconds']:.0f} s")
        for p in range(3):
            g = lambda k: o[f"p{p}_{k}"]
            ratio = g("sigma") / sig_an[None, :]                      # (T, 15)
            tot = g("counts").sum((1, 2)).astype(float)
            d = dict(sigma_ratio_mean=ratio.mean(0).tolist(),
                     sigma_ratio_sem=(ratio.std(0, ddof=1) / np.sqrt(T)).tolist(),
                     f_masked=g("f_masked").tolist(), n_masked=g("n_masked").tolist(),
                     n_peaks=g("n_peaks").tolist(), n_conf=g("n_conf").tolist(),
                     counts_mean=g("counts").mean(0).tolist(), total=tot.tolist(),
                     any_overflow=bool(g("overflow").any()))
            r["passes"].append(d)
            name = "first pass (no mask)" if p == 0 else f"after masking iteration {p}"
            log(f"--- {name}: σ/σ_analytic per scale (mean over tiles; ±SEM at 0.5', 4.5', 15'):")
            log("    " + " ".join(f"{v:.3f}" for v in d["sigma_ratio_mean"]))
            se = d["sigma_ratio_sem"]
            log(f"    SEM {se[0]:.4f} {se[7]:.4f} {se[14]:.4f}")
            if p:
                log(f"    masked fraction {np.mean(d['f_masked']):.4f} (range {np.min(d['f_masked']):.4f}-"
                    f"{np.max(d['f_masked']):.4f}); discs {np.mean(d['n_masked']):.0f}")
            log(f"    peaks {np.mean(d['n_peaks']):.0f}, confirmed {np.mean(d['n_conf']):.0f}; data vector "
                f"{tot.mean():.1f} ± {tot.std(ddof=1) / np.sqrt(T):.1f} per tile; overflow {d['any_overflow']}")
        p0, p1, p2 = (np.asarray(o[f"p{p}_sigma"]) for p in range(3))
        t_0, t_1, t_2 = (o[f"p{p}_counts"].sum((1, 2)).astype(float) for p in range(3))
        r["iter2_vs_iter1_sigma_maxdev"] = float(np.max(np.abs(p2 / p1 - 1)))
        r["iter1_vs_first_sigma_mean"] = (p1 / p0).mean(0).tolist()
        r["counts_change"] = dict(iter1_over_first=float(t_1.sum() / t_0.sum()),
                                  iter2_over_iter1=float(t_2.sum() / t_1.sum()),
                                  iter2_minus_iter1_per_tile=(t_2 - t_1).tolist())
        log(f"iteration 2 vs 1: max |Δσ/σ| over tiles and scales {r['iter2_vs_iter1_sigma_maxdev']:.4f}; "
            f"data vector ×{r['counts_change']['iter2_over_iter1']:.4f} "
            f"(per-tile differences {sorted(set((t_2 - t_1).astype(int).tolist()))})")
        log(f"iteration 1 vs first pass: data vector ×{r['counts_change']['iter1_over_first']:.4f}")
        res["modes"][mode] = r

    # timing on one tile's maps (this backend)
    X = jax.jit(lambda k: simu.tile_maps(k, cs, theta, st_goal, tb, cells, cfg)[1])(keys[0])
    tm = time_finder(X, st_goal, cfg)
    res["finder_seconds"] = {str(k): v for k, v in tm.items()}
    log(f"\nfinder on one tile ({jax.default_backend()}): one pass {tm[0]:.2f} s, with one masking "
        f"iteration {tm[1]:.2f} s (×{tm[1] / tm[0]:.2f})")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "mask_tiles.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    t = sub.add_parser("tiles")
    t.add_argument("--tiles", type=int, default=16)
    t.add_argument("--batch", type=int, default=1)
    t.add_argument("--modes", nargs="+", default=["goal", "baseline"])
    args = ap.parse_args()
    {"export": cmd_export, "tiles": cmd_tiles}[args.cmd](args)


if __name__ == "__main__":
    main()
