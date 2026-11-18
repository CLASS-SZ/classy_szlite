"""M0 throughput spike for the mapsim tile pipeline.

    python benchmarks/mapsim/spike.py check     # painter vs exact synthesis (any backend)
    python benchmarks/mapsim/spike.py run --batch 1 2 4 --reps 3

Outputs go to $MAPSIM_OUT/m0 (paint tables are cached there). The CMB is
szsim's CAMB lensed TT ($MAPSIM_REFDATA/data_real/cmb_lensed_TT_camb.npz).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import halos as hl, paint as pt, mmf, simulator as simu  # noqa: E402
from classy_szlite.sim import instrument as ins, summaries as sm  # noqa: E402
from classy_szlite.sim.flatsky import grid  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "m0")
REFDATA = os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending")


def log(*a):
    print(*a, flush=True)


DLN = float(os.environ.get("MAPSIM_PAINT_DLN", "0.004"))


def tables():
    path = os.path.join(OUT, f"paint_tables_dln{DLN:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return pt.PaintTables(lnth0=float(d["lnth0"]), dln=float(d["dln"]), rP=d["rP"],
                              tab=d["tab"], th_split=float(d["th_split"]))
    t0 = time.time()
    tb = pt.build_tables(dln=DLN)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **tb._asdict())
    os.replace(tmp, path)
    log(f"paint tables built in {time.time() - t0:.1f} s: {tb.tab.shape}")
    return tb


def expected_occupation(lam, cg, theta, cells, tb):
    """Expected halos per footprint class and on the direct path (fiducial)."""
    zc = 0.5 * (cells.z_edges[1:] + cells.z_edges[:-1])
    mc = 0.5 * (cells.lnM_edges[1:] + cells.lnM_edges[:-1])
    from classy_szlite.sz_scaling import theta500_arcmin
    H0c = theta.h * 100.0 / 299792.458
    Ez = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Hz)) / H0c
    DA = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Da))
    th = np.asarray(theta500_arcmin(np.exp(mc)[None, :], Ez[:, None], DA[:, None],
                                    theta.h, theta.one_minus_b))
    cls = np.asarray(pt.footprint_class(jnp.asarray(th.ravel()), tb,
                                        jnp.asarray(tb.rP))).reshape(th.shape)
    lam = np.asarray(lam)
    occ = np.array([lam[cls == c].sum() for c in range(len(pt.SIDES) + 1)])
    return occ


def setup(theta=simu.THETA_FID, margin=1.5):
    cells = hl.default_cells()
    tb = tables()
    cs = jax.jit(lambda th: simu.cosmology_stage(th, cells))(theta)
    lam, cg = cs.lam, cs.cg
    occ = expected_occupation(lam, cg, theta, cells, tb)
    lam_tot = float(np.sum(lam))
    halo_cap = int(np.ceil((lam_tot + 6 * np.sqrt(lam_tot) + 1024) / 4096) * 4096)
    cap_big = int(np.ceil((margin * occ[-1] + 64) / 32) * 32)
    pcfg = pt.make_config(tb, occ[:-1], cap_big, margin=margin)
    cfg = simu.TileConfig(halo_cap=halo_cap, paint=pcfg)
    cmb = ins.cmb_cl_from_table(os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb, tb)
    info = dict(lambda_total=lam_tot, halo_cap=halo_cap, occupation=occ.round(1).tolist(),
                caps=list(pcfg.caps), blocks=list(pcfg.blocks), cap_big=cap_big,
                contributions=float(sum(o * s * s for o, s in zip(occ[:-1], pt.SIDES))))
    return cells, tb, cs, cfg, st, info


def cmd_check(args):
    """Painter (scheme A) against exact synthesis, on one tile's halos."""
    theta = simu.THETA_FID
    cells, tb, cs, cfg, st, info = setup(theta)
    log(json.dumps(info))
    h = jax.jit(lambda k: hl.sample_halos(k, cs.lam, cells, cfg.halo_cap, cs.cg, theta.h, theta.A_SZ,
                                          theta.alpha_SZ, theta.sigma_lnq,
                                          theta.one_minus_b, d2N=cs.d2N))(jax.random.key(1))
    log(f"halos drawn: {int(h.count)} (lambda {info['lambda_total']:.0f}), overflow {bool(h.overflow)}")
    # a subset: every halo class, ~400 halos, painted alone by both methods
    rng = np.random.default_rng(0)
    th = np.asarray(h.theta500)
    valid = np.asarray(h.valid)
    pick = []
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 6), (6, 1e9)]:
        ids = np.flatnonzero(valid & (th >= lo) & (th < hi))
        pick += list(rng.choice(ids, size=min(60, ids.size), replace=False))
    pick = np.array(sorted(pick))
    sub = jax.tree.map(lambda x: x, h)
    keep = np.zeros(valid.shape, bool)
    keep[pick] = True
    sub = sub._replace(valid=jnp.asarray(keep), y0=jnp.where(jnp.asarray(keep), h.y0, 0.0))
    tabs = tb._replace(tab=st.paint_tab, rP=st.paint_rP)
    yk_paint, ovf = jax.jit(pt.paint_y, static_argnums=2)(sub, tabs, cfg.paint, st.T_tab)
    ref = {k: np.asarray(getattr(h, k))[pick] for k in ("iy", "ix", "oy", "ox", "y0", "theta500")}
    yk_ref = jax.jit(pt.synth_direct, static_argnums=2)(ref, st.T_tab, jnp.float64)
    G = grid()
    a, b = np.asarray(yk_paint)[G.sel], np.asarray(yk_ref)[G.sel]
    err = np.max(np.abs(a - b)) / np.max(np.abs(b))
    log(f"painter vs exact synthesis ({pick.size} halos, theta500 {th[pick].min():.2f}-"
        f"{th[pick].max():.1f}'): max |dy(k)| / max |y(k)| = {err:.2e}; rms ratio "
        f"{np.sqrt(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2)):.2e}")
    # per-class: paint each class alone
    for c, (lo, hi) in enumerate([(0, 0.5), (0.5, 1), (1, 2), (2, 6), (6, 1e9)]):
        sel = (th[pick] >= lo) & (th[pick] < hi)
        k2 = np.zeros(valid.shape, bool)
        k2[pick[sel]] = True
        s2 = h._replace(valid=jnp.asarray(k2), y0=jnp.where(jnp.asarray(k2), h.y0, 0.0))
        y2, _ = jax.jit(pt.paint_y, static_argnums=2)(s2, tabs, cfg.paint, st.T_tab)
        r2 = {k: v[sel] for k, v in ref.items()}
        y2r = jax.jit(pt.synth_direct, static_argnums=2)(r2, st.T_tab, jnp.float64)
        a, b = np.asarray(y2)[G.sel], np.asarray(y2r)[G.sel]
        log(f"   theta500 in [{lo}, {hi}): {sel.sum():3d} halos, max |dy|/max|y| = "
            f"{np.max(np.abs(a - b)) / np.max(np.abs(b)):.2e}")


def tile_keys(B, sim_index=0, base=20261118):
    """Per-tile keys: fold_in(fold_in(base, sim), tile). Tile i gets the same
    key whatever the batch size."""
    k = jax.random.fold_in(jax.random.key(base), sim_index)
    return jax.vmap(lambda i: jax.random.fold_in(k, i))(jnp.arange(B, dtype=jnp.uint32))


def cmd_invariance(args):
    """Same tile keys at several batch sizes: integer outputs must be identical
    (run with XLA_FLAGS=--xla_gpu_deterministic_ops=true on GPUs)."""
    theta = simu.THETA_FID
    cells, tb, cs, cfg, st, info = setup(theta)

    def tile(key):
        return simu.simulate_tile(key, cs, theta, st, tb, cells, cfg)
    f = jax.jit(jax.vmap(tile))
    Bmax = max(args.batch)
    outs = {}
    for B in args.batch:
        o = {}
        for a in range(0, Bmax, B):                  # cover tiles 0..Bmax-1 in chunks of B
            keys = tile_keys(Bmax)[a:a + B]
            r = jax.device_get(f(keys))
            for k in ("counts", "n_halos", "n_peaks", "n_kept", "n_conf", "overflow", "sigma"):
                o.setdefault(k, []).append(np.asarray(r[k]))
        outs[B] = {k: np.concatenate(v) for k, v in o.items()}
    ref = outs[args.batch[0]]
    ok = True
    for B in args.batch[1:]:
        for k in ("counts", "n_halos", "n_peaks", "n_kept", "n_conf", "overflow"):
            same = np.array_equal(outs[B][k], ref[k])
            ok &= same
            log(f"batch {B} vs {args.batch[0]}: {k:9s} identical: {same}")
        ds = np.max(np.abs(outs[B]["sigma"] / ref["sigma"] - 1))
        log(f"batch {B} vs {args.batch[0]}: sigma max rel diff {ds:.2e}")
    log(f"total counts over {Bmax} tiles: {ref['counts'].sum(0).ravel().tolist()}")
    log("INVARIANT" if ok else "NOT INVARIANT")


def cmd_run(args):
    theta = simu.THETA_FID
    t0 = time.time()
    cells, tb, cs, cfg, st, info = setup(theta)
    log(f"setup {time.time() - t0:.1f} s; backend {jax.default_backend()}, devices {jax.devices()}")
    log(json.dumps(info))
    tabs = tb._replace(tab=st.paint_tab, rP=st.paint_rP)

    # stage functions (one tile), for per-stage timing
    def s_halos(key):
        return hl.sample_halos(key, cs.lam, cells, cfg.halo_cap, cs.cg, theta.h, theta.A_SZ,
                               theta.alpha_SZ, theta.sigma_lnq, theta.one_minus_b, d2N=cs.d2N)

    def s_paint(h):
        return pt.paint_y(h, tabs, cfg.paint, st.T_tab)

    def s_maps(key, yk):
        return simu.channel_maps(key, yk, st)

    def s_filter(X):
        q, sig, _ = mmf.q_cube(X, st.fs)
        return q, sig

    def s_peaks(q, h):
        pk = mmf.find_peaks(q, cfg.peak_cap)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, h, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        return sm.histogram(pk.q, cf.z, keep & cf.ok), pk.count

    def tile(key):
        return simu.simulate_tile(key, cs, theta, st, tb, cells, cfg)

    res = dict(info=info, backend=jax.default_backend(), device=str(jax.devices()[0]),
               xla_flags=os.environ.get("XLA_FLAGS", ""), runs=[])
    for B in args.batch:
        keys = tile_keys(B)
        stages = dict(
            halos=jax.jit(jax.vmap(s_halos)),
            paint=jax.jit(jax.vmap(s_paint)),
            maps=jax.jit(jax.vmap(s_maps)),
            filter=jax.jit(jax.vmap(s_filter)),
            peaks=jax.jit(jax.vmap(s_peaks)),
            full=jax.jit(jax.vmap(tile)),
        )
        timings = {}

        def timed(name, f, *a):
            tc = time.time()
            out = jax.block_until_ready(f(*a))
            t_first = time.time() - tc
            ts = []
            for _ in range(args.reps):
                tc = time.time()
                out = jax.block_until_ready(f(*a))
                ts.append(time.time() - tc)
            timings[name] = dict(first=t_first, best=min(ts), per_tile=min(ts) / B)
            log(f"B={B:3d} {name:7s} first {t_first:7.2f} s, best {min(ts):8.4f} s, "
                f"per tile {1e3 * min(ts) / B:9.2f} ms")
            return out

        h = timed("halos", stages["halos"], keys)
        yk, ovf = timed("paint", stages["paint"], h)
        X = timed("maps", stages["maps"], keys, yk)
        q, sig = timed("filter", stages["filter"], X)
        counts, npk = timed("peaks", stages["peaks"], q, h)
        out = timed("full", stages["full"], keys)
        res["runs"].append(dict(batch=B, timings=timings,
                                n_halos=np.asarray(out["n_halos"]).tolist(),
                                n_peaks=np.asarray(out["n_peaks"]).tolist(),
                                n_conf=np.asarray(out["n_conf"]).tolist(),
                                counts=np.asarray(out["counts"]).sum(0).tolist(),
                                overflow=np.asarray(out["overflow"]).any(0).tolist(),
                                sigma=np.asarray(out["sigma"])[0].tolist()))
        log(f"   n_halos {res['runs'][-1]['n_halos'][:4]}, n_peaks {res['runs'][-1]['n_peaks'][:4]}, "
            f"n_conf {res['runs'][-1]['n_conf'][:4]}, any overflow {res['runs'][-1]['overflow']}")
        if args.trace and B == args.batch[-1]:
            tdir = os.path.join(OUT, f"trace_{args.tag or jax.default_backend()}_B{B}")
            with jax.profiler.trace(tdir):
                jax.block_until_ready(stages["full"](keys))
            log(f"   profiler trace in {tdir}")
        try:
            ms = jax.devices()[0].memory_stats()
            if ms:
                log(f"   peak device memory {ms.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB")
                res["runs"][-1]["peak_bytes"] = ms.get("peak_bytes_in_use", 0)
        except Exception:
            pass
    tag = args.tag or jax.default_backend()
    with open(os.path.join(OUT, f"spike_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "invariance"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--trace", action="store_true", help="jax.profiler trace of the last batch size")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"check": cmd_check, "run": cmd_run, "invariance": cmd_invariance}[args.cmd](args)


if __name__ == "__main__":
    main()
