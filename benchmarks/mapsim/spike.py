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
from classy_szlite.sim import instrument as ins, summaries as sm, noise as nz  # noqa: E402
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


def theta_of(name):
    """The fiducial point, or the adopted prior box's heaviest corner (prior_corners.py)."""
    if name == "fid":
        return simu.THETA_FID
    pc = json.load(open(os.path.join(os.environ.get("MAPSIM_OUT", "."), "priors", "prior_corners.json")))
    return simu.THETA_FID._replace(**pc["boxes"]["wide"]["heaviest"]["theta"])


def noise_for(args, B):
    """None for the 2019 forecast noise; for --noise mss2 a NoiseInput with one spectrum per tile
    (tile i takes spectrum i mod n, or all tiles the deepest one: the lowest 150 GHz white level)
    and depth 1, or depths cycling 1.0, 1.3, 0.8 with --mixed-depth."""
    if args.noise == "forecast":
        return None
    nt, info = nz.load_mss2(args.spectra, args.noise_set)
    if args.deepest:
        spec = np.full(B, int(np.argmin(np.asarray(info["white"])[:, info["bands"].index("f150")])))
    else:
        spec = np.arange(B) % nt.F.shape[0]
    depth = np.array([1.0, 1.3, 0.8])[np.arange(B) % 3] if args.mixed_depth else np.ones(B)
    log(f"noise: MSS2 '{args.noise_set}', spectra {spec[:8].tolist()}{'...' if B > 8 else ''}, depths {depth[:6].tolist()}")
    return nz.noise_input(nt, spec, depth)


def per_tile(ni, B):
    """Per-tile spectrum indices and depths (dummies for the forecast noise)."""
    if ni is None:
        return jnp.zeros(B, jnp.int32), jnp.ones(B, jnp.float32)
    return ni.spec, ni.depth


def noise_arg(ni, s, d):
    return None if ni is None else nz.NoiseInput(ni.tables, s, d)


def cmd_invariance(args):
    """Same tile keys at several batch sizes: integer outputs must be identical
    (run with XLA_FLAGS=--xla_gpu_deterministic_ops=true on GPUs)."""
    theta = theta_of(args.theta)
    cells, tb, cs, cfg, st, info = setup(theta)
    Bmax = max(args.batch)
    ni = noise_for(args, Bmax)
    specs, depths = per_tile(ni, Bmax)

    def tile(key, s, d):
        return simu.simulate_tile(key, cs, theta, st, tb, cells, cfg, noise_arg(ni, s, d))
    f = jax.jit(jax.vmap(tile))
    outs = {}
    for B in args.batch:
        o = {}
        for a in range(0, Bmax, B):                  # cover tiles 0..Bmax-1 in chunks of B
            keys = tile_keys(Bmax)[a:a + B]
            r = jax.device_get(f(keys, specs[a:a + B], depths[a:a + B]))
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
    theta = theta_of(args.theta)
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

    ni_all = noise_for(args, max(args.batch))

    def s_maps(key, yk, s, d):
        return simu.channel_maps(key, yk, st, noise_arg(ni_all, s, d))

    def s_filter(X):
        q, sig, _ = mmf.q_cube(X, st.fs)
        return q, sig

    def s_peaks(q, h):
        pk = mmf.find_peaks(q, cfg.peak_cap)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, h, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        return sm.histogram(pk.q, cf.z, keep & cf.ok), pk.count

    def s_finder(X):
        # first pass plus the masking iteration (cfg.n_iter), up to the final peaks
        return simu.find_clusters(X, st.fs, cfg)[0].count

    def tile(key, s, d):
        return simu.simulate_tile(key, cs, theta, st, tb, cells, cfg, noise_arg(ni_all, s, d))

    res = dict(info=info, backend=jax.default_backend(), device=str(jax.devices()[0]),
               xla_flags=os.environ.get("XLA_FLAGS", ""), noise=args.noise, theta=args.theta, runs=[])
    for B in args.batch:
        keys = tile_keys(B)
        specs, depths = (a[:B] for a in per_tile(ni_all, max(args.batch)))
        stages = dict(
            halos=jax.jit(jax.vmap(s_halos)),
            paint=jax.jit(jax.vmap(s_paint)),
            maps=jax.jit(jax.vmap(s_maps)),
            filter=jax.jit(jax.vmap(s_filter)),
            peaks=jax.jit(jax.vmap(s_peaks)),
            finder=jax.jit(jax.vmap(s_finder)),
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
        X = timed("maps", stages["maps"], keys, yk, specs, depths)
        q, sig = timed("filter", stages["filter"], X)
        counts, npk = timed("peaks", stages["peaks"], q, h)
        timed("finder", stages["finder"], X)
        out = timed("full", stages["full"], keys, specs, depths)
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
                jax.block_until_ready(stages["full"](keys, specs, depths))
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


FIELDS_INT = ("counts", "n_halos", "n_peaks", "n_kept", "n_conf", "overflow")


def cmd_tiles(args):
    """Tiles 0..B-1 one at a time (batch 1), outputs saved as tiles_<tag>.json: for comparisons
    between backends (run once with JAX_PLATFORMS=cpu), and the peak buffer at a given setting."""
    theta = theta_of(args.theta)
    cells, tb, cs, cfg, st, info = setup(theta)
    B = args.batch[0]
    ni = noise_for(args, B)
    specs, depths = per_tile(ni, B)
    f = jax.jit(lambda key, s, d: simu.simulate_tile(key, cs, theta, st, tb, cells, cfg, noise_arg(ni, s, d)))
    keys = tile_keys(B)
    out = {k: [] for k in FIELDS_INT + ("sigma", "sigma0")}
    for i in range(B):
        r = jax.device_get(f(keys[i], specs[i], depths[i]))
        for k in out:
            out[k].append(np.asarray(r[k]).tolist())
        log(f"tile {i}: n_halos {int(r['n_halos'])}, n_peaks {int(r['n_peaks'])} (buffer {cfg.peak_cap}), "
            f"n_kept {int(r['n_kept'])}, n_conf {int(r['n_conf'])}, data vector {int(np.asarray(r['counts']).sum())}, "
            f"overflow {np.asarray(r['overflow']).any()}")
    out.update(backend=jax.default_backend(), noise=args.noise, theta=args.theta, peak_cap=cfg.peak_cap,
               spec=np.asarray(specs).tolist(), depth=np.asarray(depths).tolist())
    with open(os.path.join(OUT, f"tiles_{args.tag}.json"), "w") as fh:
        json.dump(out, fh)


def cmd_compare(args):
    """Integer outputs and sigma of two cmd_tiles runs (e.g. GPU against CPU)."""
    a, b = (json.load(open(os.path.join(OUT, f"tiles_{t}.json"))) for t in args.tags)
    for k in FIELDS_INT:
        same = [np.array_equal(x, y) for x, y in zip(a[k], b[k])]
        log(f"{args.tags[0]} vs {args.tags[1]}: {k:9s} identical on {sum(same)} of {len(same)} tiles")
    for k in ("sigma", "sigma0"):
        d = np.max(np.abs(np.asarray(a[k]) / np.asarray(b[k]) - 1))
        log(f"{args.tags[0]} vs {args.tags[1]}: {k} max relative difference {d:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "invariance", "tiles", "compare"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tags", nargs=2, default=None, help="compare: two tiles_<tag>.json")
    ap.add_argument("--trace", action="store_true", help="jax.profiler trace of the last batch size")
    ap.add_argument("--theta", choices=["fid", "corner"], default="fid",
                    help="the fiducial point or the adopted prior box's heaviest corner")
    ap.add_argument("--noise", choices=["forecast", "mss2"], default="forecast")
    ap.add_argument("--spectra", default=os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2", "spectra", "spectra.npz"))
    ap.add_argument("--noise-set", default="mean", help="mean (four splits) or split0..split3")
    ap.add_argument("--mixed-depth", action="store_true", help="per-tile depths cycling 1.0, 1.3, 0.8")
    ap.add_argument("--deepest", action="store_true", help="every tile takes the deepest MSS2 spectrum")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"check": cmd_check, "run": cmd_run, "invariance": cmd_invariance, "tiles": cmd_tiles,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
