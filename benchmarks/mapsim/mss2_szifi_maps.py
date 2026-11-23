"""Maps and our finder's catalogues for the comparison with SZiFi on MSS2 tiles.

    python benchmarks/mapsim/mss2_szifi_maps.py [--tiles 2,16,26,43,45,50,86,97] [--sky 0]
        [--root DIR] [--no-clusters] [--no-write-maps] [--variants "ours=0,1,2,3,4,5;ours_ch4=2,3,4,5"]

Options:
  --no-clusters    the same maps without the cluster sky (MSS2 noise and the CMB only; the halos
                   are still drawn, for the same keys)
  --no-write-maps  the maps are rebuilt from the flags (which must match the run that wrote
                   --root: checked against ROOT/meta.json, and each rebuilt map is compared with
                   ROOT/maps, reported in info.json) and only the variants' folders are written
  --variants       our finder on each named set of channels (indices of 27, 39, 93, 145, 225,
                   280 GHz) into ROOT/NAME/ (one npz per map, and NAME/info.json: channels,
                   self-checks, map comparison)
  --mask-from DIR  also the noise each variant's masked estimate gives on these maps with the mask
                   of the same variant on the same map in DIR (DIR/NAME/tile_*.npz): s_same, f_same
                   (the estimator's response to that mask without the clusters under it)
  --force          write into folders that already hold outputs (refused otherwise)
The q rescaling to the periodic tiles' scale (rs) is the six-channel geometry calibration for
every variant (info.json: rs_channels).

For each tile and two depths, the data-vector test's map is rebuilt exactly as
``mss2_datavector_test.run_tile`` builds it: MSS2's noise cut-out plus the simulator's sky for
that tile and sky index (halos above 1e13 Msun to z = 3, tSZ painted in the six SO channels with
Gaussian beams, Gaussian lensed CMB). Depths:
  split0   MSS2's split 0 (one split: about 1.35x the simulator's baseline in the finder's noise,
           somewhat deeper than year 1)
  mean     the plain mean of the four splits (about the simulator's goal depth)

Written for SZiFi (``mss2_szifi_run.py``): the untapered map in uK, (1200, 1200, 6) float32,
axis 0 = rows (y), channels 27, 39, 93, 145, 225, 280 GHz, and ``taper.npy``, our cut-out taper
(cosine over the outer 30'). SZiFi's survey file tapers the map before SZiFi sees it for the
configurations set as our finder (the map our finder filters), or hands SZiFi the taper as its
mask_map and mask_ps for the colleagues' set-up.

Our finder runs on the tapered map as in the data-vector test: the masking pass at q = 4.3 (the
cut-outs' threshold), with every peak kept down to q = 4 (production's final pass keeps q > 5;
the counted set is the same, checked below). Saved per map, in ``ours/``:
  pk0_* / pk1_*   first-pass and final-pass peaks with q > 4: scale index j, row iy, column ix,
                  raw (cut-out) q, in buffer order (scale, then row-major pixel)
  s0, s1          sigma_j before and after the masking pass (cut-out scale)
  f_masked, n_masked, mask (the masking pass's binary mask, packed bits, 1 = kept)
  rs              sigma_periodic / sigma_cut per scale (the data-vector test's q rescaling)
  h_*             the halos with M >= 5e13 Msun (confirmation): position in the finder's 0.5'
                  pixels, (painting-grid row + offset) * PIXF / PIX as summaries.confirm places
                  them (and the same for columns), z, y0, ln M
  white           the map's white levels per channel [uK-arcmin] as SZiFi reads the map (square
                  0.5' pixels): spectra.npz's true-sky levels / sqrt(cos dec)
  counts_*        data vectors computed here with the simulator's own merging and confirmation:
                  prod (q rescaled: production), raw (cut-out q), raw0 (first pass, cut-out q)
  selfcheck       counts_prod against the data-vector pipeline run on the same map, and against
                  the data-vector test's stored counts for this tile, sky and noise map
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mss2_finder_test as ft  # noqa: E402
import mss2_datavector_test as dv  # noqa: E402
import spike  # noqa: E402
from classy_szlite.sim import instrument as ins, mmf, simulator as simu, summaries as sm  # noqa: E402
from classy_szlite.sim.flatsky import N, PIX, PIXF  # noqa: E402

OUT = os.path.join(dv.MSS2, "szifi_compare")
TILES = (2, 16, 26, 43, 45, 50, 86, 97)      # two at dec -35 and -25, two at -15, one at -5 and +5
BANDS = ("f030", "f040", "f090", "f150", "f230", "f290")
DEPTHS = {"split0": ("split", "split0"), "mean": ("coadd", "mean")}   # name: (data-vector depth, spectra set)
Q_LIST = 4.0                                  # lowest q kept in the exported peak lists


def log(*a):
    print(*a, flush=True)


def make_ours(fs, cfg):
    """Our finder as simulator.find_clusters (n_iter = 1), written out to keep the first pass and
    the mask, with the filter ``fs`` (all six channels or a subset; the maps passed must hold the
    same channels); the data vectors with summaries' own merging, confirmation and histogram."""
    e = ft.EDGE_DROP

    def counts_of(pk, halos):
        inner = (pk.iy >= e) & (pk.iy < N - e) & (pk.ix >= e) & (pk.ix < N - e)
        keep = sm.merge_local_max(pk, cfg.r_merge) & inner
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        return sm.histogram(pk.q, cf.z, keep & cf.ok), cf.overflow

    @jax.jit
    def run(X, halos, rs_j):
        q0, s0, C0 = mmf.q_cube(X, fs)
        pkm = mmf.find_peaks(q0, cfg.peak_cap, cfg.q_mask)            # the peaks that feed the mask
        pk0 = mmf.find_peaks(q0, cfg.peak_cap, Q_LIST)
        mask, n_masked = mmf.cluster_mask(pkm, fs, cfg.q_mask)
        C, f = mmf.masked_cov(X, mask, C0, fs)
        q1, s1, _ = mmf.q_cube(X, fs, C=C)
        pk1 = mmf.find_peaks(q1, cfg.peak_cap, Q_LIST)
        pkr = pk1._replace(q=(pk1.q / rs_j[pk1.j]).astype(pk1.q.dtype))   # the periodic tiles' q scale
        c_prod, o1 = counts_of(pkr, halos)
        c_raw, o2 = counts_of(pk1, halos)
        c_raw0, o3 = counts_of(pk0, halos)
        return dict(pk0=pk0, pk1=pk1, s0=s0, s1=s1, mask=mask, n_masked=n_masked, f_masked=1.0 - f,
                    peak_overflow=pkm.overflow | pk0.overflow | pk1.overflow, conf_overflow=o1 | o2 | o3,
                    counts_prod=c_prod, counts_raw=c_raw, counts_raw0=c_raw0)
    return run


def make_same_mask(fs):
    """sigma_j of the whitened masked estimate with a given mask (1 = kept), and the kept
    fraction: the masking step's estimator with another map's mask."""
    @jax.jit
    def run(X, mask):
        n = fs.a.shape[1]
        C0 = mmf.binned_cov(X.reshape(n, -1)[:, fs.m_idx], fs)
        C, f = mmf.masked_cov(X, mask, C0, fs)
        _, aCa = mmf.weights(C, fs)
        return 1.0 / jnp.sqrt(jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)), f
    return run


def peaks_np(pk):
    v = np.asarray(pk.valid)
    return {k: np.asarray(getattr(pk, k))[v] for k in ("j", "iy", "ix", "q")}


def stored_counts(tile, sky):
    """The data-vector test's stored counts for this tile and sky: {depth name: (4, 5)} or {}."""
    try:
        runs = {t["tile"]: t for t in dv.load_runs()}
    except Exception as exc:                      # no stored runs: the inline check still runs
        log(f"stored data-vector runs not read: {exc}")
        return {}
    if tile not in runs:
        return {}
    out = {}
    for name, (dp, _) in DEPTHS.items():
        rec = runs[tile]["data"][dp].get("raw", {}).get(sky)
        if rec is not None:
            out[name] = np.asarray(rec["counts"][0])          # map 0: split 0, or the plain mean
    return out


def parse_variants(text):
    out = {}
    for part in text.split(";"):
        if part.strip():
            name, chs = part.split("=")
            out[name.strip()] = [int(c) for c in chs.split(",") if c.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=",".join(map(str, TILES)))
    ap.add_argument("--sky", type=int, default=0, help="sky index, as the data-vector test's")
    ap.add_argument("--root", default=OUT, help="output folder")
    ap.add_argument("--no-clusters", action="store_true", help="noise and CMB only (no tSZ)")
    ap.add_argument("--no-write-maps", action="store_true", help="the maps exist in --root: run the variants only")
    ap.add_argument("--variants", default="ours=0,1,2,3,4,5", help="NAME=channels;... (our finder per channel set)")
    ap.add_argument("--mask-from", default="", help="a root whose variants' masks give s_same here")
    ap.add_argument("--force", action="store_true", help="write into folders that already hold outputs")
    args = ap.parse_args()
    root, clusters, write_maps = args.root, not args.no_clusters, not args.no_write_maps
    tiles = [int(t) for t in args.tiles.split(",") if t]
    variants = parse_variants(args.variants)
    ALL = list(range(6))
    assert variants and all(ch and set(ch) <= set(ALL) and len(set(ch)) == len(ch) for ch in variants.values()), variants
    # refuse to overwrite earlier outputs (the first job's, or a baseline) unless --force
    busy = [os.path.join(root, v) for v in variants if glob.glob(os.path.join(root, v, "tile_*.npz"))]
    if write_maps and os.path.exists(os.path.join(root, "meta.json")):
        busy.append(os.path.join(root, "meta.json"))
    if busy and not args.force:
        raise SystemExit(f"refusing to overwrite {busy} (use --force)")
    old_meta = None
    if not write_maps:                                        # the maps exist: the flags must match them
        old_meta = json.load(open(os.path.join(root, "meta.json")))
        assert old_meta["sky"] == args.sky, (old_meta["sky"], args.sky)
        assert old_meta.get("clusters", True) == clusters, "clusters flag differs from the run that wrote root"
        assert set(tiles) <= set(old_meta["tiles"]), (tiles, old_meta["tiles"])
    for sub in (["maps"] if write_maps else []) + list(variants):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    t0 = time.time()
    theta = simu.THETA_FID
    cells, tb, cs, cfg0, _, info = spike.setup(theta)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb, tb, mode="baseline")          # sets only the lensing table (the noise is MSS2's)
    cfg = cfg0._replace(q_mask=dv.Q_MASK_CUT)
    rs = dv.geometry_ratios()
    sky, datavector, _ = dv.make_pipeline(cs, theta, st, tb, cells, cfg, cfg0)
    fss = {name: (st.fs if ch == ALL else mmf.filter_static(channels=ch)) for name, ch in variants.items()}
    finders = {name: make_ours(fss[name], cfg) for name in variants}
    same_mask = {name: make_same_mask(fss[name]) for name in variants} if args.mask_from else {}
    Bf, ca = np.asarray(st.B_full, dtype=np.float64), np.asarray(st.cmb_amp, dtype=np.float64)

    spec = np.load(os.path.join(dv.MSS2, "spectra", "spectra.npz"))
    smeta = json.loads(str(spec["meta"]))
    assert tuple(smeta["bands"]) == BANDS, smeta["bands"]
    row = {t: i for i, t in enumerate(spec["index"].tolist())}

    if write_maps:
        freqs = np.asarray(ins.SO_LAT["freqs"], float)
        inst = dict(nu_eff_GHz=freqs.tolist(), fwhm_arcmin=np.asarray(ins.SO_LAT["fwhm"], float).tolist(),
                    tsz_sed_uK=np.asarray(ins.tsz_f_uK(freqs), float).tolist(), bands=list(BANDS))
        json.dump(inst, open(os.path.join(root, "instrument.json"), "w"), indent=1)
        np.save(os.path.join(root, "taper.npy"), np.asarray(ft.TAPER, dtype=np.float64))
    meta = dict(tiles=tiles, sky=args.sky, clusters=clusters, depths=list(DEPTHS), N=int(N), pix_arcmin=float(PIX),
                thetas=np.asarray(mmf.THETAS15, float).tolist(), edge_drop=int(ft.EDGE_DROP),
                edge_taper=int(ft.EDGE_TAPER), q_mask_cut=float(dv.Q_MASK_CUT), q_list=Q_LIST,
                M_conf=float(cfg.M_conf), r_conf=float(cfg.r_conf), r_merge=float(cfg.r_merge),
                q_edges=np.asarray(sm.Q_EDGES, float).tolist(), z_edges=np.asarray(sm.Z_EDGES, float).tolist(),
                rs={name: np.asarray(rs[dp], float).tolist() for name, (dp, _) in DEPTHS.items()},
                halo_positions="0.5' pixels", white_convention="square 0.5' pixels (true-sky level / sqrt(cos dec))",
                white={}, ra={}, dec={}, selfcheck={})
    import inspect
    pkg = dict(mmf_file=os.path.abspath(mmf.__file__),
               channel_subsets="channels" in inspect.signature(mmf.filter_static).parameters)
    log(f"classy_szlite filter: {pkg['mmf_file']} (channel subsets: {pkg['channel_subsets']})")
    vinfo = {name: dict(channels=ch, clusters=clusters, sky=args.sky, rs_channels=ALL, mask_from=args.mask_from,
                        package=pkg, selfcheck={}, maps_vs_root={}) for name, ch in variants.items()}
    log(f"setup {time.time() - t0:.0f} s; tiles {tiles}; sky {args.sky}; clusters {clusters}; "
        f"variants {variants}; write maps {write_maps}; root {root}; halo buffer {cfg.halo_cap}")

    for ti in tiles:
        t1 = time.time()
        path = os.path.join(dv.MSS2, "tiles", f"tile_{ti:03d}.npz")
        _, tmeta, _, raw_big, coadd_big = dv.load_big(path)
        noise = {"split0": np.asarray(ft.crop(raw_big[0]), dtype=np.float32),
                 "mean": np.asarray(ft.crop(coadd_big), dtype=np.float32)}
        del raw_big, coadd_big

        # the sky, as mss2_datavector_test.run_tile (the tSZ part left out with --no-clusters)
        r = args.sky
        key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261120), ti), r)
        halos, Xs, ovf_sky = sky(key)
        rc = np.random.default_rng([2026, 11, 20, ti, r, 99])
        Wc = ft.rfft2(rc.standard_normal((N, N)))
        cmb_map = np.stack([ft.irfft2(Bf[c] * ca * Wc, N) for c in range(6)])
        if clusters:
            sig = np.stack([ft.irfft2(np.asarray(Xs[c], dtype=np.complex128), N) for c in range(6)])
            common = sig + cmb_map
        else:
            common = cmb_map

        hv = np.asarray(halos.valid) & (np.asarray(halos.lnM) >= np.float32(np.log(cfg.M_conf)))
        to_px = np.float32(PIXF / PIX)                           # painting-grid pixels -> 0.5' pixels (exact)
        hal = dict(h_y=(np.asarray(halos.iy)[hv].astype(np.float32) + np.asarray(halos.oy)[hv]) * to_px,
                   h_x=(np.asarray(halos.ix)[hv].astype(np.float32) + np.asarray(halos.ox)[hv]) * to_px,
                   h_z=np.asarray(halos.z)[hv], h_y0=np.asarray(halos.y0)[hv], h_lnM=np.asarray(halos.lnM)[hv])
        stored = stored_counts(ti, r) if clusters else {}
        meta["ra"][ti], meta["dec"][ti] = float(tmeta["ra"]), float(tmeta["dec"])
        meta["white"][ti] = {}
        for v in vinfo.values():
            v["selfcheck"][ti] = {}
        for name, (dp, sset) in DEPTHS.items():
            m = noise[name]
            total = m + common                                        # float64, as run_tile
            m32 = np.ascontiguousarray(np.moveaxis(total, 0, -1).astype(np.float32))
            if write_maps:
                np.save(os.path.join(root, "maps", f"tile_{ti:03d}_{name}.npy"), m32)
            else:                                             # the rebuilt map against the stored one
                old = np.load(os.path.join(root, "maps", f"tile_{ti:03d}_{name}.npy"))
                cmp_ = dict(identical=bool(np.array_equal(old, m32)), n_diff=int(np.sum(old != m32)),
                            max_abs_diff=float(np.max(np.abs(old.astype(np.float64) - m32))))
                for v in vinfo.values():
                    v["maps_vs_root"].setdefault(ti, {})[name] = cmp_
                del old
            Xfull = jnp.asarray(ft.rfft2(total * ft.TAPER[None]).astype(np.complex64))
            rs_j = jnp.asarray(rs[dp], dtype=jnp.float32)
            ref = np.asarray(jax.device_get(datavector(Xfull, halos, rs_j))["counts"])
            k = smeta["sets"].index(sset)
            white = np.asarray(spec["white"][row[ti], k], float) / np.sqrt(np.cos(np.radians(float(tmeta["dec"]))))
            meta["white"][ti][name] = white.tolist()
            for vname, ch in variants.items():
                X = Xfull if ch == ALL else Xfull[jnp.asarray(ch)]
                o = jax.device_get(finders[vname](X, halos, rs_j))
                six = ch == ALL
                same_inline = bool(np.array_equal(np.asarray(o["counts_prod"]), ref)) if six else None
                same_stored = (bool(np.array_equal(np.asarray(o["counts_prod"]), stored[name]))
                               if six and name in stored else None)
                vinfo[vname]["selfcheck"][ti][name] = dict(inline=same_inline, stored=same_stored)
                p0, p1 = peaks_np(o["pk0"]), peaks_np(o["pk1"])
                extra = dict(counts_pipeline=ref) if six else {}         # the six-channel pipeline's counts
                if args.mask_from:
                    mf = np.load(os.path.join(args.mask_from, vname, f"tile_{ti:03d}_{name}.npz"))
                    mk = np.unpackbits(mf["mask"])[:N * N].reshape(N, N).astype(np.float32)
                    s_same, f_same = jax.device_get(same_mask[vname](X, jnp.asarray(mk)))
                    extra.update(s_same=np.asarray(s_same), f_same=float(f_same))
                np.savez(os.path.join(root, vname, f"tile_{ti:03d}_{name}.npz"),
                         **{f"pk0_{kk}": v for kk, v in p0.items()}, **{f"pk1_{kk}": v for kk, v in p1.items()},
                         s0=np.asarray(o["s0"]), s1=np.asarray(o["s1"]), f_masked=float(o["f_masked"]),
                         n_masked=int(o["n_masked"]), mask=np.packbits(np.asarray(o["mask"]) > 0.5),
                         rs=np.asarray(rs[dp], float), counts_prod=np.asarray(o["counts_prod"]),
                         counts_raw=np.asarray(o["counts_raw"]), counts_raw0=np.asarray(o["counts_raw0"]),
                         white=white, n_halos=int(halos.count), channels=np.asarray(ch), **extra,
                         overflow=np.array([bool(halos.overflow), bool(np.any(np.asarray(ovf_sky))),
                                            bool(o["peak_overflow"]), bool(o["conf_overflow"])]), **hal)
                log(f"tile {ti:3d} {name:6s} {vname}: peaks q>4 first {p0['q'].size}, final {p1['q'].size}; "
                    f"data vector {int(np.asarray(o['counts_prod']).sum())} (raw q {int(np.asarray(o['counts_raw']).sum())}); "
                    f"masked {float(o['f_masked']):.3f}; same as the pipeline: {same_inline}, as stored: {same_stored}; "
                    f"overflow {bool(o['peak_overflow']) or bool(o['conf_overflow']) or bool(np.any(np.asarray(ovf_sky)))}")
        log(f"tile {ti:3d} done in {time.time() - t1:.0f} s")
    for vname, v in vinfo.items():
        json.dump(v, open(os.path.join(root, vname, "info.json"), "w"), indent=1)
    six_names = [n for n, ch in variants.items() if ch == ALL]
    if write_maps:
        meta["selfcheck"] = vinfo[six_names[0]]["selfcheck"] if six_names else {}
        json.dump(meta, open(os.path.join(root, "meta.json"), "w"), indent=1)
    ok = all(c["inline"] is not False and c["stored"] is not False
             for v in vinfo.values() for d in v["selfcheck"].values() for c in d.values())
    log(f"self-check {'passed' if ok else 'FAILED'}; total {time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
