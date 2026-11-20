"""Does the noise model change the data vector? The simulator's own sky in MSS2 noise and in
model noise, through the production pipeline.

    python benchmarks/mapsim/mss2_datavector_test.py [--tiles N] [--skies R] [--first-sky R0]
                                                      [--kinds K1,K2] [--tag T] [--part I/K]
    python benchmarks/mapsim/mss2_datavector_test.py --gen-spectra       # band spectra for "gen"
    python benchmarks/mapsim/mss2_datavector_test.py --summary            # pools every run by tile

For each MSS2 tile (``mss2_tiles.py``) and R sky realisations, the simulator's fiducial sky
(halos above 1e13 Msun to z = 3, tSZ painted in the six SO channels, lensed CMB) is added to
every noise map of the tile. The production pipeline then gives the data vector: the finder
with its masking pass, merging within 1.5', confirmation by a halo above 5e13 Msun within 1.5',
counts in the paper's 4 x 5 grid of q and z, and the stacked-lensing sample. The same sky goes
into every noise map of a tile, so the differences between noise kinds are paired.

Noise kinds, built as in ``mss2_finder_test.py``, at one-split depth and at four-split depth (the
plain mean of the splits). In the finder's noise, one split is about 1.35x the simulator's
baseline (MSS2's coadd has SO's five-year baseline white levels, so a split, a quarter of the
data, is somewhat deeper than year 1 rescaled from it), and the plain mean is about the
simulator's goal depth; MSS2's own coadd, weighted by inverse variance, is deeper still:
  raw       MSS2 itself (one-split: 4 maps per tile; four-split: 1)
  mnms      mnms simulations, stitched above ell = 4000-4800 to a Gaussian copy of the MSS2
            map's own noise (one-split: N_MNMS_SPLIT per split; four-split: 5)
  gaussmod  Gaussian with MSS2's 2D spectra and depth pattern (one-split: one per split;
            four-split: N_SURR)
  iso       isotropic Gaussian with MSS2's binned 6 x 6 spectra: the simulator's kind of noise
            (as gaussmod)
These copy each tile's own spectra and noise level, so they test non-Gaussianity at perfect
two-point statistics. Two more kinds:
  gen       a practical generator (four-split depth only): isotropic Gaussian with the
            depth-normalised binned spectra averaged over the other tiles of the same declination
            band (leave one out), times the tile's own absolute depth pattern from the coadd
            inverse variance (``--gen-spectra`` writes the spectra first)
  iso_per   control of the q scale: iso noise drawn as a periodic tile with no taper, run with
            production's masking threshold (4) and no rescaling, exactly as the simulator runs

The tiles are tapered cut-outs, where the finder's q is 5-8% above the periodic tiles' scale
(``mss2_finder_test.py --geometry``). The masking pass runs at q = 4.3 on the cut-outs, and every
final-pass peak's q is divided by the geometry run's sigma_periodic / sigma_cut at its scale
before merging, confirmation and binning. Both were calibrated on noise-only maps: there 4.3
masks less than 4 does on periodic tiles (the masking pass thresholds the first-pass q, whose
ratio is 1.042-1.066, so the matching threshold is about 4.2), but with the cluster sky the
masked fraction comes back near the calibration point, and the iso_per control checks the
resulting scale. Only peaks more than 60' from the tile's edges count (0.64 of the tile).

Per map: the counts (4 x 5); the noiseless stacked-lensing signal of the selected clusters (the
sum of p_bar with no scatter drawn, so the comparison sees the selection only, not the lensing
noise) and their number; the kept peaks above q = 5 with no massive halo within 1.5'; sigma_j
after and before the masking pass; the masked fraction and the number of masked discs.

The summary compares each kind with MSS2, paired by tile and sky, with errors from a bootstrap
over tiles: total counts, counts per q bin and per z bin, and the mean p_bar of the selection.
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
import spike  # noqa: E402
from classy_szlite.sim import instrument as ins, halos as hl, lensing, paint as pt  # noqa: E402
from classy_szlite.sim import simulator as simu, summaries as sm  # noqa: E402
from classy_szlite.sim.flatsky import N  # noqa: E402

MSS2 = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2")
OUT = os.path.join(MSS2, "datavector_test")
Q_MASK_CUT = 4.3            # masks on the cut-outs what 4 masks on periodic tiles
N_MNMS_SPLIT = 2            # mnms simulations per split at one-split depth
N_SURR = 4                  # gaussmod / iso maps at four-split depth
KINDS = ("raw", "mnms", "gaussmod", "iso")           # the default run
ALL_KINDS = KINDS + ("gen", "iso_per")
DEPTHS = ("split", "coadd")


def log(*a):
    print(*a, flush=True)


def geometry_ratios():
    """sigma_periodic / sigma_cut per filter scale, per depth, from the geometry run."""
    geo = []
    for f in sorted(glob.glob(os.path.join(MSS2, "finder_test", "geometry_part_*_of_*.json"))):
        geo += json.load(open(f))
    assert geo, "run mss2_finder_test.py --geometry first"
    return {d: np.mean([np.mean(t[d]["iso_per"]["s1"], 0) / np.mean(t[d]["iso_cut"]["s1"], 0) for t in geo], 0)
            for d in DEPTHS}


def depth_abs(ivar2, obs2):
    """sigma pattern on the 1680^2 grid from the coadd inverse variance, as ft.depth_shape but
    not rescaled: it carries the tile's absolute depth (up to one factor per band, the inverse
    variance maps' normalisation, which cancels in the generator)."""
    from scipy.ndimage import gaussian_filter, zoom
    iv = np.where(obs2 > 0, ivar2, np.median(ivar2[obs2 > 0]))
    s2 = gaussian_filter(1.0 / np.sqrt(np.maximum(iv, 1e-30 * iv.max())),
                         ft.DEPTH_FWHM / 2.0 / np.sqrt(8 * np.log(2)), mode="nearest")
    return zoom(s2, 4, order=1, mode="nearest", grid_mode=True)


def load_big(path):
    """The tile's raw maps on the 1680 grid (four splits and their plain mean), weights, meta."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    raw_hf, raw_lf, obs2 = d["raw_hf"], d["raw_lf"], d["obs2"]
    wts = np.stack([np.kron(o, np.ones((4, 4))) for o in obs2.astype(float)]) * ft.T_MARGIN[None]
    raw_big = []
    for s in range(4):
        lf = [ft.fourier_upsample(raw_lf[s, b].astype(np.float64), 4) for b in range(2)]
        raw_big.append(np.stack(lf + [raw_hf[s, b].astype(np.float64) for b in range(4)]))
    return d, meta, wts, raw_big, np.mean(raw_big, axis=0)


def gen_spectra(paths):
    """Depth-normalised binned spectra of every tile's four-split mean (written for the gen kind)."""
    out = []
    for p in paths:
        d, meta, wts, raw_big, coadd_big = load_big(p)
        sa = np.stack([depth_abs(d["ivar2"][b], d["obs2"][b]) for b in range(6)])
        out.append(dict(tile=int(os.path.basename(p)[5:8]), dec=float(meta["dec"]),
                        spectra=ft.iso_spectra(coadd_big / sa, wts).tolist()))
        log(f"gen spectra: tile {out[-1]['tile']} (dec {meta['dec']:+.0f})")
    return out


def noise_maps(path, rng, kinds=KINDS, band_spectra=None):
    """The noise maps of one tile: {depth: {kind: [(6, N, N) float32]}}, and the tile's meta."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    dec = float(meta["dec"])
    raw_hf, raw_lf, mnms, ivar2, obs2 = d["raw_hf"], d["raw_lf"], d["mnms"], d["ivar2"], d["obs2"]
    assert raw_hf.shape[-1] == ft.NB and raw_lf.shape[-1] == ft.NB // 4 and mnms.shape[-1] == 300 + 2 * ft.MM
    wts = np.stack([np.kron(o, np.ones((4, 4))) for o in obs2.astype(float)]) * ft.T_MARGIN[None]
    sh = np.stack([ft.depth_shape(ivar2[b], obs2[b]) for b in range(6)])
    t_big = ft.stitch_weight(ft.true_ell(dec, ft.NB))
    raw_big = []
    for s in range(4):
        lf = [ft.fourier_upsample(raw_lf[s, b].astype(np.float64), 4) for b in range(2)]
        raw_big.append(np.stack(lf + [raw_hf[s, b].astype(np.float64) for b in range(4)]))
    coadd_big = np.mean(raw_big, axis=0)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    out = {dp: {k: [] for k in kinds} for dp in DEPTHS}

    def mnms_map(coarse6, maps_big):
        hi = ft.high_field(maps_big, wts, sh, t_big, rng)
        chans = [ft.mnms_to_tile(coarse6[b], dec, lowpass=False) for b in range(2)]
        chans += [ft.mnms_to_tile(coarse6[b], dec, lowpass=True) + hi[b - 2] for b in range(2, 6)]
        return f32(np.stack(chans))

    # the original kinds keep their random draws whatever else is asked for (same maps as the first run)
    if set(kinds) & set(KINDS):
        for s in range(4):
            out["split"].get("raw", []).append(f32(ft.crop(raw_big[s])))
            gm = f32(ft.crop(ft.surrogate(raw_big[s] / sh, wts, rng) * sh))
            io = f32(ft.crop(ft.iso_field(raw_big[s], wts, rng)))
            if "gaussmod" in kinds:
                out["split"]["gaussmod"].append(gm)
            if "iso" in kinds:
                out["split"]["iso"].append(io)
        sims = meta["sims"]
        for jx, (s, n) in enumerate(sims):
            if n < N_MNMS_SPLIT and "mnms" in kinds:
                out["split"]["mnms"].append(mnms_map(mnms[jx], raw_big[s]))
        if "raw" in kinds:
            out["coadd"]["raw"].append(f32(ft.crop(coadd_big)))
        for _ in range(N_SURR):
            gm = f32(ft.crop(ft.surrogate(coadd_big / sh, wts, rng) * sh))
            io = f32(ft.crop(ft.iso_field(coadd_big, wts, rng)))
            if "gaussmod" in kinds:
                out["coadd"]["gaussmod"].append(gm)
            if "iso" in kinds:
                out["coadd"]["iso"].append(io)
        if "mnms" in kinds:
            for n in range(max(nn for _, nn in sims) + 1):
                js = [jx for jx, (s, nn) in enumerate(sims) if nn == n]
                out["coadd"]["mnms"].append(mnms_map(np.mean(mnms[js].astype(np.float64), axis=0), coadd_big))
    r2 = np.random.default_rng([2026, 11, 20, int(os.path.basename(path)[5:8]), 7])    # the new kinds' own draws
    if "iso_per" in kinds:
        for s in range(4):
            out["split"]["iso_per"].append(f32(ft.iso_from_spectra(ft.iso_spectra(raw_big[s], wts), N, r2)))
        out["coadd"]["iso_per"].append(f32(ft.iso_from_spectra(ft.iso_spectra(coadd_big, wts), N, r2)))
    if "gen" in kinds:
        ti = int(os.path.basename(path)[5:8])
        mine = [b for b in band_spectra if b["tile"] == ti][0]
        others = [np.asarray(b["spectra"]) for b in band_spectra
                  if round(b["dec"]) == round(mine["dec"]) and b["tile"] != ti]
        Cs = np.mean(others, axis=0)
        sa = np.stack([depth_abs(ivar2[b], obs2[b]) for b in range(6)])
        for _ in range(N_SURR):
            out["coadd"]["gen"].append(f32(ft.crop(ft.iso_from_spectra(Cs, ft.NB, r2) * sa)))
    return out, meta


def make_pipeline(cs, theta, st, tb, cells, cfg, cfg_per):
    """jitted sky (halos and the tSZ channel maps) and data vector of one map, for the tapered
    cut-outs (cfg, q rescaled) and for the periodic control (cfg_per, production's)."""
    e = ft.EDGE_DROP

    @jax.jit
    def sky(key):
        k_h, _ = jax.random.split(key)
        halos = hl.sample_halos(k_h, cs.lam, cells, cfg.halo_cap, cs.cg, theta.h, theta.A_SZ, theta.alpha_SZ,
                                theta.sigma_lnq, theta.one_minus_b, d2N=cs.d2N)
        tabs = tb._replace(tab=st.paint_tab, rP=st.paint_rP)
        yk, ovf = pt.paint_y(halos, tabs, cfg.paint, st.T_tab)
        return halos, st.a_full * yk[None], jnp.concatenate([halos.overflow[None], ovf])

    def datavector_(X, halos, rs_j, cfg):
        pk, sigma, mk = simu.find_clusters(X, st.fs, cfg)
        pk = pk._replace(q=(pk.q / rs_j[pk.j]).astype(pk.q.dtype))       # the periodic tiles' q scale
        inner = (pk.iy >= e) & (pk.iy < N - e) & (pk.ix >= e) & (pk.ix < N - e)
        keep = sm.merge_local_max(pk, cfg.r_merge) & inner
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        counts = sm.histogram(pk.q, cf.z, keep & cf.ok)
        sel = sm.in_data_vector(pk.q, cf.z, keep & cf.ok)
        M = jnp.exp(halos.lnM[cf.halo].astype(jnp.float64))
        z = halos.z[cf.halo].astype(jnp.float64)
        pbar = lensing.p_mean(M, z, cs.cg, theta.h, theta.beta_lens, cs.chi_star, st.lens_lnth, st.lens_lnsig)
        return dict(counts=counts, pbar_sum=jnp.sum(jnp.where(sel, pbar, 0.0)), n_sel=jnp.sum(sel),
                    n_unconf=jnp.sum(keep & ~cf.ok & (pk.q > float(sm.Q_EDGES[0]))),
                    n_kept=jnp.sum(keep & (pk.q > float(sm.Q_EDGES[0]))), sigma=sigma, sigma0=mk["sigma0"],
                    f_masked=mk["f_masked"], n_masked=mk["n_masked"],
                    overflow=jnp.stack([mk["peak_overflow"], cf.overflow]))

    datavector = jax.jit(lambda X, halos, rs_j: datavector_(X, halos, rs_j, cfg))
    datavector_per = jax.jit(lambda X, halos, rs_j: datavector_(X, halos, rs_j, cfg_per))
    return sky, datavector, datavector_per


def run_tile(path, skies, kinds, sky, datavector, datavector_per, st, rs, band_spectra):
    t0 = time.time()
    ti = int(os.path.basename(path)[5:8])
    rng = np.random.default_rng([2026, 11, 20, ti])
    maps, meta = noise_maps(path, rng, kinds, band_spectra)
    t_build = time.time() - t0
    Bf, ca = np.asarray(st.B_full, dtype=np.float64), np.asarray(st.cmb_amp, dtype=np.float64)
    rs_j = {dp: jnp.asarray(rs[dp], dtype=jnp.float32) for dp in DEPTHS}
    one = jnp.ones(15, dtype=jnp.float32)
    fields = ("counts", "pbar_sum", "n_sel", "n_unconf", "n_kept", "sigma", "sigma0", "f_masked", "n_masked", "overflow")
    res = {dp: {k: {f: [] for f in fields} for k in kinds if maps[dp][k]} for dp in DEPTHS}
    sky_info = []
    for r in skies:
        key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261120), ti), r)
        halos, Xs, ovf_sky = sky(key)
        sig = np.stack([ft.irfft2(np.asarray(Xs[c], dtype=np.complex128), N) for c in range(6)])
        rc = np.random.default_rng([2026, 11, 20, ti, r, 99])
        Wc = ft.rfft2(rc.standard_normal((N, N)))
        common = sig + np.stack([ft.irfft2(Bf[c] * ca * Wc, N) for c in range(6)])
        sky_info.append(dict(sky=int(r), halos=int(halos.count), overflow=np.asarray(ovf_sky).tolist()))
        for dp in DEPTHS:
            for k in res[dp]:
                rows = {f: [] for f in fields}
                for m in maps[dp][k]:
                    if k == "iso_per":                         # periodic tile: no taper, production's finder
                        o = jax.device_get(datavector_per(jnp.asarray(ft.rfft2(m + common).astype(np.complex64)), halos, one))
                    else:
                        X = jnp.asarray(ft.rfft2((m + common) * ft.TAPER[None]).astype(np.complex64))
                        o = jax.device_get(datavector(X, halos, rs_j[dp]))
                    rows["counts"].append(np.asarray(o["counts"]).tolist())
                    for f in ("pbar_sum", "f_masked"):
                        rows[f].append(float(o[f]))
                    for f in ("n_sel", "n_unconf", "n_kept", "n_masked"):
                        rows[f].append(int(o[f]))
                    for f in ("sigma", "sigma0"):
                        rows[f].append(np.asarray(o[f]).tolist())
                    rows["overflow"].append(np.asarray(o["overflow"]).tolist())
                for f, v in rows.items():
                    res[dp][k][f].append(v)                      # [sky][map]
    return dict(tile=ti, ra=meta["ra"], dec=meta["dec"], skies=[int(r) for r in skies], sky_info=sky_info, res=res,
                seconds=round(time.time() - t0, 1), build_seconds=round(t_build, 1))


# ------------------------------------------------------------------ summary
SKY_40 = 0.4 * 41253.0 / 64.0          # interior tiles in 40% of the sky (64 deg2 each)


def load_runs():
    """Every run's part files, pooled by tile: {tile: dict(dec, data[depth][kind][sky] = {field: [maps]})}."""
    tiles = {}
    for f in sorted(glob.glob(os.path.join(OUT, "part_*_of_*.json"))):
        for t in json.load(open(f)):
            T = tiles.setdefault(t["tile"], dict(tile=t["tile"], dec=t["dec"], data={dp: {} for dp in DEPTHS}, sky_ovf=[]))
            T["sky_ovf"] += [int(np.any(np.asarray(si["overflow"]))) for si in t["sky_info"]]
            for dp in DEPTHS:
                for k, v in t["res"][dp].items():
                    skies = range(t["skies"]) if isinstance(t["skies"], int) else t["skies"]   # first run: 0..R-1
                    for jr, r in enumerate(skies):
                        T["data"][dp].setdefault(k, {})[int(r)] = {fld: v[fld][jr] for fld in v}
    return [tiles[k] for k in sorted(tiles)]


def summary(tiles):
    rng = np.random.default_rng(1)
    T = len(tiles)
    ix = rng.integers(0, T, (2000, T))
    decs = np.round([t["dec"] for t in tiles]).astype(int)
    qe, ze = sm.Q_EDGES, sm.Z_EDGES
    lines, out = [], dict(n_tiles=T, q_edges=list(qe), z_edges=list(ze), depth={})

    def tile_mean(t, dp, k, skies, fld, axis_map=True):
        v = [np.asarray(t["data"][dp][k][r][fld], float) for r in skies]
        return np.mean([x.mean(axis=0) for x in v], axis=0)            # mean over maps, then skies

    def common(t, dp, a, b):
        return sorted(set(t["data"][dp].get(a, {})) & set(t["data"][dp].get(b, {})))

    def pair(dp, a, b, fld="counts"):
        """per-tile means of kind a and kind b over their common skies: (T, ...) arrays."""
        A, B = [], []
        for t in tiles:
            sk = common(t, dp, a, b)
            A.append(tile_mean(t, dp, a, sk, fld)); B.append(tile_mean(t, dp, b, sk, fld))
        return np.array(A), np.array(B)

    def ratio(A, B):
        r = A.sum(0) / B.sum(0)
        e = np.std(A[ix].sum(1) / B[ix].sum(1), axis=0, ddof=1)
        return r, e

    nsk = {dp: {k: int(np.median([len(t["data"][dp].get(k, {})) for t in tiles])) for k in ALL_KINDS
                if all(k in t["data"][dp] for t in tiles)} for dp in DEPTHS}
    lines.append(f"{T} tiles (dec bands " + ", ".join(f"{b:+d}: {int(np.sum(decs == b))}" for b in sorted(set(decs)))
                 + "); skies per tile: " + "; ".join(f"{dp} " + ", ".join(f"{k} {n}" for k, n in nsk[dp].items()) for dp in DEPTHS))
    lines.append("Counts per interior tile (64 deg2) on the periodic tiles' q scale. Errors: bootstrap over tiles. "
                 "'Poisson units': the difference over the 40%-sky Poisson error of that bin.")
    for dp in DEPTHS:
        kinds = list(nsk[dp])
        lines.append(f"\n=== {'one split (about 1.35x baseline noise)' if dp == 'split' else 'plain mean of four splits (about goal depth)'}")
        Araw, _ = pair(dp, "raw", "raw")
        c = Araw
        lines.append(f"  MSS2: {c.sum((1, 2)).mean():.1f} detections per tile; by q bin "
                     + ", ".join(f"{v:.2f}" for v in c.sum(2).mean(0)) + "; by z bin " + ", ".join(f"{v:.2f}" for v in c.sum(1).mean(0)))
        dd = {}
        for k in kinds:
            if k == "raw":
                continue
            ref = "iso" if k == "iso_per" else "raw"
            A, B = pair(dp, k, ref)
            rt, et = ratio(A.sum((1, 2))[:, None], B.sum((1, 2))[:, None])
            rq, eq = ratio(A.sum(2), B.sum(2))
            rz, ez = ratio(A.sum(1), B.sum(1))
            # Poisson units for 40% of the sky, per q bin, and the 20-bin delta chi^2 (minus its noise part)
            nA, nB = A.mean(0), B.mean(0)
            pu = np.sqrt(SKY_40) * (A.sum(2).mean(0) - B.sum(2).mean(0)) / np.sqrt(B.sum(2).mean(0))
            dvar = np.var((A - B)[ix].mean(1), axis=0, ddof=1)
            ok = nB > 0
            chi2 = float(SKY_40 * np.sum(((nA - nB) ** 2)[ok] / nB[ok]))
            chi2_noise = float(SKY_40 * np.sum(dvar[ok] / nB[ok]))
            pa, na = pair(dp, k, ref, "pbar_sum")[0], pair(dp, k, ref, "n_sel")[0]
            pb, nb = pair(dp, k, ref, "pbar_sum")[1], pair(dp, k, ref, "n_sel")[1]
            mp = (pa.sum() / na.sum()) / (pb.sum() / nb.sum())
            ep = np.std((pa[ix].sum(1) / na[ix].sum(1)) / (pb[ix].sum(1) / nb[ix].sum(1)), ddof=1)
            # within-tile (skies) and between-tile parts of the paired difference's variance
            dts = [np.array([np.asarray(t["data"][dp][k][r]["counts"], float).sum((-2, -1)).mean()
                             - np.asarray(t["data"][dp][ref][r]["counts"], float).sum((-2, -1)).mean()
                             for r in common(t, dp, k, ref)]) for t in tiles]
            within = np.mean([d.var(ddof=1) / d.size for d in dts]) / T
            total = np.var([d.mean() for d in dts], ddof=1) / T
            lab = f"{k}/{'iso' if ref == 'iso' else 'MSS2'}"
            lines.append(f"  {lab:14s} counts {rt[0]:.4f}±{et[0]:.4f}; by q bin " + " ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(rq, eq))
                         + "; Poisson units by q bin " + " ".join(f"{v:+.2f}" for v in pu))
            lines.append(f"  {'':14s} by z bin " + " ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(rz, ez))
                         + f"; 20-bin delta chi2 (40% sky) {chi2:.1f}, of which noise {chi2_noise:.1f}; mean p_bar {mp:.4f}±{ep:.4f}; "
                         f"error split within/between tiles {np.sqrt(within) / B.sum((1, 2)).mean() * 100:.2f}%/"
                         f"{np.sqrt(max(total - within, 0)) / B.sum((1, 2)).mean() * 100:.2f}%")
            dd[k] = dict(ref=ref, counts_ratio=[float(rt[0]), float(et[0])], q_ratio=rq.tolist(), q_ratio_err=eq.tolist(),
                         z_ratio=rz.tolist(), z_ratio_err=ez.tolist(), poisson_units_q=pu.tolist(), chi2=chi2,
                         chi2_noise=chi2_noise, pbar_ratio=[float(mp), float(ep)])
        # kind against kind (more precise where kinds share the MSS2 map), and correlations of the ratios
        pairs = [(a, b) for a, b in (("gaussmod", "iso"), ("mnms", "iso"), ("gen", "iso"), ("gen", "gaussmod")) if a in kinds and b in kinds]
        for a, b in pairs:
            A, B = pair(dp, a, b)
            r, e = ratio(A.sum((1, 2))[:, None], B.sum((1, 2))[:, None])
            lines.append(f"  {a}/{b}: counts {r[0]:.4f}±{e[0]:.4f}")
            dd[f"{a}_over_{b}"] = [float(r[0]), float(e[0])]
        ks = [k for k in kinds if k not in ("raw", "iso_per")]
        if len(ks) > 1:
            bs = []
            for k in ks:
                A, B = pair(dp, k, "raw")
                bs.append(A.sum((1, 2))[ix].sum(1) / B.sum((1, 2))[ix].sum(1))
            cc = np.corrcoef(bs)
            lines.append("  correlation of the kinds' ratios to MSS2 (" + ", ".join(ks) + "): "
                         + "; ".join(f"{ks[i]}-{ks[j]} {cc[i, j]:.2f}" for i in range(len(ks)) for j in range(i + 1, len(ks))))
        # masking details where recorded (the later runs)
        det = []
        for k in kinds:
            v = [t["data"][dp][k][r] for t in tiles for r in t["data"][dp][k] if "sigma0" in t["data"][dp][k][r]]
            if v:
                s1 = np.mean([np.mean(np.asarray(x["sigma"]), 0) for x in v], 0)
                s0 = np.mean([np.mean(np.asarray(x["sigma0"]), 0) for x in v], 0)
                det.append(f"{k} masked {np.mean([np.mean(x['f_masked']) for x in v]):.3f}, discs {np.mean([np.mean(x['n_masked']) for x in v]):.0f}, "
                           f"sigma after/before masking {s1[2] / s0[2]:.4f} (1')")
        if det:
            lines.append("  masking: " + "; ".join(det))
        # per declination band, with errors
        for b in sorted(set(decs)):
            sel = np.flatnonzero(decs == b)
            cells = []
            for k in kinds:
                if k in ("raw", "iso_per"):
                    continue
                A, B = pair(dp, k, "raw")
                a, bb = A.sum((1, 2))[sel], B.sum((1, 2))[sel]
                jb = rng.integers(0, sel.size, (2000, sel.size))
                cells.append(f"{k} {a.sum() / bb.sum():.3f}±{np.std(a[jb].sum(1) / bb[jb].sum(1), ddof=1):.3f}")
            lines.append(f"  dec {b:+3d} ({sel.size} tiles): " + ", ".join(cells))
        out["depth"][dp] = dict(raw=dict(detections=float(c.sum((1, 2)).mean()), q_counts=c.sum(2).mean(0).tolist(),
                                         z_counts=c.sum(1).mean(0).tolist()), kinds=dd, skies=nsk[dp])
    ovf = sum(int(np.any(np.asarray(t["data"][dp][k][r]["overflow"]))) for t in tiles for dp in DEPTHS
              for k in t["data"][dp] for r in t["data"][dp][k])
    if ovf or any(sum(t["sky_ovf"]) for t in tiles):
        lines.append(f"\nWARNING: {ovf} map sets with a peak or confirmation overflow; "
                     f"{sum(sum(t['sky_ovf']) for t in tiles)} skies with a halo or paint overflow")
    return "\n".join(lines), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=0, help="number of tiles (0 = all), spread over dec bands")
    ap.add_argument("--skies", type=int, default=8, help="sky realisations per tile")
    ap.add_argument("--first-sky", type=int, default=0, help="index of the first sky (skies are keyed by index)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma-separated: " + ", ".join(ALL_KINDS))
    ap.add_argument("--tag", default="", help="suffix of this run's part files (runs are pooled by tile)")
    ap.add_argument("--part", default="0/1", help="I/K: run every K-th tile starting at I")
    ap.add_argument("--gen-spectra", action="store_true", help="write the depth-normalised spectra for the gen kind")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.summary:
        txt, nums = summary(load_runs())
        print(txt)
        open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
        json.dump(nums, open(os.path.join(OUT, "summary.json"), "w"))
        return
    paths = ft.pick(sorted(glob.glob(os.path.join(MSS2, "tiles", "tile_*.npz"))), args.tiles or 10 ** 6)
    i, k = (int(x) for x in args.part.split("/"))
    paths = paths[i::k]
    ft.ISO_BINS = ft.iso_bins()
    if args.gen_spectra:
        json.dump(gen_spectra(paths), open(os.path.join(OUT, f"gen_spectra_{i}_of_{k}.json"), "w"))
        return
    kinds = tuple(x for x in args.kinds.split(",") if x)
    assert all(x in ALL_KINDS for x in kinds), kinds
    band_spectra = None
    if "gen" in kinds:
        band_spectra = sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "gen_spectra_*_of_*.json")))], [])
        assert len(band_spectra) >= 2, "run --gen-spectra first"
    theta = simu.THETA_FID
    t0 = time.time()
    cells, tb, cs, cfg0, _, info = spike.setup(theta)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb, tb, mode="baseline")          # lensing table: MSS2's white levels are baseline's
    cfg = cfg0._replace(q_mask=Q_MASK_CUT)
    rs = geometry_ratios()
    sky, datavector, datavector_per = make_pipeline(cs, theta, st, tb, cells, cfg, cfg0)
    skies = list(range(args.first_sky, args.first_sky + args.skies))
    log(f"setup {time.time() - t0:.0f} s; {len(paths)} tiles in this part; kinds {kinds}; skies {skies[0]}-{skies[-1]}; "
        f"halo buffer {cfg.halo_cap}; sigma_periodic/sigma_cut {rs['split'].min():.3f}-{rs['split'].max():.3f}")
    out = []
    tag = f"_{args.tag}" if args.tag else ""
    for p in paths:
        r = run_tile(p, skies, kinds, sky, datavector, datavector_per, st, rs, band_spectra)
        out.append(r)
        tot = {dp: {kk: np.mean(np.asarray(r["res"][dp][kk]["counts"], float).sum((-2, -1))) for kk in r["res"][dp]}
               for dp in DEPTHS}
        log(f"tile {r['tile']:3d} (dec {r['dec']:+.0f}) {r['seconds']:.0f} s (noise {r['build_seconds']:.0f} s); detections, "
            + " | ".join(f"{dp}: " + ", ".join(f"{kk} {v:.1f}" for kk, v in tot[dp].items()) for dp in DEPTHS))
        with open(os.path.join(OUT, f"part_{i}_of_{k}{tag}.json"), "w") as fh:
            json.dump(out, fh)


if __name__ == "__main__":
    main()
