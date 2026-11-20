"""How close each finder's noise after the masking step comes to the noise without clusters, and why.

    python benchmarks/mapsim/mss2_szifi_noise.py [--root DIR]

Inputs: ROOT (default $MAPSIM_OUT/mss2/szifi_compare: the maps with the cluster sky) and
ROOT/nosky (the same maps without it: mss2_szifi_maps.py --no-clusters --mask-from ROOT; same noise
and CMB). Finders: our finder's variants (ours: six channels; ours_ch4: the four high-frequency
ones) and SZiFi's runs (match4: set as our finder with the four channels; coll: the colleagues'
settings; and their one-pass *_same runs on ROOT/nosky with the cluster masks from ROOT).
Numpy only.

Per finder, depth and scale theta_j (sigma0, sigma1: the noise before and after the masking step;
'clusters' = ROOT, 'none' = ROOT/nosky):
  r_before = sigma0(clusters) / sigma0(none)    the clusters' contamination of the first pass; the
                                                 taper's effect on sigma is the same in both maps
                                                 and cancels
  r_after  = sigma1(clusters) / sigma0(none)    the finder's final noise against the noise it would
                                                 have without clusters: its final q is too low by
                                                 this factor (too high if below 1)
  r_same   = sigma_m(none) / sigma0(none)       the masked estimator on the maps without clusters,
                                                 with the mask it used on the maps with clusters:
                                                 its own response to that mask (for our finder on
                                                 tapered cut-outs about sqrt(k), below)
  residual = r_after / r_same                    the cluster signal the masking step left in
  none     = sigma1(none) / sigma0(none)          the masking step on the maps without clusters
For our finder also k = mean(M T^2) / (mean(M) mean(T^2)), with M its mask on the maps with
clusters and T the taper: the whitened estimate divides by the kept fraction mean(M), but the discs
sit mostly where T = 1, so the estimate comes out about sqrt(k) low in sigma (none on periodic
tiles). Only the tiles present for every finder on both roots are used. SZiFi's sigma at its first
scale is the median y0 / q of that scale's detections (NaN without any); the colleagues' 100 scales
are interpolated to our 15 in log-log; at each scale all five ratios are means over the same tiles,
those where all of them are finite (the number is printed).
Checks: ROOT/nosky/meta.json has no clusters and the same sky, tiles and depths; ours_ref was made
by the earlier package (no channel subsets) and ours_rerun by the new one (info.json: package);
ours_rerun against ours_ref (same job): every array identical, else exit 1; and against ours (the
first job, maybe another node): reported.
Writes ROOT/noise_summary.txt and .json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mss2_szifi_compare as C  # noqa: E402

OURS = ("ours", "ours_ch4")
SZ = ("match4", "coll")
LABEL = {"ours": "our finder, 6 channels", "ours_ch4": "our finder, 4 channels",
         "match4": "SZiFi as ours, 4 channels", "coll": "SZiFi, colleagues' settings"}
PICK = (0, 2, 4, 7, 10, 14)


def load_ours(root, v, ti, dp):
    f = os.path.join(root, v, f"tile_{ti:03d}_{dp}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def load_szifi(root, c, ti, dp):
    f = os.path.join(root, "szifi", "runs", f"{c}_tile_{ti:03d}_{dp}.npz")
    if not os.path.exists(f):
        return None
    d = dict(np.load(f))
    return d if "it0_sig" in d else None


def compare_dirs(new_dir, ref_dir, strict):
    """Every npz in new_dir against ref_dir: identical maps, round-off only (relative 1e-5 or
    less on float arrays; integers and shapes identical), differing. NaNs compare equal only to
    NaNs. strict: any difference counts as differing."""
    new = sorted(glob.glob(os.path.join(new_dir, "tile_*.npz")))
    if not new:
        return None
    same, roundoff, diff, worst = 0, 0, [], 0.0
    for f in new:
        g = os.path.join(ref_dir, os.path.basename(f))
        if not os.path.exists(g):
            diff.append((os.path.basename(f), "missing"))
            continue
        a, b = np.load(f), np.load(g)
        bad, rel = [], 0.0
        for k in b.files:
            if k not in a.files:
                bad.append(k)
                continue
            x, y = np.asarray(a[k]), np.asarray(b[k])
            if x.shape == y.shape and np.array_equal(x, y, equal_nan=np.issubdtype(y.dtype, np.floating)):
                continue
            if strict or x.shape != y.shape or not np.issubdtype(y.dtype, np.floating):
                bad.append(k)
                continue
            r = float(np.nanmax(np.abs(x.astype(np.float64) - y)) / max(float(np.nanmax(np.abs(y))), 1e-300))
            if not np.isfinite(r) or r > 1e-5 or not np.array_equal(np.isnan(x), np.isnan(y)):
                bad.append(k)
            else:
                rel = max(rel, r)
        if bad:
            diff.append((os.path.basename(f), bad))
        elif rel > 0:
            roundoff += 1
            worst = max(worst, rel)
        else:
            same += 1
    return dict(n=len(new), identical=same, roundoff=roundoff, worst_rel=worst, differing=diff)


def taper_factor(mask_packed, T2, N):
    """k = mean(M T^2) / (mean(M) mean(T^2)) for a packed mask (1 = kept)."""
    M = np.unpackbits(mask_packed)[:N * N].reshape(N, N).astype(np.float64)
    return float(np.mean(M * T2) / (np.mean(M) * np.mean(T2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.OUT)
    ap.add_argument("--configs", default=",".join(SZ), help="SZiFi set-ups (each needs its *_same runs)")
    args = ap.parse_args()
    sz = tuple(c for c in args.configs.split(",") if c)
    root, nosky = args.root, os.path.join(args.root, "nosky")
    meta = json.load(open(os.path.join(root, "meta.json")))
    mn = json.load(open(os.path.join(nosky, "meta.json")))
    assert mn.get("clusters") is False, "ROOT/nosky must hold the maps without clusters"
    assert mn["sky"] == meta["sky"] and mn["tiles"] == meta["tiles"] and mn["depths"] == meta["depths"], "maps differ"
    N = int(meta["N"])
    thetas = np.asarray(meta["thetas"], float)
    T2 = np.load(os.path.join(root, "taper.npy")) ** 2
    lines, res = [], dict(depths={})
    P = lines.append
    P("Noise after the masking step against the noise without clusters (mss2_szifi_noise.py)")
    fail = False
    ref = compare_dirs(os.path.join(root, "ours_rerun"), os.path.join(root, "ours_ref"), strict=True)
    res["regression_same_job"] = ref
    if ref is not None:
        pk = {v: json.load(open(os.path.join(root, v, "info.json"))).get("package", {}) for v in ("ours_rerun", "ours_ref")}
        res["packages"] = pk
        ok_pkg = (pk["ours_ref"].get("channel_subsets") is False and pk["ours_rerun"].get("channel_subsets") is True
                  and pk["ours_ref"].get("mmf_file") != pk["ours_rerun"].get("mmf_file"))
        P(f"check: the reference used {pk['ours_ref'].get('mmf_file')} (no channel subsets), the rerun "
          f"{pk['ours_rerun'].get('mmf_file')}: {'as intended' if ok_pkg else 'NOT AS INTENDED'}")
        fail |= not ok_pkg
        P(f"check: our six-channel finder after the change to channel subsets against the earlier snapshot in "
          f"the same job: identical on {ref['identical']} of {ref['n']} maps"
          + (f"; DIFFERING: {ref['differing']}" if ref["differing"] else ""))
        fail |= bool(ref["differing"]) or ref["identical"] != ref["n"]
    old = compare_dirs(os.path.join(root, "ours_rerun"), os.path.join(root, "ours"), strict=False)
    res["regression_first_job"] = old
    if old is not None:
        P(f"check: against the first job's outputs (another node, maybe): identical on {old['identical']} of "
          f"{old['n']}, round-off only on {old['roundoff']} (max relative {old['worst_rel']:.1e})"
          + (f"; differing: {old['differing']}" if old["differing"] else ""))
    P("theta: " + "  ".join(f"{thetas[k]:.2f}'" for k in PICK))
    for dp in meta["depths"]:
        R = res["depths"].setdefault(dp, {})
        P("")
        P(f"=== depth {dp}")
        data = {}
        for ti in meta["tiles"]:
            row = {}
            for v in OURS:
                row[v] = (load_ours(root, v, ti, dp), load_ours(nosky, v, ti, dp))
            for c in sz:
                row[c] = (load_szifi(root, c, ti, dp), load_szifi(nosky, c, ti, dp), load_szifi(nosky, c + "_same", ti, dp))
            data[ti] = row
        ok_tiles = [ti for ti, row in data.items() if all(x is not None for t in row.values() for x in t)]
        missing = sorted(set(data) - set(ok_tiles))
        R["tiles"], R["missing"] = ok_tiles, missing
        P(f"tiles used: {ok_tiles}" + (f"; left out (a finder's run is missing): {missing}" if missing else ""))
        if not ok_tiles:
            fail = True
            continue
        for name in OURS + sz:
            rb, ra, rs, rn, kk, fm_c, fm_n = [], [], [], [], [], [], []
            for ti in ok_tiles:
                if name in OURS:
                    a, b = data[ti][name]
                    s0c, s1c, s0n, s1n, ssm = a["s0"], a["s1"], b["s0"], b["s1"], b["s_same"]
                    kk.append(taper_factor(a["mask"], T2, N))
                    fm_c.append(float(a["f_masked"])); fm_n.append(float(b["f_masked"]))
                else:
                    a, b, m = data[ti][name]
                    s0c = C.sigma_on_ours(a, 0, thetas)
                    s1c = C.sigma_on_ours(a, 1, thetas) if "it1_sig" in a else s0c      # nothing masked: one pass
                    s0n = C.sigma_on_ours(b, 0, thetas)
                    s1n = C.sigma_on_ours(b, 1, thetas) if "it1_sig" in b else s0n
                    ssm = C.sigma_on_ours(m, 0, thetas)
                    fm_c.append(1.0 - float(a["rec_fsky"][0]) if a["rec_fsky"].size else 0.0)
                    fm_n.append(1.0 - float(b["rec_fsky"][0]) if b["rec_fsky"].size else 0.0)
                rb.append(s0c / s0n); ra.append(s1c / s0n); rs.append(ssm / s0n); rn.append(s1n / s0n)
            rb, ra, rs, rn = map(np.array, (rb, ra, rs, rn))
            ok = np.isfinite(rb) & np.isfinite(ra) & np.isfinite(rs) & np.isfinite(rn)   # one tile set per scale
            rb, ra, rs, rn = (np.where(ok, x, np.nan) for x in (rb, ra, rs, rn))
            resid = ra / rs
            mean = lambda x: np.nanmean(x, 0)
            R[name] = dict(n_tiles=len(ok_tiles), r_before=mean(rb).tolist(), r_after=mean(ra).tolist(),
                           r_same=mean(rs).tolist(), residual=mean(resid).tolist(), none=mean(rn).tolist(),
                           residual_min=np.nanmin(resid, 0).tolist(), residual_max=np.nanmax(resid, 0).tolist(),
                           n_finite=ok.sum(0).tolist(),
                           masked_clusters=float(np.mean(fm_c)), masked_none=float(np.mean(fm_n)))
            if kk:
                R[name]["k"] = dict(mean=float(np.mean(kk)), min=float(np.min(kk)), max=float(np.max(kk)))
            line = lambda key: "  ".join(f"{R[name][key][j]:.3f}" for j in PICK)
            P(f"{LABEL[name]}: masked {np.mean(fm_c):.3f} with clusters, {np.mean(fm_n):.3f} without"
              + (f"; taper factor sqrt(k) {np.sqrt(np.mean(kk)):.4f} (k {np.min(kk):.3f}-{np.max(kk):.3f})" if kk else ""))
            P("   r_before " + line("r_before"))
            P("   r_after  " + line("r_after"))
            P("   r_same   " + line("r_same"))
            P("   residual " + line("residual") + "   (tiles with a value: "
              + " ".join(str(R[name]["n_finite"][j]) for j in PICK) + ")")
            P("   none     " + line("none"))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(root, "noise_summary.txt"), "w").write(txt + "\n")

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    json.dump(clean(res), open(os.path.join(root, "noise_summary.json"), "w"), indent=1)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
