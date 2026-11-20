"""Per-tile noise spectra of MSS2 in true sky multipoles, for the simulator's noise generator (M6b).

    MSS2_RAW=/mnt/ceph/users/bbolliet/mapsim/mss2/raw MSS2_OUT=/mnt/ceph/users/bbolliet/mapsim/mss2 \\
        python -I benchmarks/mapsim/mss2_spectra.py [--check] [--bands B ...]

Needs astropy (on rusty: the sz-poisson-blending environment), not JAX, so the three small numpy
helpers it shares with mss2_finder_test.py (taper, fourier_upsample, the margin-weighted Fourier
transform) are repeated here.

For every candidate tile of mss2_tiles.py (10 deg, fully observed in all six bands, >= 90% inside
the 70% Galactic mask; 107 at dec -40 to +10), and for the plain mean of the four splits and for
each split, this measures the binned 6 x 6 auto and cross spectra of the six channels:

  * on the 1680^2 cut-out (the tile with a 120' margin), weighted by a cosine taper over the margin
    times the observed pixels, mean removed, normalised by the weight's rms;
  * in true sky multipoles: the maps' pixels are 0.5' in dec but 0.5' cos(dec) in RA, so a mode's
    multipole is hypot(k_dec, k_RA / cos(dec)), with dec the tile centre's (it changes by +-6% in
    cos(dec) across a tile at dec -35);
  * as the per-mode power the simulator needs on its square 0.5' pixels: the measured per-mode power
    (pixel units) times cos(dec), the ratio of the true pixel area to the simulator's;
  * in bins of DL = 50 in multipole up to the simulator's corner, 21600 sqrt(2); each rfft column
    inside the half-plane counts for two modes, the k_RA = 0 and Nyquist columns for one;
  * 27 and 39 GHz have no power above the 2' grid's Nyquist (multipole 5400 along dec): above
    LF_FLOOR[1] their auto spectra are set to their mean over LF_FLOOR, and their cross spectra to 0.

The mismatch the tests found: treating the maps' pixels as square put up to 1/sqrt(cos dec) = 1.105
too much noise at dec -35 in amplitude, and stretched multipoles along RA by up to 1/cos(dec).

--check tests the conversion on synthetic fields built from the maps' physical pixel sizes, at dec
-35, -15 and +5, through the same functions that write the spectra: isotropic fields with a known
spectrum (all 36 entries, and the RA- and dec-dominated sectors with the multipole mapping), white
noise on rows whose pixel area follows their own cos(dec), and a square-pixel control that must fail
at dec -35 (see check()).

Output: $MSS2_OUT/spectra/spectra.npz (float32 spectra C[tile, set, bin, 6, 6], sets = mean,
split0..split3; the tiles' positions; the multipole bins; the mode counts; the tiles' median
inverse-variance depth per band) and spectra.json (per-tile white levels and positions).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mss2_tiles as mt  # noqa: E402

ARC = np.pi / 180.0 / 60.0
PIX = 0.5 * ARC                          # the simulator's (and the maps' dec) pixel, rad
NT, MH, NL, ML = mt.NT, mt.MH, mt.NL, mt.ML
NB = NT + 2 * MH                         # 1680
DL = 50.0
ELL_MAX = 21600.0 * np.sqrt(2.0)         # the simulator's corner multipole (0.5' pixels)
NBIN = int(np.ceil(ELL_MAX / DL))
LF_FLOOR = (3000.0, 4500.0)              # fit window, and the start of the 27/39 GHz floor
SETS = ["mean"] + mt.SPLITS
WORKERS = 3
OUTD = os.path.join(mt.OUT, "spectra")


def log(*a):
    print(f"[{time.time() - T0:7.1f} s]", *a, flush=True)


T0 = time.time()


def taper(n, m):
    """1 inside, a cosine to ~0 over the outer m pixels (as mss2_finder_test.py)."""
    t = np.ones(n)
    s = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    t[:m], t[-m:] = s, s[::-1]
    return np.outer(t, t)


T_MARGIN = taper(NB, MH)


def fourier_upsample(a, f, shift_rows=0.0):
    """Band-limited (zero-padding) upsampling by an integer factor (as mss2_finder_test.py), then a
    shift by shift_rows fine rows: out[j] = upsampled[j - shift_rows]. Coarse sample i lands on fine
    row f i. A 2' row k of the maps is centred on 0.5' row 4k + 1.5 (CRPIX2 1890.5 against 7560.5;
    the columns line up), so the 27/39 GHz cut-outs need shift_rows = 1.5 to sit on the 0.5' maps."""
    ny, nx = a.shape
    A = np.fft.fft2(a)
    B = np.zeros((ny * f, nx * f), complex)
    hy, hx = ny // 2, nx // 2
    B[:hy, :hx], B[:hy, -hx:] = A[:hy, :hx], A[:hy, -hx:]
    B[-hy:, :hx], B[-hy:, -hx:] = A[-hy:, :hx], A[-hy:, -hx:]
    if shift_rows:
        B *= np.exp(-2j * np.pi * np.fft.fftfreq(ny * f)[:, None] * shift_rows)
    return np.real(np.fft.ifft2(B)) * f * f


def margin_ft(maps, wts):
    """rfft of the weighted, mean-removed maps, normalised by rms(weight) (as mss2_finder_test.py)."""
    out = []
    for m, w in zip(maps, wts):
        mu = np.sum(m * w) / np.sum(w)
        out.append(sfft.rfft2((m - mu) * w, workers=WORKERS) / np.sqrt(np.mean(w ** 2)))
    return np.stack(out)


def mode_geometry(cosd, n=NB):
    """True multipole of every rfft mode of an n^2 grid of the maps' pixels at a tile centre with
    cos(dec) = cosd, the bin of each mode (-1 beyond the simulator's corner) and its weight (2 inside
    the half-plane; n is even, so the last column is Nyquist)."""
    k = 2 * np.pi * np.fft.fftfreq(n, d=PIX)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=PIX)
    ell = np.hypot(k[:, None], kx[None, :] / cosd)
    b = np.floor(ell / DL).astype(np.int64)
    b[ell >= ELL_MAX] = -1
    b[0, 0] = -1                                        # the mean
    w = np.full(ell.shape, 2.0)
    w[:, 0] = 1.0
    w[:, -1] = 1.0
    return ell, b.ravel(), w.ravel()


def binned_spectra(X, cosd, geo=None):
    """(NBIN, 6, 6) per-mode power on the simulator's square 0.5' pixels, binned in true multipole,
    and the number of modes per bin. X: (6, n, n/2+1) from margin_ft on an n^2 grid."""
    n = X.shape[1]
    ell, b, w = geo if geo is not None else mode_geometry(cosd, n)
    ok = b >= 0
    bb, ww = b[ok], w[ok]
    norm = np.bincount(bb, weights=ww, minlength=NBIN)
    C = np.zeros((NBIN, 6, 6))
    Xf = X.reshape(6, -1)[:, ok]
    for i in range(6):
        for j in range(i, 6):
            p = np.real(Xf[i] * np.conj(Xf[j])) / n ** 2 * cosd
            C[:, i, j] = C[:, j, i] = np.bincount(bb, weights=ww * p, minlength=NBIN) / np.maximum(norm, 1e-30)
    return C, norm


def row_area_weights(dec_rows, cosd):
    """sqrt(cos(dec_row) / cos(dec_centre)) per row: on CAR maps a pixel's area follows its own row's
    cos(dec). Multiplying each row by this before the transform makes one centre cos(dec) exact for
    noise of uniform sky depth (without it the white power comes out 0.2-0.35% high)."""
    return np.sqrt(np.cos(np.radians(dec_rows)) / cosd)[:, None]


LEVEL_WINDOWS = ((3000.0, 4500.0), (6000.0, 20000.0))      # 27/39 GHz (their floor's window); 90-280 GHz
TAPER_TILE = taper(NT, 60)                                   # the finder's 30' taper on the tile


def level_ratio(C_cut, C_tile, nm_cut, nm_tile):
    """Per channel, the tile's own power over the cut-out's in a high-ell window (the cut-out's 120'
    margin carries 24% of the weight and mixes its surroundings' depth into the level)."""
    r = np.ones(6)
    for i in range(6):
        lo, hi = LEVEL_WINDOWS[0] if i < 2 else LEVEL_WINDOWS[1]
        s = (ELL_C >= lo) & (ELL_C < hi)
        r[i] = (np.sum(C_tile[s, i, i] * nm_tile[s]) / np.sum(nm_tile[s])) / (np.sum(C_cut[s, i, i] * nm_cut[s]) / np.sum(nm_cut[s]))
    return r


def lf_floor(C, ell_c):
    """27/39 GHz: above LF_FLOOR[1] the auto spectra take their mean over LF_FLOOR, the cross spectra 0."""
    fit = (ell_c >= LF_FLOOR[0]) & (ell_c < LF_FLOOR[1])
    hi = ell_c >= LF_FLOOR[1]
    white = [float(np.mean(C[fit, i, i])) for i in range(2)]
    for i in range(2):
        C[hi, i, :] = 0.0
        C[hi, :, i] = 0.0
        C[hi, i, i] = white[i]
    return C, white


ELL_C = (np.arange(NBIN) + 0.5) * DL


def white_level(C, lo=8000.0, hi=12000.0):
    """uK-arcmin per channel from the mean per-mode power over lo <= ell < hi (square 0.5' pixels)."""
    s = (ELL_C >= lo) & (ELL_C < hi)
    return [float(np.sqrt(np.mean(C[s, i, i]) * PIX ** 2) / ARC) for i in range(6)]


# ------------------------------------------------------------------ the synthetic check
def check(n_real=16, decs=(-35.0, -15.0, 5.0), wide=500.0, seed=11):
    """Tests of the conversion, on synthetic fields built from the maps' physical pixel sizes (not from
    mode_geometry), measured by the same functions that write the spectra:
      1. isotropic 6-channel fields with a known spectrum: binned_spectra's output, rebinned by mode
         count into bins of `wide` over ell = 1000-20000, against the truth at each fine bin's centre:
         the mean ratio within 0.3% for the auto and for the cross spectra, at most 3% of the auto bin
         values beyond 3 standard errors (16 realisations and correlated channels give 1-2%);
      2. the same in the RA- and dec-dominated sectors, with each mode in the bin mode_geometry gives
         it and the truth at that bin's centre, so a wrong multipole mapping shows;
      3. white noise of uniform sky depth on rows whose pixel area follows their own cos(dec), with
         the row weights: the white level within 0.1% (0.2-0.35% high without them);
      4. a control: the same field measured as if the pixels were square must miss by more than 10%
         at dec -35 (it cannot fail near the equator)."""
    rng = np.random.default_rng(seed)
    corr = np.full((6, 6), 0.2) + np.eye(6) * 0.8
    corr[0, 1] = corr[1, 0] = corr[2, 3] = corr[3, 2] = corr[4, 5] = corr[5, 4] = 0.6
    amp = np.array([3.0, 2.0, 1.0, 1.2, 1.6, 2.5])
    Lc = np.linalg.cholesky(corr)

    def shape(ell):
        return 1.0 + (np.maximum(ell, 30.0) / 1500.0) ** -2.5
    edges = np.arange(1000.0, 20000.0 + wide, wide)
    nw = len(edges) - 1
    wb = np.digitize(ELL_C, edges) - 1
    inw = (wb >= 0) & (wb < nw)
    truth = shape(ELL_C)[:, None, None] * (amp[:, None] * amp[None, :] * corr)[None]
    iu = np.triu_indices(6)
    auto = iu[0] == iu[1]
    results = []
    for dec in decs:
        cosd = np.cos(np.radians(dec))
        geo = mode_geometry(cosd)
        dx, dy = PIX * cosd, PIX
        k = 2 * np.pi * np.fft.fftfreq(NB, d=dy)
        kx = 2 * np.pi * np.fft.rfftfreq(NB, d=dx)
        ell = np.hypot(k[:, None], kx[None, :])
        g = np.sqrt(shape(ell) * PIX ** 2 / (dx * dy))       # per-mode power on the maps' pixels: C_ell / (dx dy)
        sectors = {"RA": (np.abs(kx[None, :]) > 2 * np.abs(k[:, None])).ravel(),
                   "dec": (np.abs(k[:, None]) > 2 * np.abs(kx[None, :])).ravel()}
        Rall = np.zeros((n_real, nw, iu[0].size))
        Rsec = {nm_: np.zeros((n_real, nw, 6)) for nm_ in sectors}
        for r in range(n_real):
            W = np.stack([sfft.rfft2(rng.standard_normal((NB, NB)), workers=WORKERS) for _ in range(6)])
            F = np.einsum("ij,jkl->ikl", Lc * amp[:, None], W) * g[None]
            maps = np.stack([sfft.irfft2(f, s=(NB, NB), workers=WORKERS) for f in F])
            X = margin_ft(maps, np.broadcast_to(T_MARGIN, (6, NB, NB)))
            C, nm = binned_spectra(X, cosd, geo)                          # the production path
            num = np.zeros((nw, iu[0].size))
            den = np.zeros((nw, iu[0].size))
            np.add.at(num, wb[inw], (C[inw][:, iu[0], iu[1]] * nm[inw, None]))
            np.add.at(den, wb[inw], (truth[inw][:, iu[0], iu[1]] * nm[inw, None]))
            Rall[r] = num / den
            Xf = X.reshape(6, -1)
            for name, sec in sectors.items():                              # modes in mode_geometry's bins
                bb = geo[1].copy()
                bb[~sec] = -1
                okk = bb >= 0
                wgt = geo[2][okk]
                for i in range(6):
                    p = np.abs(Xf[i, okk]) ** 2 / NB ** 2 * cosd
                    num_b = np.bincount(bb[okk], weights=wgt * p, minlength=NBIN)
                    den_b = np.bincount(bb[okk], weights=wgt * truth[bb[okk], i, i], minlength=NBIN)
                    Rsec[name][r, :, i] = np.bincount(wb[inw], weights=num_b[inw], minlength=nw) / \
                        np.bincount(wb[inw], weights=den_b[inw], minlength=nw)
        ok = True
        line = []
        m, se = Rall.mean(0), Rall.std(0, ddof=1) / np.sqrt(n_real)
        b_auto, b_cross = float(m[:, auto].mean() - 1), float(m[:, ~auto].mean() - 1)
        out3 = float(np.mean(np.abs(m[:, auto] - 1) > 3 * se[:, auto]))
        ok &= abs(b_auto) < 0.003 and abs(b_cross) < 0.003 and out3 <= 0.03
        line.append(f"autos {1 + b_auto:.4f}, crosses {1 + b_cross:.4f}, beyond 3 SE {100 * out3:.1f}%")
        for name in sectors:
            ms, ss = Rsec[name].mean(0), Rsec[name].std(0, ddof=1) / np.sqrt(n_real)
            bias, o3 = float(ms.mean() - 1), float(np.mean(np.abs(ms - 1) > 3 * ss))
            ok &= abs(bias) < 0.003 and o3 <= 0.03
            line.append(f"{name} sector {1 + bias:.4f} ({100 * o3:.1f}% beyond 3 SE)")
        # 4. the square-pixel control on the last realisation
        Csq, nsq = binned_spectra(X, 1.0, mode_geometry(1.0))
        sel = (ELL_C >= 1000) & (ELL_C < 20000)
        sq = float(np.sum(Csq[sel, 3, 3] * nsq[sel]) / np.sum(truth[sel, 3, 3] * nsq[sel]))
        if dec <= -30:
            ok &= abs(sq - 1) > 0.10
        line.append(f"as if square pixels {sq:.3f}")
        # 3. rows whose pixel area follows their own cos(dec): white noise of uniform sky depth
        dec_rows = dec + (np.arange(NB) - NB / 2 + 0.5) * 0.5 / 60.0
        wrat = []
        for use_w in (True, False):
            acc = 0.0
            for _ in range(4):
                sig = 1.0 / np.sqrt(np.cos(np.radians(dec_rows)))[:, None]   # pixel variance ~ 1/area
                f = rng.standard_normal((6, NB, NB)) * sig[None]
                if use_w:
                    f = f * row_area_weights(dec_rows, cosd)[None]
                Cw, nmw = binned_spectra(margin_ft(f, np.broadcast_to(T_MARGIN, (6, NB, NB))), cosd, geo)
                s2 = (ELL_C >= 2000) & (ELL_C < 20000)
                acc += np.mean([np.sum(Cw[s2, i, i] * nmw[s2]) / np.sum(nmw[s2]) for i in range(6)]) / 4
            wrat.append(acc)                       # truth: per-mode power 1 on square pixels (sky depth fixed)
        ok &= abs(wrat[0] - 1) < 0.001
        line.append(f"row areas: white level {wrat[0]:.4f} with the row weights, {wrat[1]:.4f} without")
        results.append(ok)
        log(f"check dec {dec:+.0f} ({n_real} realisations, ell 1000-20000): " + "; ".join(line) + (" -- passed" if ok else " -- FAILED"))
    ok = all(results)
    log("check " + ("PASSED" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------ the measurement
def measure_band(args):
    dec_c, cands = args
    h090, h030, hsim = mt.headers()
    r0 = cands[0]["r0"]
    k0 = r0 // 4
    hrows, lrows = slice(r0 - MH, r0 + NT + MH), slice(k0 - ML, k0 + NL + ML)
    raw = {(b, s): np.asarray(mt.hdu(mt.raw_path(b, s, "noise_map")).data[0, hrows, :], dtype=np.float32)
           for b in mt.HF for s in mt.SPLITS}
    ivhf = {b: np.asarray(mt.hdu(mt.raw_path(b, "coadd", "sky_ivar")).data[hrows, :], dtype=np.float32) for b in mt.HF}
    rawlf = {(b, s): np.asarray(mt.hdu(mt.raw_path(b, s, "noise_map")).data[0, lrows, :], dtype=np.float32)
             for b in mt.LF for s in mt.SPLITS}
    ivlf = {b: np.asarray(mt.hdu(mt.raw_path(b, "coadd", "sky_ivar")).data[lrows, :], dtype=np.float32) for b in mt.LF}
    dec090 = mt.rows_dec(h090)
    log(f"dec {dec_c:+.0f}: slabs read, {len(cands)} tiles")
    out = []
    for c in cands:
        c0, ra_c, dc = c["c0"], c["ra"], c["dec"]
        cosd = float(np.cos(np.radians(dc)))
        geo, geo_t = mode_geometry(cosd), mode_geometry(cosd, NT)
        _, mc = mt.pix_of(h030, ra_c, dc)
        m0 = mc - NL // 2
        lf_cols = slice(m0 - ML, m0 + NL + ML)
        hcols = slice(c0 - MH, c0 + NT + MH)
        obs = np.stack([np.kron((ivlf[b][:, lf_cols] > 0).astype(float), np.ones((4, 4))) for b in mt.LF]
                       + [(ivhf[b][:, hcols] > 0).astype(float) for b in mt.HF])
        wts = obs * T_MARGIN[None]
        rw = row_area_weights(dec090[r0 - MH:r0 + NT + MH], cosd)        # (NB, 1)
        maps = {}
        for s in mt.SPLITS:
            lf = [fourier_upsample(rawlf[(b, s)][:, lf_cols].astype(np.float64), 4, shift_rows=1.5) for b in mt.LF]
            maps[s] = np.stack(lf + [raw[(b, s)][:, hcols].astype(np.float64) for b in mt.HF]) * rw[None]
        maps["mean"] = np.mean([maps[s] for s in mt.SPLITS], axis=0)
        C = np.zeros((len(SETS), NBIN, 6, 6))
        ratios, whites = np.zeros((len(SETS), 6)), np.zeros((len(SETS), 6))
        wt_tile = obs[:, MH:MH + NT, MH:MH + NT] * TAPER_TILE[None]
        for i_s, s in enumerate(SETS):
            Cs, nm = binned_spectra(margin_ft(maps[s], wts), cosd, geo)
            Ct, nt_ = binned_spectra(margin_ft(maps[s][:, MH:MH + NT, MH:MH + NT], wt_tile), cosd, geo_t)
            r = level_ratio(Cs, Ct, nm, nt_)
            Cs = Cs * np.sqrt(r[:, None] * r[None, :])[None]            # the tile's own level, the cut-out's shape
            Cs, lfw = lf_floor(Cs, ELL_C)
            C[i_s], ratios[i_s], whites[i_s] = Cs, r, white_level(Cs)
        out.append(dict(index=int(c["index"]), ra=ra_c, dec=dc, dec_band=dec_c, r0=int(c["r0"]), c0=int(c0),
                        mask_frac=c["mask_frac"], C=C.astype(np.float32), nmodes=nm.astype(np.float32),
                        white=whites, level_ratio=ratios))
        log(f"tile {c['index']:3d} (ra {ra_c:6.1f}, dec {dc:+6.1f}): white levels of the four-split mean "
            + ", ".join(f"{b} {w:.2f}" for b, w in zip(mt.BANDS, whites[0])) + " uK-arcmin; tile/cut-out level "
            + " ".join(f"{x:.4f}" for x in ratios[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only run the synthetic-field check")
    ap.add_argument("--bands", type=float, nargs="*", default=None, help="declination bands to measure (default all)")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    os.makedirs(OUTD, exist_ok=True)
    h090, h030, _ = mt.headers()
    cands = mt.candidates(h090, h030)
    for i, c in enumerate(cands):
        c["index"] = i                                   # the tile number used by mss2_tiles.py
    bands = args.bands if args.bands else mt.DEC_CENTRES
    assert all(b in mt.DEC_CENTRES for b in bands), f"bands must be among {mt.DEC_CENTRES}"
    jobs = [(d, [c for c in cands if c["dec_band"] == d]) for d in bands]
    with Pool(len(jobs)) as pool:
        res = sum(pool.map(measure_band, jobs), [])
    res.sort(key=lambda r: r["index"])
    stem = "spectra" if not args.bands else "spectra_partial_" + "_".join(f"{b:+.0f}" for b in bands)
    if stem == "spectra":
        assert [r["index"] for r in res] == list(range(len(cands))), "a full run must hold every candidate"
    np.savez(os.path.join(OUTD, stem + ".npz"),
             C=np.stack([r["C"] for r in res]), nmodes=np.stack([r["nmodes"] for r in res]),
             index=np.array([r["index"] for r in res]), ra=np.array([r["ra"] for r in res]),
             dec=np.array([r["dec"] for r in res]), dec_band=np.array([r["dec_band"] for r in res]),
             white=np.stack([r["white"] for r in res]), level_ratio=np.stack([r["level_ratio"] for r in res]),
             ell_edges=np.arange(NBIN + 1) * DL,
             meta=json.dumps(dict(sets=SETS, bands=mt.BANDS, dl=DL, ell_max=ELL_MAX, lf_floor=LF_FLOOR,
                                  units="per-mode power on the simulator's square 0.5' pixels [uK^2]; "
                                        "C_ell [uK^2 sr] = C * (0.5 arcmin in rad)^2",
                                  geometry=("true multipole hypot(k_dec, k_RA/cos(dec_centre)); each row weighted by "
                                            "sqrt(cos(dec_row)/cos(dec_centre)); power times cos(dec_centre); the shape "
                                            "from the cut-out with its 120' margin, the level (per channel) from the "
                                            "tile under the finder's 30' taper; 27/39 GHz shifted 1.5 fine rows"),
                                  white="uK-arcmin at ell 8000-12000, [tile, set, band]",
                                  level_ratio="tile over cut-out power, [tile, set, band]")))
    with open(os.path.join(OUTD, stem + ".json"), "w") as fh:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k not in ("C", "nmodes")}
                   for r in res], fh)
    w = np.array([r["white"][0] for r in res])
    lr = np.array([r["level_ratio"][0] for r in res])
    log("tile/cut-out level ratios (four-split mean), rms over tiles: " + ", ".join(
        f"{b} {np.sqrt(np.mean((x - 1) ** 2)):.4f}" for b, x in zip(mt.BANDS, lr.T)))
    log(f"saved {len(res)} tiles to {OUTD}; white levels of the four-split mean, median over tiles: "
        + ", ".join(f"{b} {v:.1f}" for b, v in zip(mt.BANDS, np.median(w, axis=0))) + " uK-arcmin")


if __name__ == "__main__":
    main()
