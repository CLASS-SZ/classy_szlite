"""Cut 10-degree tiles of MSS2 noise for noise-model tests, and measure its non-Gaussianity
(own code; numpy, scipy, astropy).

    MSS2_RAW=/mnt/ceph/users/bbolliet/mapsim/mss2/raw MSS2_OUT=/mnt/ceph/users/bbolliet/mapsim/mss2 \
        python -I benchmarks/mapsim/mss2_tiles.py

MSS2_RAW holds the copied products: maps/ (sobs_RC1.r01_LAT_mission_* noise maps and coadd
inverse variance), sims/ (mnms fdw simulations) and masks/ (Kaper's 70% Galactic mask); on
Perlmutter set MSS2_RAW="" to read them in place. Needs astropy (on rusty: the
sz-poisson-blending environment).

Tiles: 1200 x 1200 pixels of 0.5' in the native CAR grid (rows = dec, columns = RA; a
tile is 600' x 600' cos(dec) on the sky), fully observed in all six bands and >= 90%
inside Kaper's 70% Galactic mask, centred on dec -35..+5 deg. Rows of the full maps are
read once per declination band and all tiles of that band are cut from them. Every 2'
grid (raw f030/f040, their inverse variance, the three mnms tubes) has CRPIX2 = 1890.5
against 7560.5 at 0.5', so 2' row k holds 0.5' rows 4k..4k+3: the 2' rows are taken
from the 0.5' rows with integer arithmetic (rounding the tile's mean declination, which
sits on a 2' pixel edge, put them one row south in the southern bands).

Stored for N_STORE tiles (one .npz each, float32), each with a 120' margin around the
tile so that surrogates can be built on a larger cut-out and cropped:
  raw_hf  (4 splits, 4 bands f090-f290, 1680, 1680)   raw MSS2 noise, Stokes I (margin 240)
  raw_lf  (4 splits, 2 bands f030-f040, 420, 420)      raw MSS2 noise at its native 2' (margin 60)
  mnms    (20, 6 bands, 332, 332)                      mnms "fdw" sims at 2' with a 16-pixel
                                                       margin; sims 0-4 of sets (splits) 0-3
  ivar2   (6 bands, 420, 420)                          coadd inverse variance at 2'
                                                       (4x4 block means for f090-f290)
  obs2    (6 bands, 420, 420) uint8                    1 where every 0.5' pixel is observed
  mask2   (420, 420)                                   70% Galactic mask at 2' (block means)
  meta    JSON string: centre, the slices of every product in its own grid, the grids'
          reference pixels, row declinations, sims, band order.

Diagnostics for N_DIAG tiles (JSON), f090 and f150, in ell bands 300-1000, 1000-3000,
3000-5000 (the mnms range), for the 4 raw splits, the 20 mnms sims (2', central 300 x
300), and a Gaussian reference for every one of these maps (white noise shaped by that
map's 2D amplitude: same anisotropic spectrum, stationary, Gaussian). Every map is
first high-passed locally (minus its Gaussian smoothing, FWHM 30', reflected at the
edges), so that the noise below ell ~ 250, which is 100-10,000 times the in-band level
and differs between raw and mnms, cannot leak into the bands through the taper:
  K     excess kurtosis of the band-filtered map divided by its local rms (Gaussian
        smoothing of f^2, FWHM 60'): non-Gaussianity beyond depth variations. The
        normalisation biases K (about -0.5 at ell 300-1000, where the window holds few
        modes), so K is only read against the same map's Gaussian reference
  S     skewness of the same
  tail  fraction of |z| > 4 rms (printed in units of 1e-5; read against the reference)
  K0    excess kurtosis without the local normalisation (depth variations included)
  A     power with |k_dec| > 2|k_RA| over power with |k_RA| > 2|k_dec| (scan stripes)
  bp    mean band power, C_ell in uK^2 sr
Edges: 30' cosine taper; statistics over the interior (60' margin dropped). The summary
gives per-tile paired differences (map minus its own reference; raw minus mnms) with
errors from their tile-to-tile scatter.
"""
import os
import json
import time

import numpy as np
import scipy.fft as sfft
from astropy.io import fits
from scipy.ndimage import gaussian_filter

RAW = os.environ.get("MSS2_RAW", "/mnt/ceph/users/bbolliet/mapsim/mss2/raw")
if RAW:
    R, SIMS = os.path.join(RAW, "maps"), os.path.join(RAW, "sims")
    MASK = os.path.join(RAW, "masks", "mss0002_mask_enhanced_mask_70.fits")
else:                                    # Perlmutter, in place
    R = "/global/cfs/cdirs/sobs/sims/mss-0002/RC1.r01"
    SIMS = "/global/cfs/cdirs/sobs/v4_sims/mbs/mbs_s0015_20240504/sims"
    MASK = "/global/cfs/projectdirs/sobs/users/kaper/mss-0002/masks/mss0002_mask_enhanced_mask_70.fits"
OUT = os.environ.get("MSS2_OUT", os.path.join(os.environ.get("SCRATCH", "."), "mss2_tiles"))
LF, HF = ["f030", "f040"], ["f090", "f150", "f230", "f290"]
BANDS = LF + HF
SPLITS = [f"split{i}" for i in range(4)]
TUBE = {"f030": ("lf", 0), "f040": ("lf", 1), "f090": ("mf", 0), "f150": ("mf", 1), "f230": ("uhf", 0), "f290": ("uhf", 1)}
PAIR = {"lf": ("f030", "f040"), "mf": ("f090", "f150"), "uhf": ("f230", "f290")}
SIMS_PER_SET = 5
N_STORE, N_DIAG = 32, 64
DEC_CENTRES = [-35.0, -25.0, -15.0, -5.0, 5.0]
NT, NL, MARG = 1200, 300, 16
MH, ML = 240, 60                    # stored margins: 120' at 0.5' (HF) and at 2' (LF, ivar, mask)
HP_FWHM = 30.0                      # arcmin: local high-pass before the statistics
LBANDS = [(300, 1000), (1000, 3000), (3000, 5000)]
DIAG_BANDS = ["f090", "f150"]
ARC = np.pi / 180.0 / 60.0
W = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
RNG = np.random.default_rng(20261118)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f} s]", *a, flush=True)


def hdu(path):
    return fits.open(path, memmap=True)[0]


def raw_path(band, split, kind):
    return f"{R}/sobs_RC1.r01_LAT_mission_{band}_4way_{split}_{kind}_car.fits"


def sim_path(tube, s, n):
    fa, fb = PAIR[tube]
    return f"{SIMS}/so_lat_mbs_mss0002_fdw_{tube}_{tube}_{fa}_{tube}_{fb}_lmax5400_4way_set{s}_noise_sim_map{n:04d}.fits"


def rows_dec(h):
    return h["CRVAL2"] + (np.arange(h["NAXIS2"]) + 1 - h["CRPIX2"]) * h["CDELT2"]


def cols_ra(h):
    return np.mod(h["CRVAL1"] + (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) * h["CDELT1"], 360.0)


def pix_of(h, ra, dec):
    dra = (ra - h["CRVAL1"] + 180.0) % 360.0 - 180.0
    return (int(round(h["CRPIX2"] - 1 + (dec - h["CRVAL2"]) / h["CDELT2"])),
            int(round(h["CRPIX1"] - 1 + dra / h["CDELT1"])))


def taper(ny, nx, m):
    def t(n):
        v = np.ones(n)
        s = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
        v[:m], v[-m:] = s, s[::-1]
        return v
    return np.outer(t(ny), t(nx))


def kgrid(ny, nx, pix_arcmin, dec_c):
    dy, dx = pix_arcmin * ARC, pix_arcmin * ARC * np.cos(np.radians(dec_c))
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)[None, :]
    return ky, kx, np.hypot(ky, kx)


def diag(m, pix_arcmin, dec_c):
    ny, nx = m.shape
    w = taper(ny, nx, int(round(30.0 / pix_arcmin)))
    m = (m - np.sum(m * w) / np.sum(w)) * w
    X = sfft.rfft2(m, workers=W)
    ky, kx, L = kgrid(ny, nx, pix_arcmin, dec_c)
    sig = 60.0 * ARC / np.sqrt(8 * np.log(2))
    smooth = np.exp(-0.5 * (L * sig) ** 2)
    e = int(round(60.0 / pix_arcmin))
    P = np.abs(X) ** 2
    wt = np.full(P.shape, 2.0)
    wt[:, 0] = 1.0
    if nx % 2 == 0:
        wt[:, -1] = 1.0
    out = []
    for lo, hi in LBANDS:
        sel = (L >= lo) & (L < hi)
        f = sfft.irfft2(X * sel, s=(ny, nx), workers=W)
        v = sfft.irfft2(sfft.rfft2(f * f, workers=W) * smooth, s=(ny, nx), workers=W)
        fi, vi = f[e:-e, e:-e], v[e:-e, e:-e]
        z = fi / np.sqrt(np.maximum(vi, 1e-30 * vi.max()))
        z2 = np.mean(z * z)
        K = float(np.mean(z ** 4) / z2 ** 2 - 3.0)
        S = float(np.mean(z ** 3) / z2 ** 1.5)
        tail = float(np.mean(np.abs(z) > 4.0 * np.sqrt(z2)))
        f2 = np.mean(fi * fi)
        K0 = float(np.mean(fi ** 4) / f2 ** 2 - 3.0)
        dec_dom = sel & (np.abs(ky) > 2 * np.abs(kx))
        ra_dom = sel & (np.abs(kx) > 2 * np.abs(ky))
        A = float(np.sum((P * wt)[dec_dom]) / np.sum((P * wt)[ra_dom]))
        omega = (pix_arcmin * ARC) ** 2 * np.cos(np.radians(dec_c))
        bp = float(np.sum((P * wt)[sel]) / np.sum(wt[sel]) * omega / (ny * nx * np.mean(w ** 2)))
        out.append([K, S, tail, K0, A, bp])
    return out


def highpass(m, pix_arcmin, dec_c):
    """m minus its Gaussian smoothing (FWHM 30' on the sky; RA pixels are cos(dec) narrower)."""
    s = HP_FWHM / pix_arcmin / np.sqrt(8 * np.log(2))
    return m - gaussian_filter(m, (s, s / np.cos(np.radians(dec_c))), mode="reflect")


def gauss_like(m, pix_arcmin):
    """Gaussian, stationary reference with the tile's 2D spectrum (the tapered map's Fourier
    amplitudes times a white field), scaled to the map's own power."""
    ny, nx = m.shape
    w = taper(ny, nx, int(round(30.0 / pix_arcmin)))
    amp = np.abs(sfft.rfft2((m - m.mean()) * w, workers=W))
    g = sfft.irfft2(sfft.rfft2(RNG.standard_normal((ny, nx)), workers=W) * amp, s=(ny, nx), workers=W)
    return g / np.sqrt(ny * nx * np.mean(w ** 2))


def main():
    os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)
    h090 = hdu(raw_path("f090", "coadd", "sky_ivar")).header
    h030 = hdu(raw_path("f030", "coadd", "sky_ivar")).header
    hsim = {t: hdu(sim_path(t, 0, 0)).header for t in PAIR}
    hmask = hdu(MASK).header
    assert all(hmask[k] == h090[k] for k in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2", "CRVAL1", "CDELT1", "CDELT2"))
    dec090, ra090 = rows_dec(h090), cols_ra(h090)
    for h in [h030] + list(hsim.values()):          # 2' row k holds 0.5' rows 4k..4k+3
        assert h["CRPIX2"] == (h090["CRPIX2"] + 1.5) / 4 and abs(h["CDELT2"] - 4 * h090["CDELT2"]) < 1e-12

    # ---- candidates per declination band
    cands = []
    for dec_c in DEC_CENTRES:
        rc, _ = pix_of(h090, 180.0, dec_c)
        r0 = 4 * int(round(rc / 4)) - NT // 2
        ivs = {b: np.asarray(hdu(raw_path(b, "coadd", "sky_ivar")).data[r0:r0 + NT, :], dtype=np.float32) for b in HF}
        msk = np.asarray(hdu(MASK).data[r0:r0 + NT, :], dtype=np.float32)
        for c0 in range(NT, h090["NAXIS1"] - 2 * NT + 1, NT):
            ok = all(np.all(ivs[b][:, c0:c0 + NT] > 0) for b in HF)
            frac = float(np.mean(msk[:, c0:c0 + NT] > 0.5))
            if not ok or frac < 0.9:
                continue
            ra_c = float(ra090[c0 + NT // 2])
            dc = float(np.mean(dec090[r0:r0 + NT]))
            k0 = r0 // 4
            _, mc = pix_of(h030, ra_c, dc)
            iv_lf = [np.asarray(hdu(raw_path(b, "coadd", "sky_ivar")).data[k0:k0 + NL, mc - NL // 2:mc + NL // 2]) for b in LF]
            if all(np.all(v > 0) for v in iv_lf):
                cands.append(dict(dec_band=dec_c, r0=r0, c0=c0, ra=ra_c, dec=dc, mask_frac=frac))
        log(f"dec {dec_c:+.0f}: {sum(c['dec_band'] == dec_c for c in cands)} candidates")
    order = RNG.permutation(len(cands))
    # spread the selection across declination bands: round-robin over bands in random order
    by_band = {d: [i for i in order if cands[i]["dec_band"] == d] for d in DEC_CENTRES}
    picked = []
    while len(picked) < min(N_DIAG, len(cands)):
        for d in DEC_CENTRES:
            if by_band[d] and len(picked) < N_DIAG:
                picked.append(by_band[d].pop(0))
    stored = set(picked[:N_STORE])
    log(f"{len(cands)} candidates; {len(picked)} for diagnostics, {len(stored)} stored")

    res = dict(lbands=LBANDS, stats=["K", "S", "tail", "K0", "A", "bp"], tiles=[])
    for dec_c in DEC_CENTRES:
        idx = [i for i in picked if cands[i]["dec_band"] == dec_c]
        if not idx:
            continue
        r0 = cands[idx[0]]["r0"]
        k0 = r0 // 4
        # slabs of full rows for this band, with the margins (raw[...][MH + i] is tile row i)
        hrows, lrows = slice(r0 - MH, r0 + NT + MH), slice(k0 - ML, k0 + NL + ML)
        raw = {(b, s): np.asarray(hdu(raw_path(b, s, "noise_map")).data[0, hrows, :], dtype=np.float32)
               for b in HF for s in SPLITS}
        ivhf = {b: np.asarray(hdu(raw_path(b, "coadd", "sky_ivar")).data[hrows, :], dtype=np.float32) for b in HF}
        msk = np.asarray(hdu(MASK).data[hrows, :], dtype=np.float32)
        rawlf = {(b, s): np.asarray(hdu(raw_path(b, s, "noise_map")).data[0, lrows, :], dtype=np.float32)
                 for b in LF for s in SPLITS}
        ivlf = {b: np.asarray(hdu(raw_path(b, "coadd", "sky_ivar")).data[lrows, :], dtype=np.float32) for b in LF}
        sim_list = [[s_, n_] for s_ in range(4) for n_ in range(SIMS_PER_SET)]
        simslab = {}
        for tube in PAIR:
            for (s_, n_) in sim_list:
                d = hdu(sim_path(tube, s_, n_)).data
                for q in range(2):
                    simslab[(tube, q, s_, n_)] = np.asarray(d[q, 0, 0, k0 - MARG:k0 + NL + MARG, :], dtype=np.float32)
        log(f"dec {dec_c:+.0f}: slabs read (raw {len(raw)}, raw 2' {len(rawlf)}, sims {len(simslab)})")
        for i in idx:
            c = cands[i]
            c0, ra_c, dc = c["c0"], c["ra"], c["dec"]
            _, mc = pix_of(h030, ra_c, dc)
            m0 = mc - NL // 2
            lf_cols = slice(m0 - ML, m0 + NL + ML)
            hcols = slice(c0 - MH, c0 + NT + MH)
            tile = (slice(MH, MH + NT), slice(c0, c0 + NT))                 # tile in the HF slabs
            raw_lf = np.stack([np.stack([rawlf[(b, s)][:, lf_cols] for b in LF]) for s in SPLITS])
            sims = []
            for (s_, n_) in sim_list:
                per_band = []
                for b in BANDS:
                    tube, q = TUBE[b]
                    _, m = pix_of(hsim[tube], ra_c, dc)
                    per_band.append(simslab[(tube, q, s_, n_)][:, m - NL // 2 - MARG:m - NL // 2 + NL + MARG])
                sims.append(np.stack(per_band))
            sims = np.stack(sims)                                           # (20, 6, 332, 332)
            assert sims.shape[-2:] == (NL + 2 * MARG, NL + 2 * MARG) and raw_lf.shape[-2:] == (NL + 2 * ML, NL + 2 * ML)
            entry = dict(index=int(i), ra=ra_c, dec=dc, mask_frac=c["mask_frac"], stored=i in stored, diag={})
            for b in DIAG_BANDS:
                bi = BANDS.index(b)
                rmaps = [highpass(raw[(b, s)][tile], 0.5, dc) for s in SPLITS]
                mmaps = [highpass(sims[j, bi, MARG:-MARG, MARG:-MARG], 2.0, dc) for j in range(sims.shape[0])]
                entry["diag"][b] = dict(raw=[diag(m, 0.5, dc) for m in rmaps],
                                        mnms=[diag(m, 2.0, dc) for m in mmaps],
                                        gauss_raw=[diag(gauss_like(m, 0.5), 0.5, dc) for m in rmaps],
                                        gauss_mnms=[diag(gauss_like(m, 2.0), 2.0, dc) for m in mmaps])
            if i in stored:
                raw_hf = np.stack([np.stack([raw[(b, s)][:, hcols] for b in HF]) for s in SPLITS])
                block = lambda a: a.reshape(a.shape[0] // 4, 4, a.shape[1] // 4, 4).mean((1, 3))
                bmin = lambda a: a.reshape(a.shape[0] // 4, 4, a.shape[1] // 4, 4).min((1, 3))
                ivar2 = np.stack([ivlf[b][:, lf_cols] for b in LF] + [block(ivhf[b][:, hcols]) for b in HF])
                obs2 = np.stack([(ivlf[b][:, lf_cols] > 0) for b in LF] + [bmin(ivhf[b][:, hcols] > 0) for b in HF]).astype(np.uint8)
                meta = dict(ra=ra_c, dec=dc, margin_hf=MH, margin_lf=ML, margin_mnms=MARG,
                            slices=dict(hf=[int(r0 - MH), int(r0 + NT + MH), int(c0 - MH), int(c0 + NT + MH)],
                                        lf=[int(k0 - ML), int(k0 + NL + ML), int(m0 - ML), int(m0 + NL + ML)],
                                        mnms={t: [int(k0 - MARG), int(k0 + NL + MARG),
                                                  int(pix_of(hsim[t], ra_c, dc)[1] - NL // 2 - MARG),
                                                  int(pix_of(hsim[t], ra_c, dc)[1] - NL // 2 + NL + MARG)] for t in PAIR}),
                            grids=dict(hf=[h090["CRPIX1"], h090["CRPIX2"], h090["CDELT1"], h090["CDELT2"]],
                                       lf=[h030["CRPIX1"], h030["CRPIX2"], h030["CDELT1"], h030["CDELT2"]],
                                       **{f"mnms_{t}": [hsim[t]["CRPIX1"], hsim[t]["CRPIX2"], hsim[t]["CDELT1"], hsim[t]["CDELT2"]] for t in PAIR}),
                            obs_margin=[float(o.mean()) for o in obs2],
                            dec_rows_hf=dec090[r0:r0 + NT].tolist(), ra_cols_hf=ra090[c0:c0 + NT].tolist(),
                            bands_hf=HF, bands_lf=LF, bands_mnms=BANDS, splits=SPLITS, sims=sim_list,
                            note=("slices are [row0, row1, col0, col1] in each product's own grid, margins included; "
                                  "dec is the tile centre, ra the RA of tile column 600 (half a pixel from the centre); "
                                  "the f090-f290 mnms tubes sit on the 4x4 block grid of the 0.5' maps (CRPIX1 5400.625), "
                                  "raw f030/f040 and the lf tube 1.5 fine pixels away (CRPIX1 5401); Stokes I"))
                np.savez(os.path.join(OUT, "tiles", f"tile_{i:03d}.npz"), raw_hf=raw_hf, raw_lf=raw_lf, mnms=sims,
                         ivar2=ivar2, obs2=obs2, mask2=block(msk[:, hcols]), meta=json.dumps(meta))
            res["tiles"].append(entry)
            log(f"tile {i:3d} (ra {ra_c:6.1f}, dec {dc:+6.1f}) done{' and stored' if i in stored else ''}")
        with open(os.path.join(OUT, "tile_diagnostics.json"), "w") as fh:
            json.dump(res, fh)
        del raw, ivhf, msk, rawlf, ivlf, simslab

    # ---- summary: per-tile paired differences, errors from their tile-to-tile scatter
    def per_tile(src, b, li, si):
        return np.array([np.mean([r[li][si] for r in t["diag"][b][src]]) for t in res["tiles"]])

    def fmt(x, scale=1.0):
        x = np.asarray(x) * scale
        return f"{x.mean():+.4f}±{x.std(ddof=1) / np.sqrt(x.size):.4f}"

    for b in DIAG_BANDS:
        for li, (lo, hi) in enumerate(LBANDS):
            line = []
            for si, name in enumerate(res["stats"]):
                sc = 1e5 if name == "tail" else 1.0
                r, m = per_tile("raw", b, li, si), per_tile("mnms", b, li, si)
                gr, gm = per_tile("gauss_raw", b, li, si), per_tile("gauss_mnms", b, li, si)
                if name == "bp":
                    line.append(f"mnms/raw power {fmt(m / r)}")
                    continue
                line.append(f"{name}{' [1e-5]' if sc != 1 else ''}: raw-ref {fmt(r - gr, sc)} mnms-ref {fmt(m - gm, sc)} "
                            f"raw-mnms {fmt(r - m, sc)} (ref raw {np.mean(gr) * sc:+.4f})")
            log(f"{b} ell {lo}-{hi}: " + " | ".join(line))
    log("saved", OUT)


if __name__ == "__main__":
    main()
