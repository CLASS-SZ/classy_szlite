"""Does the noise model matter for the finder? MSS2 against mnms and Gaussian surrogates.

    python benchmarks/mapsim/mss2_finder_test.py [--tiles N] [--part I/K] [--q-mask Q]
    python benchmarks/mapsim/mss2_finder_test.py --geometry [--part I/K]   # cut-out against periodic tile
    python benchmarks/mapsim/mss2_finder_test.py --summary                 # merge the parts, print

Outputs go to $MAPSIM_OUT/mss2/finder_test (finder_test_qmask<Q> with --q-mask). On the
2026-11-19 run the four parts ran side by side, each pinned to four of the job's CPUs.
--q-mask sets the masking threshold on the cut-outs. The masking pass thresholds the first-pass
q, whose cut-out/periodic ratio is 1.042-1.066 (the final pass's is 1.055-1.077), so for noise
alone about 4.2 matches 4 on periodic tiles, and 4.3 masks less (0.063 of the tile against
0.094-0.102). The 2026-11-19 run at 4.3 is therefore lighter than production for noise alone.
With a full cluster sky, 4.3 brings the masked fraction back near the calibration point
(mss2_datavector_test.py checks this against periodic tiles). diagnostics_data.py converts the
counts to the periodic tiles' q scale.

Input: the MSS2 tiles cut by ``mss2_tiles.py`` ($MAPSIM_OUT/mss2/tiles): raw MSS2 noise
(4 splits, native CAR pixels, Stokes I) with a 120' margin, 20 mnms "fdw" simulations at
2', the coadd inverse variance at 2'.

Kinds of six-channel noise tile, at the same sky positions, at split depth (one split) and
at coadd depth (the mean of the four splits):
  raw       the MSS2 noise itself (split: 4 per tile; coadd: 1)
  mnms      mnms simulations (split: 20 per tile, sims 0-4 of each split; coadd: 5, the mean
            of sets 0-3 for each sim index), Fourier-upsampled 2' -> 0.5' after apodising the
            16-pixel margin (so the patch is periodic and nothing rings at its edges). Above
            ell = 4000-4800 (true ell on the sky) a cos/sin crossover to a fresh gaussmod
            field of the matching raw map: mnms stops at ell 5400 and the finder uses f090-f290
            up to ell ~ 8000. Those multipoles are therefore an oracle, not a model.
  gauss     Gaussian and stationary, with the raw map's 2D auto and cross spectra: the phases
            of the margin-tapered 1680^2 cut-out rotated by one random field for all channels,
            then cropped (4 per raw split; 4 at coadd depth)
  gaussmod  the same on the raw map divided by its depth pattern (coadd inverse
            variance^-1/2, 8' smoothing), multiplied back: depth changes kept, else Gaussian
  iso       Gaussian, stationary and isotropic, with the raw map's azimuthally averaged 6 x 6
            auto and cross spectra (bins of 51 in ell): the simulator's kind of noise, with
            the MSS2 spectra (4 per raw split; 4 at coadd depth)
Building the surrogates on the cut-out with its margin and cropping them makes them, like the
raw tile, non-periodic: the finder's taper then treats them alike (built on the tile itself,
the taper's leakage of the large-scale noise was counted twice).

Every map gets the same Gaussian lensed-CMB realisation of its tile (the simulator's: CMB
emulator C_ell, SO beams), so that the finder's channel weights are the production ones, and
then the finder's 30' cosine taper. The CAR pixels are used as if square (all kinds share
the geometry). The finder is the production one: masking pass, final pass at q > 4, peaks
merged within 1.5' over all scales (sm.merge_local_max). Peaks within 60' of an edge are
dropped.

Measured per map:
  counts    merged interior peaks with q > 4, 4.5, 5 and 6, with their position and scale
  qstats    the final q cube over the interior, per scale: rms, excess kurtosis, P(q > 3),
            P(q > 4), P(q < -3)
  sigma     the finder's sigma_j, first pass and final pass
and, at coadd depth, for 64 injected clusters per tile (the inner 8 x 8 sites of a 60' grid,
jittered by +-3', random sub-pixel offsets; theta500 log-uniform in 0.5'-6'; y0 = q_nom
sigma_raw(theta500) with q_nom uniform in [2, 10] and sigma_raw from the raw coadd's
final-pass sigma_j; the same clusters in every coadd map of the tile):
  qn_ref - qs_ref   the noise q at the nearest pixel and scale (qs: the noiseless signal
                    through the map's own final-pass covariance)
  q_det             the q of the highest kept peak within 1.5' (NaN if none)

The q scale: the finder estimates its covariance from the tapered tile (mean taper^2 =
0.879), so interior q is up to ~1.065 times too dispersed at small scales, less at large
ones (see qstats rms for iso). All kinds share this; the counts compare kinds and are not
the survey's false-detection rate.

The summary pairs the kinds tile by tile (the mean over a tile's maps of a kind, against
raw), with errors from a bootstrap over tiles, for all tiles and per declination band.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter, zoom
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classy_szlite.sim import instrument as ins, mmf, simulator as simu, paint as pt  # noqa: E402
from classy_szlite.sim import summaries as sm  # noqa: E402
from classy_szlite.sim.flatsky import N, NF, ARCMIN  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2")
EDGE_TAPER, EDGE_DROP = 60, 120          # pixels (30', 60')
MH, ML, MM = 240, 60, 16                 # stored margins: 0.5' HF, 2' LF/ivar, 2' mnms
NB = N + 2 * MH                          # 1680: the cut-out with its margin
L_STITCH = (4000.0, 4800.0)              # true ell (mnms is good to ~4400, stops at 5400)
DEPTH_FWHM = 8.0                         # arcmin: smoothing of the depth pattern
THRESH = (4.0, 4.5, 5.0, 6.0)
N_SURR = 4                               # gauss/gaussmod/iso draws per raw map
ISO_DL = 2                               # iso bins: 2 grid steps of the 1680 grid (51 in ell)
SITES, SPACING_F, JIT_F = 10, NF // 10, 12
R_MATCH = 1.5                            # arcmin
WORKERS = 4


def rfft2(a):
    return sfft.rfft2(a, workers=WORKERS)


def irfft2(a, n):
    return sfft.irfft2(a, s=(n, n), workers=WORKERS)


def taper(n, m):
    t = np.ones(n)
    s = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    t[:m], t[-m:] = s, s[::-1]
    return np.outer(t, t)


TAPER = taper(N, EDGE_TAPER)
T_MARGIN = taper(NB, MH)                 # 1 on the tile, cosine to 0 over the margin


def crop(a):
    return a[..., MH:MH + N, MH:MH + N]


def fourier_upsample(a, f):
    """Band-limited (zero-padding) upsampling of a 2D patch by an integer factor."""
    ny, nx = a.shape
    A = np.fft.fft2(a)
    B = np.zeros((ny * f, nx * f), complex)
    hy, hx = ny // 2, nx // 2
    B[:hy, :hx], B[:hy, -hx:] = A[:hy, :hx], A[:hy, -hx:]
    B[-hy:, :hx], B[-hy:, -hx:] = A[-hy:, :hx], A[-hy:, -hx:]
    return np.real(np.fft.ifft2(B)) * f * f


def true_ell(dec_deg, n):
    """|ell| on the sky for an n x n 0.5' CAR patch at declination dec (RA pixels are cos(dec)
    narrower than dec pixels), on the rfft half-plane."""
    k = 2 * np.pi * np.fft.fftfreq(n, d=0.5 * ARCMIN)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=0.5 * ARCMIN)
    return np.hypot(k[:, None], kx[None, :] / np.cos(np.radians(dec_deg)))


def stitch_weight(lt):
    return np.clip((lt - L_STITCH[0]) / (L_STITCH[1] - L_STITCH[0]), 0.0, 1.0)


def mnms_to_tile(coarse, dec_deg, lowpass):
    """A 2' mnms patch with its margin -> the 0.5' tile. The margin is apodised to zero first,
    so the patch is periodic: Fourier upsampling and the stitch low-pass then do not ring on
    the patch edges (a low-pass applied to the cropped, non-periodic tile put ~10% excess
    power at ell 4000-5000 along the axes). With lowpass, the mnms part of the stitch,
    cos(pi t / 2) in true ell, is applied on the large grid before cropping."""
    n = coarse.shape[-1]
    mc = (n - 300) // 2
    A = np.fft.fft2((coarse - coarse.mean()) * taper(n, mc))
    nf, h = 4 * n, n // 2
    B = np.zeros((nf, nf), complex)
    B[:h, :h], B[:h, -h:], B[-h:, :h], B[-h:, -h:] = A[:h, :h], A[:h, -h:], A[-h:, :h], A[-h:, -h:]
    if lowpass:
        k = 2 * np.pi * np.fft.fftfreq(nf, d=0.5 * ARCMIN)
        lt = np.hypot(k[:, None], k[None, :] / np.cos(np.radians(dec_deg)))
        B *= np.cos(0.5 * np.pi * stitch_weight(lt))
    m = 4 * mc
    return (np.real(np.fft.ifft2(B)) * 16.0)[m:m + N, m:m + N]


def depth_shape(ivar2, obs2):
    """sigma pattern on the 1680^2 grid from the 2' coadd inverse variance: unobserved pixels
    filled with the median, 8' Gaussian smoothing (removes the 2' block steps and the
    hit-count scatter, which would spread the multiplied field's power at high ell),
    bilinear upsampling; rms 1 on the tile."""
    iv = np.where(obs2 > 0, ivar2, np.median(ivar2[obs2 > 0]))
    s2 = gaussian_filter(1.0 / np.sqrt(np.maximum(iv, 1e-30 * iv.max())),
                         DEPTH_FWHM / 2.0 / np.sqrt(8 * np.log(2)), mode="nearest")
    s = zoom(s2, 4, order=1, mode="nearest", grid_mode=True)
    return s / np.sqrt(np.mean(crop(s) ** 2))


def margin_ft(maps, wts):
    """rfft of the weighted, mean-removed maps on the 1680 grid, normalised by rms(weight)."""
    out = []
    for m, w in zip(maps, wts):
        mu = np.sum(m * w) / np.sum(w)
        out.append(rfft2((m - mu) * w) / np.sqrt(np.mean(w ** 2)))
    return np.stack(out)


def surrogate(maps, wts, rng):
    """Gaussian, stationary field on the 1680 grid with the margin-tapered maps' 2D auto and
    cross spectra: one random phase field (from a real white field, so the rfft symmetry
    holds on the k_x = 0 and Nyquist columns) for all channels."""
    ph = rfft2(rng.standard_normal((NB, NB)))
    ph /= np.maximum(np.abs(ph), 1e-30)
    X = margin_ft(maps, wts) * ph[None]
    return np.stack([irfft2(x, NB) for x in X])


ISO_BINS = None                          # (order, bounds) of the iso ell bins on the 1680 grid


def iso_bins():
    k = 2 * np.pi * np.fft.fftfreq(NB, d=0.5 * ARCMIN)
    kx = 2 * np.pi * np.fft.rfftfreq(NB, d=0.5 * ARCMIN)
    L = np.hypot(k[:, None], kx[None, :])
    b = np.floor(L / (ISO_DL * 2 * np.pi / (NB * 0.5 * ARCMIN))).astype(np.int64).ravel()
    order = np.argsort(b, kind="stable")
    return order, np.searchsorted(b[order], np.arange(b.max() + 2))


def iso_field(maps, wts, rng):
    """Gaussian, stationary, isotropic field on the 1680 grid with the maps' azimuthally
    averaged 6 x 6 spectra (square-pixel ell, bins of ISO_DL grid steps)."""
    order, bounds = ISO_BINS
    Xf = margin_ft(maps, wts).reshape(6, -1)
    W = np.stack([rfft2(rng.standard_normal((NB, NB))) for _ in range(6)]).reshape(6, -1) / NB
    Y = np.zeros_like(Xf)
    for b in range(len(bounds) - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        if idx.size == 0:
            continue
        xb = Xf[:, idx]
        C = np.real(xb @ xb.conj().T) / idx.size
        C += np.eye(6) * (1e-9 * np.trace(C) / 6 + 1e-300)
        Y[:, idx] = np.linalg.cholesky(C) @ W[:, idx]
    Y[:, 0] = 0.0
    return np.stack([irfft2(y.reshape(NB, NB // 2 + 1), NB) for y in Y])


def iso_spectra(maps, wts):
    """Binned 6 x 6 per-mode power (pixel units) of the margin-tapered maps, bins of ISO_DL
    steps of the 1680 grid (51 in square-pixel ell)."""
    order, bounds = ISO_BINS
    Xf = margin_ft(maps, wts).reshape(6, -1)
    Cs = np.zeros((len(bounds) - 1, 6, 6))
    for b in range(len(bounds) - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        if idx.size:
            xb = Xf[:, idx]
            Cs[b] = np.real(xb @ xb.conj().T) / idx.size / NB ** 2
    return Cs


def iso_from_spectra(Cs, n, rng):
    """Gaussian, isotropic, periodic field on an n x n 0.5' grid with the binned power Cs."""
    k = 2 * np.pi * np.fft.fftfreq(n, d=0.5 * ARCMIN)
    kx = 2 * np.pi * np.fft.rfftfreq(n, d=0.5 * ARCMIN)
    b = np.minimum(np.floor(np.hypot(k[:, None], kx[None, :]) / (ISO_DL * 2 * np.pi / (NB * 0.5 * ARCMIN))),
                   len(Cs) - 1).astype(np.int64).ravel()
    W = np.stack([rfft2(rng.standard_normal((n, n))) for _ in range(6)]).reshape(6, -1)
    Y = np.zeros_like(W)
    for bb in np.unique(b):
        idx = np.nonzero(b == bb)[0]
        C = Cs[bb] + np.eye(6) * (1e-9 * np.trace(Cs[bb]) / 6 + 1e-300)
        Y[:, idx] = np.linalg.cholesky(C) @ W[:, idx]
    Y[:, 0] = 0.0
    return np.stack([irfft2(y.reshape(n, n // 2 + 1), n) for y in Y])


def high_field(maps, wts, sh, t_big, rng):
    """The sin(pi t / 2) part of the mnms stitch: a fresh gaussmod field of the raw maps,
    filtered on the 1680 grid and cropped (HF channels only)."""
    g = surrogate(maps / sh, wts, rng) * sh
    return crop(np.stack([irfft2(np.sin(0.5 * np.pi * t_big) * rfft2(g[c]), NB) for c in range(2, 6)]))


def make_injections(rng, lnth15, lnsig15):
    """64 clusters at the inner 8 x 8 sites of a 60' grid (fine pixels), the optbias recipe."""
    gy, gx = np.meshgrid(np.arange(1, SITES - 1), np.arange(1, SITES - 1), indexing="ij")
    n = gy.size
    iy = (gy.ravel() * SPACING_F + SPACING_F // 2 + rng.integers(-JIT_F, JIT_F + 1, n)) % NF
    ix = (gx.ravel() * SPACING_F + SPACING_F // 2 + rng.integers(-JIT_F, JIT_F + 1, n)) % NF
    oy, ox = rng.random(n), rng.random(n)
    th = np.exp(rng.uniform(np.log(0.5), np.log(6.0), n))
    qn = rng.uniform(2.0, 10.0, n)
    y0 = qn * np.exp(np.interp(np.log(th), lnth15, lnsig15))
    cy = np.rint((iy + oy) / 2.0).astype(np.int64) % N
    cx = np.rint((ix + ox) / 2.0).astype(np.int64) % N
    jref = np.argmin(np.abs(np.log(th)[:, None] - lnth15[None, :]), axis=1)
    inj = dict(iy=iy.astype(np.int32), ix=ix.astype(np.int32), oy=oy, ox=ox, y0=y0, theta500=th)
    aux = dict(q_nom=qn, theta500=th, cy=cy.astype(np.int32), cx=cx.astype(np.int32), jref=jref.astype(np.int32))
    return inj, aux


def make_finder(st, cfg):
    fs = st.fs
    e = EDGE_DROP

    def passes(X):
        """find_clusters (n_iter = 1), keeping the final q cube and covariance."""
        q0, s0, C0 = mmf.q_cube(X, fs)
        pk0 = mmf.find_peaks(q0, cfg.peak_cap, cfg.q_mask)
        mask, _ = mmf.cluster_mask(pk0, fs, cfg.q_mask)
        C, f = mmf.masked_cov(X, mask, C0, fs)
        q, s1, _ = mmf.q_cube(X, fs, C=C)
        pk = mmf.find_peaks(q, cfg.peak_cap, cfg.q_final)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        return q, s0, s1, C, pk, keep, 1.0 - f, pk0.overflow | pk.overflow

    @jax.jit
    def noise_run(X):
        q, s0, s1, C, pk, keep, fm, ovf = passes(X)
        qi = q[:, e:N - e, e:N - e].astype(jnp.float64)
        m2 = jnp.mean(qi * qi, axis=(1, 2))
        qstats = jnp.stack([jnp.sqrt(m2), jnp.mean(qi ** 4, axis=(1, 2)) / m2 ** 2 - 3.0,
                            jnp.mean(qi > 3.0, axis=(1, 2)), jnp.mean(qi > 4.0, axis=(1, 2)),
                            jnp.mean(qi < -3.0, axis=(1, 2))], axis=1)
        return dict(s0=s0, s1=s1, j=pk.j, iy=pk.iy, ix=pk.ix, q=pk.q, ok=pk.valid & keep,
                    qstats=qstats, f_masked=fm, overflow=ovf)

    @jax.jit
    def inj_run(X, Xs, cy, cx, jref):
        q, s0, s1, C, pk, keep, fm, ovf = passes(X)
        qs, _, _ = mmf.q_cube(Xs, fs, C=C)
        dy = sm._wrap(pk.iy[None, :] - cy[:, None], N)
        dx = sm._wrap(pk.ix[None, :] - cx[:, None], N)
        m = (pk.valid & keep)[None, :] & ((dy * dy + dx * dx) <= int((R_MATCH / 0.5) ** 2))
        qd = jnp.max(jnp.where(m, pk.q[None, :], -jnp.inf), axis=1)
        return dict(qn_ref=q[jref, cy, cx], qs_ref=qs[jref, cy, cx],
                    q_det=jnp.where(jnp.isfinite(qd), qd, jnp.nan), overflow=ovf)

    return noise_run, inj_run


def run_tile(path, st, noise_run, inj_run):
    t0 = time.time()
    ti = int(os.path.basename(path)[5:8])
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    dec = float(meta["dec"])
    rng = np.random.default_rng([2026, 11, 19, ti])
    raw_hf, raw_lf, mnms, ivar2, obs2 = d["raw_hf"], d["raw_lf"], d["mnms"], d["ivar2"], d["obs2"]
    assert raw_hf.shape[-1] == NB and raw_lf.shape[-1] == NB // 4 and mnms.shape[-1] == 300 + 2 * MM
    wts = np.stack([np.kron(o, np.ones((4, 4))) for o in obs2.astype(float)]) * T_MARGIN[None]
    sh = np.stack([depth_shape(ivar2[b], obs2[b]) for b in range(6)])
    t_big = stitch_weight(true_ell(dec, NB))

    # raw maps on the big grid: LF upsampled from 2' (their own margin keeps the ringing off the tile)
    raw_big = []
    for s in range(4):
        lf = [fourier_upsample(raw_lf[s, b].astype(np.float64), 4) for b in range(2)]
        raw_big.append(np.stack(lf + [raw_hf[s, b].astype(np.float64) for b in range(4)]))
    coadd_big = np.mean(raw_big, axis=0)

    # one lensed-CMB realisation for every map of the tile (the simulator's recipe)
    rc = np.random.default_rng([2026, 11, 19, ti, 99])
    Wc = rfft2(rc.standard_normal((N, N)))
    Bf, ca = np.asarray(st.B_full, dtype=np.float64), np.asarray(st.cmb_amp, dtype=np.float64)
    cmb = np.stack([irfft2(Bf[c] * ca * Wc, N) for c in range(6)])

    sims = meta["sims"]
    res = {"split": {}, "coadd": {}}

    def record(depth, kind, maps, inj=None):
        X = jnp.asarray(rfft2((maps + cmb) * TAPER[None]).astype(np.complex64))
        o = jax.device_get(noise_run(X))
        ok = np.asarray(o["ok"])
        iy, ix, j, q = (np.asarray(o[k])[ok] for k in ("iy", "ix", "j", "q"))
        inner = (iy >= EDGE_DROP) & (iy < N - EDGE_DROP) & (ix >= EDGE_DROP) & (ix < N - EDGE_DROP)
        iy, ix, j, q = iy[inner], ix[inner], j[inner], q[inner]
        r = res[depth].setdefault(kind, dict(counts=[], peaks=[], qstats=[], s0=[], s1=[], f_masked=[], overflow=[]))
        r["counts"].append([int(np.sum(q > t)) for t in THRESH])
        r["peaks"].append([[int(a), int(b), int(c), round(float(v), 3)] for a, b, c, v in zip(iy, ix, j, q)])
        r["qstats"].append(np.asarray(o["qstats"]).tolist())
        r["s0"].append(np.asarray(o["s0"]).tolist())
        r["s1"].append(np.asarray(o["s1"]).tolist())
        r["f_masked"].append(float(o["f_masked"]))
        r["overflow"].append(bool(o["overflow"]))
        if inj is not None:
            Xs_np, aux = inj
            sig = np.stack([irfft2(Xs_np[c], N) for c in range(6)])
            Xi = jnp.asarray(rfft2((maps + cmb + sig) * TAPER[None]).astype(np.complex64))
            oi = jax.device_get(inj_run(Xi, jnp.asarray(Xs_np.astype(np.complex64)), jnp.asarray(aux["cy"]),
                                        jnp.asarray(aux["cx"]), jnp.asarray(aux["jref"])))
            ri = res[depth].setdefault(kind + "_inj", dict(qn_ref=[], qs_ref=[], q_det=[], overflow=[]))
            for k in ("qn_ref", "qs_ref", "q_det"):
                ri[k].append(np.asarray(oi[k], dtype=float).tolist())
            ri["overflow"].append(bool(oi["overflow"]))
        return o

    # ---- split depth
    for s in range(4):
        record("split", "raw", crop(raw_big[s]))
        for _ in range(N_SURR):
            record("split", "gauss", crop(surrogate(raw_big[s], wts, rng)))
            record("split", "gaussmod", crop(surrogate(raw_big[s] / sh, wts, rng) * sh))
            record("split", "iso", crop(iso_field(raw_big[s], wts, rng)))
    for jx, (s, n) in enumerate(sims):
        hi = high_field(raw_big[s], wts, sh, t_big, rng)
        chans = [mnms_to_tile(mnms[jx, b], dec, lowpass=False) for b in range(2)]
        chans += [mnms_to_tile(mnms[jx, b], dec, lowpass=True) + hi[b - 2] for b in range(2, 6)]
        record("split", "mnms", np.stack(chans))

    # ---- coadd depth: noise only for the raw coadd's sigma_j, then every kind with the clusters
    o_raw = record("coadd", "raw", crop(coadd_big))
    lnth15 = np.log(np.asarray(mmf.THETAS15, dtype=float))
    inj, aux = make_injections(rng, lnth15, np.log(np.asarray(o_raw["s1"], dtype=float)))
    yk = pt.synth_direct(inj, st.T_tab, jnp.float32)
    Xs_np = np.asarray(st.a_full * yk[None], dtype=np.complex128)
    res["coadd"]["injections"] = {k: np.asarray(v).tolist() for k, v in aux.items()}
    Xs_aux = (Xs_np, aux)

    def inj_only(kind, maps):
        """The noise-only record and the run with the clusters."""
        record("coadd", kind, maps, inj=Xs_aux)

    # the raw coadd with the clusters (its noise-only record is above)
    sig = np.stack([irfft2(Xs_np[c], N) for c in range(6)])
    Xi = jnp.asarray(rfft2((crop(coadd_big) + cmb + sig) * TAPER[None]).astype(np.complex64))
    oi = jax.device_get(inj_run(Xi, jnp.asarray(Xs_np.astype(np.complex64)), jnp.asarray(aux["cy"]),
                                jnp.asarray(aux["cx"]), jnp.asarray(aux["jref"])))
    res["coadd"]["raw_inj"] = {k: [np.asarray(oi[k], dtype=float).tolist()] for k in ("qn_ref", "qs_ref", "q_det")}
    res["coadd"]["raw_inj"]["overflow"] = [bool(oi["overflow"])]
    for _ in range(N_SURR):
        inj_only("gauss", crop(surrogate(coadd_big, wts, rng)))
        inj_only("gaussmod", crop(surrogate(coadd_big / sh, wts, rng) * sh))
        inj_only("iso", crop(iso_field(coadd_big, wts, rng)))
    nsim = max(n for _, n in sims) + 1
    for n in range(nsim):
        js = [jx for jx, (s, nn) in enumerate(sims) if nn == n]
        co = np.mean(mnms[js].astype(np.float64), axis=0)
        hi = high_field(coadd_big, wts, sh, t_big, rng)
        chans = [mnms_to_tile(co[b], dec, lowpass=False) for b in range(2)]
        chans += [mnms_to_tile(co[b], dec, lowpass=True) + hi[b - 2] for b in range(2, 6)]
        inj_only("mnms", np.stack(chans))
    res.update(tile=ti, ra=meta["ra"], dec=dec, seconds=round(time.time() - t0, 1))
    return res


def run_tile_geometry(path, st, noise_run, inj_run):
    """The q-scale gap between the test's geometry and the simulator's: the same isotropic
    Gaussian noise (the raw map's binned spectra) as a cut-out with the finder's 30' taper
    (iso_cut, as the iso kind) and as a periodic tile with no taper (iso_per, exactly what
    simulator.simulate_tile runs on); same CMB, same clusters."""
    t0 = time.time()
    ti = int(os.path.basename(path)[5:8])
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    dec = float(meta["dec"])
    rng = np.random.default_rng([2026, 11, 19, ti, 7])
    raw_hf, raw_lf, obs2 = d["raw_hf"], d["raw_lf"], d["obs2"]
    wts = np.stack([np.kron(o, np.ones((4, 4))) for o in obs2.astype(float)]) * T_MARGIN[None]
    raw_big = []
    for s in range(4):
        lf = [fourier_upsample(raw_lf[s, b].astype(np.float64), 4) for b in range(2)]
        raw_big.append(np.stack(lf + [raw_hf[s, b].astype(np.float64) for b in range(4)]))
    coadd_big = np.mean(raw_big, axis=0)
    rc = np.random.default_rng([2026, 11, 19, ti, 99])                     # the main run's CMB
    Wc = rfft2(rc.standard_normal((N, N)))
    Bf, ca = np.asarray(st.B_full, dtype=np.float64), np.asarray(st.cmb_amp, dtype=np.float64)
    cmb = np.stack([irfft2(Bf[c] * ca * Wc, N) for c in range(6)])
    res = {"split": {}, "coadd": {}}

    def record(depth, kind, maps, tapered, inj=None):
        w = TAPER[None] if tapered else 1.0
        o = jax.device_get(noise_run(jnp.asarray(rfft2((maps + cmb) * w).astype(np.complex64))))
        ok = np.asarray(o["ok"])
        iy, ix, j, q = (np.asarray(o[k])[ok] for k in ("iy", "ix", "j", "q"))
        inner = (iy >= EDGE_DROP) & (iy < N - EDGE_DROP) & (ix >= EDGE_DROP) & (ix < N - EDGE_DROP)
        r = res[depth].setdefault(kind, dict(counts=[], peaks=[], qstats=[], s0=[], s1=[], f_masked=[], overflow=[]))
        r["counts"].append([int(np.sum(q[inner] > t)) for t in THRESH])
        r["peaks"].append([[int(a), int(b), int(c), round(float(v), 3)] for a, b, c, v in
                           zip(iy[inner], ix[inner], j[inner], q[inner])])
        r["qstats"].append(np.asarray(o["qstats"]).tolist())
        r["s0"].append(np.asarray(o["s0"]).tolist())
        r["s1"].append(np.asarray(o["s1"]).tolist())
        r["f_masked"].append(float(o["f_masked"]))
        r["overflow"].append(bool(o["overflow"]))
        if inj is not None:
            Xs_np, aux = inj
            sig = np.stack([irfft2(Xs_np[c], N) for c in range(6)])
            oi = jax.device_get(inj_run(jnp.asarray(rfft2((maps + cmb + sig) * w).astype(np.complex64)),
                                        jnp.asarray(Xs_np.astype(np.complex64)), jnp.asarray(aux["cy"]),
                                        jnp.asarray(aux["cx"]), jnp.asarray(aux["jref"])))
            ri = res[depth].setdefault(kind + "_inj", dict(qn_ref=[], qs_ref=[], q_det=[], overflow=[]))
            for k in ("qn_ref", "qs_ref", "q_det"):
                ri[k].append(np.asarray(oi[k], dtype=float).tolist())
            ri["overflow"].append(bool(oi["overflow"]))
        return o

    for s in range(4):
        Cs = iso_spectra(raw_big[s], wts)
        for _ in range(N_SURR):
            record("split", "iso_cut", crop(iso_from_spectra(Cs, NB, rng)), True)
            record("split", "iso_per", iso_from_spectra(Cs, N, rng), False)
    o_raw = record("coadd", "raw", crop(coadd_big), True)
    lnth15 = np.log(np.asarray(mmf.THETAS15, dtype=float))
    inj, aux = make_injections(rng, lnth15, np.log(np.asarray(o_raw["s1"], dtype=float)))
    yk = pt.synth_direct(inj, st.T_tab, jnp.float32)
    Xs_np = np.asarray(st.a_full * yk[None], dtype=np.complex128)
    res["coadd"]["injections"] = {k: np.asarray(v).tolist() for k, v in aux.items()}
    Cs = iso_spectra(coadd_big, wts)
    for _ in range(N_SURR):
        record("coadd", "iso_cut", crop(iso_from_spectra(Cs, NB, rng)), True, inj=(Xs_np, aux))
        record("coadd", "iso_per", iso_from_spectra(Cs, N, rng), False, inj=(Xs_np, aux))
    res.update(tile=ti, ra=meta["ra"], dec=dec, seconds=round(time.time() - t0, 1))
    return res


def pick(paths, n):
    """n tiles spread over the declination bands (round-robin), in a fixed order."""
    by = {}
    for p in paths:
        dec = round(json.loads(str(np.load(p)["meta"]))["dec"])
        by.setdefault(dec, []).append(p)
    out, bands = [], sorted(by)
    while len(out) < n and any(by.values()):
        for b in bands:
            if by[b] and len(out) < n:
                out.append(by[b].pop(0))
    return out


# ------------------------------------------------------------------ summary
def boot(f, arrays, nb=2000, seed=1):
    """f over the tiles and its bootstrap standard error (resampling tiles)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    est = f(*arrays)
    bs = np.array([f(*[a[ix] for a in arrays]) for ix in rng.integers(0, n, (nb, n))])
    return est, np.std(bs, axis=0, ddof=1)


def summary(tiles):
    decs = np.round([t["dec"] for t in tiles]).astype(int)
    bands = sorted(set(decs))
    lines = [f"{len(tiles)} tiles (dec bands: " + ", ".join(f"{b:+d}: {int(np.sum(decs == b))}" for b in bands) + ")"]
    area = ((N - 2 * EDGE_DROP) / N) ** 2
    th = np.asarray(mmf.THETAS15)
    for depth in ("split", "coadd"):
        kinds = [k for k in ("raw", "mnms", "gauss", "gaussmod", "iso") if k in tiles[0][depth]]
        lines.append(f"\n=== {depth} depth: merged noise peaks per interior tile ({area:.2f} of a tile), CMB included")
        per = {k: np.array([np.mean(t[depth][k]["counts"], axis=0) for t in tiles]) for k in kinds}    # (tiles, 4)
        for k in kinds:
            c = per[k]
            cells = []
            for ti_, thr in enumerate(THRESH):
                m, e = boot(lambda a: a.mean(), [c[:, ti_]])
                cells.append(f"q>{thr:g} {m:6.2f}±{e:.2f}")
            rat = []
            if k != "raw":
                for ti_ in range(3):
                    m, e = boot(lambda a, b: a.mean() / max(b.mean(), 1e-9), [c[:, ti_], per["raw"][:, ti_]])
                    rat.append(f"{m:.2f}±{e:.2f}")
            n_maps = len(tiles[0][depth][k]["counts"])
            lines.append(f"  {k:9s} ({n_maps:2d}/tile) " + "  ".join(cells)
                         + (("   kind/raw at q>4, 4.5, 5: " + ", ".join(rat)) if rat else ""))
        lines.append("  median over tiles of the q>5 counts: " + ", ".join(f"{k} {np.median(per[k][:, 2]):.2f}" for k in kinds))
        # by the scale of the peak (q > 4.5): theta500 < 1', 1-2.5', 2.5-6', > 6'
        groups = [(0.0, 1.0), (1.0, 2.5), (2.5, 6.0), (6.0, 99.0)]
        jg = np.digitize(th, [g[1] for g in groups[:-1]])
        lines.append("  q>4.5 peaks by scale (theta < 1', 1-2.5', 2.5-6', > 6'), kind/raw paired by tile:")
        byscale = {}
        for k in kinds:
            byscale[k] = np.array([np.mean([np.bincount(jg[[pk[2] for pk in m if pk[3] > 4.5]], minlength=4)
                                            for m in t[depth][k]["peaks"]], axis=0) for t in tiles])
        for k in kinds:
            cells = []
            for g in range(4):
                if k == "raw":
                    cells.append(f"{byscale[k][:, g].mean():.2f}")
                else:
                    m, e = boot(lambda a, b: a.mean() / max(b.mean(), 1e-9), [byscale[k][:, g], byscale["raw"][:, g]])
                    cells.append(f"{m:.2f}±{e:.2f}")
            lines.append(f"    {k:9s} " + "  ".join(cells) + ("  (counts per tile)" if k == "raw" else "  (/raw)"))
        if depth == "split":
            def split_mean(k, t, s_):
                c = np.asarray(t[depth][k]["counts"], float)
                return c.reshape(4, -1, 4)[s_].mean(0)          # maps are recorded split by split
            lines.append("  per split, q>4 / q>5 (mean over tiles):")
            for k in kinds:
                cells = [f"{np.mean([split_mean(k, t, s_)[0] for t in tiles]):.1f} / {np.mean([split_mean(k, t, s_)[2] for t in tiles]):.2f}"
                         for s_ in range(4)]
                lines.append(f"    {k:9s} " + "   ".join(f"split{s_} {c}" for s_, c in enumerate(cells)))
        worst = np.argsort(-per["raw"][:, 2])[:3]
        lines.append("  raw q>5 is largest in tiles " + ", ".join(
            f"{tiles[i]['tile']} ({per['raw'][i, 2]:.1f}; mnms {per['mnms'][i, 2]:.1f})" for i in worst))
        keep = np.setdiff1d(np.arange(len(tiles)), worst)
        lines.append("  without those 3 tiles, q>4.5 / q>5: " + ", ".join(
            f"{k} {per[k][keep, 1].mean():.2f} / {per[k][keep, 2].mean():.2f}" for k in kinds))
        for b in bands:
            sel = decs == b
            lines.append(f"  dec {b:+3d} ({sel.sum():2d} tiles) q>4 / q>5: "
                         + ", ".join(f"{k} {per[k][sel, 0].mean():.1f} / {per[k][sel, 2].mean():.2f}" for k in kinds))
        s1 = {k: np.array([np.mean(t[depth][k]["s1"], axis=0) for t in tiles]) for k in kinds}
        s0 = {k: np.array([np.mean(t[depth][k]["s0"], axis=0) for t in tiles]) for k in kinds}
        jsel = [int(np.argmin(np.abs(th - x))) for x in (1.0, 3.0, 8.0)]
        for k in kinds:
            r0 = np.mean(s0[k] / s0["raw"], axis=0)
            r1 = np.mean(s1[k] / s1["raw"], axis=0)
            qs = np.mean([np.mean(t[depth][k]["qstats"], axis=0) for t in tiles], axis=0)       # (15, 5)
            lines.append(f"  {k:9s} sigma/raw first {r0.min():.3f}-{r0.max():.3f}, final {r1.min():.3f}-{r1.max():.3f}; q cube "
                         + "; ".join(f"{th[j]:.1f}': rms {qs[j, 0]:.3f} K {qs[j, 1]:+.3f} P(q>4) {qs[j, 3] * 1e5:.2f}e-5" for j in jsel)
                         + f"; masked {np.mean([np.mean(t[depth][k]['f_masked']) for t in tiles]):.3f}")
        ovf = sum(sum(t[depth][k]["overflow"]) for t in tiles for k in kinds)
        if ovf:
            lines.append(f"  WARNING: {ovf} maps overflowed the peak buffer")
    lines.append("\n=== coadd depth, injected clusters (64 per tile; q_nom 2-10, theta500 0.5'-6')")
    kinds = [k for k in ("raw", "mnms", "gauss", "gaussmod", "iso") if k + "_inj" in tiles[0]["coadd"]]
    edges = [2, 3, 4, 5, 6, 8, 11]
    for k in kinds:
        qn = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["qn_ref"]) for t in tiles])
        qs = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["qs_ref"]) for t in tiles])
        qd = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["q_det"]) for t in tiles])
        dq = qn - qs
        kurt = np.mean((dq - dq.mean()) ** 4) / np.var(dq) ** 2 - 3
        comp = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (qs >= a) & (qs < b)
            comp.append(f"{np.mean(np.nan_to_num(qd[s], nan=-np.inf) > 5):.2f}")
        lines.append(f"  {k:9s} noise q at the clusters: mean {dq.mean():+.3f} rms {dq.std():.3f} K {kurt:+.2f} "
                     f"P(|dq|>3) {np.mean(np.abs(dq) > 3):.4f}; P(q_det>5) for qs in "
                     + " ".join(f"[{a},{b})" for a, b in zip(edges[:-1], edges[1:])) + ": " + " ".join(comp))
    return "\n".join(lines)


def summary_numbers(tiles, geo):
    """The numbers the progress page shows (diagnostics_data.py reads summary.json)."""
    th = np.asarray(mmf.THETAS15)
    kinds_all = ("raw", "mnms", "gauss", "gaussmod", "iso")
    out = dict(n_tiles=len(tiles), thresholds=list(THRESH), thetas=th.tolist(), depth={})
    for depth in ("split", "coadd"):
        kinds = [k for k in kinds_all if k in tiles[0][depth]]
        per = {k: np.array([np.mean(t[depth][k]["counts"], axis=0) for t in tiles]) for k in kinds}
        dd = {}
        for k in kinds:
            mean, err, ratio, ratio_err = [], [], [], []
            for i_ in range(len(THRESH)):
                m, e = boot(lambda a: a.mean(), [per[k][:, i_]])
                mean.append(float(m)); err.append(float(e))
                r, re = boot(lambda a, b: a.mean() / max(b.mean(), 1e-9), [per[k][:, i_], per["raw"][:, i_]])
                ratio.append(float(r)); ratio_err.append(float(re))
            s1 = np.mean([np.mean(t[depth][k]["s1"], axis=0) / np.mean(t[depth]["raw"]["s1"], axis=0) for t in tiles], axis=0)
            qs = np.mean([np.mean(t[depth][k]["qstats"], axis=0) for t in tiles], axis=0)
            dd[k] = dict(n_maps=len(tiles[0][depth][k]["counts"]), counts=mean, counts_err=err, over_raw=ratio,
                         over_raw_err=ratio_err, sigma_final_over_raw=s1.tolist(), q_rms=qs[:, 0].tolist(),
                         q_kurt=qs[:, 1].tolist(), p_q4=qs[:, 3].tolist(),
                         f_masked=float(np.mean([np.mean(t[depth][k]["f_masked"]) for t in tiles])))
            if k + "_inj" in tiles[0][depth]:
                qn = np.concatenate([np.ravel(t[depth][k + "_inj"]["qn_ref"]) for t in tiles])
                qsr = np.concatenate([np.ravel(t[depth][k + "_inj"]["qs_ref"]) for t in tiles])
                qd = np.concatenate([np.ravel(t[depth][k + "_inj"]["q_det"]) for t in tiles])
                dq = qn - qsr
                edges = [2, 3, 4, 5, 6, 8, 11]
                comp = []
                for a, b in zip(edges[:-1], edges[1:]):
                    s = (qsr >= a) & (qsr < b)
                    p = float(np.mean(np.nan_to_num(qd[s], nan=-np.inf) > 5))
                    comp.append(dict(lo=a, hi=b, n=int(s.sum()), p=p, se=float(np.sqrt(max(p * (1 - p), 1e-6) / max(s.sum(), 1)))))
                dd[k]["injections"] = dict(noise_q_rms=float(dq.std()), noise_q_mean=float(dq.mean()),
                                           noise_q_kurt=float(np.mean((dq - dq.mean()) ** 4) / np.var(dq) ** 2 - 3),
                                           p_abs_gt3=float(np.mean(np.abs(dq) > 3)), completeness=comp)
        out["depth"][depth] = dd
    if geo:
        g = {}
        for depth in ("split", "coadd"):
            ca = np.array([np.mean(t[depth]["iso_cut"]["counts"], axis=0) for t in geo])
            cb = np.array([np.mean(t[depth]["iso_per"]["counts"], axis=0) for t in geo])
            s1 = np.mean([np.mean(t[depth]["iso_per"]["s1"], axis=0) / np.mean(t[depth]["iso_cut"]["s1"], axis=0) for t in geo], axis=0)
            qa = np.mean([np.mean(t[depth]["iso_cut"]["qstats"], axis=0) for t in geo], axis=0)
            qb = np.mean([np.mean(t[depth]["iso_per"]["qstats"], axis=0) for t in geo], axis=0)
            g[depth] = dict(cut=ca.mean(0).tolist(), periodic=cb.mean(0).tolist(),
                            ratio=[float(boot(lambda x, y: x.mean() / max(y.mean(), 1e-9), [ca[:, i_], cb[:, i_]])[0]) for i_ in range(len(THRESH))],
                            ratio_err=[float(boot(lambda x, y: x.mean() / max(y.mean(), 1e-9), [ca[:, i_], cb[:, i_]])[1]) for i_ in range(len(THRESH))],
                            sigma_periodic_over_cut=s1.tolist(), q_rms_cut=qa[:, 0].tolist(), q_rms_periodic=qb[:, 0].tolist())
        qs_a = np.concatenate([np.ravel(t["coadd"]["iso_cut_inj"]["qs_ref"]) for t in geo])
        qs_b = np.concatenate([np.ravel(t["coadd"]["iso_per_inj"]["qs_ref"]) for t in geo])
        th_i = np.concatenate([np.tile(t["coadd"]["injections"]["theta500"], N_SURR) for t in geo])
        g["qs_periodic_over_cut"] = {f"{x}-{y}": float(np.mean(qs_b[(th_i >= x) & (th_i < y)]) / np.mean(qs_a[(th_i >= x) & (th_i < y)]))
                                     for x, y in ((0.5, 1.0), (1.0, 2.5), (2.5, 6.0))}
        out["geometry"] = g
    return out


def summary_geometry(tiles):
    """Tapered cut-out against periodic, untapered tile, the same isotropic Gaussian noise."""
    th = np.asarray(mmf.THETAS15)
    lines = [f"\n=== geometry: iso_cut (cut-out, 30' taper, as in this test) against iso_per (periodic tile, "
             f"no taper, as in the simulator), {len(tiles)} tiles"]
    for depth in ("split", "coadd"):
        a, b = "iso_cut", "iso_per"
        ca = np.array([np.mean(t[depth][a]["counts"], axis=0) for t in tiles])
        cb = np.array([np.mean(t[depth][b]["counts"], axis=0) for t in tiles])
        cells = []
        for i_, thr in enumerate(THRESH[:3]):
            m, e = boot(lambda x, y: x.mean() / max(y.mean(), 1e-9), [ca[:, i_], cb[:, i_]])
            cells.append(f"q>{thr:g}: cut {ca[:, i_].mean():.2f}, periodic {cb[:, i_].mean():.2f}, ratio {m:.2f}±{e:.2f}")
        lines.append(f"  {depth}: " + "; ".join(cells))
        s1 = np.mean([np.mean(t[depth][b]["s1"], axis=0) / np.mean(t[depth][a]["s1"], axis=0) for t in tiles], axis=0)
        qa = np.mean([np.mean(t[depth][a]["qstats"], axis=0) for t in tiles], axis=0)
        qb = np.mean([np.mean(t[depth][b]["qstats"], axis=0) for t in tiles], axis=0)
        lines.append(f"  {depth}: sigma_j periodic/cut at theta " + " ".join(f"{th[j]:.1f}' {s1[j]:.3f}" for j in range(0, 15, 2)))
        lines.append(f"  {depth}: interior q rms cut / periodic at theta " + " ".join(f"{th[j]:.1f}' {qa[j, 0]:.3f}/{qb[j, 0]:.3f}" for j in range(0, 15, 2)))
        lines.append(f"  {depth}: masked fraction cut {np.mean([np.mean(t[depth][a]['f_masked']) for t in tiles]):.3f}, "
                     f"periodic {np.mean([np.mean(t[depth][b]['f_masked']) for t in tiles]):.3f}")
    edges = [2, 3, 4, 5, 6, 8, 11]
    for k in ("iso_cut", "iso_per"):
        qn = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["qn_ref"]) for t in tiles])
        qs = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["qs_ref"]) for t in tiles])
        qd = np.concatenate([np.ravel(t["coadd"][k + "_inj"]["q_det"]) for t in tiles])
        dq = qn - qs
        comp = [f"{np.mean(np.nan_to_num(qd[(qs >= x) & (qs < y)], nan=-np.inf) > 5):.2f}" for x, y in zip(edges[:-1], edges[1:])]
        lines.append(f"  injections {k}: noise q rms {dq.std():.3f}, mean qs {qs.mean():.3f}; P(q_det>5) for qs in "
                     + " ".join(f"[{x},{y})" for x, y in zip(edges[:-1], edges[1:])) + ": " + " ".join(comp))
    qs_a = np.concatenate([np.ravel(t["coadd"]["iso_cut_inj"]["qs_ref"]) for t in tiles])
    qs_b = np.concatenate([np.ravel(t["coadd"]["iso_per_inj"]["qs_ref"]) for t in tiles])
    th_i = np.concatenate([np.tile(t["coadd"]["injections"]["theta500"], N_SURR) for t in tiles])
    for x, y in ((0.5, 1.0), (1.0, 2.5), (2.5, 6.0)):
        sel = (th_i >= x) & (th_i < y)
        lines.append(f"  same clusters, theta500 {x}'-{y}': noiseless q periodic/cut = {np.mean(qs_b[sel]) / np.mean(qs_a[sel]):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=0, help="number of tiles (0 = all), spread over dec bands")
    ap.add_argument("--part", default="0/1", help="I/K: run every K-th tile starting at I")
    ap.add_argument("--summary", action="store_true", help="merge the parts' outputs and print the summary")
    ap.add_argument("--geometry", action="store_true", help="run the cut-out vs periodic comparison instead")
    ap.add_argument("--q-mask", type=float, default=float(mmf.Q_MASK),
                    help="masking threshold on the finder's q (on these tapered cut-outs about 4.2 matches 4 on "
                         "periodic tiles for noise alone; see the docstring)")
    args = ap.parse_args()
    odir = os.path.join(OUT, "finder_test" if args.q_mask == mmf.Q_MASK else f"finder_test_qmask{args.q_mask:g}")
    os.makedirs(odir, exist_ok=True)
    if args.summary:
        tiles, geo = [], []
        for f in sorted(glob.glob(os.path.join(odir, "part_*.json"))):
            tiles += json.load(open(f))
        for f in sorted(glob.glob(os.path.join(odir, "geometry_part_*.json"))):
            geo += json.load(open(f))
        tiles.sort(key=lambda t: t["tile"])
        geo.sort(key=lambda t: t["tile"])
        txt = summary(tiles) if tiles else ""
        if geo:
            txt += "\n" + summary_geometry(geo)
        print(txt)
        if tiles:
            with open(os.path.join(odir, "summary.json"), "w") as fh:
                json.dump(summary_numbers(tiles, geo), fh)
        open(os.path.join(odir, "summary.txt"), "w").write(txt + "\n")
        return
    paths = sorted(glob.glob(os.path.join(OUT, "tiles", "tile_*.npz")))
    paths = pick(paths, args.tiles or len(paths))
    i, k = (int(x) for x in args.part.split("/"))
    paths = paths[i::k]
    global ISO_BINS
    ISO_BINS = iso_bins()
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(ins.cmb_cl_emulator(), tb)
    cfg = simu.TileConfig(halo_cap=1, paint=None, q_final=4.0, q_mask=args.q_mask)
    noise_run, inj_run = make_finder(st, cfg)
    out = []
    for p in paths:
        if args.geometry:
            r = run_tile_geometry(p, st, noise_run, inj_run)
            out.append(r)
            print(f"tile {r['tile']:3d} (dec {r['dec']:+.0f}) {r['seconds']:.0f} s; q>4 split cut/periodic "
                  f"{np.mean(r['split']['iso_cut']['counts'], axis=0)[0]:.1f}/{np.mean(r['split']['iso_per']['counts'], axis=0)[0]:.1f}",
                  flush=True)
            with open(os.path.join(odir, f"geometry_part_{i}_of_{k}.json"), "w") as fh:
                json.dump(out, fh)
            continue
        r = run_tile(p, st, noise_run, inj_run)
        out.append(r)
        c = {dp: {kd: np.mean(v["counts"], axis=0)[2] for kd, v in r[dp].items() if isinstance(v, dict) and "counts" in v}
             for dp in ("split", "coadd")}
        print(f"tile {r['tile']:3d} (dec {r['dec']:+.0f}) {r['seconds']:.0f} s; q>5 split: "
              + ", ".join(f"{kd} {v:.2f}" for kd, v in c["split"].items()) + " | coadd: "
              + ", ".join(f"{kd} {v:.2f}" for kd, v in c["coadd"].items()), flush=True)
        with open(os.path.join(odir, f"part_{i}_of_{k}.json"), "w") as fh:
            json.dump(out, fh)


if __name__ == "__main__":
    main()
