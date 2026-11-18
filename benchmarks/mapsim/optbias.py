"""Optimisation bias of our matched-filter finder, from isolated injections.

    python benchmarks/mapsim/optbias.py --tiles 128 --modes goal baseline

Each tile holds lensed CMB, white and atmospheric noise (one depth) and a
10 × 10 grid of isolated clusters synthesised exactly. The sites are 60'
apart, jittered by up to ±3', with random sub-pixel offsets. θ500 is
log-uniform in [0.5', 6'], and the target q_true is uniform in [1, 12]:

    q_true = y0 / σ(θ500),

where σ(θ) is the matched-filter noise of the exact template at θ500 for the
true covariance (CMB + noise). This is the q of the catalogue-level model.

Per injection, both for the true (analytic) covariance and for the covariance
estimated from the tile (one pass, unmasked):

    qs_max  noiseless q: maximum over a 7 × 7 pixel window (±1.5') and all 15 scales
    qn_max  noisy q, same window and scales: the finder's value at its maximum
    qs_ref  noiseless q at the nearest pixel and the nearest scale
    qn_ref  noisy q at the same pixel and scale (no maximisation)
    q_det   q of the highest kept peak (q > 4, merged within 1.5') within 1.5'
            of the injection; NaN if there is none

The filter is linear, so for a given covariance qn = qs + (noise q) exactly.

Expectations from the optimisation-bias model:
- ⟨qn_ref − qs_ref⟩ = 0 with unit scatter.
- Maximising over d parameters near a peak gives ⟨qn_max²⟩ ≈ qs² + d + 1,
  i.e. q_obs ≈ N(sqrt(q² + d), 1), with d = 3 for position plus scale in
  the continuum limit.
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
from scipy.interpolate import CubicSpline
from scipy.stats import norm as _norm

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike  # noqa: E402
from counts_modes import analytic_binned_cov, sigma_from_cov  # noqa: E402

from classy_szlite import profiles  # noqa: E402
from classy_szlite.sim import instrument as ins, mmf, simulator as simu  # noqa: E402
from classy_szlite.sim import paint as pt, summaries as sm, flatsky as fsk  # noqa: E402
from classy_szlite.sim.flatsky import N, NF, ARCMIN  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "optbias")
SITES = 10
SPACING_F = NF // SITES          # 240 fine pixels = 60'
JIT_F = 12                       # ±3' in fine pixels
WIN = 3                          # ±3 coarse pixels = ±1.5'
R_MATCH = 1.5                    # arcmin


def log(*a):
    print(*a, flush=True)


def sigma_continuous(Cb, fs, thetas):
    """σ(θ) [y0 units] of the exact template at arbitrary θ (arcmin)."""
    G = fsk.grid()
    _, aCa = mmf.weights(jnp.asarray(Cb), fs)
    aCa, W = np.asarray(aCa), np.asarray(fs.m_W)
    out = []
    for t in np.atleast_1d(thetas):
        tau = (t * ARCMIN) ** 2 * profiles.T_interp(G.m_L * t * ARCMIN) / G.Om
        out.append(np.sum(W * tau ** 2 * aCa) ** -0.5)
    return np.array(out)


def make_injections(rng, sig_spline, th_lo, th_hi, q_lo, q_hi):
    n = SITES * SITES
    gy, gx = np.meshgrid(np.arange(SITES), np.arange(SITES), indexing="ij")
    iy = (gy.ravel() * SPACING_F + SPACING_F // 2 + rng.integers(-JIT_F, JIT_F + 1, n)) % NF
    ix = (gx.ravel() * SPACING_F + SPACING_F // 2 + rng.integers(-JIT_F, JIT_F + 1, n)) % NF
    oy, ox = rng.random(n), rng.random(n)
    th = np.exp(rng.uniform(np.log(th_lo), np.log(th_hi), n))
    qt = rng.uniform(q_lo, q_hi, n)
    y0 = qt * np.exp(sig_spline(np.log(th)))
    cy = np.rint((iy + oy) / 2.0).astype(np.int64) % N            # nearest coarse pixel
    cx = np.rint((ix + ox) / 2.0).astype(np.int64) % N
    jref = np.argmin(np.abs(np.log(th)[:, None] - np.log(mmf.THETAS15)[None, :]), axis=1)
    inj = dict(iy=iy.astype(np.int32), ix=ix.astype(np.int32), oy=oy, ox=ox, y0=y0, theta500=th)
    aux = dict(q_true=qt, theta500=th, cy=cy.astype(np.int32), cx=cx.astype(np.int32),
               jref=jref.astype(np.int32))
    return inj, aux


def _measure(qn, qs, cy, cx, jref):
    d = jnp.arange(-WIN, WIN + 1)
    yy = (cy[:, None, None] + d[None, :, None]) % N
    xx = (cx[:, None, None] + d[None, None, :]) % N
    wn = jnp.moveaxis(qn[:, yy, xx], 0, 1).reshape(cy.shape[0], -1)        # (n, 15*49)
    ws = jnp.moveaxis(qs[:, yy, xx], 0, 1).reshape(cy.shape[0], -1)
    kn, ks = jnp.argmax(wn, axis=1), jnp.argmax(ws, axis=1)
    win2 = (2 * WIN + 1) ** 2
    return dict(qn_max=jnp.max(wn, axis=1), qs_max=jnp.max(ws, axis=1),
                jn=kn // win2, js=ks // win2,
                qn_ref=qn[jref, cy, cx], qs_ref=qs[jref, cy, cx])


def _detect(qn, cy, cx):
    pk = mmf.find_peaks(qn, 2048)
    keep = sm.merge_local_max(pk, R_MATCH)
    dy = sm._wrap(pk.iy[None, :] - cy[:, None], N)
    dx = sm._wrap(pk.ix[None, :] - cx[:, None], N)
    m = keep[None, :] & ((dy * dy + dx * dx) <= int((R_MATCH / fsk.PIX) ** 2))
    q = jnp.max(jnp.where(m, pk.q[None, :], -jnp.inf), axis=1)
    n_other = jnp.sum(keep) - jnp.sum(jnp.any(m, axis=0))
    return jnp.where(jnp.isfinite(q), q, jnp.nan), n_other, pk.overflow


def tile_fn(key, inj, aux, st, Cb):
    yk = pt.synth_direct(inj, st.T_tab, jnp.float32)
    X = simu.channel_maps(key, yk, st)
    Xs = st.a_full * yk[None]
    out = {}
    for name, C in (("an", Cb), ("est", None)):
        qn, sig, Cu = mmf.q_cube(X, st.fs, C=C)
        qs, _, _ = mmf.q_cube(Xs, st.fs, C=Cu)
        r = _measure(qn, qs, aux["cy"], aux["cx"], aux["jref"])
        r["q_det"], r["n_other"], r["peak_overflow"] = _detect(qn, aux["cy"], aux["cx"])
        r["sigma"] = sig
        out[name] = r
    return out


def summarise(rec, sig_ratio):
    """Text summary of one mode."""
    lines = []
    qt = rec["q_true"]
    edges = np.arange(1.0, 13.0, 1.0)
    for name in ("an", "est"):
        r = rec[name]
        dref = r["qn_ref"] - r["qs_ref"]
        lines.append(f"--- covariance: {'true (analytic)' if name == 'an' else 'estimated from the tile, one pass'}")
        if name == "est":
            lines.append("    sigma_est / sigma_true per scale (mean over tiles): "
                         + " ".join(f"{x:.3f}" for x in sig_ratio))
        lines.append(f"    no maximisation: <qn_ref - qs_ref> = {dref.mean():+.4f} ± "
                     f"{dref.std() / np.sqrt(dref.size):.4f}, std {dref.std():.3f}")
        hi = qt >= 4
        lines.append(f"    noiseless losses (q_true >= 4): <qs_max/q_true> = {np.mean(r['qs_max'][hi] / qt[hi]):.4f},"
                     f" <qs_ref/q_true> = {np.mean(r['qs_ref'][hi] / qt[hi]):.4f}")
        # fitted k: q_obs = sqrt(q^2 + k) + N(0,1), least squares over q_true >= 4
        for ref_name, qref in (("q_true", qt), ("qs_max", r["qs_max"])):
            ks = np.linspace(-2, 8, 1001)
            sse = [np.sum((r["qn_max"][hi] - np.sqrt(np.maximum(qref[hi] ** 2 + k, 0))) ** 2) for k in ks]
            kbest = ks[int(np.argmin(sse))]
            resid = r["qn_max"][hi] - np.sqrt(np.maximum(qref[hi] ** 2 + kbest, 0))
            lines.append(f"    fit qn_max ~ sqrt({ref_name}^2 + k) + N(0,1), q_true >= 4: k = {kbest:.2f}, "
                         f"residual mean {resid.mean():+.3f}, std {resid.std():.3f}")
        lines.append("    q_true bin  n     <qn_max-q_true>  k=2      k=3      <qn_max-qs_max>  "
                     "k_eff(qs)  P(q_det>5)  Phi k=0  k=2    k=3")
        for a, b in zip(edges[:-1], edges[1:]):
            s = (qt >= a) & (qt < b)
            if s.sum() < 10:
                continue
            q0 = qt[s]
            dq = r["qn_max"][s] - q0
            m2, m3 = np.mean(np.sqrt(q0 ** 2 + 2) - q0), np.mean(np.sqrt(q0 ** 2 + 3) - q0)
            dqs = r["qn_max"][s] - r["qs_max"][s]
            keff = np.mean(r["qn_max"][s] ** 2) - np.mean(r["qs_max"][s] ** 2) - 1.0
            det = np.nan_to_num(r["q_det"][s], nan=-np.inf) > 5
            phi = [np.mean(_norm.cdf(np.sqrt(q0 ** 2 + k) - 5)) for k in (0, 2, 3)]
            lines.append(f"    [{a:4.1f},{b:4.1f}) {s.sum():5d}  {dq.mean():+.3f}±{dq.std() / np.sqrt(s.sum()):.3f}"
                         f"     {m2:+.3f}   {m3:+.3f}   {dqs.mean():+.3f}±{dqs.std() / np.sqrt(s.sum()):.3f}"
                         f"     {keff:5.2f}     {det.mean():.3f}       {phi[0]:.3f}   {phi[1]:.3f}  {phi[2]:.3f}")
        lines.append(f"    other kept peaks (q > 4) per tile, not matched to an injection: "
                     f"{np.mean(r['n_other']):.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=128)
    ap.add_argument("--modes", nargs="+", default=["goal"])
    ap.add_argument("--th", type=float, nargs=2, default=[0.5, 6.0])
    ap.add_argument("--q", type=float, nargs=2, default=[1.0, 12.0])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cmb = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    dummy = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    f = jax.jit(tile_fn)
    summary = {}
    for mode in args.modes:
        t0 = time.time()
        st = simu.tile_static(cmb, dummy, mode=mode)
        Cb = analytic_binned_cov(cmb, mode)
        th_tab = np.geomspace(0.8 * args.th[0], 1.25 * args.th[1], 160)
        sig_tab = sigma_continuous(Cb, st.fs, th_tab)
        sig15 = sigma_from_cov(Cb, st.fs)
        chk = np.max(np.abs(sigma_continuous(Cb, st.fs, mmf.THETAS15) / sig15 - 1))
        spl = CubicSpline(np.log(th_tab), np.log(sig_tab))
        mid = np.sqrt(th_tab[1:] * th_tab[:-1])[::10]
        ierr = np.max(np.abs(np.exp(spl(np.log(mid))) / sigma_continuous(Cb, st.fs, mid) - 1))
        log(f"\n=== {mode}: sigma(theta) table ready in {time.time() - t0:.0f} s; "
            f"check vs filter scales {chk:.1e}, spline error {ierr:.1e}")
        base = jax.random.fold_in(jax.random.key(20261119), {"goal": 0, "baseline": 1}[mode])
        rec = {}
        sig_est = []
        for t in range(args.tiles):
            rng = np.random.default_rng([2026, 11, 19, t])
            inj, aux = make_injections(rng, spl, *args.th, *args.q)
            o = jax.device_get(f(jax.random.fold_in(base, t), inj, aux, st, jnp.asarray(Cb)))
            for k, v in aux.items():
                rec.setdefault(k, []).append(v)
            for name in ("an", "est"):
                for k, v in o[name].items():
                    if k in ("sigma",):
                        continue
                    rec.setdefault(name, {}).setdefault(k, []).append(np.atleast_1d(np.asarray(v)))
            sig_est.append(np.asarray(o["est"]["sigma"]) / np.asarray(o["an"]["sigma"]))
            if t == 0:
                log(f"first tile done at {time.time() - t0:.0f} s")
        rec = {k: (np.concatenate(v) if isinstance(v, list) else {kk: np.concatenate(vv) for kk, vv in v.items()})
               for k, v in rec.items()}
        txt = summarise(rec, np.mean(sig_est, axis=0))
        log(f"{mode}: {args.tiles} tiles, {rec['q_true'].size} injections, {time.time() - t0:.0f} s; "
            f"peak overflow {bool(rec['est']['peak_overflow'].any() or rec['an']['peak_overflow'].any())}")
        log(txt)
        np.savez(os.path.join(OUT, f"optbias_{mode}.npz"), q_true=rec["q_true"], theta500=rec["theta500"],
                 **{f"{n}_{k}": v for n in ("an", "est") for k, v in rec[n].items()})
        summary[mode] = txt
    with open(os.path.join(OUT, "optbias_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
