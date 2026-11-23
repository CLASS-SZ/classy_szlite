"""One tile through every stage of the simulator and the finder, keeping the intermediate maps (for
figures, notebooks and checks). Forecast noise (public).

    from classy_szlite.sim import tile
    out = tile.stages()                 # fiducial point, first tile of the first mock
    out["y"], out["m145"], out["q_final"], out["peaks"], out["halos"]

Cost: about 3-6 min on a laptop CPU the first time (paint tables, compilation), then a few seconds;
about 1-2 GB of memory.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import simulator as simu, summaries as sm, mmf, catalogue as ct, paint as pt, priors
from .flatsky import N, PIX, PIXF
from .setup import tile_setup, default_cmb

I145 = 3          # channel index of 145 GHz


def stages(theta=None, seed=priors.SEEDS_MAP["truth"], mock=0, tile=0, q_final=4.0, log=print):
    """The maps of one tile (1200 x 1200 pixels of 0.5', numpy float32) and its catalogue:

    y                 Compton-y of every halo above 1e13 Msun
    tsz145, cmb145, noise145, m145
                      the 145 GHz map's three parts and their sum [uK]
    q_first, q_final  the finder's S/N, maximised over its 15 sizes, before and after the masking pass
    mask              the masking pass's discs (0 = masked)
    sigma_first, sigma_final, sigma_model
                      the finder's noise at the 15 sizes (first pass, final pass) and the analytic noise of
                      the instrument and the CMB
    peaks             kept peaks with q > q_final: dict of x, y [pixels], q, size [arcmin], confirmed, halo
    halos             halos above 5e13 Msun: dict of x, y [pixels], z, M500, theta500, y0, q_model

    With the default seed this is tile ``tile`` of mock ``mock`` of the published 'truth' set."""
    theta = simu.THETA_FID if theta is None else theta
    cmb = default_cmb()
    ts = tile_setup(theta, cmb, log=log)
    cfg = ts.cfg._replace(q_final=q_final, peak_cap=6144)
    cells, tb, cs, st = ts.cells, ts.tables, ts.cosmo, ts.static
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(seed), mock), tile)

    @jax.jit
    def run(key):
        halos, X, ovf = simu.tile_maps(key, cs, theta, st, tb, cells, cfg, None)
        yk, _ = pt.paint_y(halos, tb._replace(tab=st.paint_tab, rP=st.paint_rP), cfg.paint, st.T_tab)
        k_h, k_m = jax.random.split(key)
        k_cmb = jax.random.split(k_m, 3)[0]
        cmbk = jnp.fft.rfft2(jax.random.normal(k_cmb, (N, N), dtype=jnp.float32)) * st.cmb_amp
        tsz, cmbm = st.a_full[I145] * yk, st.B_full[I145] * cmbk
        q0, sig0, C0 = mmf.q_cube(X, st.fs)
        pk0 = mmf.find_peaks(q0, cfg.peak_cap, cfg.q_mask)
        mask, _ = mmf.cluster_mask(pk0, st.fs, cfg.q_mask)
        C, _ = mmf.masked_cov(X, mask, C0, st.fs)
        q1, sig1, _ = mmf.q_cube(X, st.fs, C=C)
        pk = mmf.find_peaks(q1, cfg.peak_cap, cfg.q_final)
        keep = sm.merge_local_max(pk, cfg.r_merge)
        cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
        ir = lambda a: jnp.fft.irfft2(a, s=(N, N))
        return dict(y=ir(yk), m145=ir(X[I145]), tsz145=ir(tsz), cmb145=ir(cmbm), noise145=ir(X[I145] - tsz - cmbm),
                    q_first=jnp.max(q0, 0), q_final=jnp.max(q1, 0), mask=mask, sigma_first=sig0, sigma_final=sig1,
                    pk=pk, keep=keep & pk.valid, conf=cf.ok, halo=cf.halo, halos=halos,
                    overflow=jnp.any(ovf) | halos.overflow | pk.overflow)
    o = jax.device_get(run(key))
    if o["overflow"]:
        raise RuntimeError("buffer overflow")
    h = o["halos"]
    hv = np.asarray(h.valid) & (np.asarray(h.lnM) >= np.log(cfg.M_conf))
    slot = np.flatnonzero(hv)
    sig_model, lnth = ct.analytic_sigma_y0_forecast(cmb, "goal")
    th500 = np.asarray(h.theta500)[hv]
    qmod = np.asarray(h.y0)[hv] / np.exp(np.interp(np.log(np.clip(th500, np.exp(lnth[0]), np.exp(lnth[-1]))), lnth, np.log(sig_model[0])))
    kp = np.asarray(o["keep"])
    pk = o["pk"]
    conf = np.asarray(o["conf"])[kp]
    hidx = np.full(kp.sum(), -1)
    hidx[conf] = np.searchsorted(slot, np.asarray(o["halo"])[kp][conf])
    f32 = lambda a: np.asarray(a, np.float32)
    out = {k: f32(o[k]) for k in ("y", "m145", "tsz145", "cmb145", "noise145", "q_first", "q_final", "mask")}
    out.update(sigma_first=np.asarray(o["sigma_first"]), sigma_final=np.asarray(o["sigma_final"]), sigma_model=sig_model[0],
               thetas=np.asarray(mmf.THETAS15), pix_arcmin=PIX,
               peaks=dict(x=np.asarray(pk.ix)[kp], y=np.asarray(pk.iy)[kp], q=np.asarray(pk.q)[kp],
                          size=np.asarray(mmf.THETAS15)[np.asarray(pk.j)[kp]], confirmed=conf, halo=hidx),
               halos=dict(x=(np.asarray(h.ix)[hv] + np.asarray(h.ox)[hv]) * PIXF / PIX,
                          y=(np.asarray(h.iy)[hv] + np.asarray(h.oy)[hv]) * PIXF / PIX,
                          z=np.asarray(h.z)[hv], M500=np.exp(np.asarray(h.lnM)[hv]), theta500=th500,
                          y0=np.asarray(h.y0)[hv], q_model=qmod))
    return out
