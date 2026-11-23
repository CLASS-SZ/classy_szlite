"""Maps of one simulated tile for the report: each stage of the map-level simulator and of the finder.

    python benchmarks/mapsim/report_maps.py [--mock 0] [--tile 0]

The tile is tile `--tile` of map-level mock `--mock` (maps_trainset.py --set truth: same keys, same
MSS2 spectrum), at the fiducial point. It is simulated twice with the same halos and CMB: with its
MSS2 noise and with the 2019 forecast noise (goal). Saved (float16 where it is a map, 1200 x 1200
at 0.5'): the Compton-y map; the 145 GHz map and its three parts (tSZ, CMB, noise); the finder's
S/N maximised over its 15 sizes, before and after the masking pass; the mask; the final kept
peaks with q > 4 (position, size, q, confirmed, matched halo) and the halos above 5e13 Msun
(position, z, M, theta500, the catalogue model's q). Output: $MAPSIM_OUT/report/tile_maps.npz.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spike  # noqa: E402
import maps_trainset as mt  # noqa: E402
from classy_szlite.sim import simulator as simu, noise as nz, instrument as ins, summaries as sm  # noqa: E402
from classy_szlite.sim import catalogue as ct, mmf  # noqa: E402
from classy_szlite.sim.flatsky import N, PIXF, PIX  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "report")
I145 = 3                                       # channel index of 145 GHz (instrument.SO_LAT freqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", type=int, default=0)
    ap.add_argument("--tile", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nt, _ = nz.load_mss2(mt.SPECTRA, "mean")
    specs = mt.survey_spectra(nt.F.shape[0])
    spec = int(specs[args.tile])
    cells, tb, cs, cfg0, st, _ = spike.setup(simu.THETA_FID)
    cmb_cl = ins.cmb_cl_from_table(os.path.join(spike.REFDATA, "data_real", "cmb_lensed_TT_camb.npz"))
    st = simu.tile_static(cmb_cl, tb, mode="goal")
    cfg = cfg0._replace(q_final=4.0, peak_cap=6144)
    th = simu.THETA_FID
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(mt.SEED["truth"]), args.mock), args.tile)
    sig_model, lnth = ct.analytic_sigma_y0(nt, [spec], cmb_cl)

    out = {}
    for label, noise in (("mss2", nz.NoiseInput(nt, jnp.int32(spec), jnp.float32(1.0))), ("forecast", None)):
        @jax.jit
        def run(key):
            k_h, k_m = jax.random.split(key)                 # as simulator.tile_maps
            halos, X, ovf = simu.tile_maps(key, cs, th, st, tb, cells, cfg, noise)
            tabs = tb._replace(tab=st.paint_tab, rP=st.paint_rP)
            from classy_szlite.sim import paint as pt
            yk, _ = pt.paint_y(halos, tabs, cfg.paint, st.T_tab)
            k_cmb, k_w, k_a = jax.random.split(k_m, 3)       # as simulator.channel_maps
            cmbk = jnp.fft.rfft2(jax.random.normal(k_cmb, (N, N), dtype=jnp.float32)) * st.cmb_amp
            tsz = st.a_full[I145] * yk
            cmb = st.B_full[I145] * cmbk
            noise145 = X[I145] - tsz - cmb
            q0, sig0, C0 = mmf.q_cube(X, st.fs)
            pk0 = mmf.find_peaks(q0, cfg.peak_cap, cfg.q_mask)
            mask, n_masked = mmf.cluster_mask(pk0, st.fs, cfg.q_mask)
            C, _ = mmf.masked_cov(X, mask, C0, st.fs)
            q1, sig1, _ = mmf.q_cube(X, st.fs, C=C)
            pk = mmf.find_peaks(q1, cfg.peak_cap, cfg.q_final)
            keep = sm.merge_local_max(pk, cfg.r_merge)
            cf = sm.confirm(pk, keep, halos, cfg.M_conf, cfg.r_conf, cfg.conf_cap)
            ir = lambda a: jnp.fft.irfft2(a, s=(N, N))
            return dict(y=ir(yk), m145=ir(X[I145]), tsz145=ir(tsz), cmb145=ir(cmb), noise145=ir(noise145),
                        q0max=jnp.max(q0, 0), q1max=jnp.max(q1, 0), mask=mask, sig0=sig0, sig1=sig1,
                        pk_iy=pk.iy, pk_ix=pk.ix, pk_j=pk.j, pk_q=pk.q, pk_keep=keep & pk.valid, pk_conf=cf.ok,
                        pk_halo=cf.halo, h_iy=halos.iy, h_ix=halos.ix, h_oy=halos.oy, h_ox=halos.ox, h_z=halos.z,
                        h_lnM=halos.lnM, h_y0=halos.y0, h_th=halos.theta500, h_valid=halos.valid,
                        n_masked=n_masked, overflow=jnp.any(ovf) | halos.overflow | pk.overflow)
        o = jax.device_get(run(key))
        assert not bool(o["overflow"]), label
        # halos above M_conf, positions in coarse pixels, and the catalogue model's q
        hv = o["h_valid"] & (o["h_lnM"] >= np.log(cfg.M_conf))
        hy = (o["h_iy"][hv] + o["h_oy"][hv]) * PIXF / PIX
        hx = (o["h_ix"][hv] + o["h_ox"][hv]) * PIXF / PIX
        sig = sig_model[0] if label == "mss2" else ct.analytic_sigma_y0_forecast(cmb_cl, "goal")[0][0]
        qmod = o["h_y0"][hv] / np.exp(np.interp(np.log(np.clip(o["h_th"][hv], np.exp(lnth[0]), np.exp(lnth[-1]))), lnth, np.log(sig)))
        slot = np.flatnonzero(hv)
        kp = o["pk_keep"]
        halo_idx = np.full(kp.sum(), -1)
        conf = o["pk_conf"][kp]
        halo_idx[conf] = np.searchsorted(slot, o["pk_halo"][kp][conf])
        f16 = lambda a: np.asarray(a, np.float16)
        out.update({f"{label}_{k}": f16(o[k]) for k in ("m145", "tsz145", "cmb145", "noise145", "q0max", "q1max")})
        out.update({f"{label}_mask": o["mask"].astype(np.uint8), f"{label}_sig0": o["sig0"], f"{label}_sig1": o["sig1"],
                    f"{label}_pk_iy": o["pk_iy"][kp], f"{label}_pk_ix": o["pk_ix"][kp], f"{label}_pk_j": o["pk_j"][kp],
                    f"{label}_pk_q": o["pk_q"][kp], f"{label}_pk_conf": conf, f"{label}_pk_halo": halo_idx,
                    f"{label}_h_qmodel": qmod, f"{label}_sig_model": sig})
        if label == "mss2":
            out.update(y=f16(o["y"]), h_y=hy, h_x=hx, h_z=o["h_z"][hv], h_lnM=o["h_lnM"][hv], h_th=o["h_th"][hv],
                       h_y0=o["h_y0"][hv])
        print(f"{label}: kept peaks q > 4: {kp.sum()}, confirmed with q > 5: {np.sum(conf & (o['pk_q'][kp] > 5))}; "
              f"halos above 5e13: {hv.sum()}; masked discs {int(o['n_masked'])}; "
              f"sigma finder / model at 2': {o['sig1'][7] / sig[7]:.3f}", flush=True)
    np.savez_compressed(os.path.join(OUT, "tile_maps.npz"), **out, thetas=np.asarray(mmf.THETAS15), spec=spec,
                        mock=args.mock, tile=args.tile, pix_arcmin=PIX)
    print("saved", os.path.join(OUT, "tile_maps.npz"), flush=True)


if __name__ == "__main__":
    main()
