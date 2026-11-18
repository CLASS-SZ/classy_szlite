"""Halo load, emulator range and peak counts at the corners of candidate priors.

    python benchmarks/mapsim/prior_corners.py --tiles 4

Three boxes in (σ8, ω_c, A_SZ); h at ±3σ of its N(0.674, 0.01) prior:

  paper   σ8 0.700-0.900, ω_c 0.0804-0.204, A_SZ -4.41 to -4.21 (Table 1 of
          arXiv:2504.10230)
  wide    σ8 0.770-0.852, ω_c 0.0804-0.169, A_SZ -4.53 to -4.08 (proposed for
          the 10%-sky runs: about ±8 of the paper's posterior widths)
  narrow  σ8 0.780-0.842, ω_c 0.085-0.157, A_SZ -4.48 to -4.13 (proposed for
          the 40%-sky runs: about ±6 of the paper's posterior widths)

The paper's likelihood widths are 0.0051 (σ8), 0.0060 (ω_c) and 0.028 (A_SZ).

For every corner: ln(10^10 A_s), expected halos per tile with M ≥ 1e13 and
M ≥ 5e13, and the painter's load. At each box's heaviest corner, with A_SZ at
the top of the box, α_SZ = 1 and σ_lnq at +3σ (the most peaks), a few
goal-depth tiles are simulated, recording peaks, merged peaks, confirmed
detections and buffer overflows.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike  # noqa: E402

from classy_szlite.cosmology import ln10As_from_sigma8  # noqa: E402
from classy_szlite.sim import halos as hl, paint as pt, simulator as simu  # noqa: E402

OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "priors")
H_EDGES = (0.644, 0.704)                       # h at ±3σ
ALPHA_LOW = 1.0
SIGMA_LNQ_HIGH = 0.173 + 3 * 0.05
BOXES = {
    "paper": dict(sigma8=(0.700, 0.900), omega_cdm=(0.0804, 0.204), A_SZ=(-4.41, -4.21)),
    "wide": dict(sigma8=(0.770, 0.852), omega_cdm=(0.0804, 0.169), A_SZ=(-4.53, -4.08)),
    "narrow": dict(sigma8=(0.780, 0.842), omega_cdm=(0.085, 0.157), A_SZ=(-4.48, -4.13)),
}


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = hl.default_cells()
    tb = spike.tables()
    stage = jax.jit(lambda th: simu.cosmology_stage(th, cells))
    lnMc = 0.5 * (cells.lnM_edges[1:] + cells.lnM_edges[:-1])
    base = simu.THETA_FID

    def load(th):
        cs = stage(th)
        lam = np.asarray(cs.lam)
        occ = spike.expected_occupation(cs.lam, cs.cg, th, cells, tb)
        p = {"omega_cdm": th.omega_cdm, "H0": 100.0 * th.h, "omega_b": th.omega_b,
             "n_s": th.n_s}
        return dict(ln10As=float(ln10As_from_sigma8(p, th.sigma8, as_reference=simu.LN10AS_REFERENCE)),
                    halos=float(lam.sum()),
                    halos_5e13=float(lam[:, lnMc >= np.log(5e13)].sum()),
                    footprint=float(sum(o * s * s for o, s in zip(occ[:-1], pt.SIDES))),
                    direct=float(occ[-1]))

    fid = load(base)
    log(f"fiducial: ln10As {fid['ln10As']:.3f}, halos {fid['halos']:.0f}, M >= 5e13 "
        f"{fid['halos_5e13']:.0f}, footprint {fid['footprint']:.3e}, direct {fid['direct']:.0f}")
    res = dict(fiducial=fid, boxes={}, tiles=args.tiles, h_edges=H_EDGES,
               alpha_low=ALPHA_LOW, sigma_lnq_high=SIGMA_LNQ_HIGH)
    for name, box in BOXES.items():
        rows = []
        for s8, oc, h in itertools.product(box["sigma8"], box["omega_cdm"], H_EDGES):
            r = load(base._replace(sigma8=s8, omega_cdm=oc, h=h))
            r.update(sigma8=s8, omega_cdm=oc, h=h)
            rows.append(r)
        log(f"\n=== {name}: " + ", ".join(f"{k} {v}" for k, v in box.items()))
        log("   sigma8  omega_cdm      h   ln10As    halos  rel.   M>=5e13   footprint  direct")
        for r in rows:
            log(f"   {r['sigma8']:.3f}  {r['omega_cdm']:.4f}  {r['h']:.3f}  {r['ln10As']:6.3f} "
                f"{r['halos']:8.0f} {r['halos'] / fid['halos']:5.2f} {r['halos_5e13']:9.0f} "
                f"{r['footprint']:11.3e} {r['direct']:6.0f}")
        heavy = max(rows, key=lambda r: r["halos"])
        th = base._replace(sigma8=heavy["sigma8"], omega_cdm=heavy["omega_cdm"], h=heavy["h"],
                           A_SZ=box["A_SZ"][1], alpha_SZ=ALPHA_LOW, sigma_lnq=SIGMA_LNQ_HIGH)
        t0 = time.time()
        _, _, cs, cfg, st, info = spike.setup(th)
        f = jax.jit(jax.vmap(lambda key, s: simu.simulate_tile(key, cs, th, s, tb, cells, cfg),
                             in_axes=(0, None)))
        keys = spike.tile_keys(args.tiles)
        o = jax.device_get(f(keys[:1], st))
        t1 = time.time()
        outs = [o] + [jax.device_get(f(keys[a:a + 1], st)) for a in range(1, args.tiles)]
        t2 = time.time()
        o = {k: np.concatenate([np.asarray(x[k]) for x in outs]) for k in outs[0]}
        sim = dict(theta=th._asdict(), halo_cap=cfg.halo_cap, peak_cap=cfg.peak_cap,
                   conf_cap=cfg.conf_cap, caps=list(cfg.paint.caps), cap_big=info["cap_big"],
                   n_halos=o["n_halos"].tolist(), n_peaks=o["n_peaks"].tolist(),
                   n_kept=o["n_kept"].tolist(), n_conf=o["n_conf"].tolist(),
                   detections=o["counts"].sum((1, 2)).tolist(),
                   overflow=o["overflow"].any(0).tolist(),
                   compile_and_first_s=t1 - t0, per_tile_s=(t2 - t1) / max(1, args.tiles - 1))
        log(f"   heaviest corner, A_SZ {th.A_SZ}, alpha_SZ {th.alpha_SZ}, sigma_lnq "
            f"{th.sigma_lnq:.3f}: halos {np.mean(sim['n_halos']):.0f} (cap {cfg.halo_cap}), "
            f"peaks {sim['n_peaks']} (cap {cfg.peak_cap}), kept {sim['n_kept']}, confirmed "
            f"{sim['n_conf']}, in the data vector {sim['detections']}")
        log(f"   overflow flags (halos, painter classes..., peaks, confirmation): {sim['overflow']}; "
            f"{sim['per_tile_s']:.1f} s per tile after {sim['compile_and_first_s']:.0f} s")
        res["boxes"][name] = dict(box=box, corners=rows, heaviest=sim)
    with open(os.path.join(OUT, "prior_corners.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"saved {os.path.join(OUT, 'prior_corners.json')}")


if __name__ == "__main__":
    main()
