"""A calibration of the catalogue model that follows the parameters (variant (iii)).

    python benchmarks/mapsim/qcal_grid.py run [--points 24] [--mocks 5] [--noise forecast]   # GPU
    python benchmarks/mapsim/qcal_grid.py fit [--noise forecast]                             # CPU

Why: the catalogue model calibrated on map-level mocks at one point (variant (ii), qcal.py) is right
there but miscalibrated across the prior, because the finder's noise (the confusion of faint
clusters), search lift, scatter and losses change with the amount of tSZ on the sky (LOG
2026-11-21). Here the calibration is measured at --points parameter points spread over the
map-level prior box in (sigma8, omega_cdm, A_SZ, alpha_SZ) (a Latin hypercube; the other parameters
at their fiducial values) plus the fiducial point, each from --mocks map-level mocks of the 41-tile
survey (qcal.table, own keys), and fitted as linear functions of x = (theta - fid) / prior width:

  ln sigma_finder at each of the 15 filter sizes, mean over tiles, relative to the fiducial
  calibration's per-tile table: d0 + g . x;
  per redshift bin, s, k, w, f of the censored fit against the finder's own noise (qcal.censored_fit):
  c0 + c . x.

run: per point, the per-halo table and the fits; written to calib_grid.npz (one row per point) in
$MAPSIM_OUT/qcal_grid[_forecast]/. fit: the linear fits, their residuals at the points (and a
leave-one-out check), and calib_theta.npz, read by catalogue_trainset.py --variant calibt41[f].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ZB = [0.01, 0.31, 0.61, 0.90, 1.20, 1.50, 99.0]
SB = [0.0, 1.0, 2.0, 4.0, 1e9]          # theta500 size bins [arcmin], for --size
N_MIN = 400                              # halos with q > 2 needed to fit a (z, size) cell
PARAMS = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ")
SEED = 20261127


def out_dir(noise):
    return os.path.join(os.environ.get("MAPSIM_OUT", "."), "qcal_grid" + ("_forecast" if noise == "forecast" else ""))


def points(n, box):
    """Latin hypercube in the box (n points, 4 parameters), numpy default_rng(SEED)."""
    rng = np.random.default_rng(SEED)
    u = np.stack([(rng.permutation(n) + rng.uniform(size=n)) / n for _ in PARAMS], 1)
    lo = np.array([box[p][0] for p in PARAMS])
    hi = np.array([box[p][1] for p in PARAMS])
    return lo + u * (hi - lo)


def calib_fit_cells(H):
    """Per (z, size) cell (s, k, w, f, n) against q_fsig on halos with q_fsig > 2; NaN where fewer than
    N_MIN halos."""
    from qcal import censored_fit
    qf, qo, z, th = (H[k].astype(float) for k in ("q_fsig", "q_obs", "z", "theta500"))
    out = np.full((len(ZB) - 1, len(SB) - 1, 5), np.nan)
    for i, (a, b) in enumerate(zip(ZB[:-1], ZB[1:])):
        for j, (c, d) in enumerate(zip(SB[:-1], SB[1:])):
            m = (z > a) & (z <= b) & (th > c) & (th <= d) & (qf > 2)
            out[i, j, 4] = m.sum()
            if m.sum() >= N_MIN:
                r = censored_fit(qf[m], qo[m])
                out[i, j, :4] = (r["s"], r["k"], r["w"], r["f"])
    return out


def calib_fit(H, T):
    """Per z bin (s, k, w, f) against q_fsig on halos with q_fsig > 2, and the tile-mean sigma."""
    from qcal import censored_fit
    qf, qo, z = H["q_fsig"].astype(float), H["q_obs"].astype(float), H["z"].astype(float)
    rows = []
    for a, b in zip(ZB[:-1], ZB[1:]):
        m = (z > a) & (z <= b) & (qf > 2)
        r = censored_fit(qf[m], qo[m])
        rows.append((r["s"], r["k"], r["w"], r["f"], r["n"]))
    sig_tile = np.array([T["sigma"][T["tile"] == t].mean(0) for t in range(int(T["tile"].max()) + 1)])
    return np.array(rows), sig_tile


def run(args):
    import jax
    import maps_trainset as mt
    import qcal
    from classy_szlite.sim import simulator as simu
    out = out_dir(args.noise)
    os.makedirs(out, exist_ok=True)
    P = points(args.points, mt.UNIFORM)
    fid = np.array([getattr(simu.THETA_FID, p) for p in PARAMS])
    P = np.vstack([fid, P])                                    # row 0: the fiducial point
    thetas = [simu.THETA_FID._replace(**dict(zip(PARAMS, map(float, row)))) for row in P]
    # one set of buffers, sized at the heaviest point
    from classy_szlite.sim import halos as hl
    cells = hl.default_cells()
    lam_f = jax.jit(lambda t: simu.cosmology_stage(t, cells).lam.sum())
    lam = [float(lam_f(t)) for t in thetas]
    heavy = thetas[int(np.argmax(lam))]
    print(f"{len(P)} points (the fiducial one first); halos per tile {min(lam):.0f}-{max(lam):.0f}; buffers at {dict(zip(PARAMS, np.round(P[int(np.argmax(lam))], 4)))}", flush=True)
    path = os.path.join(out, "calib_grid_zs.npz" if args.size else "calib_grid.npz")
    os.makedirs(os.path.join(out, "points"), exist_ok=True)
    done = {}
    if os.path.exists(path):                                   # resume
        old = np.load(path)
        if np.array_equal(old["theta"], P):
            done = {int(i): (old["rows"][i], old["sigma"][i], old["cells"][i] if args.size else None) for i in np.flatnonzero(old["done"])}
            print(f"resuming: {len(done)} points done", flush=True)
    rows = np.full((len(P), len(ZB) - 1, 5), np.nan)
    sig = np.full((len(P), mt.N_TILES, 15), np.nan)
    flag = np.zeros(len(P), bool)
    cells = np.full((len(P), len(ZB) - 1, len(SB) - 1, 5), np.nan)
    for i, (r, sg, cl) in done.items():
        rows[i], sig[i], flag[i] = r, sg, True
        if cl is not None:
            cells[i] = cl
    t0 = time.time()
    for i, th in enumerate(thetas):
        if flag[i]:
            continue
        H, T = qcal.table(args.noise, args.mocks, theta=th, base_key=jax.random.fold_in(jax.random.key(SEED), i),
                          setup_theta=heavy, cap=args.cap, light=True, log=lambda m: None)
        rows[i], sig[i] = calib_fit(H, T)
        if args.size:
            cells[i] = calib_fit_cells(H)
        # keep the halos that matter (q_fsig > 1.5) for later refits without the GPU
        keep = H["q_fsig"] > 1.5
        np.savez_compressed(os.path.join(out, "points", f"point_{i:02d}.npz"),
                            **{k: H[k][keep] for k in ("mock", "tile", "z", "lnM", "theta500", "y0", "q_model", "q_fsig", "q_obs")},
                            sigma=T["sigma"], sigma_tile=T["tile"], theta=P[i])
        flag[i] = True
        np.savez(path, theta=P, rows=rows, sigma=sig, cells=cells, done=flag, params=np.array(PARAMS), z_upper=np.array(ZB[1:]),
                 size_upper=np.array(SB[1:]), lnth=T["lnth"], specs=T["specs"], noise=args.noise, mocks=args.mocks)
        r = rows[i]
        print(f"[{time.time() - t0:6.0f} s] point {i}: {dict(zip(PARAMS, np.round(P[i], 4)))}; {len(H)} halos; "
              f"sigma/fid at 2' {np.mean(sig[i][:, 7]) / np.mean(sig[0][:, 7]) if flag[0] else float('nan'):.4f}; "
              f"s {r[:5, 0].round(3).tolist()} k {r[:5, 1].round(2).tolist()} w {r[:5, 2].round(3).tolist()} f {r[:5, 3].round(4).tolist()}",
              flush=True)


def fit(args):
    import maps_trainset as mt
    from classy_szlite.sim import simulator as simu
    out = out_dir(args.noise)
    d = np.load(os.path.join(out, "calib_grid.npz"))
    assert d["done"].all(), "not all points done"
    P, rows, sig = d["theta"], d["rows"], d["sigma"]
    fid = np.array([getattr(simu.THETA_FID, p) for p in PARAMS])
    width = np.array([mt.UNIFORM[p][1] - mt.UNIFORM[p][0] for p in PARAMS])
    X = np.c_[np.ones(len(P)), (P - fid) / width]                       # (n, 5)
    lines = []

    def Pr(s=""):
        print(s, flush=True)
        lines.append(s)

    # ln sigma: tile mean, relative to the fiducial point's per-tile table
    lsig = np.log(sig.mean(1)) - np.log(sig[0].mean(0))[None, :]        # (n, 15)
    coef_sig, *_ = np.linalg.lstsq(X, lsig, rcond=None)                 # (5, 15)
    res_sig = lsig - X @ coef_sig
    Pr(f"ln sigma (relative to the fiducial point): spread over the points {lsig.std(0).min():.4f}-{lsig.std(0).max():.4f};"
       f" rms residual of the linear fit {np.sqrt((res_sig ** 2).mean(0)).min():.4f}-{np.sqrt((res_sig ** 2).mean(0)).max():.4f}")
    Pr("   gradient per prior width at 0.5', 2', 8' (sigma8, omega_cdm, A_SZ, alpha_SZ): "
       + "; ".join(" ".join(f"{v:+.3f}" for v in coef_sig[1:, j]) for j in (0, 7, 12)))
    # observation parameters per z bin
    names = ("s", "k", "w", "f")
    coef_obs = np.zeros((len(ZB) - 1, 4, 5))
    for b in range(len(ZB) - 1):
        for c in range(4):
            y = rows[:, b, c]
            cf, *_ = np.linalg.lstsq(X, y, rcond=None)
            coef_obs[b, c] = cf
            res = y - X @ cf
            # leave-one-out residuals
            loo = []
            for i in range(len(P)):
                m = np.arange(len(P)) != i
                cfi, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
                loo.append(y[i] - X[i] @ cfi)
            Pr(f"   z <= {ZB[b + 1]:5.2f} {names[c]}: fid {y[0]:+.4f}; spread {y.std():.4f}; fit {cf[0]:+.4f} + "
               + " ".join(f"{v:+.4f}" for v in cf[1:]) + f"; rms residual {np.sqrt((res ** 2).mean()):.4f}, leave-one-out {np.sqrt(np.mean(np.square(loo))):.4f}")
    # the CatalogueConfig.qcal_theta tuple pieces: sigma relative to the fiducial per-tile table, so the
    # base PatchTable is the fiducial point's per-tile sigma and d0 the fit's intercept
    np.savez(os.path.join(out, "calib_theta.npz"), fid=fid, width=width, g_sig=coef_sig[1:].T, d0_sig=coef_sig[0],
             rows=np.concatenate([np.array(ZB[1:])[:, None], coef_obs.reshape(len(ZB) - 1, 20)], 1),
             sigma=sig[0], lnth=d["lnth"], specs=d["specs"], noise=str(d["noise"]))
    open(os.path.join(out, "calib_theta.txt"), "w").write("\n".join(lines) + "\n")
    Pr(f"written {os.path.join(out, 'calib_theta.npz')}")


def fit_conf(args):
    """Variant (iv): the calibration through the finder's noise level u = ln(sigma at 2' / fiducial),
    quadratic in the scaled parameters; ln sigma at each size linear in u; s, k, w, f per z bin
    quadratic in u. Leave-one-out residuals against the linear-in-parameters model."""
    import maps_trainset as mt
    from classy_szlite.sim import simulator as simu
    out = out_dir(args.noise)
    d = np.load(os.path.join(out, "calib_grid_zs.npz" if args.size else "calib_grid.npz"))
    assert d["done"].all(), "not all points done"
    P, rows, sig = d["theta"], d["rows"], d["sigma"]
    n = len(P)
    if args.size:
        # one row per (z, size) cell, z-major; a cell with too few halos at a point takes that point's z-bin value
        C = d["cells"]                                                  # (n, nz, ns, 5)
        nz, ns = C.shape[1], C.shape[2]
        filled = np.where(np.isnan(C[..., :4]), rows[:, :, None, :4], C[..., :4])
        Pr0 = [f"(z, size) cells: halos with q > 2 at the fiducial point " + "; ".join(
            f"z<={ZB[i + 1]:.2f}: " + "/".join(str(int(C[0, i, j, 4])) for j in range(ns)) for i in range(nz))]
        Pr0.append("cells fitted at all 25 points: " + " ".join(f"z{i}s{j}" for i in range(nz) for j in range(ns) if not np.isnan(C[:, i, j, 0]).any()))
        rows = filled.reshape(n, nz * ns, 4)
        cell_upper = [(ZB[i + 1], SB[j + 1]) for i in range(nz) for j in range(ns)]
    else:
        Pr0 = []
        cell_upper = [(ZB[b + 1],) for b in range(len(ZB) - 1)]
    fid = np.array([getattr(simu.THETA_FID, p) for p in PARAMS])
    width = np.array([mt.UNIFORM[p][1] - mt.UNIFORM[p][0] for p in PARAMS])
    x = (P - fid) / width
    Fq = np.c_[np.ones(n), x, np.array([x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)]).T]   # (n, 15)
    Fl = np.c_[np.ones(n), x]
    lsig = np.log(sig.mean(1)) - np.log(sig[0].mean(0))[None, :]
    u = lsig[:, 7]
    lines = []

    def Pr(s=""):
        print(s, flush=True)
        lines.append(s)

    def loo(F, y):
        e = []
        for i in range(n):
            m = np.arange(n) != i
            c, *_ = np.linalg.lstsq(F[m], y[m], rcond=None)
            e.append(y[i] - F[i] @ c)
        return float(np.sqrt(np.mean(np.square(e))))
    for m_ in Pr0:
        Pr(m_)
    u_coef, *_ = np.linalg.lstsq(Fq, u, rcond=None)
    Pr(f"u = ln(noise at 2' / fiducial): spread {u.std():.4f}; leave-one-out rms: linear in the parameters {loo(Fl, u):.4f},"
       f" quadratic {loo(Fq, u):.4f}")
    Fu = np.c_[np.ones(n), u]
    sig_ab = np.array([np.linalg.lstsq(Fu, lsig[:, j], rcond=None)[0] for j in range(15)])
    Pr("ln sigma at each size against u: leave-one-out rms " + " ".join(f"{loo(Fu, lsig[:, j]):.4f}" for j in range(15)))
    Fu2 = np.c_[np.ones(n), u, u * u]
    out_rows = []
    for b, up in enumerate(cell_upper):
        r = list(up)
        msg = []
        for c, nm in enumerate("skwf"):
            y = rows[:, b, c]
            r += list(np.linalg.lstsq(Fu2, y, rcond=None)[0])
            msg.append(f"{nm} {y[0]:.3f} ({loo(Fl, y):.4f}->{loo(Fu2, y):.4f})")
        out_rows.append(r)
        Pr(f"   cell {' '.join(f'{v:g}' for v in up)}: fiducial value (leave-one-out rms, linear in the parameters -> quadratic in u): "
           + "; ".join(msg))
    name = "calib_conf_zs" if args.size else "calib_conf"
    np.savez(os.path.join(out, name + ".npz"), fid=fid, width=width, u_coef=u_coef, sig_ab=sig_ab, rows=np.array(out_rows),
             sigma=sig[0], lnth=d["lnth"], specs=d["specs"], noise=str(d["noise"]),
             size_upper=np.array(SB[1:]) if args.size else np.array([]))
    open(os.path.join(out, name + ".txt"), "w").write("\n".join(lines) + "\n")
    Pr(f"written {os.path.join(out, name + '.npz')}")


def load_calib_conf(path):
    """(CatalogueConfig.qcal_conf, PatchTable of the fiducial per-tile sigma) from calib_conf.npz."""
    import jax.numpy as jnp
    from classy_szlite.sim import catalogue as ct
    d = np.load(path)
    tup = lambda a: tuple(tuple(float(v) for v in r) for r in np.atleast_2d(a))
    qc = (tuple(map(float, d["fid"])), tuple(map(float, d["width"])), tuple(map(float, d["u_coef"])), tup(d["sig_ab"]), tup(d["rows"]))
    size = tuple(map(float, d["size_upper"])) if "size_upper" in d.files and len(d["size_upper"]) else None
    return qc, ct.PatchTable(lnth=jnp.asarray(d["lnth"]), lnsig=jnp.asarray(np.log(d["sigma"]))), size


def load_calib_theta(path):
    """(CatalogueConfig.qcal_theta, PatchTable of the fiducial per-tile sigma) from calib_theta.npz."""
    import jax.numpy as jnp
    from classy_szlite.sim import catalogue as ct
    d = np.load(path)
    tup = lambda a: tuple(tuple(float(v) for v in r) for r in np.atleast_2d(a))
    qt = (tuple(map(float, d["fid"])), tuple(map(float, d["width"])), tup(d["g_sig"]), tuple(map(float, d["d0_sig"])), tup(d["rows"]))
    return qt, ct.PatchTable(lnth=jnp.asarray(d["lnth"]), lnsig=jnp.asarray(np.log(d["sigma"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "fit"))
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--mocks", type=int, default=5)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--noise", choices=("mss2", "forecast"), default="forecast")
    ap.add_argument("--size", action="store_true", help="run: also fit per (z, size) cell (calib_grid_zs.npz); "
                    "fit --model conf: per (z, size) cell (calib_conf_zs.npz)")
    ap.add_argument("--model", choices=("theta", "conf"), default="theta",
                    help="theta: linear in the parameters (calib_theta.npz); conf: through the noise level (calib_conf.npz)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        fit(args) if args.model == "theta" else fit_conf(args)


if __name__ == "__main__":
    main()
