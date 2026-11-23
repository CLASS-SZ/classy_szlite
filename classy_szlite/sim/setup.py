"""Everything a 10-degree tile needs before it can be simulated: the paint tables, the CMB spectrum,
the static arrays and buffer sizes for a given parameter point.

    from classy_szlite.sim import setup
    ts = setup.tile_setup()            # fiducial point, emulator CMB, forecast-noise statics
    # ts.cells, ts.tables, ts.cosmo, ts.cfg, ts.static, ts.info

The paint tables (radial y profiles on a grid of sizes; about 17 MB, 1-3 min of scipy work the first
time) are cached in ``$CLASSY_SZLITE_CACHE`` (default ``~/.cache/classy_szlite``).

The CMB: by default the lensed TT spectrum of the package's CMB emulator at its default (Planck 2018)
point (instrument.cmb_cl_emulator). ``MAPSIM_CMB_TABLE`` may point to a CAMB table in the format of
instrument.cmb_cl_from_table instead (the published runs used one; it agrees with the emulator to
0.4% for l <= 3000).
"""
from __future__ import annotations

import os
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from . import halos as hl, paint as pt, simulator as simu, instrument as ins

DLN = float(os.environ.get("MAPSIM_PAINT_DLN", "0.004"))


def cache_dir():
    d = os.environ.get("CLASSY_SZLITE_CACHE", os.path.expanduser("~/.cache/classy_szlite"))
    os.makedirs(d, exist_ok=True)
    return d


def paint_tables(dln=None, directory=None, log=print):
    """The painter's radial tables (paint.build_tables), built once and cached."""
    dln = DLN if dln is None else dln
    path = os.path.join(directory or cache_dir(), f"paint_tables_dln{dln:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return pt.PaintTables(lnth0=float(d["lnth0"]), dln=float(d["dln"]), rP=d["rP"], tab=d["tab"],
                              th_split=float(d["th_split"]))
    t0 = time.time()
    tb = pt.build_tables(dln=dln)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **tb._asdict())
    os.replace(tmp, path)
    log(f"paint tables built in {time.time() - t0:.1f} s: {tb.tab.shape}, cached in {path}")
    return tb


_CMB = {}


def default_cmb():
    """C_l^TT [uK^2] of the lensed CMB as a function of l: the table at $MAPSIM_CMB_TABLE if set, else
    the emulator's (cached per process)."""
    path = os.environ.get("MAPSIM_CMB_TABLE", "")
    if path not in _CMB:
        _CMB[path] = ins.cmb_cl_from_table(path) if path else ins.cmb_cl_emulator()
    return _CMB[path]


def expected_occupation(lam, cg, theta, cells, tb):
    """Expected halos per painter footprint class, and on the direct (large-halo) path."""
    from ..sz_scaling import theta500_arcmin
    zc = 0.5 * (cells.z_edges[1:] + cells.z_edges[:-1])
    mc = 0.5 * (cells.lnM_edges[1:] + cells.lnM_edges[:-1])
    H0c = theta.h * 100.0 / 299792.458
    Ez = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Hz)) / H0c
    DA = np.interp(zc, np.asarray(cg.z), np.asarray(cg.Da))
    th = np.asarray(theta500_arcmin(np.exp(mc)[None, :], Ez[:, None], DA[:, None], theta.h, theta.one_minus_b))
    cls = np.asarray(pt.footprint_class(jnp.asarray(th.ravel()), tb, jnp.asarray(tb.rP))).reshape(th.shape)
    lam = np.asarray(lam)
    return np.array([lam[cls == c].sum() for c in range(len(pt.SIDES) + 1)])


class TileSetup(NamedTuple):
    cells: hl.CellGrid            # halo cells (z, ln M)
    tables: pt.PaintTables        # paint tables
    cosmo: simu.CosmoStage        # halo rates per cell and cosmology grids at theta
    cfg: simu.TileConfig          # buffer sizes (halos, painter, peaks)
    static: simu.TileStatic       # beams, CMB amplitude, forecast noise, filter, lensing table
    info: dict                    # expected halo numbers and buffer sizes


def tile_setup(theta=simu.THETA_FID, cmb_cl=None, margin=1.5, mode="goal", dln=None, log=print):
    """Buffers sized for theta (with ``margin`` on the painter's expected occupation and 6 sigma plus
    1024 slots on the halo count), the static arrays with the CMB ``cmb_cl`` (default_cmb()) and the
    forecast noise at depth ``mode`` ("goal" or "baseline")."""
    cells = hl.default_cells()
    tb = paint_tables(dln, log=log)
    cs = jax.jit(lambda th: simu.cosmology_stage(th, cells))(theta)
    occ = expected_occupation(cs.lam, cs.cg, theta, cells, tb)
    lam_tot = float(np.sum(cs.lam))
    halo_cap = int(np.ceil((lam_tot + 6 * np.sqrt(lam_tot) + 1024) / 4096) * 4096)
    cap_big = int(np.ceil((margin * occ[-1] + 64) / 32) * 32)
    pcfg = pt.make_config(tb, occ[:-1], cap_big, margin=margin)
    cfg = simu.TileConfig(halo_cap=halo_cap, paint=pcfg)
    st = simu.tile_static(cmb_cl or default_cmb(), tb, mode=mode)
    info = dict(lambda_total=lam_tot, halo_cap=halo_cap, occupation=occ.round(1).tolist(), caps=list(pcfg.caps),
                blocks=list(pcfg.blocks), cap_big=cap_big,
                contributions=float(sum(o * s * s for o, s in zip(occ[:-1], pt.SIDES))))
    return TileSetup(cells, tb, cs, cfg, st, info)
