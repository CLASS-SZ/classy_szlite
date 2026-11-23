"""Command-line pipeline of the cluster-count simulations: ``mapsim <command>`` (or
``python -m classy_szlite.sim <command>``).

    mapsim tile        --out OUT                       one tile through every stage (maps and catalogue)
    mapsim maps        --out OUT --set truth|heldout|train [--tiles 41] [--n N]
    mapsim calibrate   --out OUT point|grid|fit-noise|fit-theta [--size] [--mocks M] [--points P]
    mapsim catalogues  --out OUT --model published|calibrated-point|calibrated-noise|calibrated-theta|paper
                       --set truth|heldout|train [--tiles 41]
    mapsim train       --out OUT --data maps|<catalogue model> [--tiles 41] [--seed S] [--name NAME]
    mapsim compare     --out OUT [--tiles 41]

Everything is written under OUT:

    maps_t{tiles}/{set}.npz                      map-level sets (forecast noise)
    calibration/point.npz, grid/, noise.npz, theta.npz
    catalogues/{model}_t{tiles}/{set}.npz        catalogue-level sets
    npe/{data}_t{tiles}/{name}.json, {name}_{applied set}.npz
    compare_t{tiles}.json                        bias and coverage of every trained posterior

Each step resumes where it stopped. The noise is the public 2019 SO forecast (goal depth) throughout;
``--noise mss2 --spectra PATH`` selects measured per-tile noise spectra instead (maps, calibrate).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


def log(*a):
    print(*a, flush=True)


def cmd_tile(a):
    _jax()
    from . import tile
    o = tile.stages(seed=a.seed, mock=a.mock, tile=a.tile, log=log)
    path = os.path.join(a.out, f"tile_m{a.mock}_t{a.tile}.npz")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in o.items() if not isinstance(v, dict)},
                        **{f"peaks_{k}": v for k, v in o["peaks"].items()}, **{f"halos_{k}": v for k, v in o["halos"].items()})
    p = o["peaks"]
    log(f"{len(o['halos']['z'])} halos above 5e13 Msun; {int(np.sum(p['confirmed'] & (p['q'] > 5)))} confirmed detections "
        f"with q > 5; finder noise at 2' / instrument + CMB: {o['sigma_final'][7] / o['sigma_model'][7]:.3f}; written {path}")


def cmd_maps(a):
    _jax()
    from . import survey, priors
    theta = priors.draws(a.set, a.n, box="map", seeds=priors.SEEDS_MAP)
    path = os.path.join(a.out, f"maps_t{a.tiles}" + ("" if a.noise == "forecast" else f"_{a.noise}"), f"{a.set}.npz")
    survey.simulate_map_set(path, theta, priors.SEEDS_MAP[a.set], n_tiles=a.tiles, noise=a.noise, spectra=a.spectra,
                            chunk=a.chunk, log=log)


def cmd_calibrate(a):
    _jax()
    from . import calibration as cal
    d = os.path.join(a.out, "calibration")
    os.makedirs(d, exist_ok=True)
    if a.what == "point":
        H, T = cal.halo_table(a.mocks, n_tiles=a.tiles, noise=a.noise, spectra=a.spectra, log=log)
        rows = cal.calibrate_point(H, T, os.path.join(d, "point.npz"), noise=a.noise)
        log("calibration at the fiducial point (z_upper, s, k, w, f):")
        for z, r in zip(cal.ZB[1:], rows):
            log(f"   {z:5.2f}  " + "  ".join(f"{v:.4f}" for v in r[:4]))
    elif a.what == "grid":
        cal.calibrate_grid(os.path.join(d, "grid"), n_points=a.points, n_mocks=a.mocks, size=a.size, n_tiles=a.tiles,
                           noise=a.noise, spectra=a.spectra, log=log)
    elif a.what == "fit-noise":
        loo = cal.fit_noise_level(os.path.join(d, "grid", "calib_grid.npz"), os.path.join(d, "noise.npz"), size=a.size, log=log)
        json.dump(loo, open(os.path.join(d, "noise_loo.json"), "w"), indent=1)
    elif a.what == "fit-theta":
        cal.fit_theta(os.path.join(d, "grid", "calib_grid.npz"), os.path.join(d, "theta.npz"), log=log)


CAL_FILES = {"calibrated-point": "point.npz", "calibrated-noise": "noise.npz", "calibrated-theta": "theta.npz"}


def cmd_catalogues(a):
    _jax()
    from . import survey, priors
    cal = os.path.join(a.out, "calibration", CAL_FILES[a.model]) if a.model in CAL_FILES else None
    sz, le, cfg = survey.catalogue_model(a.model, n_tiles=a.tiles, calibration=cal)
    box = "paper" if a.model == "paper" else "map"
    theta = priors.draws(a.set, a.n, box=box, seeds=priors.SEEDS_CATALOGUE)
    path = os.path.join(a.out, "catalogues", f"{a.model}_t{a.tiles}", f"{a.set}.npz")
    survey.simulate_catalogue_set(path, theta, priors.SEEDS_CATALOGUE[a.set], sz, le, cfg, log=log,
                                  info=dict(model=a.model, n_tiles=a.tiles))


def _data_dir(a):
    return (os.path.join(a.out, f"maps_t{a.tiles}") if a.data == "maps"
            else os.path.join(a.out, "catalogues", f"{a.data}_t{a.tiles}"))


def cmd_train(a):
    from . import npe, priors
    data = _data_dir(a)
    th, x = npe.load_set(os.path.join(data, "train.npz"))
    th, x = th[:a.n_train], x[:a.n_train]
    box = "paper" if a.data == "paper" else "map"
    post, info = npe.train(th, x, box=box, seed=a.seed, log=log)
    out = os.path.join(a.out, "npe", f"{a.data}_t{a.tiles}")
    os.makedirs(out, exist_ok=True)
    name = a.name or f"seed{a.seed}"
    # applied to the map-level mocks and prior draws (the test), and to its own held-out draws (calibration)
    maps = os.path.join(a.out, f"maps_t{a.tiles}")
    sets = {"mocks": os.path.join(maps, "truth.npz"), "prior": os.path.join(maps, "heldout.npz"),
            "own": os.path.join(data, "heldout.npz")}
    for k, path in sets.items():
        if not os.path.exists(path):
            log(f"{k}: {path} absent, skipped")
            continue
        t, xx = npe.load_set(path)
        s = npe.summarise(post, t, xx, a.samples)
        np.savez(os.path.join(out, f"{name}_{k}.npz"), **s)
        b, cov = npe.bias(s), npe.coverage(s)
        info[k] = dict(bias=b["bias"].tolist(), err=b["err"].tolist(), bias_derived=b["bias_derived"].tolist(),
                       coverage=cov.tolist())
        log(f"{k}: bias " + " ".join(f"{priors.NAMES[j]} {b['bias'][j]:+.2f}" for j in range(4))
            + "; 90% coverage " + " ".join(f"{cov[-1, j]:.2f}" for j in range(4)))
    json.dump(info, open(os.path.join(out, f"{name}.json"), "w"), indent=1)


def cmd_compare(a):
    from . import npe, priors
    res = {}
    for d in sorted(glob.glob(os.path.join(a.out, "npe", f"*_t{a.tiles}"))):
        model = os.path.basename(d)[: -len(f"_t{a.tiles}")]
        entry = {}
        for k in ("mocks", "prior", "own"):
            files = sorted(glob.glob(os.path.join(d, f"*_{k}.npz")))
            if not files:
                continue
            s = npe.ensemble([dict(np.load(f)) for f in files])
            b, cov = npe.bias(s), npe.coverage(s)
            entry[k] = dict(networks=len(files), bias=b["bias"].tolist(), err=b["err"].tolist(),
                            bias_derived=b["bias_derived"].tolist(), coverage=cov.tolist())
        res[model] = entry
        if "mocks" in entry:
            e = entry["mocks"]
            log(f"{model:18s} ({e['networks']} networks) bias at the truth: "
                + " ".join(f"{priors.LABELS[j]} {e['bias'][j]:+.2f}±{e['err'][j]:.2f}" for j in range(4))
                + ("; 90% coverage over the prior: " + " ".join(f"{entry['prior']['coverage'][-1][j]:.2f}" for j in range(4))
                   if "prior" in entry else ""))
    json.dump(res, open(os.path.join(a.out, f"compare_t{a.tiles}.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mapsim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, tiles=True):
        p.add_argument("--out", required=True, help="output folder")
        if tiles:
            p.add_argument("--tiles", type=int, default=41, help="survey size in 10-degree tiles (41: 10%% of the sky; 168: 40.7%%)")
        return p
    p = common(sub.add_parser("tile", help="one tile through every stage"), tiles=False)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--mock", type=int, default=0)
    p.add_argument("--tile", type=int, default=0)
    p = common(sub.add_parser("maps", help="map-level simulations"))
    p.add_argument("--set", choices=("truth", "heldout", "train"), required=True)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--chunk", type=int, default=21, help="tiles per vectorised batch")
    p.add_argument("--noise", choices=("forecast", "mss2"), default="forecast")
    p.add_argument("--spectra", default=None)
    p = common(sub.add_parser("calibrate", help="calibrate the catalogue model on map-level mocks"))
    p.add_argument("what", choices=("point", "grid", "fit-noise", "fit-theta"))
    p.add_argument("--mocks", type=int, default=None, help="mocks per point (default 20 for point, 5 for grid)")
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--size", action="store_true", help="also per (redshift, size) cell")
    p.add_argument("--noise", choices=("forecast", "mss2"), default="forecast")
    p.add_argument("--spectra", default=None)
    p = common(sub.add_parser("catalogues", help="catalogue-level simulations"))
    p.add_argument("--model", choices=("published", "calibrated-point", "calibrated-noise", "calibrated-theta", "paper"), required=True)
    p.add_argument("--set", choices=("truth", "heldout", "train"), required=True)
    p.add_argument("--n", type=int, default=None)
    p = common(sub.add_parser("train", help="train a posterior and apply it to the map-level mocks and prior draws"))
    p.add_argument("--data", required=True, help="'maps' or a catalogue model")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--name", default="")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--samples", type=int, default=4000)
    common(sub.add_parser("compare", help="bias and coverage of every trained posterior (ensembles over seeds)"))
    a = ap.parse_args(argv)
    if a.cmd == "calibrate" and a.mocks is None:
        a.mocks = 20 if a.what == "point" else 5
    {"tile": cmd_tile, "maps": cmd_maps, "calibrate": cmd_calibrate, "catalogues": cmd_catalogues,
     "train": cmd_train, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
