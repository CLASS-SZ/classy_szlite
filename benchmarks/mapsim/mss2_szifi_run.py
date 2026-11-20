"""SZiFi itself on the MSS2 comparison maps (``mss2_szifi_maps.py``), in three configurations.

    $SZIFI_PY -I -B benchmarks/mapsim/mss2_szifi_run.py [--workers 12] [--configs match,match4,coll]
                                                         [--tiles ...] [--depths split0,mean]

SZiFi (github.com/inigozubeldia/szifi, commit 73626fb; the clone at $SZIFI_DIR) is imported
unchanged and only called; no SZiFi code is copied (it has no licence). Run it with SZiFi's own
environment and -I -B (no bytecode written into the clone). One SZiFi run per process, one thread.

Configurations (all: blind search, detection at q > 4, SZiFi's noise iteration with max_it 1,
masking discs of 3 theta_500, inpainting with l_sep 12000, spectra in bins of 72 in ell
(bin_fac 4) with no decoupling, the taper as mask_map and mask_ps):
  match    set as our finder: the map our finder filters (tapered first, then the multipole band;
           SZiFi on its own cuts the band before it tapers, which removes MSS2's very red noise
           below ell 100 before the taper can leak it); the six channels; our 15 scales
           (0.5'-15'); isotropic binned covariance; ell 100-20000 with SZiFi's inverse covariance
           set to 0 outside it (our finder uses in-band modes only; without it SZiFi amplifies
           float32 round-off there, SZIFI_REVIEW.md E4); masking at q > 4.3 (the cut-outs'
           threshold, as ours) with uncapped discs; diffusive inpainting, 100 steps (the set-up
           whose counts matched our finder's within 2% in sz-poisson-blending's SZiFi check).
  match4   match with the four high-frequency channels (93, 145, 225, 280 GHz).
  coll     the colleagues' settings (run_szifi_so_full_car_FP.py run as "0 lowmass
           fgbl_0p8_m260923 8 ag 3 100 20000 C2 25 gal 4.0"): freqs [0, 1, 2, 3] of SZiFi's SO
           experiment, i.e. the four high-frequency channels; 100 scales (70 from 0.5' to 5', 30
           more to 80'); anisotropic Gaussian covariance with a kernel of 3 Fourier pixels and no
           decoupling (their script switches it off for this covariance); lrange [100, 20000] and
           powspec_lmax1d 20000, as configured (no out-of-band fix); q_th_noise 3.8, below the
           detection threshold 4, so every detection is masked (effectively q > 4, as in their
           runs), with discs of 3 theta_500 clipped to 1'-10'; orphics inpainting (n_inpaint -1,
           its random fill seeded per run here). SZiFi cuts the band, then tapers (the taper as
           mask_map and mask_ps), as with an apodised survey mask.
           Not applicable here, or not in SZiFi 73626fb: CAR tiles with a 30' catalogue buffer and a
           C2 25' apodisation (our 10-degree cut-out with its 30' cosine taper instead, square
           pixels, kernel [3, 3]); the Galactic mask (these maps have no Galaxy); the Websky
           cosmology (Planck15 here: in find mode SZiFi converts theta_500 to a mass at fixed z and
           normalises the template to its centre, so the cosmology cancels); their fork's
           'template_eval_type' hankel and 'use_q_tensor_interp' (SZiFi's standard templates);
           fixed mode (a blind search here).

Recorded per run (``szifi/runs/{cfg}_tile_{tile}_{depth}.npz``): both iterations' catalogues
(q, y0, theta_500, the scale index, row i and column j from SZiFi's own coordinates), sigma per
scale (SZiFi's 'maxima_lomem' leaves sigma at the first scale at 0: there it is the median
y0 / q of that scale's detections), the unmasked fraction and the number of masked detections of
the masking step, its mask (packed bits), the number of modes set to 0 outside the band, wall
times, peak memory, and the full parameter dictionary. SZiFi's printout goes to szifi/stdout/.
"""
import argparse
import zlib
import concurrent.futures as cfu
import contextlib
import copy
import io
import json
import multiprocessing as mp
import os
import resource
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("MAPSIM_OUT", "."), "mss2", "szifi_compare")
SZIFI = os.environ.get("SZIFI_DIR", "/mnt/ceph/users/bbolliet/szifi")
SURVEY = os.path.join(HERE, "mss2_szifi_survey.py")
RUND, STDD = os.path.join(OUT, "szifi", "runs"), os.path.join(OUT, "szifi", "stdout")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(OUT, "szifi", "numba_cache"))
if SZIFI not in sys.path:
    sys.path.insert(0, SZIFI)

import numpy as np  # noqa: E402

N, DX = 1200, 0.5 / 60. / 180. * np.pi
CH6, CH4 = [0, 1, 2, 3, 4, 5], [2, 3, 4, 5]
THETAS15 = np.geomspace(0.5, 15.0, 15)                     # our finder's (mmf.THETAS15)
THETAS_COLL = np.concatenate([np.exp(np.linspace(np.log(0.5), np.log(5.), 70)),
                              np.exp(np.linspace(np.log(5.), np.log(80.), 31))[1:]])
BASE = dict(q_th=4.0, iterative=True, max_it=1, mask_radius=3.0, beam="gaussian", integrate_bandpass=False,
            get_lonlat=False, save_and_load_template=False, detection_method="maxima_lomem", extraction_mode="find",
            snr_weighting=False, estimate_spec="estimate", interp_type="nearest", norm_type="centre", apod_type="old",
            mmf_type="standard", cmmf_type=None, a_matrix=None, comp_to_calculate=[0], deproject_cib=None,
            tile_type="healpix", tilemask_mode="field", get_q_true=False, powspec_bin_fac=4, powspec_new_shape=None,
            inpaint=True, lsep=12000, decouple_type="none", coupling_matrix_needed=False,
            compute_coupling_matrix=False, save_coupling_matrix=False, lrange=[100, 20000], map_dtype=np.float32,
            save_snr_maps=False, rSZ=False)
CONFIGS = {
    "match": dict(channels=CH6, theta_500_vec_arcmin=THETAS15, cov_type="isotropic", powspec_lmax1d=None,
                  q_th_noise=4.3, min_radius_mask_arcmin=0.0, max_radius_mask_arcmin=np.inf,
                  inpaint_type="diffusive", n_inpaint=100, zero_oob=True, pretaper=True),
    "coll": dict(channels=CH4, theta_500_vec_arcmin=THETAS_COLL, cov_type="anisotropic_gaussian",
                 cov_kernel_shape=[3, 3], powspec_lmax1d=20000, q_th_noise=3.8, min_radius_mask_arcmin=1.0,
                 max_radius_mask_arcmin=10.0, inpaint_type="orphics", n_inpaint=-1, zero_oob=False, pretaper=False),
}
CONFIGS["match4"] = dict(CONFIGS["match"], channels=CH4)
COST = {"coll": 10.0, "match": 1.5, "match4": 1.2}          # relative run time, for the order of submission

FLAGS = dict(zero_oob=False, lrange=None)
REC = {}


def rec_reset():
    REC.clear()
    REC.update(fsky=[], n_masked=[], mask=None, n_oob=[], t_invcov=[], t_filter=[])


def install_wrappers():
    """Thin wrappers around SZiFi functions that SZiFi looks up at call time: they record what
    passes through and, for zero_oob, set the inverse covariance to 0 outside lrange (exactly the
    modes SZiFi zeroes in the maps: l < lmin or l > lmax). Return values are otherwise unchanged."""
    from szifi import spec, mmf, maps
    if getattr(spec.cross_spec.get_inv_cov, "_mss2", False):
        return
    orig_invcov, orig_mask, orig_find = spec.cross_spec.get_inv_cov, mmf.get_cluster_mask, mmf.filter_maps.find_clusters

    def w_invcov(self, pix, *a, **k):
        t = time.time()
        out = orig_invcov(self, pix, *a, **k)
        if FLAGS["zero_oob"]:
            ell = maps.rmap(pix).get_ell()
            if out.shape[:2] != ell.shape:
                raise ValueError("zero_oob needs a full-resolution inverse covariance")
            lmin, lmax = FLAGS["lrange"]
            oob = (ell < lmin) | (ell > lmax)
            out[oob] = 0.0
            REC["n_oob"].append(int(oob.sum()))
        REC["t_invcov"].append(time.time() - t)
        return out

    def w_mask(pix, catalogue, q_th_noise, mask_radius, **k):
        m = orig_mask(pix, catalogue, q_th_noise, mask_radius, **k)
        REC["fsky"].append(float(np.mean(m)))
        REC["n_masked"].append(int(np.sum(np.asarray(catalogue.catalogue["q_opt"]) > q_th_noise)))
        if REC["mask"] is None:                             # the mask of the second pass (computed once, max_it 1)
            REC["mask"] = np.packbits(np.asarray(m) > 0.5)
        return m

    def w_find(self, *a, **k):
        t = time.time()
        out = orig_find(self, *a, **k)
        REC["t_filter"].append(time.time() - t)
        return out
    w_invcov._mss2 = True
    spec.cross_spec.get_inv_cov = w_invcov
    mmf.get_cluster_mask = w_mask
    mmf.filter_maps.find_clusters = w_find


def make_params(cfg):
    import szifi
    p = copy.deepcopy(szifi.params_szifi_default)
    p.update(copy.deepcopy(BASE))
    c = CONFIGS[cfg]
    p.update({k: copy.deepcopy(v) for k, v in c.items() if k not in ("channels", "zero_oob", "pretaper")})
    p["freqs"] = list(range(len(c["channels"])))             # the survey file passes only these channels
    p.update(survey_file=SURVEY, path=SZIFI + "/", path_data=os.path.join(OUT, "maps") + "/", path_template=None)
    return p


def jsonable(p):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        if isinstance(v, type):
            return v.__name__
        return v
    return {k: conv(v) for k, v in p.items()}


def cat_arrays(c, thetas):
    """SZiFi catalogue -> arrays. Pixel indices from SZiFi's own coordinates (theta_x = (j + 1/2) dx,
    theta_y = (N - i - 1/2) dx): i = row (our iy), j = column (our ix)."""
    d = c.catalogue
    q = np.asarray(d["q_opt"], float)
    tx, ty = np.asarray(d["theta_x"], float), np.asarray(d["theta_y"], float)
    jj, ii = tx / DX - 0.5, N - 0.5 - ty / DX
    j, i = np.rint(jj).astype(np.int64), np.rint(ii).astype(np.int64)
    if q.size:
        assert np.abs(jj - j).max() < 1e-6 and np.abs(ii - i).max() < 1e-6
    th = np.asarray(d["theta_500"], float)
    jth = np.abs(np.log(th)[:, None] - np.log(thetas)[None, :]).argmin(1) if q.size else np.zeros(0, np.int64)
    return dict(q=q, y0=np.asarray(d["y0"], float), theta=th, j_theta=jth, i=i, j=j)


def run_one(job):
    cfg, ti, depth = job
    import szifi
    install_wrappers()
    rec_reset()
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    c = CONFIGS[cfg]
    p = make_params(cfg)
    FLAGS.update(zero_oob=bool(c["zero_oob"]), lrange=list(p["lrange"]))
    fid = f"tile_{ti:03d}_{depth}"
    pdata = {"data_set": "mapsim MSS2 comparison", "field_ids": [fid],
             "map_files": {fid: os.path.join(OUT, "maps", f"{fid}.npy")}, "channels": list(c["channels"]),
             "instrument_file": os.path.join(OUT, "instrument.json"), "white": meta["white"][str(ti)][depth],
             "taper_file": os.path.join(OUT, "taper.npy"), "pretaper": bool(c["pretaper"])}
    seed = zlib.crc32(f"{cfg}_{fid}".encode())               # orphics inpainting draws from numpy's global generator
    np.random.seed(seed)
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = szifi.input_data(params_szifi=p, params_data=pdata)
        cf = szifi.cluster_finder(params_szifi=p, params_model=copy.deepcopy(szifi.params_model_default),
                                  data_file=data, rank=0)
        cf.find_clusters()
    t_run = time.time() - t0
    with open(os.path.join(STDD, f"{cfg}_{fid}.txt"), "w") as fh:
        fh.write(buf.getvalue())
    res = cf.results_dict[fid]
    thetas = np.asarray(p["theta_500_vec_arcmin"], float)
    out = dict(cfg=cfg, tile=ti, depth=depth, thetas=thetas, channels=np.asarray(c["channels"]),
               pretaper=bool(c["pretaper"]), seed=seed, params_json=json.dumps(jsonable(p)), t_run=t_run,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6))
    n_it = 0
    for it in (0, 1):
        key = f"catalogue_find_{it}"
        if key not in res.catalogues:
            continue
        n_it = it + 1
        ca = cat_arrays(res.catalogues[key], thetas)
        for k, v in ca.items():
            out[f"it{it}_{k}"] = v
        sv = np.asarray(res.sigma_vec[f"find_{it}"], float)
        sc = np.full(thetas.size, np.nan)
        for jj in range(thetas.size):
            m = ca["j_theta"] == jj
            if m.any():
                sc[jj] = np.median(ca["y0"][m] / ca["q"][m])
        sig = sv.copy()
        sig[0] = sc[0]
        out[f"it{it}_sigvec"], out[f"it{it}_sig_cat"], out[f"it{it}_sig"] = sv, sc, sig
    out["n_it"] = n_it
    out["rec_fsky"] = np.asarray(REC["fsky"], float)
    out["rec_n_masked"] = np.asarray(REC["n_masked"], int)
    out["rec_n_oob"] = np.asarray(REC["n_oob"], int)
    out["rec_t_invcov"] = np.asarray(REC["t_invcov"], float)
    out["rec_t_filter"] = np.asarray(REC["t_filter"], float)
    if REC["mask"] is not None:
        out["mask"] = REC["mask"]
    np.savez(os.path.join(RUND, f"{cfg}_{fid}.npz"), **out)
    q1 = out.get("it1_q", np.zeros(0))
    return dict(cfg=cfg, tile=ti, depth=depth, t_run=round(t_run, 1), peak_rss_gb=round(out["peak_rss_gb"], 2),
                n_it=n_it, n_det_it1=int(q1.size), n_det_it1_q5=int(np.sum(q1 > 5)),
                unmasked=out["rec_fsky"].tolist(), n_oob=out["rec_n_oob"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--configs", default="coll,match,match4")
    ap.add_argument("--tiles", default="")
    ap.add_argument("--depths", default="", help="default: the depths in meta.json")
    args = ap.parse_args()
    for d in (RUND, STDD, os.environ["NUMBA_CACHE_DIR"]):
        os.makedirs(d, exist_ok=True)
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    tiles = [int(t) for t in args.tiles.split(",") if t] or meta["tiles"]
    cfgs = [c for c in args.configs.split(",") if c]
    assert all(c in CONFIGS for c in cfgs), cfgs
    depths = [d for d in args.depths.split(",") if d] or meta["depths"]
    jobs = [(c, t, d) for c in cfgs for t in tiles for d in depths]
    jobs.sort(key=lambda j: -COST[j[0]])                     # the longest first
    print(f"{len(jobs)} SZiFi runs on {args.workers} workers: configs {cfgs}, tiles {tiles}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with cfu.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        summ = []
        for f in cfu.as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
                summ.append(r)
                print(f"[{time.time() - t0:7.0f} s] {r['cfg']:6s} tile {r['tile']:3d} {r['depth']:6s}: "
                      f"{r['t_run']:6.0f} s, {r['peak_rss_gb']:.2f} GB, iterations {r['n_it']}, detections "
                      f"{r['n_det_it1']} (q>5: {r['n_det_it1_q5']}), unmasked {r['unmasked']}, out of band {r['n_oob']}",
                      flush=True)
            except Exception as exc:
                print(f"[{time.time() - t0:7.0f} s] FAILED {j}: {exc!r}", flush=True)
                summ.append(dict(cfg=j[0], tile=j[1], depth=j[2], error=repr(exc)))
    json.dump(summ, open(os.path.join(OUT, "szifi", "runs_summary.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s; {sum('error' in s for s in summ)} failed", flush=True)


if __name__ == "__main__":
    main()
