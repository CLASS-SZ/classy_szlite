"""SZiFi 'survey file' for the MSS2 comparison maps (``mss2_szifi_maps.py``). SZiFi's input_data
loads it from params_szifi['survey_file'] and needs the class input_data_survey. Own code: the
dictionary keys are the ones SZiFi's cluster_finder reads; no SZiFi code is copied.

One field per call, params_data:
  field_ids        [fid]
  map_files        {fid: (1200, 1200, 6) float32 .npy in uK, axis 0 = rows (y), axis 1 = columns}
  channels         indices of the channels to use, of 27, 39, 93, 145, 225, 280 GHz
  instrument_file  instrument.json: nu_eff_GHz, fwhm_arcmin, tsz_sed_uK (the simulator's, six channels)
  white            (6,) the map's white levels in uK-arcmin per square 0.5' pixel, as SZiFi reads the
                   map (SZiFi uses them in orphics inpainting)
  taper_file       taper.npy: our cut-out taper (cosine over the outer 30')
  pretaper         True: the map is tapered here and SZiFi gets ones as mask_map and mask_ps, so it
                   filters the map our finder filters (taper, then the multipole band). False: SZiFi
                   gets the untapered map and the taper as mask_map and mask_ps, as a survey with an
                   apodised mask would (SZiFi then cuts the band before it tapers).

Masks: the taper (on the map, or as mask_map and mask_ps); no correction for the taper's power loss
either way, as in our finder; ones for point sources and the tile; ones for peak finding and
selection (the comparison keeps detections more than 60' from the edges afterwards, as our data
vector does).
Pixels: square, 0.5', SZiFi's flat tile ('healpix' tile type), as our finder treats the cut-outs.
Experiment: SZiFi's 'SOgoal_simple' object with nu_eff, FWHM, noise levels and tSZ SED replaced by
ours for the chosen channels.
"""
import json

import numpy as np
from szifi import maps, expt

N_PIX = 1200
DX_ARCMIN = 0.5
KEYS = ("nx", "ny", "dx_arcmin", "dy_arcmin", "pix", "t_obs", "t_noi", "mask_map", "mask_point", "mask_select",
        "mask_select_no_tile", "mask_select_buffer", "mask_ps", "mask_peak_finding", "mask_peak_finding_no_tile",
        "mask_tile", "coupling_matrix_name")


class input_data_survey:

    def __init__(self, params_szifi=None, params_data=None):
        self.data = {"params_data": params_data}
        for k in KEYS:
            self.data[k] = {}
        inst = json.load(open(params_data["instrument_file"]))
        ch = [int(c) for c in params_data["channels"]]
        taper = np.load(params_data["taper_file"])
        if taper.shape != (N_PIX, N_PIX):
            raise ValueError(f"unexpected taper {taper.shape}")
        dx = DX_ARCMIN / 60. / 180. * np.pi
        for fid in params_data["field_ids"]:
            tmap = np.load(params_data["map_files"][fid])
            if tmap.shape != (N_PIX, N_PIX, 6) or tmap.dtype != np.float32:
                raise ValueError(f"unexpected map {tmap.shape} {tmap.dtype}")
            tmap = np.ascontiguousarray(tmap[:, :, ch])
            pre = bool(params_data.get("pretaper", False))
            if pre:
                tmap = (tmap.astype(np.float64) * taper[:, :, None]).astype(np.float32)
            self.data["nx"][fid] = N_PIX
            self.data["ny"][fid] = N_PIX
            self.data["dx_arcmin"][fid] = DX_ARCMIN
            self.data["dy_arcmin"][fid] = DX_ARCMIN
            self.data["pix"][fid] = maps.pixel(N_PIX, dx)
            self.data["t_obs"][fid] = tmap
            self.data["t_noi"][fid] = tmap            # noise estimated from the data, as our finder does
            ones = np.ones((N_PIX, N_PIX))
            self.data["mask_map"][fid] = ones.copy() if pre else taper.copy()
            self.data["mask_ps"][fid] = ones.copy() if pre else taper.copy()
            self.data["mask_point"][fid] = ones.copy()
            self.data["mask_tile"][fid] = ones.copy()
            for k in ("mask_select", "mask_select_no_tile", "mask_peak_finding", "mask_peak_finding_no_tile"):
                self.data[k][fid] = ones.copy()
            self.data["mask_select_buffer"][fid] = 0
            self.data["coupling_matrix_name"][fid] = None
        e = expt.experiment(experiment_name="SOgoal_simple", params_szifi=params_szifi)
        e.nu_eff = np.asarray(inst["nu_eff_GHz"], float)[ch] * 1e9
        e.FWHM = np.asarray(inst["fwhm_arcmin"], float)[ch]
        e.noise_levels = np.asarray(params_data["white"], float)[ch]
        e.tsz_sed = np.asarray(inst["tsz_sed_uK"], float)[ch]
        e.n_freqs = len(ch)
        self.data["experiment"] = e
