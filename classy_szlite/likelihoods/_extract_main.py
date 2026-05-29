"""Extract ALL numerical tables from cobaya likelihoods into one npz.

Covers:
  - planck_2018_lowl.TT    : spline tables, covinv, mu, offset
  - planck_2018_lowl.EE_sroll2 : probEE table
  - act_dr6_cmbonly.PlanckActCut : X_data, invcov, weights, blmin/blmax
  - act_dr6_mflike.ACTDR6MFLike  : data_vec, inv_cov, l_bpws, spec_meta,
                                    bands, beams, fg totals at best-fit
                                    + window function matrices per spectrum
"""
import os, sys
from pathlib import Path
import numpy as np
import yaml

# Resolve paths from the environment if possible (overrideable from the
# extract_data CLI wrapper) so users on a different machine don't need to
# edit the source.
CHAIN_YAML = Path(os.environ.get(
    "CLASSY_SZLITE_LIK_CHAIN_YAML",
    "/Users/boris/GitHub/act-dr6-ede-analysis/p-actbase_ede+n3_classsz/p-actbase_ede+n3_classsz.input.yaml"
))
PACKAGES_PATH = os.environ.get(
    "CLASSY_SZLITE_LIK_PACKAGES_PATH",
    "/Users/boris/cobaya_packages"
)
OUT = Path(os.environ.get(
    "CLASSY_SZLITE_LIK_OUT_NPZ",
    str(Path.home() / ".classy_szlite" / "likelihood_data.npz")
))
OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"  reading chain yaml: {CHAIN_YAML}")
print(f"  cobaya packages_path: {PACKAGES_PATH}")
print(f"  writing npz to:     {OUT}")

with open(CHAIN_YAML) as f:
    info = yaml.safe_load(f)

mfl = info["likelihood"]["act_dr6_mflike.ACTDR6MFLike"]
mfl["data_folder"] = "ACTDR6MFLike/v1.0"
mfl["input_file"] = "dr6_data.fits"
info["packages_path"] = PACKAGES_PATH
info["sampler"] = {"evaluate": None}
info.pop("output", None); info.pop("resume", None); info.pop("force", None)
for block in ("theory", "likelihood"):
    for cname, cval in info.get(block, {}).items():
        if isinstance(cval, dict):
            for k in ("output_params", "version", "aliases", "speed", "type",
                      "stop_at_error", "use_renames", "ignore_obsolete", "renames"):
                cval.pop(k, None)
info["theory"]["classy_szfast.classy_sz.classy_sz"]["extra_args"].pop("r", None)

print("Building cobaya model …")
from cobaya.model import get_model
model = get_model(info)
lks = model.likelihood
print("Likelihoods:", list(lks))

BESTFIT = dict(
    omega_b=0.022751597, omega_cdm=0.13242228, H0=71.113655,
    fEDE=0.11678514, log10z_c=3.5517577, thetai_scf=2.6478792,
    logA=3.0722822, n_s=0.98501417, tau_reio=0.05798782,
    a_tSZ=3.4983345, alpha_tSZ=-0.62001806, a_kSZ=1.2858443,
    a_p=7.6669211, beta_p=1.8798252, a_c=4.0215689,
    beta_c=1.8798252,
    a_s=2.9096742, beta_s=-2.7694932,
    a_gtt=7.9189003, a_gte=0.41908216, a_gee=0.16687108,
    a_psee=0.0096106134, a_pste=-0.025321767, xi=0.08599627,
    T_d=9.6, beta_d=1.5,
    bandint_shift_dr6_pa4_f220=5.2853662,
    bandint_shift_dr6_pa5_f090=-0.028006735,
    bandint_shift_dr6_pa5_f150=-1.478886,
    bandint_shift_dr6_pa6_f090=0.55546381,
    bandint_shift_dr6_pa6_f150=-0.81019193,
    calG_all=1.0013228,
    calE_dr6_pa5_f090=0.98883908, calE_dr6_pa5_f150=0.99891013,
    calE_dr6_pa6_f090=0.99875697, calE_dr6_pa6_f150=0.99889845,
    cal_dr6_pa4_f220=0.97804664, cal_dr6_pa5_f090=1.0004643,
    cal_dr6_pa5_f150=0.99900579, cal_dr6_pa6_f090=1.0001087,
    cal_dr6_pa6_f150=1.0011954,
)

dump = {}

# ─── 1. planck_2018_lowl.TT ──────────────────────────────────────────────
lowTT = lks["planck_2018_lowl.TT"]
lmin_TT = lowTT.lmin  # 2
lmax_TT = lowTT.lmax  # 29
n = lmax_TT - lmin_TT + 1

dump["lowTT_lmin"] = np.array(lmin_TT)
dump["lowTT_lmax"] = np.array(lmax_TT)
dump["lowTT_mu"] = lowTT._mu.copy()            # (n,)
dump["lowTT_covinv"] = lowTT._covinv.copy()    # (n, n)
dump["lowTT_prior_bounds"] = lowTT._prior_bounds.copy()  # (n, 2)
dump["lowTT_offset"] = np.array(lowTT._offset)

# Store spline tables for each ell on a well-resolved grid.
# Use log-spaced grid to resolve both the low-Cl and high-Cl ends.
# 10000 points over [lo, hi] in log space ensures <0.001 interpolation error.
spline_cls_list = []
spline_vals_list = []
spline_dvals_list = []
N_GRID = 10000
for i, (sp, dsp) in enumerate(zip(lowTT._spline, lowTT._spline_derivative)):
    lo, hi = lowTT._prior_bounds[i]
    # Use log-spacing so we have dense coverage at the low end where Cl values live
    cl_fine = np.exp(np.linspace(np.log(max(lo, 1e-10)), np.log(hi), N_GRID))
    val_fine = sp(cl_fine)
    dval_fine = dsp(cl_fine)
    spline_cls_list.append(cl_fine)
    spline_vals_list.append(val_fine)
    spline_dvals_list.append(dval_fine)
dump["lowTT_spline_cl"] = np.array(spline_cls_list)    # (n, N_GRID)
dump["lowTT_spline_val"] = np.array(spline_vals_list)   # (n, N_GRID)
dump["lowTT_spline_dval"] = np.array(spline_dvals_list)  # (n, N_GRID)
print(f"lowTT: lmin={lmin_TT}, lmax={lmax_TT}, n={n}, offset={lowTT._offset:.6f}")

# ─── 2. planck_2018_lowl.EE_sroll2 ───────────────────────────────────────
sroll2 = lks["planck_2018_lowl.EE_sroll2"]
dump["sroll2_probEE"] = sroll2.probEE.copy()   # (nsteps, n_ell) = (3000, 28)
dump["sroll2_lmin"] = np.array(sroll2._lmin)   # 2
dump["sroll2_lmax"] = np.array(sroll2._lmax)   # 29
dump["sroll2_nsteps"] = np.array(sroll2._nstepsEE)  # 3000
dump["sroll2_step"] = np.array(sroll2._stepEE)       # 0.0001
print(f"sroll2: probEE shape={sroll2.probEE.shape}, step={sroll2._stepEE}")

# ─── 3. act_dr6_cmbonly.PlanckActCut ─────────────────────────────────────
plac = lks["act_dr6_cmbonly.PlanckActCut"]
# used_bins: list of arrays for [TT, TE, EE]
# used_indices: flat indices into full data vector for used bins

# Build a dense window function matrix (n_used_bins x lmax+1) per spectrum
# W[b, l] = weights[l] for l in [blmin[b], blmax[b]], else 0
# We pack all three into a single matrix over the full ell range

# The weights are indexed from bin_lmin_offset up to max(blmax)+1
lmax_plac = plac.lmax   # from the dataset
L0 = 0  # cobaya calls with L0=0

all_weights_T = []
all_weights_TE = []
all_weights_EE = []
nbins_per_spec = [len(plac.used_bins[i]) for i in range(3)]

# Build full binning matrices for each spectrum for used bins
# ell range: 0..lmax_plac (so index == ell)
n_ell_plac = lmax_plac + 1

# For each spectrum, build weight matrix W[b_idx, ell]
spec_win_list = []
for tp_idx, ubins in enumerate(plac.used_bins):
    n_used = len(ubins)
    W = np.zeros((n_used, n_ell_plac), dtype=np.float64)
    spec_names = ["tt", "te", "ee"]
    for row, b in enumerate(ubins):
        lo = plac.blmin[b]
        hi = plac.blmax[b]
        W[row, lo:hi+1] = plac.weights[lo:hi+1]
    spec_win_list.append(W)
    print(f"PlanckActCut [{spec_names[tp_idx]}]: {n_used} bins, ell [{plac.blmin[plac.used_bins[tp_idx][0]] if n_used else 'n/a'}..{plac.blmax[plac.used_bins[tp_idx][-1]] if n_used else 'n/a'}]")

dump["plac_win_TT"] = spec_win_list[0]   # (nTT_used, lmax+1)
dump["plac_win_TE"] = spec_win_list[1]   # (nTE_used, lmax+1)
dump["plac_win_EE"] = spec_win_list[2]   # (nEE_used, lmax+1)
dump["plac_X_data"] = plac.X_data.copy()   # bandpower data vector (D_ell units)
dump["plac_invcov"] = plac.invcov.copy()
dump["plac_lmax"] = np.array(lmax_plac)
print(f"PlanckActCut: X_data shape={plac.X_data.shape}, invcov shape={plac.invcov.shape}")

# ─── 4. act_dr6_mflike.ACTDR6MFLike ──────────────────────────────────────
act = lks["act_dr6_mflike.ACTDR6MFLike"]

dump["mflike_data_vec"] = act.data_vec.copy()      # (n_bpw,)
dump["mflike_inv_cov"] = act.inv_cov.copy()        # (n_bpw, n_bpw)
dump["mflike_l_bpws"] = act.l_bpws.copy()          # (n_ell,) ell grid for theory
dump["mflike_logp_const"] = np.array(act.logp_const)

print(f"mflike: data_vec={act.data_vec.shape}, l_bpws={act.l_bpws.shape}")
print(f"mflike: experiments={act.experiments}")
print(f"mflike: requested_cls={act.requested_cls}")

# Store spec_meta as flat arrays (one entry per spectrum component)
# spec_meta[i] = {"ids": array, "pol": str, "hasYX_xsp": bool, "t1": str, "t2": str, "bpw": BPW}
meta_ids = []     # list of 1D arrays of data-vec indices
meta_pol = []     # str pol type
meta_hasYX = []   # bool
meta_t1 = []      # exp1 name
meta_t2 = []      # exp2 name
meta_bpw_weights = []   # shape (n_ell_bpws, n_bins_this_spec) window functions
meta_leff = []    # effective ells

for m in act.spec_meta:
    meta_ids.append(m["ids"].copy())
    meta_pol.append(m["pol"])
    meta_hasYX.append(bool(m["hasYX_xsp"]))
    meta_t1.append(m["t1"])
    meta_t2.append(m["t2"])
    meta_leff.append(m["leff"].copy())
    # bpw.weight: shape (n_ell_bpws, n_bins_this_spec)
    meta_bpw_weights.append(m["bpw"].weight.copy())

# Save spec_meta as object arrays (they vary in size)
# Store them as separate prefix-keyed arrays
for i, (ids, pol, hasYX, t1, t2, leff, bpw_w) in enumerate(
    zip(meta_ids, meta_pol, meta_hasYX, meta_t1, meta_t2, meta_leff, meta_bpw_weights)
):
    dump[f"meta_{i}_ids"] = ids
    dump[f"meta_{i}_pol"] = np.array(pol)
    dump[f"meta_{i}_hasYX"] = np.array(hasYX)
    dump[f"meta_{i}_t1"] = np.array(t1)
    dump[f"meta_{i}_t2"] = np.array(t2)
    dump[f"meta_{i}_leff"] = leff
    dump[f"meta_{i}_bpw_weight"] = bpw_w

dump["mflike_n_meta"] = np.array(len(meta_ids))
print(f"mflike: {len(meta_ids)} spec_meta entries")

# ─── Evaluate at best-fit to get fg_totals ───────────────────────────────
# We need to get the foreground totals at the best-fit. Run a model eval.
sampled = list(model.parameterization.sampled_params())
point = {k: BESTFIT[k] for k in sampled if k in BESTFIT}
print("\nRunning model.loglikes at best-fit to capture fg_totals …")
loglikes, derived = model.loglikes(point, as_dict=True)
print("loglikes:", {k: -2*v for k, v in loglikes.items()})

# Now capture fg_totals from the foreground provider
# They are stored in the theory block after calculate()
fgbp = model.theory.get("mflike.BandpowerForeground")
if fgbp is not None:
    fg_totals = fgbp.get_fg_totals()  # list of arrays [tt_fg, te_fg, ee_fg], shape (n_exp, n_exp, n_ell_bpws)
    for s_idx, s in enumerate(act.requested_cls):
        fg_arr = fg_totals[s_idx]  # (n_exp, n_exp, n_ell_bpws)
        dump[f"mflike_fg_bestfit_{s}"] = fg_arr
        print(f"fg_totals[{s}] shape: {fg_arr.shape}")
else:
    print("WARNING: BandpowerForeground theory block not found, no fg_totals saved")

# Also save the list of experiments and requested_cls
dump["mflike_experiments"] = np.array(act.experiments)
dump["mflike_requested_cls"] = np.array(act.requested_cls)

# ─── Save ────────────────────────────────────────────────────────────────
np.savez(OUT, **dump)
print(f"\nSaved {len(dump)} arrays to {OUT}")
print("Keys:", sorted(dump.keys()))
