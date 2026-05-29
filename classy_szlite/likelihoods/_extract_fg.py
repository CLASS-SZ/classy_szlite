"""Extract per-component foreground templates from mflike at chain best-fit.

For each amplitude parameter, computes the unit foreground spectrum (amplitude=1).
Saves to npz so the JAX likelihood can sample amplitudes via linear combination.

Output keys:
  mflike_fg_template_<comp>_<pol>  shape (5, 5, 8500)
    where <comp> ∈ {kSZ, cibp, radio, tSZ, cibc, dust, tSZxCIB_unit, dustEE, radioEE, dustTE, radioTE}

The tSZ_and_CIB cross-term is nonlinear: -xi*sqrt(a_tSZ*a_c)*template_cross.
We store template_cross by varying xi while keeping a_tSZ, a_c fixed.
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import yaml
from pathlib import Path

CHAIN_YAML = Path(os.environ.get(
    "CLASSY_SZLITE_LIK_CHAIN_YAML",
    "/Users/boris/GitHub/act-dr6-ede-analysis/p-actbase_ede+n3_classsz/p-actbase_ede+n3_classsz.input.yaml"
))
OUT_NPZ = Path(os.environ.get(
    "CLASSY_SZLITE_LIK_FG_NPZ",
    str(Path.home() / ".classy_szlite" / "mflike_fg_components.npz")
))
OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
PACKAGES_PATH = os.environ.get(
    "CLASSY_SZLITE_LIK_PACKAGES_PATH",
    "/Users/boris/cobaya_packages"
)

# Best-fit values from the chain
BF = dict(
    omega_b=0.022751597, omega_cdm=0.13242228, H0=71.113655,
    fEDE=0.11678514, log10z_c=3.5517577, thetai_scf=2.6478792,
    logA=3.0722822, n_s=0.98501417, tau_reio=0.05798782,
    a_tSZ=3.4983345, alpha_tSZ=-0.62001806, a_kSZ=1.2858443,
    a_p=7.6669211, beta_p=1.8798252, a_c=4.0215689, beta_c=1.8798252,
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

# Amplitude params that scale linearly. xi/a_tSZ/a_c are handled specially.
LINEAR_AMP_PARAMS = ["a_kSZ", "a_p", "a_s", "a_tSZ", "a_c",
                      "a_gtt", "a_gee", "a_psee", "a_pste", "a_gte"]

with open(CHAIN_YAML) as f:
    info = yaml.safe_load(f)
info["likelihood"]["act_dr6_mflike.ACTDR6MFLike"]["data_folder"] = "ACTDR6MFLike/v1.0"
info["likelihood"]["act_dr6_mflike.ACTDR6MFLike"]["input_file"] = "dr6_data.fits"
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

from cobaya.model import get_model
print("Building cobaya model …")
model = get_model(info)

# Best-fit point (sampled subset)
sampled = list(model.parameterization.sampled_params())
bf_sampled = {k: BF[k] for k in sampled if k in BF}
loglikes, _ = model.loglikes(bf_sampled, as_dict=True)
print(f"Best-fit chi2: total={sum(-2*v for v in loglikes.values()):.3f}")

# Access mflike's BandpowerForeground theory to get fg per component
fg_theory = model.theory["mflike.BandpowerForeground"]
print(f"fg_theory: {fg_theory}, ells {len(fg_theory.ells)}")

# Build foreground component breakdown at best-fit. Tilt values that the
# chain holds fixed are read directly from the input.yaml via cobaya's
# parameter routing, so we don't accidentally hardcode a wrong default.
import re
_chain_tilts = {}
with open(CHAIN_YAML) as f_yaml:
    yaml_src = f_yaml.read()
for _tilt in ("alpha_s", "alpha_p", "alpha_dT", "alpha_dE",
              "T_d", "T_effd", "beta_d"):
    m = re.search(rf"^\s*{_tilt}:\s*\n\s*value:\s*([0-9.\-+eE]+)",
                  yaml_src, flags=re.MULTILINE)
    if m:
        _chain_tilts[_tilt] = float(m.group(1))
print(f"chain tilt fixed values: {_chain_tilts}")

fg_params_bf = {
    "a_tSZ": BF["a_tSZ"], "alpha_tSZ": BF["alpha_tSZ"], "a_kSZ": BF["a_kSZ"],
    "a_p": BF["a_p"], "beta_p": BF["beta_p"], "a_c": BF["a_c"],
    "beta_c": BF["beta_c"],
    "a_s": BF["a_s"], "beta_s": BF["beta_s"],
    "a_gtt": BF["a_gtt"], "a_gte": BF["a_gte"], "a_gee": BF["a_gee"],
    "a_psee": BF["a_psee"], "a_pste": BF["a_pste"], "xi": BF["xi"],
    **_chain_tilts,
}

# Pass bandpass shifts via fg theory's fg_params (mflike reads them from must_provide)
# Use the public API to compute the component-wise dictionary
ell = np.asarray(fg_theory.ells)
print(f"ell range: {ell[0]}..{ell[-1]}")

# Include bandint_shifts in fg_params so _get_foreground_model_arrays applies them
bandint_shifts_bf = {k: BF[k] for k in BF if "bandint_shift" in k}
all_fg_params = {**fg_params_bf, **bandint_shifts_bf}
print(f"Passing bandint_shifts: {bandint_shifts_bf}")

# Call get_foreground_model so bandpasses are constructed with the shifts
fg_dict = fg_theory.get_foreground_model(**all_fg_params)
print(f"foreground component keys: {len(fg_dict)} items, sample: {list(fg_dict.keys())[:5]}")

# Group by (pol, comp): fg_dict[(pol, comp, exp1, exp2)] = array shape (n_ell,)
# Convert to (5,5,n_ell) per (pol, comp)
n_exp = 5
experiments = ["dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150"]
n_ell = len(ell)

dump = {"mflike_fg_ell": ell, "mflike_fg_experiments": np.asarray(experiments)}

components_per_pol = {"tt": [], "te": [], "ee": []}
for key in fg_dict:
    pol, comp, e1, e2 = key
    if pol in components_per_pol and comp not in components_per_pol[pol]:
        components_per_pol[pol].append(comp)
print(f"\nComponents per pol: {components_per_pol}")

for pol, comps in components_per_pol.items():
    for comp in comps:
        arr = np.zeros((n_exp, n_exp, n_ell))
        for i, e1 in enumerate(experiments):
            for j, e2 in enumerate(experiments):
                key = (pol, comp, e1, e2)
                if key in fg_dict:
                    arr[i, j] = fg_dict[key]
        dump[f"mflike_fg_{pol}_{comp}_bf"] = arr

# Best-fit values of each amplitude for the linear scaling
for k, v in fg_params_bf.items():
    dump[f"fg_param_bf_{k}"] = np.asarray(v)

print(f"\nSaving {len(dump)} arrays to {OUT_NPZ}")
np.savez(OUT_NPZ, **dump)
print(f"Done. File size: {OUT_NPZ.stat().st_size / 1e6:.1f} MB")
