# classy_szlite

A pure-JAX subset of [`classy_szfast`](https://github.com/CLASS-SZ/classy_szfast)
for fast, differentiable evaluation of CMB Cls, matter Pk, distances,
derived params, and halo-model tSZ Cl^yy — without dragging in keras,
tensorflow, scipy, or the class_sz C library.

**Default cosmology emulator:** `ede-v2`. LCDM is also supported.

## Capabilities

| Function | Returns |
| --- | --- |
| `cl_TTTEEE(cosmo)` | CMB angular power spectra: `{'tt', 'te', 'ee', 'ell'}` |
| `Pk(cosmo, z_arr)` | linear matter power spectrum `P(k, z)` |
| `Pnl(cosmo, z_arr)` | non-linear `P(k, z)` (HMcode) |
| `distances(cosmo, z_arr)` | `(Hz, chi, Da)` |
| `derived(cosmo)` | `{'sigma_8', 'Omega_m', 'S8', 'der_full'}` |
| `cl_yy(cosmo, profile, ell)` | halo-model tSZ Cl^yy (1h, 2h), Arnaud 10 GNFW |

## Install

```bash
git clone https://github.com/CLASS-SZ/classy_szlite
cd classy_szlite
pip install -e .
```

Runtime dependencies: `jax`, `numpy`, `mcfit`. Nothing else.

## Emulator data

`classy_szlite` does not bundle the CosmoPower emulator `.npz` files
(they're ~100 MB per cosmo_model). Get them from the CLASS-SZ data
distribution and either:

1. Place them at `~/class_sz_data/` (recommended), or
2. Re-use an existing `~/class_sz_data_directory/` from `classy_szfast`
   (auto-detected as fallback), or
3. Point the `CLASSY_SZLITE_DATA_DIR` environment variable at the root.

Required tree per cosmo_model:
```
$CLASSY_SZLITE_DATA_DIR/
├── lcdm/              # for cosmo_model='lcdm'
│   ├── PK/{PKL_v1,PKNL_v1}.npz
│   ├── TTTEEE/{TT_v1,TE_v1,EE_v1}.npz
│   ├── PP/PP_v1.npz
│   ├── growth-and-distances/{HZ_v1,DAZ_v1,S8Z_v1}.npz
│   └── derived-parameters/DER_v1.npz
└── ede/               # for cosmo_model='ede-v2'
    ├── PK/{PKL_v2,PKNL_v2}.npz
    └── ... (same subdirs, _v2.npz files)
```

## Quick start

```python
import jax.numpy as jnp
import classy_szlite as csl

# Defaults: Planck-18-like cosmology, ede-v2 ν convention (3 deg. ν, Σmν=0.06 eV)
cosmo = csl.CosmoParams()

# Derived params
d = csl.derived(cosmo)
print(f"σ8 = {d['sigma_8']:.4f}, Ω_m = {d['Omega_m']:.4f}, S8 = {d['S8']:.4f}")

# CMB Cls (dimensionless; multiply by Tcmb² to get μK²)
out = csl.cl_TTTEEE(cosmo)
ell, tt = out['ell'], out['tt']

# Matter Pk at multiple z
k, pk = csl.Pk(cosmo, [0., 0.5, 1., 2.])

# Distances
Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])

# tSZ Cl^yy (halo-model, Arnaud 10 profile, B = 1.25 hydrostatic mass bias)
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell=jnp.geomspace(2, 9000, 80))

# Switch to lcdm cosmology — note ν convention differs (1 massive ν of 0.06 eV)
cosmo_lcdm = csl.CosmoParams().for_lcdm()
d_lcdm = csl.derived(cosmo_lcdm, cosmo_model='lcdm')
```

## Key design notes

- **`ede-v2` is the default** because it's a strict superset of LCDM (set
  `fEDE=0.001` to recover the LCDM-equivalent point) and supports the
  full early-dark-energy parameter space.
- **Neutrino conventions differ between emulators**: the `ede-v2`
  emulator was trained with 3 degenerate ν of 0.02 eV (Σmν=0.06,
  N_ur=0.00441); LCDM with 1 massive ν of 0.06 eV (N_ur=2.0328). The
  `CosmoParams` defaults match `ede-v2`; use
  `cosmo.for_lcdm()` to switch.
- **Low-k Pk extrapolation**: the `ede-v2` Pk emulator has
  `k_min = 5e-4 / Mpc`; we extend down to `1e-4` using the
  primordial-slope asymptote `P(k) ∝ k^{n_s}` (valid at all z).
- **ede-v2 DAZ z=0 anchor**: the `ede-v2` DAZ emulator omits z=0 and
  returns 4999 values; we prepend `chi(0)=0` (matching what
  classy_szfast's cobaya wrapper does).
- **Pure JAX forward pass**: no keras dependency. The `.npz` files are
  loaded with numpy; the α-β-sigmoid CosmoPower forward pass is a few
  `jnp` ops in `classy_szlite/_emulator.py`.

## Parity with classy_szfast

`classy_szlite` is a faithful port — at the emulator level it produces
bit-identical raw NN output to `classy_szfast.cosmopower_jax`. End-to-end:

| Quantity | classy_szlite vs classy_szfast |
| --- | --- |
| Raw NN forward pass | identical (~5e-9 floating-point noise) |
| Linear `Pk(k, z)` | identical to 1.00000 across all k |
| Distances (`Hz, chi, Da`) | identical |
| Derived (`σ8, Ω_m, S8`) | identical to ~0.01% |
| `Cl^yy` (halo-model) | ~1–2% (within default n_z/n_m differences) |
| **CMB Cls — lcdm and ede-v2** | identical to ~0.1%: D_ell^TT @ ell=220 = 5766 μK² for both. The per-cosmo_model normalisation factor (lcdm: 1/[ell(ell+1)/(2π)]; ede-v2: 1/ell²) is applied internally so the user always gets D_ell or Cl in the standard convention. |

## Relationship to `classy_szfast`

`classy_szlite` is the **lite JAX subset** — ported, tested, and
de-dependent. `classy_szfast` remains the full-featured non-JAX +
cobaya-wrapper distribution and supports more cosmologies (mnu, neff,
wcdm, ede) and more observables (cluster counts, kSZ², CIB, etc.).

For MCMC work where you want fast (~10 ms / call), differentiable,
JAX-jittable, dependency-minimal evaluation in either LCDM or EDE-v2,
use `classy_szlite`.

## License

MIT.
