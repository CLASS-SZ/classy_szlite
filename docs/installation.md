# Installation

## From PyPI (recommended)

```bash
pip install classy_szlite
```

## From source

```bash
git clone https://github.com/CLASS-SZ/classy_szlite
cd classy_szlite
pip install -e .
```

Runtime dependencies: `jax >= 0.4`, `numpy >= 1.24`, `mcfit >= 0.0.21`.

## Emulator data

`classy_szlite` does not bundle the CosmoPower emulator `.npz` files
(~290 MB), so you fetch them once after installing.

### Quick way (recommended)

```bash
classy-szlite-get-data
# or, equivalently:
python -m classy_szlite.fetch_data
```

This downloads exactly the pickle-free `*_v3_plain.npz` files `classy_szlite`
needs into `~/class_sz_data/ede/` (or
`$CLASSY_SZLITE_DATA_DIR` if you've set it), and **verifies every file
loads with `allow_pickle=False`** before committing it into place. It is
safe to re-run: already-valid files are skipped, and any *invalid* file
(e.g. an accidentally-downloaded pickled `_v3.npz`, or a
truncated download) is detected and re-fetched. Options:

```bash
classy-szlite-get-data --dest /custom/path   # put data elsewhere
classy-szlite-get-data --force               # re-download everything
classy-szlite-get-data --quiet               # only warnings/errors
```

If you use a custom `--dest` that isn't an auto-detected location, export
it so `classy_szlite` finds it:

```bash
export CLASSY_SZLITE_DATA_DIR=/custom/path
```

```{admonition} Common error
:class: warning
`ValueError: Cannot load file containing pickled data when
allow_pickle=False` (or `Object arrays cannot be loaded …`) means a
**pickled** `_v3.npz` file is sitting where a pickle-free
`_v3_plain.npz` should be — usually from hand-copying the wrong file out
of the GitHub web UI. Just run `classy-szlite-get-data`; it replaces the
bad files automatically.
```

### Manual layout

If you'd rather place the files yourself, put them at one of:

1. The path in the `$CLASSY_SZLITE_DATA_DIR` environment variable, **or**
2. `~/class_sz_data/` (default new location), **or**
3. `~/class_sz_data_directory/` (legacy location, still honoured)

Expected layout — `classy_szlite` reads the pickle-free
`*_v3_plain.npz` files (loaded with `allow_pickle=False`, no
TensorFlow / cosmopower needed):

```
$CLASSY_SZLITE_DATA_DIR/
└── ede/
    ├── PK/
    │   ├── PKL_v3_plain.npz
    │   └── PKNL_v3_plain.npz
    ├── TTTEEE/
    │   ├── TT_v3_plain.npz
    │   ├── rTE_v3_plain.npz
    │   └── EE_v3_plain.npz
    ├── PP/
    │   └── PP_v3_plain.npz
    ├── growth-and-distances/
    │   ├── HZ_v3_plain.npz
    │   ├── DAZ_v3_plain.npz
    │   └── S8Z_v3_plain.npz
    └── derived-parameters/
        └── DER_v3_plain.npz
```

The files live in the
[`cosmopower-organization/ede`](https://github.com/cosmopower-organization/ede)
repository. Clone it and either point `$CLASSY_SZLITE_DATA_DIR` at
the checkout or symlink the contents into `~/class_sz_data/`:

```bash
git clone https://github.com/cosmopower-organization/ede ~/cosmopower-ede
export CLASSY_SZLITE_DATA_DIR=~/cosmopower-ede
```

### Emulator suite

**ede-v3** (October 2026; since v0.2.15 the only suite) retrains every
emulator on the CLASS-EDE training set of the ede-v2 emulators, with all
training samples, a common train/test split for the CMB spectra and full
float32 precision. Its TE emulator predicts the correlation coefficient
$r_\ell = C_\ell^{TE}/\sqrt{C_\ell^{TT} C_\ell^{EE}}$ (as $\operatorname{atanh} r_\ell$),
and `classy_szlite` reconstructs $C_\ell^{TE} = r_\ell \sqrt{C_\ell^{TT} C_\ell^{EE}}$.
Test errors are 1.3 to 20 times lower than ede-v2 (TT 0.078% against 0.164%
at the 95th percentile); accuracy tables in the
[`cosmopower-organization/ede` README](https://github.com/cosmopower-organization/ede#ede-v3-october-2026).

ede-v2 are the emulators used in
[*The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological
Models*](https://arxiv.org/abs/2503.14454) (2025) and in
[*Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble
tension*](https://arxiv.org/abs/2505.08051) (Poulin et al. 2025). They
match the CAMB-based
[Jense et al. (2024) emulators](https://github.com/cosmopower-organization/jense_2024_emulators)
to well under 0.1 $\sigma$ in $\Lambda$CDM, where $\sigma$ is the
typical 68% CL on parameter constraints from ACT DR6 data. To reproduce
those analyses exactly, use `classy_szlite <= 0.2.14` with
`set_emulator_version("v2")`, or class_sz with `cosmo_model: 6`.

Coverage:

- $\Lambda$CDM, $m_\nu$-$\Lambda$CDM, $w$CDM, $N_{\rm eff}$-$\Lambda$CDM,
  and combinations of these — plus the EDE direction needed for the EDE
  results above
- Outputs: distances ($H(z)$, $\chi(z)$, $D_A(z)$), $C_\ell^{TT,TE,EE,BB}$,
  $C_\ell^{\phi\phi}$, linear and non-linear $P(k,z)$

`classy_szlite` exposes these emulators through a single pure-JAX pipeline
suitable for `jax.grad`-based inference (Fisher, HMC/NUTS, MAP).

### Default cosmology and EDE

The emulators are trained on an EDE parameter range that
includes the LCDM-equivalent point. `classy_szlite` exploits this:
``CosmoParams()`` defaults to ``fEDE = 0.001`` so the very first call
gives standard $\Lambda$CDM to emulator precision — you don't have to
think about EDE at all. If you do want to explore EDE, set ``fEDE``,
``log10z_c`` and ``thetai_scf`` to non-default values and the same
pipeline handles it.

## Verification

```python
import classy_szlite as csl
print(csl.__version__)
print(csl.derived(csl.CosmoParams()))
# Expect: {'sigma_8': 0.8120, 'Omega_m': 0.3110, 'S8': 0.8268, ...}
```
