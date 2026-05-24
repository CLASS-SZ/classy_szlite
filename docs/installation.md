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

`classy_szlite` does not bundle the CosmoPower emulator `.npz` files.
Place them at one of:

1. The path in the `$CLASSY_SZLITE_DATA_DIR` environment variable, **or**
2. `~/class_sz_data/` (default new location)

Expected layout:

```
$CLASSY_SZLITE_DATA_DIR/
└── ede/
    ├── PK/
    │   ├── PKL_v2.npz
    │   └── PKNL_v2.npz
    ├── TTTEEE/
    │   ├── TT_v2.npz
    │   ├── TE_v2.npz
    │   └── EE_v2.npz
    ├── PP/
    │   └── PP_v2.npz
    ├── growth-and-distances/
    │   ├── HZ_v2.npz
    │   ├── DAZ_v2.npz
    │   └── S8Z_v2.npz
    └── derived-parameters/
        └── DER_v2.npz
```

The ede-v2 emulator `.npz` files live in the
[`cosmopower-organization/ede`](https://github.com/cosmopower-organization/ede)
GitHub repository. Clone it and either point
`$CLASSY_SZLITE_DATA_DIR` at the checkout or symlink the contents into
`~/class_sz_data/`:

```bash
git clone https://github.com/cosmopower-organization/ede ~/cosmopower-ede
export CLASSY_SZLITE_DATA_DIR=~/cosmopower-ede
```

### `_v2.npz` vs `_v2_plain.npz`

Each emulator is available in two on-disk formats with **identical
weights**:

- **`<name>_v2.npz`** — original CosmoPower output. Loading requires
  `cosmopower` to be importable, which transitively imports
  `tensorflow`. Use this if you already have the CosmoPower stack
  installed.

- **`<name>_v2_plain.npz`** — pickle-free re-saved form
  (`np.savez_compressed` with flat keys, ~8 % smaller). Loads with
  `allow_pickle=False`, **no TensorFlow / cosmopower needed**. Use
  this for slim deploy targets, CI, or anyone who doesn't want a TF
  install on their critical path.

`classy_szlite` **prefers `_v2_plain.npz` automatically** when present
and falls back to `_v2.npz` otherwise — no configuration needed.

### Why the v2 emulators

The `v2` emulators are the emulators used in
[*The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological
Models*](https://arxiv.org/abs/2503.14454) (2025) and in
[*Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble
tension*](https://arxiv.org/abs/2505.08051) (Poulin et al. 2025).

They are the **highest-accuracy CLASS-based emulators** available in the
[`cosmopower-organization`](https://github.com/cosmopower-organization)
namespace, and match the CAMB-based
[Jense et al. (2024) emulators](https://github.com/cosmopower-organization/jense_2024_emulators)
to **well under 0.1 $\sigma$ in $\Lambda$CDM**, where $\sigma$ is the
typical 68% CL on parameter constraints from ACT DR6 data.

Coverage:

- $\Lambda$CDM, $m_\nu$-$\Lambda$CDM, $w$CDM, $N_{\rm eff}$-$\Lambda$CDM,
  and combinations of these — plus the EDE direction needed for the EDE
  results above
- Outputs: distances ($H(z)$, $\chi(z)$, $D_A(z)$), $C_\ell^{TT,TE,EE,BB}$,
  $C_\ell^{\phi\phi}$, linear and non-linear $P(k,z)$

`classy_szlite` exposes these emulators through a single pure-JAX pipeline
suitable for `jax.grad`-based inference (Fisher, HMC/NUTS, MAP, SBI).

### Default cosmology and EDE

The `v2` emulator suite is trained on an EDE parameter range that
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
# Expect: {'sigma_8': 0.8119, 'Omega_m': 0.3110, 'S8': 0.8267, ...}
```
