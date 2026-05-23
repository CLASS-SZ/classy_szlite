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

Download the data tarball from the CLASS-SZ data distribution (see
the [CLASS-SZ github org](https://github.com/CLASS-SZ)).

## Verification

```python
import classy_szlite as csl
print(csl.__version__)
print(csl.derived(csl.CosmoParams()))
# Expect: {'sigma_8': 0.8119, 'Omega_m': 0.3110, 'S8': 0.8267, ...}
```
