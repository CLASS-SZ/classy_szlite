# classy_szlite — Documentation

Pure-JAX subset of `classy_szfast` for fast, differentiable evaluation of
CMB Cls, matter Pk, distances, derived parameters, and halo-model tSZ
Cl^yy. Default cosmo_model: `ede-v2`. LCDM emulators also available.

Runtime deps: `jax`, `numpy`, `mcfit`. That's it. No keras, no tensorflow,
no scipy, no class_sz, no cobaya.

**Tested on:** macOS arm64, JAX 0.4+, Python 3.12. (Other JAX backends
should work — CPU only is fine; GPU would need a JAX install with CUDA.)

## Contents
- [Quick start](#quick-start)
- [Installation](#installation)
- [API reference](#api-reference)
  - [`CosmoParams`](#cosmoparams)
  - [`ProfileParamsA10`](#profileparamsa10)
  - [`derived`](#derived)
  - [`cl_TTTEEE`](#cl_ttteee)
  - [`Pk` / `Pnl`](#pk--pnl)
  - [`distances`](#distances)
  - [`cl_yy`](#cl_yy)
  - [`cl_yy_factory`](#cl_yy_factory)
- [Throughput benchmarks](#throughput-benchmarks)
- [JAX gradients](#jax-gradients)
- [Cosmo_model defaults & overrides](#cosmo_model-defaults--overrides)
- [Cobaya integration pattern](#cobaya-integration-pattern)
- [Common pitfalls](#common-pitfalls)

---

## Quick start

```python
import jax.numpy as jnp
import classy_szlite as csl

# Default cosmology (ede-v2 emulator-equivalent LCDM point)
cosmo = csl.CosmoParams()

# Derived params
print(csl.derived(cosmo))                # {'sigma_8': 0.812, 'Omega_m': 0.311, 'S8': 0.827, ...}

# CMB Cls (D_ell; multiply by Tcmb² for μK²)
out = csl.cl_TTTEEE(cosmo)
ell, tt = out['ell'], out['tt']

# Matter Pk at multiple z
k, pk = csl.Pk(cosmo, [0., 0.5, 1., 2.])         # linear
k, pnl = csl.Pnl(cosmo, [0., 0.5, 1., 2.])       # non-linear

# Distances
Hz, chi, Da = csl.distances(cosmo, [0.1, 0.5, 1.0])

# tSZ Cl^yy
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell=jnp.geomspace(2, 9000, 80))

# Fixed-cosmo fast path (recommended for MCMC)
ev = csl.cl_yy_factory(cosmo, ell=jnp.geomspace(2, 9000, 80))
cl_1h, cl_2h = ev(profile)                       # ~5 ms / call after warmup
```

---

## Installation

```bash
git clone https://github.com/CLASS-SZ/classy_szlite
cd classy_szlite
pip install -e .
```

You need the CosmoPower emulator data. Either:
1. Set `CLASSY_SZLITE_DATA_DIR=/path/to/emulators` (recommended), or
2. Place at `~/class_sz_data/` (default new location), or
3. Reuse an existing `~/class_sz_data_directory/` from `classy_szfast` (auto-detected fallback)

Required subdirectories under that root: `lcdm/`, `ede/` (for ede-v2);
each with `PK/`, `growth-and-distances/`, `TTTEEE/`, `PP/`,
`derived-parameters/` containing `*.npz` files.

---

## API reference

### `CosmoParams`

```python
class CosmoParams(NamedTuple):
    # Standard six
    omega_b:      float | jax.Array = 0.02242
    omega_cdm:    float | jax.Array = 0.11933
    H0:           float | jax.Array = 67.66
    tau_reio:     float | jax.Array = 0.054
    ln10_10_As:   float | jax.Array = 3.047
    n_s:          float | jax.Array = 0.9665

    # EDE (silently used when cosmo_model='ede-v2'; ignored for 'lcdm')
    fEDE:         float | jax.Array = 0.001       # ≈ LCDM
    log10z_c:     float | jax.Array = 3.562
    thetai_scf:   float | jax.Array = 2.83
    r:            float | jax.Array = 0.0

    # Neutrinos. Defaults = ede-v2 convention (3 deg. ν × 0.02 eV)
    m_ncdm:       float | jax.Array = 0.02
    N_ur:         float | jax.Array = 0.00441
```

**Defaults give the ede-v2 LCDM-equivalent point.** For LCDM:

```python
cosmo = csl.CosmoParams(omega_b=0.02242, ...).for_lcdm()
# .for_lcdm() switches ν fields to m_ncdm=0.06, N_ur=2.0328
```

This is a JAX pytree — pass it directly into `jax.grad` / `jax.jit`.

### `ProfileParamsA10`

```python
class ProfileParamsA10(NamedTuple):
    P0:    float | jax.Array = 8.130
    c500:  float | jax.Array = 1.156
    gamma: float | jax.Array = 0.3292
    alpha: float | jax.Array = 1.0620
    beta:  float | jax.Array = 5.4807
    B:     float | jax.Array = 1.0       # hydrostatic mass bias M_true/M_HSE
```

`B` = 1.25 is the typical ACT-DR4/DR6 value. Default 1.0 = no bias.

---

### `derived`

```python
derived(cosmo: CosmoParams, cosmo_model: str = 'ede-v2') -> dict
```

Returns: `{'sigma_8': float, 'Omega_m': float, 'S8': float, 'der_full': np.ndarray}`.

`der_full` is the full CosmoPower DER emulator output (14 derived params
for lcdm, 17 for ede-v2; σ8 at index 1).

**Example:**

```python
>>> csl.derived(csl.CosmoParams())
{'sigma_8': 0.8119, 'Omega_m': 0.3110, 'S8': 0.8267, 'der_full': array([1.04, 0.81, ...])}
>>> csl.derived(csl.CosmoParams(ln10_10_As=2.91), cosmo_model='ede-v2')   # lower σ8 cosmology
{'sigma_8': 0.7508, 'Omega_m': 0.3110, 'S8': 0.7641, ...}
```

---

### `cl_TTTEEE`

```python
cl_TTTEEE(cosmo: CosmoParams,
          cosmo_model: str = 'ede-v2',
          spectra: tuple[str,...] = ('tt', 'te', 'ee'),
          ell_factor: bool = True) -> dict
```

CMB angular power spectra. Returns `{'ell': np.ndarray, 'tt': np.ndarray, ...}`.

Values are **dimensionless** — multiply by `Tcmb_uK² = (2.7255e6)²` to
get μK². With `ell_factor=True` (default) values are `D_ell = ell(ell+1) Cl / (2π)`;
with `ell_factor=False` they are raw Cl.

Per-cosmo_model normalisation (lcdm uses `1/[ell(ell+1)/(2π)]` recovery
factor, ede-v2 uses `1/ell²`) is applied internally; the user just gets
D_ell or Cl in the standard convention either way.

**Example:**

```python
>>> out = csl.cl_TTTEEE(csl.CosmoParams())
>>> ell, tt = out['ell'], out['tt']
>>> Tcmb_uK = 2.7255e6
>>> D_TT_at_220 = tt[218] * Tcmb_uK**2     # in μK²
5766.9
```

---

### `Pk` / `Pnl`

```python
Pk(cosmo, z_arr,  cosmo_model='ede-v2') -> (k, pk(z, k))    # linear
Pnl(cosmo, z_arr, cosmo_model='ede-v2') -> (k, pk(z, k))    # nonlinear (HMcode)
```

Returns `(k, pk)` with shapes `(n_k,)` and `(n_z, n_k)`. `k` in `1/Mpc`,
`pk` in `Mpc³`.

For ede-v2 the linear Pk extends down to `k = 1e-4 /Mpc` via primordial
slope `P(k) ∝ k^n_s` extrapolation (emulator native kmin is `5e-4`); for
lcdm the emulator covers `1e-4` natively.

**Example:**

```python
>>> k, pk = csl.Pk(csl.CosmoParams(), [0., 0.5, 1., 2.])
>>> pk.shape
(4, 1162)
>>> pk[0, 0]      # P_lin(k=1e-4, z=0)
2010.7
```

---

### `distances`

```python
distances(cosmo, z_arr, cosmo_model='ede-v2') -> (Hz, chi, Da)
```

Returns three 1-d arrays:
- `Hz`: `H(z)/c` in `1/Mpc`. Multiply by `c = 299792.458 km/s` for `H(z)` in km/s/Mpc.
- `chi`: comoving distance in `Mpc`.
- `Da`: angular-diameter distance `chi/(1+z)` in `Mpc`.

**Example:**

```python
>>> Hz, chi, Da = csl.distances(csl.CosmoParams(), [0.005, 0.5, 1.0, 2.0])
>>> [float(c) for c in chi]
[22.13, 1946.5, 3395.9, 5308.6]
```

---

### `cl_yy`

```python
cl_yy(cosmo: CosmoParams,
      profile: ProfileParamsA10,
      ell,
      cosmo_model: str = 'ede-v2',
      z_grid=None, n_z=100, m_min=1e10, m_max=3.5e15, n_m=200,
      delta_crit=500.0) -> (cl_1h, cl_2h)
```

Halo-model tSZ angular power spectrum (Arnaud 10 GNFW profile). Returns
dimensionless `(cl_1h, cl_2h)`; multiply by `ell(ell+1)/(2π) × 1e12` for
the conventional `D_ell × 1e12` form matching Planck / ACT tSZ bandpowers.

**Full pipeline:** runs the emulators (Pk, Hz, Da), HMF (Tinker 08),
σ(R) via mcfit, then the cl_yy_1h_2h integration — every call. Cost:
~15 ms (LCDM) to ~18 ms (ede-v2) per call after warmup. For MCMC sampling
only profile params, **use `cl_yy_factory` instead** — ~3× faster.

**Example:**

```python
>>> import jax.numpy as jnp
>>> profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
>>> ell = jnp.geomspace(2, 9000, 80)
>>> cl_1h, cl_2h = csl.cl_yy(csl.CosmoParams(), profile, ell)
>>> dl = ell * (ell + 1) / (2 * jnp.pi) * (cl_1h + cl_2h) * 1e12
>>> float(dl[40])     # D_ell^yy × 1e12 at mid-ell
0.51
```

---

### `cl_yy_factory`

```python
cl_yy_factory(cosmo: CosmoParams,
              ell,
              cosmo_model: str = 'ede-v2',
              z_grid=None, n_z=100, m_min=1e10, m_max=3.5e15, n_m=200,
              delta_crit=500.0) -> Callable
```

Returns a closure `ev(profile) -> (cl_1h, cl_2h)`. Precomputes CosmoGrids
+ HaloGrids **once** (heavy: emulators + σ(R) + HMF setup). Per-step
calls only `cl_yy_1h_2h` — ~5 ms / eval after warmup.

**Use this for MCMC over profile / nuisance parameters at fixed cosmology**
(the dominant tSZ Cl^yy use case). Call `cl_yy_factory` once in
`Theory.initialize`; call the closure in each `Likelihood.logp` step.

**Example:**

```python
>>> import jax.numpy as jnp
>>> cosmo = csl.CosmoParams()
>>> ell = jnp.geomspace(2, 9000, 80)
>>> ev = csl.cl_yy_factory(cosmo, ell)                # ~1 s setup
>>> # MCMC loop: only profile sampled
>>> for P0, beta in samples:
...     prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
...     cl_1h, cl_2h = ev(prof)                       # ~5 ms / call
```

---

## Throughput benchmarks

Measured on macOS arm64 (M-series), JAX CPU, single thread, 8 ACT-DR6
bandpower ells, may26 cosmology + profile.

```
                                          cold call   warm  (evals/s)
LCDM:
  cl_yy   (full pipeline)                  986 ms     15.4 ms     65
  cl_yy_factory   (fixed-cosmo fast path)    5 ms      5.2 ms    193

ede-v2:
  cl_yy   (full pipeline)                   72 ms     18.5 ms     54
  cl_yy_factory   (fixed-cosmo fast path)    6 ms      5.4 ms    185
```

Compare:
- classy_szfast `ClyyTheoryV2` under MPI MCMC: ~20 ms/eval
- classy_szlite `cl_yy_factory` under MPI MCMC: **~8.5 ms/eval** (measured in cobaya: 11,797 evals × 8.5 ms ≈ 100 s wall time for full convergence)
- classy_szfast `classy_sz` cobaya wrapper (non-JAX): ~58 ms/eval

End-to-end **ACT-DR6 may26 MCMC, Rminus1_stop=0.01, 4-way MPI**:
- classy_szfast `ClyyTheoryV2`: 6:05 min
- **classy_szlite `cl_yy_factory`: 1:56 min** (~3× faster wall time)

To reproduce:

```python
import time, numpy as np, jax.numpy as jnp
import classy_szlite as csl

ell = jnp.asarray(np.geomspace(1000, 6000, 8))
cosmo = csl.CosmoParams().for_lcdm()
ev = csl.cl_yy_factory(cosmo, ell, cosmo_model='lcdm')

# Cold
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
t0 = time.perf_counter()
ev(profile)[0].block_until_ready()
print(f"cold: {(time.perf_counter()-t0)*1e3:.0f} ms")

# Warm
rng = np.random.default_rng(0)
times = []
for _ in range(50):
    P0 = float(rng.uniform(1, 12)); beta = float(rng.uniform(3, 7))
    profile = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    t0 = time.perf_counter()
    cl1, cl2 = ev(profile)
    cl1.block_until_ready(); cl2.block_until_ready()
    times.append((time.perf_counter()-t0)*1e3)
print(f"warm: {np.mean(times):.1f} ± {np.std(times):.1f} ms/eval")
```

---

## JAX gradients

All public functions are JAX-traceable. Use `jax.grad`, `jax.jacfwd`,
`jax.jacrev`, `jax.vmap` as you would on any pure-JAX function. The
`CosmoParams` and `ProfileParamsA10` NamedTuples are JAX pytrees, so you
can differentiate w.r.t. them directly.

### Gradient via the fast path (recommended for inference)

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import classy_szlite as csl

cosmo = csl.CosmoParams()
ell = jnp.geomspace(2, 5000, 30)
ev = csl.cl_yy_factory(cosmo, ell)

def loss(P0, beta):
    prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl_1h, cl_2h = ev(prof)
    return jnp.sum(cl_1h + cl_2h)

# First call: ~1.7 s (autodiff compile)
# Warm calls: ~24 ms / eval (5× the forward pass)
g = jax.grad(loss, argnums=(0, 1))(8.13, 5.48)
print(f"d(loss)/dP0 = {float(g[0]):.4e}")
print(f"d(loss)/dβ  = {float(g[1]):.4e}")
```

### Gradient through the full pipeline (cosmology + profile)

```python
def full_loss(omega_b, omega_cdm, P0, beta):
    c = csl.CosmoParams(omega_b=omega_b, omega_cdm=omega_cdm, H0=68.22,
                        tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    prof = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl_1h, cl_2h = csl.cl_yy(c, prof, ell)
    return jnp.sum(cl_1h + cl_2h)

# ~8 s cold (the emulators get JIT'd during autodiff trace)
g = jax.grad(full_loss, argnums=(0,1,2,3))(0.0226, 0.118, 8.13, 5.48)
# d/d(omega_b)   = -1.83e-13
# d/d(omega_cdm) = +3.80e-13
# d/dP0          = +1.18e-15
# d/dβ           = -7.11e-15
```

### Gradient w.r.t. CosmoParams as a pytree

```python
def cl_loss(cosmo):
    prof = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    cl_1h, cl_2h = csl.cl_yy(cosmo, prof, ell)
    return jnp.sum(cl_1h + cl_2h)

# grads is a CosmoParams with each field a jax.Array of the same shape
grads = jax.grad(cl_loss)(csl.CosmoParams())
print(f"d(loss)/d(omega_b)   = {float(grads.omega_b):.4e}")
print(f"d(loss)/d(omega_cdm) = {float(grads.omega_cdm):.4e}")
print(f"d(loss)/d(fEDE)      = {float(grads.fEDE):.4e}")
```

### Caveats
- **Don't wrap `cl_yy_from_params` or `cl_yy` in `jax.jit` directly** — the CosmoPower emulators do lazy JIT-warmup with a `block_until_ready` call on `forward(dummy)`, which fails under an outer trace. Both the factory and the full pipeline already exploit the emulators' internal JIT.
- For inference at fixed cosmology, the **factory path is preferred** — gradients are 5× the forward pass (~24 ms) rather than ~250 ms through the full pipeline.

---

## Cosmo_model defaults & overrides

- Default `cosmo_model='ede-v2'` everywhere. To use LCDM: pass `cosmo_model='lcdm'` AND call `.for_lcdm()` on your `CosmoParams` to switch the ν convention.
- ede-v2 emulator was trained with 3 degenerate ν of 0.02 eV (Σmν=0.06, N_ur=0.00441). LCDM with 1 massive ν of 0.06 eV (N_ur=2.0328). Same Σmν but different distributions.
- EDE-specific params (`fEDE, log10z_c, thetai_scf, r`) default to the LCDM-equivalent point (fEDE=0.001). Set them explicitly to explore EDE space.

---

## Cobaya integration pattern

```python
from cobaya.theory import Theory
import classy_szlite as csl
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class MyTSZTheory(Theory):
    # Only standard 6 cosmology params surfaced as class attrs — EDE
    # parameters are silent (use classy_szlite defaults).
    omega_b: float = 0.0226
    omega_cdm: float = 0.118
    H0: float = 68.22
    tau_reio: float = 0.0561
    ln10_10_As: float = 3.06
    n_s: float = 0.9743
    cosmo_model: str = "ede-v2"
    multipoles_file: str = None

    params = {"P0GNFW": 8.13, "c500": 1.156, "gammaGNFW": 0.3292,
              "alphaGNFW": 1.062, "betaGNFW": 5.48, "B": 1.25}

    def initialize(self):
        ell = jnp.asarray(np.loadtxt(self.multipoles_file))
        cosmo = csl.CosmoParams(omega_b=self.omega_b, omega_cdm=self.omega_cdm,
                                 H0=self.H0, tau_reio=self.tau_reio,
                                 ln10_10_As=self.ln10_10_As, n_s=self.n_s)
        if self.cosmo_model == "lcdm":
            cosmo = cosmo.for_lcdm()
        self._eval = csl.cl_yy_factory(cosmo, ell, cosmo_model=self.cosmo_model)
        self._ell = np.asarray(ell)
        self._dl_factor = ell * (ell + 1) / (2 * jnp.pi) * 1e12

    def get_can_provide(self): return ["Cl_sz"]

    def calculate(self, state, want_derived=True, **p):
        prof = csl.ProfileParamsA10(P0=p["P0GNFW"], c500=p["c500"],
            gamma=p["gammaGNFW"], alpha=p["alphaGNFW"],
            beta=p["betaGNFW"], B=p["B"])
        cl1, cl2 = self._eval(prof)
        state["Cl_sz"] = {"ell": self._ell,
                          "1h": np.asarray(self._dl_factor * cl1),
                          "2h": np.asarray(self._dl_factor * cl2)}

    def get_Cl_sz(self): return self._current_state["Cl_sz"]
```

End-to-end ACT-DR6 may26 MCMC with this pattern: **1:56 min for 10,412
samples to R-1=0.0078** (4-way MPI, Rminus1_stop=0.01).

---

## Common pitfalls

1. **cwd collision** — don't `python` from a directory containing the
   `classy_szlite/` repo subfolder; PEP 420 namespace resolution will pick
   it up before the installed package. `cd` to your workdir first.

2. **ν convention** — LCDM and ede-v2 emulators were trained with
   different neutrino setups. Always call `.for_lcdm()` on `CosmoParams`
   when you switch to `cosmo_model='lcdm'`.

3. **No outer `jax.jit`** on `cl_yy_factory` returned closures — the
   internal cl_yy_1h_2h calls into mcfit's TophatVar which may not be
   fully jit-safe. The fast path is already plenty fast without it.

4. **EDE params** are silent by default. To sample/set them, edit your
   `CosmoParams(...)` call directly. They default to `fEDE=0.001`
   (LCDM-equivalent point).
