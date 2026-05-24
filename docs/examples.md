# Tutorials & examples

Each section pairs a runnable snippet with the figure it produces. All
examples assume:

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import classy_szlite as csl

cosmo = csl.CosmoParams()                      # Planck-18-ish defaults
```

## CMB angular power spectra

```python
cls = csl.cl_TTTEEE(cosmo, spectra=("tt", "te", "ee"))
T_CMB_uK2 = (2.7255e6) ** 2

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, key, ylab, scale in [
    (axes[0], "tt", r"$D_\ell^{TT}\;[\mu K^2]$", "log"),
    (axes[1], "te", r"$D_\ell^{TE}\;[\mu K^2]$", "linear"),
    (axes[2], "ee", r"$D_\ell^{EE}\;[\mu K^2]$", "log"),
]:
    ax.plot(cls["ell"], cls[key] * T_CMB_uK2)
    ax.set(xscale="log", yscale=scale,
           xlabel=r"$\ell$", ylabel=ylab, xlim=(2, 3000))
    ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
```

![CMB TT, TE, EE](_static/cmb_ttteee.png)

## Matter Pk (linear + nonlinear)

```python
z_arr = jnp.array([0.0, 0.5, 1.0, 2.0])
k, pk  = csl.Pk(cosmo,  z_arr)
_, pnl = csl.Pnl(cosmo, z_arr)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for i, z in enumerate(np.asarray(z_arr)):
    axes[0].loglog(k, pk[i],  label=f"z = {z}")
    axes[1].loglog(k, pnl[i], label=f"z = {z}")
axes[0].set(xlabel=r"$k\;[h/\mathrm{Mpc}]$", ylabel=r"$P(k)\;[(\mathrm{Mpc}/h)^3]$",
            title="Linear")
axes[1].set(xlabel=r"$k\;[h/\mathrm{Mpc}]$", title="Non-linear (HMcode)")
for ax in axes:
    ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout()
```

![Linear and non-linear matter power spectrum at four redshifts](_static/pk.png)

## Cosmological distances

```python
z = jnp.geomspace(0.01, 5.0, 60)
Hz, chi, Da = csl.distances(cosmo, z)
c = 299_792.458

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
axes[0].semilogx(z, np.asarray(Hz) * c)
axes[0].set(xlabel="z", ylabel=r"$H(z)\;[\mathrm{km/s/Mpc}]$", title="Hubble rate")
axes[1].loglog(z, chi)
axes[1].set(xlabel="z", ylabel=r"$\chi(z)\;[\mathrm{Mpc}]$", title="Comoving distance")
axes[2].semilogx(z, Da)
axes[2].set(xlabel="z", ylabel=r"$D_A(z)\;[\mathrm{Mpc}]$",
            title="Angular-diameter distance")
for ax in axes: ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
```

![Hubble rate, comoving + angular-diameter distance](_static/distances.png)

## Linear growth σ₈(z)

Using the linear $P_k$ amplitude as a fixed-shape proxy:

```python
z = jnp.geomspace(0.01, 4.0, 30)
k, pk = csl.Pk(cosmo, z)
sigma8_0 = csl.derived(cosmo)["sigma_8"]
amp = np.sqrt(np.trapezoid(pk, k, axis=1))
sigma8_z = sigma8_0 * amp / amp[0]

plt.plot(z, sigma8_z); plt.xscale("log")
plt.xlabel("z"); plt.ylabel(r"$\sigma_8(z)$"); plt.grid(True, alpha=0.3, which="both")
```

![Linear growth σ8(z)](_static/growth.png)

## Halo-model tSZ Cl^yy (1h + 2h decomposition)

```python
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
ell = jnp.geomspace(2, 9000, 80)
cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell)

prefac = np.asarray(ell * (ell + 1) / (2 * np.pi)) * 1e12
plt.loglog(ell, prefac * cl_1h,         label="1-halo")
plt.loglog(ell, prefac * cl_2h,         label="2-halo", ls="--")
plt.loglog(ell, prefac * (cl_1h+cl_2h), label="total", color="k", lw=2)
plt.xlabel(r"$\ell$"); plt.ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
plt.grid(True, alpha=0.3, which="both"); plt.legend()
```

![Halo-model tSZ Cl^yy: 1h + 2h decomposition](_static/cl_yy.png)

For the dependence on `n_z`, `n_m`, `m_min`, `m_max`, see the
[convergence study](convergence.md).

## Bestfit + NUTS sampling on Cl^yy bandpowers (σ8 sweep)

The factory closure makes both gradient-based optimisation (L-BFGS,
Adam, …) and Hamiltonian-style samplers (NUTS, HMC) a natural fit:
each forward pass is one ~5 ms `ev(profile)` call, gradients are
exact via `jax.grad`, and there is no proposal-covariance tuning.

The example below loads a tSZ Cl^yy bandpower dataset and, for each of
**three fitting cosmologies that differ only in σ8** (`ln10_10_As`),
does two things:

1. **L-BFGS bestfit** of (P₀, β) using `scipy.optimize.minimize` with
   `jax.grad` gradients — converges in ~20–40 function evaluations.
2. **NumPyro NUTS** for the full posterior, initialised at the bestfit
   for fast warmup (~40 s for 8000 samples × 4 chains).

This is a clean demonstration of the well-known **σ8 ↔ P₀ degeneracy**:
lowering σ8 in the fitting cosmology means fewer / lighter clusters, so
the bestfit pressure normalisation has to move up to match the same
bandpower amplitude.

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np, scipy.optimize as so
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import classy_szlite as csl

# --- load bandpowers + covariance ---
ell, y, cov = load_bandpowers()                       # (N,) (N,) (N, N)
inv_cov     = jnp.asarray(np.linalg.inv(cov))

# --- per-σ8 fit ---
def build_forward(cosmo, ell_np):
    ell = jnp.asarray(ell_np)
    ev  = csl.cl_yy_factory(cosmo, ell)
    dl_factor = jnp.asarray(ell * (ell + 1) / (2 * np.pi) * 1e12)
    def forward(P0, beta):
        prof = csl.ProfileParamsA10(P0=P0, c500=1.156, gamma=0.3292,
                                     alpha=1.062, beta=beta, B=1.25)
        c1, c2 = ev(prof)
        return dl_factor * (c1 + c2)
    return forward

for As in [3.060, 2.950, 2.850]:                     # high / med / low σ8
    cosmo   = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                              tau_reio=0.0561, ln10_10_As=As, n_s=0.9743)
    s8      = csl.derived(cosmo)["sigma_8"]
    forward = build_forward(cosmo, ell)

    # ---- L-BFGS bestfit with JAX gradients ----
    def neg_log_like(x):
        r = jnp.asarray(y) - forward(x[0], x[1])
        return 0.5 * r @ inv_cov @ r
    nll, gnll = jax.jit(neg_log_like), jax.jit(jax.grad(neg_log_like))
    bf = so.minimize(lambda x: float(nll(x)), [8.13, 5.48],
                     jac=lambda x: np.asarray(gnll(x)),
                     method="L-BFGS-B", bounds=[(0.1, 20), (0.5, 10)])
    print(f"σ8={s8:.3f}  bestfit P0={bf.x[0]:.2f}  β={bf.x[1]:.2f}  χ²={2*bf.fun:.1f}")

    # ---- NUTS, initialised at the bestfit ----
    def model():
        P0   = numpyro.sample("P0",   dist.Uniform(0.0, 20.0))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 10.0))
        r    = jnp.asarray(y) - forward(P0, beta)
        numpyro.factor("loglike", -0.5 * r @ inv_cov @ r)
    mcmc = MCMC(NUTS(model, dense_mass=True),
                num_warmup=500, num_samples=2000, num_chains=4,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0),
             init_params={"P0":   jnp.full(4, float(bf.x[0])),
                          "beta": jnp.full(4, float(bf.x[1]))})
```

Output on a laptop (single-thread JAX, sequential chains; bestfit
time excludes the NUTS run):

```
high σ8 (≈0.81)     L-BFGS bestfit in 0.4 s, 38 fn evals → P0=1.20  β=2.74  χ²=12.3/6
                    NUTS in 41 s  →  posterior P0 = 1.92 ± 1.60  β = 3.19 ± 0.77
medium σ8 (≈0.77)   L-BFGS bestfit in 0.4 s, 24 fn evals → P0=1.47  β=2.71  χ²=12.3/6
                    NUTS in 38 s  →  posterior P0 = 2.34 ± 1.85  β = 3.16 ± 0.74
low σ8 (≈0.74)      L-BFGS bestfit in 0.4 s, 36 fn evals → P0=3.42  β=3.46  χ²=15.8/6
                    NUTS in 35 s  →  posterior P0 = 2.75 ± 1.90  β = 3.11 ± 0.66
```

**Bestfit curves on the bandpowers** — the low-σ8 case visibly needs a
much higher P₀ to match the bandpower amplitude:

![L-BFGS bestfit on Cl^yy bandpowers across a σ8 sweep](_static/synthetic_bestfit.png)

**Posterior triangle plot** (`getdist`) — the P₀ mode shifts to higher
values as σ8 decreases; β is much less sensitive:

![NUTS posteriors for three fitting cosmologies in a σ8 sweep](_static/synthetic_corner.png)

The full runnable script (loader + bestfit + NUTS + plotting) is at
[`examples/nuts_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/nuts_clyy_profile.py).

## End-to-end MCMC pattern (cobaya Theory)

For the RW-MH cobaya baseline that the NUTS example above reproduces,
the natural cobaya `Theory` shape is:

```python
from cobaya.theory import Theory
import classy_szlite as csl
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class MyTSZTheory(Theory):
    # The standard 6 cosmology parameters — fixed for this Theory
    omega_b:    float = 0.0226
    omega_cdm:  float = 0.118
    H0:         float = 68.22
    tau_reio:   float = 0.0561
    ln10_10_As: float = 3.06
    n_s:        float = 0.9743

    multipoles_file: str = None       # required: 1 ell per line

    params = {"P0GNFW": 8.13, "c500": 1.156, "gammaGNFW": 0.3292,
              "alphaGNFW": 1.062, "betaGNFW": 5.48, "B": 1.25}

    def initialize(self):
        ell = jnp.asarray(np.loadtxt(self.multipoles_file))
        cosmo = csl.CosmoParams(
            omega_b=self.omega_b, omega_cdm=self.omega_cdm,
            H0=self.H0, tau_reio=self.tau_reio,
            ln10_10_As=self.ln10_10_As, n_s=self.n_s,
        )
        self._csl = csl
        self._eval = csl.cl_yy_factory(cosmo, ell)   # heavy work, done once
        self._ell_np = np.asarray(ell)
        self._dl_factor = ell * (ell + 1) / (2 * jnp.pi) * 1e12

    def get_can_provide(self): return ["Cl_sz"]

    def calculate(self, state, want_derived=True, **p):
        prof = self._csl.ProfileParamsA10(
            P0=p["P0GNFW"], c500=p["c500"],
            gamma=p["gammaGNFW"], alpha=p["alphaGNFW"],
            beta=p["betaGNFW"], B=p["B"],
        )
        cl1, cl2 = self._eval(prof)
        state["Cl_sz"] = {
            "ell": self._ell_np,
            "1h":  np.asarray(self._dl_factor * cl1),
            "2h":  np.asarray(self._dl_factor * cl2),
        }

    def get_Cl_sz(self):
        return self._current_state["Cl_sz"]
```

A complete worked example with this `MyTSZTheory` paired with a Gaussian
likelihood (ACT-DR6 may26 setup) converges in **~2 min wall** for
~10,000 samples (R−1 = 0.008, 4-way MPI, `Rminus1_stop = 0.01`).

## Cosmology scan

```python
import classy_szlite as csl
import numpy as np

omega_cdm_vals = np.linspace(0.10, 0.14, 5)
for omega_cdm in omega_cdm_vals:
    cosmo = csl.CosmoParams(omega_cdm=float(omega_cdm))
    d = csl.derived(cosmo)
    print(f"omega_cdm = {omega_cdm:.3f}  →  σ8 = {d['sigma_8']:.4f}, "
          f"Ω_m = {d['Omega_m']:.4f}")
```

## Exploring EDE space

The `v2` emulator suite spans early-dark-energy parameter space; set
`fEDE`, `log10z_c`, `thetai_scf` to non-default values to leave the
LCDM-equivalent point:

```python
ede_cosmo = csl.CosmoParams(
    fEDE=0.10, log10z_c=3.5, thetai_scf=2.83,
)
csl.derived(ede_cosmo)
# → σ8 drops as fEDE rises (more EDE → less time for growth)
```

## Pre-compiling a forward + gradient function

For a parameter-inference pipeline that calls both the forward and the
gradient many times, JAX naturally caches the compiled trace:

```python
import jax, jax.numpy as jnp
import classy_szlite as csl

cosmo = csl.CosmoParams()
ell = jnp.geomspace(2, 9000, 80)
ev = csl.cl_yy_factory(cosmo, ell)

def D_ell(P0, beta):
    profile = csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25)
    cl1, cl2 = ev(profile)
    return ell * (ell + 1) / (2 * jnp.pi) * (cl1 + cl2) * 1e12

dl     = D_ell(8.13, 5.48)                                                       # ~5 ms
g_P0   = jax.grad(lambda P0, b: jnp.sum(D_ell(P0, b)), argnums=0)(8.13, 5.48)    # ~17 ms warm
g_beta = jax.grad(lambda P0, b: jnp.sum(D_ell(P0, b)), argnums=1)(8.13, 5.48)
```
