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

## Bestfit + NUTS sampling on Cl^yy bandpowers (baseline vs lows8)

The factory closure makes both gradient-based optimisation (L-BFGS,
Adam, …) and Hamiltonian-style samplers (NUTS, HMC) a natural fit:
each forward pass is one ~5 ms `ev(profile)` call, gradients are
exact via `jax.grad`, and there is no proposal-covariance tuning.

The example fits a tSZ Cl^yy bandpower dataset at **two fixed
cosmologies** that share the same ω_b, ω_cdm, n_s but differ in σ8
through (ln10_10_As, H0):

| Cosmology | ln10_10_As | H0   | σ8     |
| --- | --- | --- | --- |
| **baseline** | 3.060 | 68.22 | ≈ 0.81 |
| **lows8** (Flamingo low-S8) | 2.910 | 67.14 | ≈ 0.75 |

For each cosmology we run:

1. **L-BFGS bestfit** of (P₀, β) via `scipy.optimize.minimize` with
   exact `jax.grad` gradients — converges in ~20–40 fn evals,
   < 0.5 s.
2. **NumPyro NUTS** for the full posterior, initialised at the
   bestfit — ~35–40 s for 8000 samples × 4 chains.
3. Loads the matching **cobaya RW-MH** chain (if present on disk) for
   a sampler-vs-sampler overlay.

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

FIT_COSMOS = [
    dict(label="baseline (σ8≈0.81)", ln10_10_As=3.060, H0=68.22),
    dict(label="lows8 (σ8≈0.75)",    ln10_10_As=2.910, H0=67.14116850291264),
]

for cfg in FIT_COSMOS:
    cosmo   = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118,
                              tau_reio=0.0561, n_s=0.9743,
                              ln10_10_As=cfg["ln10_10_As"], H0=cfg["H0"])
    s8      = csl.derived(cosmo)["sigma_8"]
    forward = build_forward(cosmo, ell)

    # L-BFGS bestfit with JAX gradients
    def neg_log_like(x):
        r = jnp.asarray(y) - forward(x[0], x[1])
        return 0.5 * r @ inv_cov @ r
    nll, gnll = jax.jit(neg_log_like), jax.jit(jax.grad(neg_log_like))
    bf = so.minimize(lambda x: float(nll(x)), [8.13, 5.48],
                     jac=lambda x: np.asarray(gnll(x)),
                     method="L-BFGS-B", bounds=[(0.1, 20), (0.5, 10)])
    print(f"{cfg['label']:25s}  bestfit P0={bf.x[0]:.2f} β={bf.x[1]:.2f}  χ²={2*bf.fun:.1f}")

    # NUTS, init at bestfit
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

Output on a laptop (single-thread JAX):

```
baseline (σ8≈0.81)   L-BFGS bestfit in 0.4 s, 38 fn evals → P0=1.20  β=2.74  χ²=12.3/6
                     NUTS in 41 s  →  posterior P0 = 1.92 ± 1.60  β = 3.19 ± 0.77
lows8    (σ8≈0.75)   L-BFGS bestfit in 0.4 s, 27 fn evals → P0=1.54  β=2.71  χ²=12.3/6
                     NUTS in 36 s  →  posterior P0 = 2.49 ± 1.91  β = 3.18 ± 0.74
```

**Bandpowers + bestfit curves + NUTS 68% bands** for both cosmologies.
The two bestfit curves are almost indistinguishable in the data range,
but the underlying GNFW shapes — and especially the P₀ values that
NUTS uncovers — are very different:

![Bestfit + NUTS 68% band on Cl^yy bandpowers, baseline vs lows8](_static/synthetic_bestfit.png)

**Triangle plot** with **4 contours** — NUTS (solid filled) +
cobaya RW-MH (dashed) for each cosmology. The NUTS and MH posteriors
overlap to within sampling noise (good sampler-vs-sampler agreement),
and the σ8 ↔ P₀ degeneracy clearly shifts the lows8 (red) posterior
to higher P₀ than the baseline (blue):

![Triangle plot: NUTS + cobaya MH posteriors for baseline and lows8 cosmologies](_static/synthetic_corner.png)

The full runnable script (loader + bestfit + NUTS + MH overlay +
plotting) is at
[`examples/nuts_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/nuts_clyy_profile.py).

## Posterior bands on the GNFW pressure profile

Since (P₀, β) are the only GNFW parameters sampled in the fit above,
each posterior sample maps to a different dimensionless profile

$$
p(x) = P_0\,(c_{500}\,x)^{-\gamma}\,\bigl[1 + (c_{500}\,x)^\alpha\bigr]^{-(\beta-\gamma)/\alpha},
\qquad x = r / r_{500}.
$$

Drawing 500 random samples per chain and taking the 16/50/84
percentiles gives a median curve + 1σ band per cosmology. Plotted
together with the **fiducial A10 profile** and a $\,p(x)\,x^2$
y-axis (which flattens the inner power-law fall-off and makes the
outer slope β easy to read):

![GNFW pressure profile from Cl^yy NUTS posteriors](_static/profile_bands.png)

Two physical observations stand out:

- **The data prefer a much shallower outer profile than A10.**
  Median β ≈ 3.2 in both fits vs the A10 fiducial β = 5.48.
- **lows8 ↔ higher pressure**, as expected — at lower σ8 you need
  more pressure per cluster to match the same Cl^yy amplitude, so
  the red band sits above the blue across the full radial range.

The runnable script is at
[`examples/profile_bands.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/profile_bands.py).

## Simulation-based inference (SBI / NPE via flowjax)

Neural Posterior Estimation (NPE) is a natural fit for `classy_szlite`:
the bottleneck of SBI is the ability to simulate $(\theta, y)$ pairs
quickly, which is exactly what `cl_yy_factory` delivers (~5 ms/eval,
trivially `jax.vmap`-batchable). We trade gradients (which SBI does
not need) for an **amortised posterior** — a single trained
conditional normalizing flow returns the posterior at any new
bandpower realisation in O(ms).

The example uses [`flowjax`](https://github.com/danielward27/flowjax)
to train a conditional Masked Autoregressive Flow
$q_\phi(\theta \mid y)$ on $\sim$8000 simulations per cosmology,
with a Gaussian proposal centred on the L-BFGS bestfit (sequential
NPE without importance reweighting — the simplest variant). It
overlays NUTS + cobaya RW-MH + SBI on the same `(P_0, \beta)` corner
for the baseline + lows8 cosmologies (6 contours total):

![NUTS + cobaya MH + SBI corner with 6 contours](_static/sbi_corner_6contours.png)

And it shows the amortisation in action — five fresh noise
realisations passed through the SAME trained flow give five 1-D
posteriors in O(ms), with the truth (vertical dashed line) sitting
within the bulk of each:

![SBI amortisation over fresh noise realisations](_static/sbi_amortised.png)

Typical wall time per cosmology on a laptop CPU: ~5 s simulation
(vmap-batched) + ~10 s flow training + O(ms) per posterior
evaluation. The runnable script is at
[`examples/sbi_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/sbi_clyy_profile.py).

```python
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.train import fit_to_data

# theta_train.shape = (N, 2)   y_train.shape = (N, n_bandpowers)
flow = masked_autoregressive_flow(
    key=jr.key(0),
    base_dist=Normal(jnp.zeros(2)),
    cond_dim=y_train.shape[1],
    nn_width=128, nn_depth=3, flow_layers=8,
)
flow, _ = fit_to_data(jr.key(0), flow, data=(theta_train, y_train),
                      max_epochs=500, batch_size=512, learning_rate=5e-4)

# Amortised: evaluate at any new y in O(ms)
samples = flow.sample(jr.key(1), sample_shape=(3000,), condition=y_obs)
```

## Fisher matrix in one autodiff sweep

For a Gaussian likelihood with fixed covariance $\Sigma$, the Fisher
matrix at parameter point $\boldsymbol{\theta}$ is

$$
F_{ij}(\boldsymbol{\theta}) = (\partial_i \mu)^\top\, \Sigma^{-1}\, (\partial_j \mu),
$$

with $\mu(\boldsymbol{\theta}) = $ forward$(P_0, \beta)$. The Jacobian
$J = \partial \mu / \partial \boldsymbol{\theta}$ is exactly what
`jax.jacfwd` returns in a single forward-mode autodiff sweep — no
finite-difference loop, no $\varepsilon$ tuning.

```python
import jax, jax.numpy as jnp
import classy_szlite as csl

forward = build_forward(cosmo, ell)   # see nuts_clyy_profile.py

def mu(x):
    return forward(x[0], x[1])

J = jax.jit(jax.jacfwd(mu))(jnp.asarray([P0_bf, beta_bf]))    # (n_bp, 2)
F = J.T @ inv_cov @ J                                          # (2, 2)
cov_fisher = jnp.linalg.inv(F)
```

The `examples/fisher_clyy_profile.py` script runs this end-to-end and
overlays the 68%/95% Fisher ellipses on the NUTS posterior:

![Fisher matrix ellipse + L-BFGS bestfit overlaid on the NUTS posterior](_static/fisher_overlay.png)

Wall time: **~135 ms per Fisher matrix** after JAX warmup (10-run
average, including jit dispatch). The autodiff Fisher matches a
2-point central finite-difference reference ($\varepsilon = 10^{-3}$)
to $|\Delta F|/|F| \sim 10^{-6}$. The Fisher ellipse here is much
tighter than the NUTS posterior (σ_Fisher ≈ 0.35 vs σ_NUTS ≈ 1.5 for
$P_0$) — Fisher captures only the local quadratic curvature at the
bestfit and misses the heavy tail toward larger $P_0$ that NUTS
readily explores. This is a useful sanity check for forecasting: the
Gaussian Fisher approximation will under-estimate the uncertainty
when the true posterior is skewed.

The runnable script is at
[`examples/fisher_clyy_profile.py`](https://github.com/CLASS-SZ/classy_szlite/blob/main/examples/fisher_clyy_profile.py).

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
