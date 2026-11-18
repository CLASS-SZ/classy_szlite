# M0 log

Results of milestone M0 (PLAN.md §6). Runs on rusty: CPU dev job on
worker5052 (rome, 16 cores), JAX 0.12.0.1, float64.

## 2026-11-18: refactor of the cosmology stage

### What changed

- `constants.py`: physical constants moved out of `hmf.py` and
  `power_spectrum.py` (the old private names stay importable from `hmf`).
  It also records the emulators' neutrino convention: one species with
  degeneracy 3, Σmν = 3 m_ncdm, ω_ν = Σmν / 93.14 eV.
- `_emulator.py`:
  - `Emulator` is a JAX pytree: its weights are the leaves, and the
    parameter names and modes are static. It can be passed as an argument to
    jitted functions.
  - `predict` always returns `(B, n_out)`; `squeeze()` is gone.
  - Weights are created under `jax.ensure_compile_time_eval()`, so the
    `lru_cache` can never store tracers.
- `cosmology.py`:
  - The k grid is static numpy, built at import.
  - σ(R) uses the pure-JAX FFTLog in float64, set up once. The per-call mcfit
    `TophatVar` and the `float()`/`int()` calls are gone.
  - `build(params, z_grid, emus=None)` is jit- and vmap-able.
  - New helpers: `omega_nu`, `Omega_m0`, `omega_cdm_from_Omega_m`,
    `sigma8_der`, and `ln10As_from_sigma8`. The last is traceable: A_s ∝ σ8²,
    then 8 Newton steps through the DER network, as cosmocnc_jax does.
- `hmf.py`:
  - **Bug fix:** the HMF's Ω_m now includes Σmν = 3 m_ncdm, as `derived()`
    always did. It used to include only m_ncdm, a third of it.
  - Switches, with classy_szlite's defaults unchanged:
    - `tinker_interp`: `'log10'` (default) or `'linear'` (cosmocnc);
    - `nu_in_rho_m` and `nu_in_delta`: True by default, False for cosmocnc's
      cb-only ρ̄_m and Ω_m(z);
    - `round_delta_mean`: colossus's integer Δ_m.
  - New pure core `sigma_dndlnm_tinker08`, so that M2 code tests can inject
    σ(R, z), ρ̄_m and Δ_mean.
- `power_spectrum.py`: the same Ω_m fix in f_b = Ω_b/Ω_m (Battaglia profile).
- Tests:
  - pytest markers `gpu`, `slow` and `refdata`, deselected by default.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `conftest.py`.
  - On rusty, a persistent compilation cache in
    `JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim`.
- Not done: a P_cb option. The PKL emulator gives the total-matter P(k)
  only; production uses P_tot, as cosmocnc_jax does.

### Checks (all on the CPU node)

| check | result | tolerance |
|---|---|---|
| test suite (`pytest tests`) | 81 passed, 5 skipped (likelihood data not extracted, as before) | |
| jit(vmap) vs eager, fresh process, jitted first, batch 1 and 7, used domain | ≤ 3.1e-13 | 1e-12 |
| batch 1 vs batch 7, used domain | ≤ 1.2e-13 | 1e-12 |
| same, full grids (record only) | 4.9e-11, in dσ²/dR at R ≈ 1e4 Mpc | |
| new vs old code: z, H(z), χ, D_A | bit-identical | |
| new vs old code: k, P(k), R, σ (R = 0.5–60 Mpc) | ≤ 5.0e-15 | |
| m_ncdm fix: new HMF vs old code given m_ncdm = 0.06 in the HMF only | ρ̄_m and Ω_m(z) identical; dn/dlnM 1.6e-13 | |
| Ω_m in the HMF vs `derived()` | < 1e-14 | 1e-14 |
| σ(R) vs mcfit, numpy backend (an independent FFT) | pass | 1e-10 |
| σ(R) vs direct quadrature, R = 3–25 Mpc | pass | 1e-4 |
| σ8 map round trip (σ8 = 0.70, 0.811, 0.95) | pass | 1e-12 |
| ln A_s(0.9) − ln A_s(0.8) vs 2 ln(9/8) | pass | 2e-3 |
| σ8 from DER vs PKL + FFTLog at 8/h Mpc (two emulators) | −1.2e-4 | 5e-3 |
| Tinker f(σ), both interpolations, vs hand-interpolated tables | pass | 1e-13 |

"Used domain": σ(R) and dσ²/dR on R = 0.5–60 Mpc (M ≈ 4e10–7e16 M_sun),
and dn/dlnM where it is at least 1e-12 of its maximum.
- At the FFTLog grid edges (R ~ 1e4 Mpc), σ² is about 1e-12 of its peak, so
  the transform's round-off is amplified there.
- In the far exponential tail of the HMF (dn/dlnM ~ 1e-41), round-off is
  amplified by about 2c/σ².
- Neither region enters the simulator.

Size of the m_ncdm fix at the fiducial point: ρ̄_m rises by 0.30%; dn/dlnM
changes by up to 27%, in the far tail at M = 5e15 M_sun and z = 3.
Production uses cosmocnc's cb-only convention, which the fix does not touch.

## 2026-11-18: first tile pipeline (throughput spike, CPU part)

Code: `classy_szlite/profiles.py`, `sz_scaling.py`, `sim/` (flatsky,
instrument, halos, paint, mmf, summaries, simulator), and
`benchmarks/mapsim/spike.py`.

### Profile (A10 eq. 12, unit centre, truncated at 5 R500)

| check | result |
|---|---|
| T(u) by the 3D sinc transform: 4000 vs 8000 quadrature panels | 1.2e-15 of T(0) |
| vs an independent 2D Hankel transform of the projected profile (u ≤ 300) | 5.2e-7 of T(0), the accuracy of that check |
| vs szsim's table (unit-Y500 normalisation, divided by fn(0)) | 3.1e-5, szsim's trapezoid accuracy |
| fn(0) · T(0) vs szsim's Y5R500/Y500 | 1.796440 = 1.796440 |
| spline table vs direct quadrature at off-node u | 1.4e-8 of T(0) |

T(0) = 0.630462 (in θ500² units); p0 = 0.873294 (central projected shape).

### Tile census at the paper's fiducial point (σ8 = 0.811)

These figures are for one 10° × 10° tile:
- 73,400 halos with M500c ≥ 1e13 at 0.01 < z < 3;
- θ500: 92% below 1′, 798 between 2′ and 5′, 43 between 5′ and 10′, 6 above
  10′;
- 4,470 halos with M ≥ 5e13 (confirmation candidates);
- 2.0e8 footprint pixels for the scatter-add painter.

### Painter (scheme A: scatter-add of beam-convolved radial tables)

- **Halos with θ500 ≤ 6′** are scatter-added on the 0.25′ grid, in 14
  footprint classes (17–385 pixels).
- **Halos above 6′** are synthesised exactly on the in-band modes. The
  phases use integer arithmetic, so float32 keeps about 1e-7 rad.

Error against exact synthesis, as max |Δy(k)| / max |y(k)|, for 360–392
halos in all classes:

| ln θ table spacing | error |
|---|---|
| 0.02 | 1.55e-4 |
| 0.01 | 4.1e-5 |
| 0.005 | 1.08e-5 |
| direct path (θ500 > 6′) | 2.5e-7 |

The error falls quadratically with the table spacing, so it comes from the
linear interpolation in ln θ. A spacing of 0.004 (≈ 7e-6, an 8 MB table) will
be the default.

### Matched filter

| test | result |
|---|---|
| noise only (CMB + noise): q mean and std at all 15 scales | ≈ 0 and 1 ± 0.05 |
| noise only: σ_j from the tile's covariance vs the analytic covariance | within 5% |
| injection, noise-free, exact synthesis, at the true position and scale (4 scales) | \|ŷ0/y0 − 1\| ≤ 4.4e-7 |
| injection, painter with 0.005 tables (2 scales) | +2.5e-6, −1.4e-6 |

### Halo sampler

- `jax.random.poisson` matches its expectation in every λ class (|z| < 1.6)
  and gives var/mean = 1.
- For the drawn halos, χ² is 1.3–8.5 over 6 z bins and 95–136 over 150 mass
  cells (64 tiles).
- One first seed gave χ² = 25.9 over 6 bins, with no systematic pattern in
  four other seeds. The test now uses 16 tiles.

### CPU timing (16 rome cores, one tile, painter tables at 0.02)

| stage | time |
|---|---|
| halos | 74 ms |
| painting | 2.71 s |
| maps | 214 ms |
| filter (covariance and 15 inverse FFTs) | 96 ms |
| peaks, merging, confirmation and histogram | 394 ms |
| whole tile | 3.46 s |

That tile had 426 peaks at q > 4, 391 of them kept and confirmed, and no
overflow. The integer outputs are identical at batch sizes 1 and 2.

## 2026-11-18: M1, instrument against szsim

`tests/test_m1_instrument.py` (marked refdata) compares against a dump from
szsim (`benchmarks/mapsim/refgen_szsim.py`, run in the sz-poisson-blending
environment). All 10 tests pass.

| quantity | agreement |
|---|---|
| constants, grid, bins, in-band modes | identical |
| SED | 1e-14 |
| beams, N_ij(ℓ) (goal, baseline, no atmosphere) | 1e-12 |
| binned analytic covariance | 1e-11 |

An independent check puts the SED null at x = 3.8300, i.e. 217.5 GHz.

The lensed TT from the emulator (`instrument.cmb_cl_emulator`) against
szsim's CAMB table, at the table's cosmology:

| ℓ range | max deviation |
|---|---|
| ≤ 3000 | 0.42% (plan bound: 0.5%) |
| 3000–9000 | 0.5–0.7% |
| 9000–9500 | 3.8% |

The emulator's last range, 9000–9500, is never used: beyond ℓ = 9000 the
spectrum is a power law fitted on 7000–9000, with slope −3.99. The two codes
also differ in σ8 by 0.2% (DER 0.81183, CAMB 0.81025). They use different
neutrino set-ups: three of 0.02 eV in CLASS, one of 0.06 eV in CAMB.

## 2026-11-18: GPU spike, first run (commit 3103321)

Jobs 7493762 (A100-SXM4-80GB, workergpu047) and 7493767 (RTX PRO 6000
Blackwell, workergpu175) ran `benchmarks/mapsim/spike_gpu.sh`; each took
18 min.

**Note:** the jobs imported the rusty clone, which was updated during the run
with additions only (new tests, `cmb_cl_emulator`). From the second run on,
jobs use frozen snapshots (`mapsim-runs/<sha>`).

| check | A100 | RTX Blackwell |
|---|---|---|
| CPU vs GPU, cosmology stage, used domain | ≤ 5.1e-13 | ≤ 3.0e-13 |
| test suite on GPU | 81 passed | 81 passed |
| painter vs exact synthesis | identical to CPU (9.33e-5 with 0.02 tables) | same |
| batch invariance 1/4/16, deterministic ops | integer outputs identical | identical |
| σ_j, batch 16 vs 1 | 9e-10 | 9e-10 |
| 16-tile histogram | identical across the two GPU types | |

Per tile in ms at batch 16 (batch 1 in brackets):

| stage | A100 | RTX Blackwell | 16-core CPU (batch 1) |
|---|---|---|---|
| halos | 0.23 (1.9) | 0.18 (1.4) | 74 |
| painting | 11.1 (15.2) | 5.1 (8.0) | 2710 |
| maps | 1.2 (2.0) | 0.6 (0.9) | 214 |
| filter | 2.2 (2.3) | 1.4 (1.5) | 96 |
| peaks + summaries | 19.2 (22.3) | 12.4 (13.1) | 394 |
| whole tile | 34.3 (38.7) | 19.5 (22.9) | 3460 |

- Peak device memory: 0.84 GiB at batch 1, 9.24 GiB at batch 16.
- Compiling a whole tile takes 30–40 s.
- Per-tile time barely changes beyond batch 2: the GPU is already saturated.
- With deterministic ops (batch 8), painting costs 661 ms (A100) and 294 ms
  (RTX) per tile: identity tests only.

**Profile** (A100, batch 16; `trace_a100_B16`): of 554 ms of kernel time,
one scatter kernel took 307 ms (55%). It was the `bincount` inside
`jnp.nonzero(size=K)` on the 21.6M-element peak mask, heavily contended
because almost every element hits one bin. Painting scatters took about
130 ms (8 ms per tile), and the 128-step direct-path scan 27 ms (1.7 ms per
tile).

**Fixed in 9623b81:**
- a prefix sum plus a binary search per output slot (`sim/_util.compact`),
  with the same output as `jnp.nonzero` (tested);
- boolean scale comparisons;
- the direct path vectorised 16 halos per step.

The second run measures the effect.

## 2026-11-18: GPU spike, second run (snapshot 49a5e50)

Jobs 7493804 (A100) and 7493807 (RTX PRO 6000 Blackwell) ran from the frozen
snapshot `mapsim-runs/49a5e50`; the log confirms the import path. The run
used `sim/_util.compact` and the 0.004 paint tables.

| check | result |
|---|---|
| CPU vs GPU parity | pass |
| test suite on GPU | 94 passed, 5 skipped (including the new simulator tests) |
| painter vs exact synthesis (RTX) | 4.69e-6 (CPU: 4.63e-6) |

Per tile in ms, before → after the fix:

| stage | A100, batch 16 | RTX, batch 16 | RTX, batch 64 |
|---|---|---|---|
| halos | 0.23 → 0.24 | 0.18 → 0.19 | 0.11 |
| painting | 11.1 → 10.1 | 5.1 → 4.6 | 4.7 |
| maps | 1.2 → 1.1 | 0.6 → 0.6 | 0.6 |
| filter | 2.2 → 2.2 | 1.4 → 1.4 | 1.3 |
| peaks + summaries | 19.2 → 0.5 | 12.4 → 0.34 | 0.30 |
| whole tile | 34.3 → 14.4 | 19.5 → 6.8 | 6.8 |
| peak memory | 9.2 → 7.9 GiB | → 7.9 GiB | 30.4 GiB |

- Painting is now about 70% of a tile.
- The RTX paints 2.2× faster than the A100 (scatter-add and gathers), and
  the whole tile runs 2.1× faster.
- Throughput saturates from batch 4. Compiling a whole tile takes about 33 s.

### Prior corners (paper Table 1)

The SO-like priors: σ8 U(0.7, 0.9), ω_c U(0.0804, 0.204), A_SZ
U(−4.41, −4.21), α_SZ U(1, 1.24), h N(0.674, 0.01), ω_b N(0.0222, 0.00015),
n_s N(0.96, 0.0042), σ_SZ N(0.173, 0.05), β_CMBlens N(0.92, 0.01),
σ_CMBlens N(0.22, 0.05). Fixed: (1−b) = 0.8, r = 0. 40% of the sky, about
16,000 clusters at q > 5.

Halos with M ≥ 1e13 per tile (h at ±3σ):

| corner | halos | M ≥ 5e13 | footprint pixels | direct path | relative to fiducial |
|---|---|---|---|---|---|
| low (σ8 0.7, ω_c 0.0804, h 0.704) | 19.2k | 0.7k | 5.4e7 | | 0.26× |
| fiducial | 73.4k | 4.5k | 2.0e8 | 28 | 1× |
| high (σ8 0.9, ω_c 0.204, h 0.644) | 197k | 16.6k | 5.1e8 | 95 | 2.7× |

The capacity classes (M2) must span these. The conf_cap of 16,384 used in
the spike would overflow at the high corner.

**Emulator range.** The ede-v3 training box, inferred from the input
standardisation, is ln(10¹⁰A_s) 2.50–3.50 and ω_cdm 0.0801–0.2000. The prior
corners map to ln A_s = 2.12–3.81.
- `build(..., as_reference=3.0)` evaluates the emulators at ln A_s = 3 and
  rescales the linear P(k) by A_s/A_s,ref, and σ8 by √(A_s/A_s,ref). This is
  exact in linear theory, and the simulator uses it.
- Inside the box it agrees with direct evaluation to 1.0e-4 (P) and 2.0e-5
  (σ8). At the corners the halo counts differ by ≤ 6.4e-4: the networks
  extrapolate gracefully in A_s.
- ω_c > 0.2000 (2% of the prior) is still an extrapolation, to be checked in
  M2.

### Counts

- At SO goal with no foregrounds, the 16 spike tiles average 237 confirmed
  detections per tile at 5 < q < 40, 0.01 < z < 1.5: about 39,000 over 40%
  of the sky, 2.4× the paper's ~16,000.
- Expected contributors: goal depth and no foregrounds against the paper's σ
  table (Websky-based, a different noise model), and the finder's search
  lift. M7 quantifies this.

### M0 decisions

- **Painter:** scheme A (scatter-add of radial tables, θ500 ≤ 6′; exact
  synthesis above). It meets the budget by a wide margin, so the
  Fourier-deposit scheme and a Pallas kernel are not needed now.
- **Production GPU:** RTX PRO 6000 Blackwell, batch 16 (8 GiB).
- **Deterministic ops** only for identity tests: painting is 20–60× slower
  with them.
