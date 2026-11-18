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
