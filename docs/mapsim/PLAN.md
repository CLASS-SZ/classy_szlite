# mapsim: a map-level simulator for cluster-count SBI

Branch `mapsim` of classy_szlite. Plan v2, 2026-11-18, revised twice:
- after an independent review of v1 from three angles (physics conventions, GPU engineering, verification), which raised 44 issues;
- after a completeness check and a fact check of the revision, which found 15 issues only partly addressed and 13 factual errors.

All are fixed below.

Updated 2026-11-20 (v2.1) after the tests with SO's MSS2 noise simulation (`LOG.md`, 2026-11-19 and 2026-11-20) and a readiness audit:
- realistic noise is a new milestone, M6b, with the design decided on 2026-11-20 (§3.5);
- the geometry for real data is decided: simulate the data pipeline's tapered cut-outs (§3.5, M9);
- inference does not wait for the noise generator: M7 and the library choice start now.

## 1. Goal and headline tests

We extend the simulation-based inference (SBI) of cluster counts in Zubeldia, Bolliet, Challinor & Handley (2025, arXiv:2504.10230) from a catalogue-level simulator to a map-level one. For each parameter draw:

1. sample halos;
2. paint their tSZ signal into the six SO LAT channels;
3. add lensed CMB and noise;
4. run a matched-filter cluster finder;
5. reduce what it detects to the paper's data vector (§3.4).

The map-level simulator captures two things the paper's catalogue-level model leaves out:

- **Blending of close clusters.**
- **The finder's own selection.** Its search over position and 15 scales lifts faint clusters (optimisation bias). Its noise level, estimated from the maps, includes unmasked tSZ (confusion): at SO goal, σ_finder/σ_analytic = 1.035–1.07 over 0.5′–15′ after masking in the earlier project, and 1.04–1.08 in our simulator (2026-11-18, 16 tiles).

Our step 3b showed that the second effect dominates for an analytic selection. With q_obs = q_true + N(0,1) (the paper's dof = 0) and the finder-calibrated noise σ_eff = r̄σ_Y, the selection model's own error is +15.6σ / +17.8σ in S8 (baseline / goal) in config a, where the Y–M relation is fixed. An analytic σ_y0 would add the noise-level error on top. Blending there is −2.1σ / −4.0σ. Config d frees (1−b), α and σlnY and absorbs most of it: −1.2σ / −1.9σ against +0.3σ / +0.4σ (sz-poisson-blending `analysis/realistic/step3b/RESULTS.md:42,327-328,345`; production finder, credit limit, 40% of the sky).

Headline tests, each on map-level mock data at a known cosmology:

- **T1.** The map-level SBI recovers the truth, with posterior coverage.
- **T2.** The paper's catalogue-level SBI on the same mocks gives the total bias.
- **T3.** Split that bias into three parts, as step 3b did:
  - *Search selection:* the control runs the same finder on halo-free tiles with each halo inserted alone, as sparse isolated injections. There σ_finder ≈ σ_y0, so the catalogue-level bias on these mocks is the search and optimisation part alone.
  - *Confusion noise:* the change from the analytic σ_y0 to the finder-calibrated σ_eff = r̄(θ)σ_y0, with r̄ measured on full tiles.
  - *Blending:* what remains with the finder-calibrated baseline. That baseline uses σ_eff and the control's measured isolated-cluster kernel, step 3b's analyst. Without this second baseline, "full minus control" would mix blending with confusion noise.
- The paper's baseline (variant (i)) uses *our* analytic SO-goal σ_y0(θ500), not the paper's Websky table, which comes from a different noise model.
- **T4.** The map-level SBI on mocks from an independent generator (the legacy szsim pipeline, §4), to test the simulator's own systematics.

## 2. Ground rules

- **We write it ourselves, in JAX, for GPUs.**
  - SZiFi, cosmocnc_jax and JXPaint are reference code: we read them, rewrite, and benchmark against them, never copy. SZiFi and JXPaint have no licence. cosmocnc_jax declares MIT in its `pyproject.toml` but ships no LICENSE file, so we treat it as unlicensed until its authors confirm.
  - classy_szlite (MIT) is the base, and the simulator lives in it.
- **Simple problem first:** flat periodic 10° × 10° tiles (1200² pixels of 0.5′), Poisson halos, uniform depth, SO goal, the six LAT channels, lensed CMB, white and atmospheric noise, and no foregrounds. Full-sky HEALPix, clustering, masks and foregrounds come later.
  - The noise is moving to per-tile noise calibrated on MSS2 (§3.5, M6b). The 2019 forecast model (white plus atmospheric, correlated within a tube) stays as an option.
- **Every component has two independent checks where possible,** each with a stated tolerance and reference. Code tests (identical inputs, round-off tolerances) are kept separate from physics comparisons (different inputs, documented, no pass/fail).
- **Flatiron's rules** (`/etc/claude-code/CLAUDE.md` on rusty):
  - every Slurm submission needs the user's explicit approval;
  - Slurm queries are spaced at least 30 s apart and scoped to the user;
  - file-tree traversals are scoped, with `timeout 120` and `-maxdepth 7`;
  - code and environments live in home, bulk data on ceph.

## 3. Conventions (pinned)

### 3.1 Numerics

- **Units:** h-free physical (M_sun, Mpc). Maps are in μK_CMB, angles in radians in code and arcmin at interfaces.
- **dtypes by stage:**

  | dtype | used for |
  |---|---|
  | f64 | cosmology, HMF, λ cells, bin accumulation, the 6×6 solves, the per-scale normalisations |
  | f32 / c64 | maps, FFTs, τ_θ(ℓ), beams, the paint table, filter weights |
  | int32 | every index |

  - `classy_szlite/sim` uses `jax.numpy_dtype_promotion('strict')`.
  - A jaxpr audit test fails on any f64, c128 or s64 value outside an allowlist.
  - Float32 paths run under `jax.default_matmul_precision('highest')`, because TF32 would break the float32 tolerances.
  - Halo positions are stored as an int32 pixel plus a float32 offset.
- **Randomness:** `fold_in(base_key, sim_index)`, then the tile index. Every random draw passes an explicit dtype.
- **Determinism:**
  - Identity tests run with XLA's deterministic-ops flag.
  - Covariance binning uses a sorted, segmented reduction, not an atomic `segment_sum`.
  - Comparisons near thresholds and ties are tie-aware.
  - We promise bitwise invariance across batch sizes on one GPU type, and only statistical equivalence across GPU types.

### 3.2 Model (production: the paper's SO-like model)

- **y0–M relation** (cosmocnc_jax `survey_sr_so_sim.py:13-20,55-62`; scatter at 639-641): **median** ȳ0 = 10^A_SZ · E(z)² · [h70 (1−b) M / 3e14 M_sun]^α_SZ · h70^(−1/2), with E(z)² hard-wired. Each halo draws y0 = ȳ0 · exp(σ_lnq · N(0,1)).
- **θ500** = 6.997′ · h70^(−2/3) · [(1−b) M / 3e14]^(1/3) · E^(−2/3) · (500 Mpc / D_A). The exact R500/D_A gives 6.9957′. The production model uses the 6.997′ formula, as cosmocnc_jax does; M3 and the legacy benchmarks use the exact R500/D_A, as szsim does.
- **Fiducial point** (paper Table 1 and Sec. II.1, read from the arXiv HTML on 2026-11-18; we use cosmocnc_jax's unrounded A_SZ and α_SZ):

  | parameter | value |
  |---|---|
  | σ8 | 0.811 |
  | ω_c | 0.121 |
  | h | 0.674 |
  | ω_b | 0.0222 |
  | n_s | 0.96 |
  | A_SZ | −4.3054 (the paper rounds to −4.31) |
  | α_SZ | 1.1233 (the paper rounds to 1.12) |
  | σ_lnq | 0.173 |
  | (1−b) | 0.8 (the paper's β_SZ; cosmocnc_jax's `bias_sz`) |
  | Σmν | 0.06 eV (3 × 0.02 eV, as the ede-v3 emulators assume) |

  Note that cosmocnc_jax's default `bias_sz` is 0.62, and its `beta_szifi` is a different parameter (an E(z) exponent, unused).
- **SBI priors** (decided 2026-11-18; numbers in LOG.md, "priors: halo load at the edges of narrower boxes"):
  - Map-level runs use one box, sized for the 10%-sky runs:
    - σ8 U(0.770, 0.852), ω_c U(0.0804, 0.169), A_SZ U(−4.53, −4.08) and α_SZ U(1, 1.24);
    - the paper's Gaussian priors for h, ω_b, n_s, σ_lnq, β_CMBlens and σ_CMBlens.
  - **The rule:** each edge is 8 of the paper's posterior widths (its likelihood standard deviations) from the fiducial, clipped at the paper's prior.
    - Posteriors at 10% of the sky are about 1.3× the paper's, so the 95% intervals stay inside with room for the mock-to-mock scatter.
    - Compared with the paper, σ8 and ω_c are narrower and A_SZ is wider. The paper's A_SZ 95% interval ended less than 2 widths from its edges.
  - **The 40%-sky runs** may narrow to 6 widths once the first map-level posteriors show their real widths: σ8 U(0.780, 0.842), ω_c U(0.085, 0.157), A_SZ U(−4.48, −4.13).
  - **Catalogue-level analyses of map-level mocks** use the paper's Table 1 priors with A_SZ widened to U(−4.53, −4.08). Their posteriors are expected to shift, and a tight box would hide the shift.
- **Pressure profile** (`profiles.py`, kept separate from `cl_yy`'s defaults):
  - The Arnaud et al. 2010 eq. 12 set, as in SZiFi and szsim: P0 = 8.403 h70^(−3/2), c500 = 1.177, γ = 0.3081, α = 1.0510, β = 5.4905.
  - Truncated at 5 R500 in 3D, and normalised to unit central value before the beam.
  - classy_szlite's existing `_A10_*` constants (8.130 / 1.156 / 0.3292 / 1.0620 / 5.4807) are a different set and are not used here.
  - A unit test checks that 10^A_SZ with A_SZ = −4.3054 reproduces the eq. 12 central y at the pivot, to ~1e-3.
- **Halo mass function:** Tinker08 at Δ = 500c, with two conventions switchable:
  - the interpolation in Δ: linear in Δ (cosmocnc) or linear in log10 Δ (classy_szlite today);
  - the neutrino treatment in ρ_m and in Δ_mean = 500/Ω_m(z);
  - rounding Δ_m to an integer, as colossus does (used only for the colossus code test);
  - P_tot or P_cb in σ(M).
  - Production (decided 2026-11-18): Tinker08 interpolated in log10 Δ, ν excluded from Ω_m, P_tot.
    - The paper does not state its choices.
    - cosmocnc_jax's default, linear in Δ, gives 0.3–2.3% more halos at 1e13. That difference is accepted.
    - Until 2026-11-18 production used linear in Δ, so earlier logged numbers carry it.
  - **classy_szlite bug, fixed in M0:** `hmf.py:222` adds m_ncdm/(93.14h²) to Ω_m, but m_ncdm is the per-species mass (0.02 eV × 3). The HMF therefore holds a third of Σmν, while `derived()` (`api.py:59`) uses all three; Ω_m in the HMF is about 0.3% low.
- **Halos:** M500c ≥ 1e13 M_sun, which matters for confusion, and 0.01 < z < 3. The data vector counts only detections.
- **CMB:** lensed TT, held fixed at the fiducial for the headline tests. A cosmology-dependent CMB is a separate test.
- **Lensing observable** (catalogue level, for the matched halo; cosmocnc_jax's `p_so_sim` model):
  - parameters: β_CMBlens = 0.92, σ_lnp = 0.22, corr(ln q, ln p) = 0 (the paper's r = 0), a_lens = 1;
  - **σ_κ(θ) at SO goal lensing depth** (decided 2026-11-18, to match the goal tSZ maps).
    - **Source:** the minimum-variance quadratic-estimator noise curve of the SO noise models with the tSZ deprojected (deproj1), goal version (SENS2). The paper cites the deproj0 file, but its table is reproduced only by deproj1 (2026-11-18).
    - **Filter:** our own truncated-NFW matched filter (c500 = 3, 5 R500), as the paper describes.
    - **Check:** the same code on the SENS1 curve must reproduce cosmocnc_jax's baseline table (`so_sim_lensing_mf_noise.npy`).
    - **Baseline comparisons:** the paper comparison and SO_sim_0 keep cosmocnc_jax's cubic log-log fit to its baseline table. We hold that fit at its end values outside the table (0.739′–5.87′), because SO_sim_0's mean p favours that (0.297 against 0.307 extrapolated; observed 0.2986 ± 0.0080);
  - a detection without a matched halo gets p ~ N(0,1);
  - the summary is the mean p over the data-vector sample;
  - blending and mis-centring of the lensing signal are not modelled in v1.
- **Population caveat:** the legacy population of our previous project had ~2.65× SPT's measured D_3000^tSZ (step 1). The production model has θ500 7% smaller and y0 ~15% higher at fixed M. Its D_3000 is computed in M3 and is expected to be similar. A lower-amplitude variant (M8) shows how the results scale.

### 3.3 Painting and template (one band-limited convention)

Paint one y map per tile, never one per channel:

1. Paint by scatter-adding a radial table, at 0.25′ (2400²) with a 0.8′ painting beam.
2. Take the rfft2 and crop to the 0.5′ in-band modes.
3. Divide out the painting beam.
4. Form each channel as a_i · B_i(ℓ) · y(ℓ) on the in-band modes. This is szsim's design, validated against exact synthesis.

The finder's template τ_θ(ℓ) is the exact Fourier transform of the same truncated profile: continuous, with no pixel window. The painter and the template are therefore consistent by construction. SZiFi-compatible (pixel-sampled, corner-centred) templates exist only inside the M4 tests.

### 3.4 Data vector (production)

- **Detections:** 3D local maxima of q over position and the 15 scales (geomspace 0.5′–15′), with q > 4, from the final pass after masking. Entries within a merge radius of 1.5′ merge into the highest-q entry, across scales. The final pass's buffer keeps only q > 5, the data vector's lower edge: merging makes this exact, and it holds ~45% fewer peaks (buffer 3,072, enough for the adopted box's heaviest corner with 1.6× to spare).
- **Confirmation and redshift**, which mimics follow-up. This is the one step that uses the true halos, standing in for optical confirmation; everything else uses only the maps.
  - A detection is confirmed if a halo with M500 ≥ M_conf lies within r_conf of it. It takes the redshift of the matched halo with the largest y0.
  - Unconfirmed detections are dropped, as in a real survey.
  - The default is (M_conf, r_conf) = (5e13 M_sun, 1.5′), applied identically to simulations and mocks.
  - False detections that line up with a massive halo are kept, which is how they enter by construction.
- **Histogram:**
  - Edges are the paper's (Sec. II.3): four q bins with edges logspace(log10 5, log10 40, 5), and five z bins with edges linspace(0.01, 1.5, 6).
  - Edge handling follows cosmocnc_jax's binning (open intervals; q ≥ 40 and z outside 0.01–1.5 are dropped), so the catalogue baseline and the paper's data vectors match.
  - With the mean p, that makes 21 numbers.
- **Legacy credit rule:** the filter-response attribution of steps 3a/3b is kept only to reproduce step-3b numbers (M6).

### 3.5 Noise and geometry (decided 2026-11-20)

What the MSS2 tests showed (details and numbers in `LOG.md`, 2026-11-19 and 2026-11-20):
- In true sky units the plain mean of MSS2's four splits has white levels of 6.9 μK-arcmin at 90 GHz and 7.9 at 150 GHz (ℓ = 8000–12000), between SO's goal and baseline; MSS2's own weighted coadd is deeper still. An earlier reading put the coadd at SO baseline because its window (ℓ = 12000–16000) sits on a bump in MSS2's spectra at ℓ ≈ 15000. In the finder's noise, the four-split mean is about our goal depth, and one split about 1.35× baseline.
- With the simulator's own sky in MSS2 noise and in model noise (paired, 32 tiles × 16 skies):
  - at four-split depth, Gaussian noise with each tile's own spectra reproduces MSS2's data vector to 0.1–0.3% (40%-sky Poisson error 0.45%);
  - at one-split depth, MSS2 gives about 1% more detections, mostly at q = 5–8.4, because the masking pass removes more of its streaks (single noisy observations) and so lowers its noise estimate by 0.27%. Gaussian models give 1.0–1.2% fewer detections, and mnms, fitted to MSS2 itself, 0.8% fewer.
- Within-tile depth variation and 2D anisotropy changed the counts by less than 0.2%. A generator from band-averaged spectra times the depth map gave −0.9% ± 0.65%: the depth map sets each tile's noise level only roughly.
- Tapered cut-outs, which is how real data come, give q 5–8% above periodic tiles. A conversion calibrated on noise alone is 0.5–0.8% off in counts once clusters are in the map.
- The tests treated MSS2's equal-angle pixels (RA pixels cos δ narrower) as square. Ratios between noise models are unaffected, but the spectra must be re-measured in true sky multipoles before the simulator uses them: as measured, they would add up to 10% noise at dec −35° and stretch multipoles along RA.

Decisions:
- **What the generator draws:** each tile's own stationary, isotropic Gaussian noise, with binned 6 × 6 auto and cross spectra in true sky multipoles and true pixel area. They come from MSS2 now and from the data's split differences later. No depth map and no 2D spectra in v1. The 27 and 39 GHz bands get a noise floor above ℓ ≈ 5400, where their 2′ maps are empty.
- **Depth:** the four-split mean of MSS2 for the next runs (the depth at which Gaussian noise passes), with depth as one scale per tile, so year 1 is a one-number change. Year 1 and the streaks are decided after the colleague's answer on how their year-1 maps were made and after an accuracy calculation: what the one-split residual does to σ8 and Ω_m with A_SZ, α_SZ and σ_lnq free. No streak model before that. Year 1 would also need a year-1 lensing noise curve and a wider prior box (one split gives 0.41× the detections).
- **Tiles and sky positions:** a per-tile table (tile id, spectrum index; later depth scale, taper and mask) passed beside the shared static arrays. That table is also the start of the multi-tile driver M8 needs.
- **Geometry for real data:** simulate what the data pipeline does (the same taper, overlapping tiles, only each tile's interior counted), and reproject the data to square-pixel tiles, rather than correcting the data's q. It costs about 1.6× the GPU time (§7). It is built at the real-data stage (M9), but the noise draw is written for any grid size now.

## 4. Modes

- **production:** §3.2–3.4.
- **legacy:** the previous project's model, for statistical reproductions (M5 r(θ), M6, T4): Planck 2013 Y500–M without the h70 factor, θ500 from the true M (exact R500/D_A), spherical-Y500 normalisation, and colossus planck18 HMF tables.
- **szsim-compatible finder:** for identity tests on exported tiles (M4–M6):
  - template table `data/template_hankel_arnaud_5R500.npz`, interpolated linearly in u;
  - ℓ bins of width 72, with edge modes in the upper bin and averages over in-band modes only;
  - a sharp band, ℓ ∈ [100, 20000];
  - scales from `tile_PID['thetas']`;
  - the disc rule distance ≤ 3θ_j, periodic.

## 5. Architecture (in classy_szlite)

```
classy_szlite/constants.py      physical constants (moved out of hmf.py and power_spectrum.py)
classy_szlite/sz_scaling.py     y0–M, theta500–M (production and legacy modes), lensing kappa0–M
classy_szlite/profiles.py       A10 eq.12 projected profile (3D truncation, unit centre), Hankel tables
classy_szlite/sim/flatsky.py    tile geometry, l grids, FFT normalisation, Hermitian Gaussian modes, beams
classy_szlite/sim/instrument.py SO LAT channels: beams, white + atmospheric noise (tube correlations), tSZ SED
classy_szlite/sim/noise.py      per-tile noise calibrated on MSS2 (M6b): binned 6x6 spectra in true sky multipoles,
                                correlated Gaussian draws on any grid size, the per-tile table
classy_szlite/sim/halos.py      lambda(z, ln M) cells per tile, Poisson draws into capacity-class buffers + overflow flags
classy_szlite/sim/paint.py      band-limited painter (lax.scan over halo blocks into one donated 2400^2 buffer;
                                int32 indices, mode='drop'; separate path for theta500 > ~10')
classy_szlite/sim/mmf.py        covariance from the maps (sorted segmented binning), per-bin 6x6 Cholesky,
                                shared weights w(l) = C^-1 a for all scales, sigma_y0, q cube, 3D peaks
                                (roll comparisons + ordered compaction), FFT disc masks, whitened masked covariance
classy_szlite/sim/match.py      confirmation and redshift (production); lean credit rule (legacy, for M6)
classy_szlite/sim/summaries.py  merge, histogram, stacked lensing
classy_szlite/sim/catalogue.py  catalogue-level baselines (the paper's, and finder-calibrated)
classy_szlite/sim/control.py    blending-free map-level control (isolated injections on halo-free tiles)
classy_szlite/sim/simulator.py  simulate(key, theta): lax.map over tile chunks, capacity-class routing
```

The refactor needed before any of this (M0) makes cosmology jit- and vmap-able:
- remove the `float()`/`int()` calls on traced values (`cosmology.py:117-121`);
- replace the per-call mcfit `TophatVar` with the pure-JAX FFTLog (`cosmology.py:176`);
- load the emulator weights eagerly into a pytree that is passed as an argument, because the lru_cache could otherwise store tracers;
- replace `squeeze()` with explicit reshapes;
- add a traceable σ8 → ln10^10 A_s map (A_s ∝ σ8² at fixed other parameters) and Ω_m ↔ ω_cdm under the pinned neutrino convention.

**Capacities and overflow.** Every fixed buffer returns a count and an overflow flag: halos, painter slots, peaks and matching candidates. A draw that overflows any of them is rerun in the next capacity class, never truncated. Draws are routed to a class by their expected halo count Λ, computed before sampling.

## 6. Milestones and benchmarks

References come from sz-poisson-blending: float64 computations, verified twice. Some stored references are float32: the truth tables, and the SZiFi runs, whose maps were float32. Paths are relative to it. Statistical criteria are stated in units of the combined standard error (pass if |Δ| < 3 SE). Count tests are Poisson tests. Float32 tolerances are relative (|Δq| ≤ 1e-5 · max(1, q)), with a written near-tie rule for peak identity.

**M0. Environment, refactor, throughput spike, data.**

Refactor:
- The §5 refactor: `jit(vmap(build))` equals the eager outputs to ≤1e-12 on one device and ≤1e-10 between CPU and GPU, on the domain the simulator uses (σ(R) on R = 0.5–60 Mpc; dn/dlnM where ≥ 1e-12 of its maximum; the FFTLog grid edges and the far HMF tail amplify round-off and are recorded, not tested). The test runs in a fresh process with the jitted call first, then eager, at batch sizes 1 and 7. That catches the cache-tracer and `squeeze()` hazards. Done on CPU 2026-11-18 (`M0_LOG.md`).
- The m_ncdm fix, with a test that Ω_m in the HMF equals `derived()`.
- Switchable Tinker and ν conventions.

Throughput spike, built before the exact benchmarks so that its results can still change the design:
- A skeleton `jit(vmap)` tile pipeline: Poisson halos in fixed buffers, the painter, CMB and noise drawn as Fourier modes, one factorised matched-filter pass, peaks, simple matching and the histogram.
- Run at fiducial and prior-corner capacities on an A100 and an RTX Blackwell, with `jax.profiler` and `memory_stats`.
- Two painter schemes are compared: scatter-add of the radial table (lax.scan over halo blocks), and a Fourier-space deposit (halo positions binned onto the grid per θ500 class, then convolved with the class template by FFT). A Pallas kernel is tried if both are too slow.
- Its results set the painter scheme, the batch size per GPU type, the capacity classes and the budget.
- Invariance test: the same keys give the same integer data vector at batch sizes 1, 4 and 16 on one GPU type, in deterministic mode.

Test infrastructure:
- pytest markers `gpu`, `slow` and `refdata`, deselected by default and in CI; a fast CPU subset stays in CI.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` for tests.
- A persistent compilation cache on rusty.

Data:
- Reference data on rusty: done on 2026-11-18. `data_real/`, `data_cosmo/`, `data/blending`, `data/catalogs`, the HMF and ρ tables and `template_hankel_arnaud_5R500.npz` (11,373 files, 12.2 GB) are in `/mnt/ceph/users/bbolliet/sz-poisson-blending`, with SHA-256 identical to the Mac's.
- An export step on the Mac, in the old `.venv`, regenerates the float64 tSZ, CMB and noise maps of the tiles used in M4–M6, plus SZiFi-format float32 maps, stored with checksums.

**M1. Instrument, CMB, noise.**
- N_ij(ℓ), B_ℓ and the SED against `szsim/instrument.py` to ≤1e-12. The second method is `analysis/realistic/step1/s02_noise_check.py`. Also the band-centre g values and the 217.51 GHz null.
- Lensed TT from the emulator, run at the reference file's cosmology (H0 = 67.66, Σmν = 0.06, A_s = 2.1045e-9), against `data_real/cmb_lensed_TT_camb.npz`: ≤0.5% for ℓ ≤ 3000. Higher ℓ is documented, not tested; CAMB and CLASS themselves differ there. The emulator stops at ℓ = 9500: state the extension above it. Its effect on σ is bounded by `analysis/realistic/step1/RESULTS.md:104`: zeroing the CMB above ℓ = 6000 changes σ_Y by ≤ 4.4e-4.
- Simulated tiles: binned auto- and cross-spectra against the input, |Δ| < 3 SE per bin.

**M2. Halos on tiles.**
- Code tests with identical σ(M,z), ρ_m and Ω_m(z): f(σ) and dn/dlnM against cosmocnc_jax (interpolation set to match) and against colossus (with the integer-Δ_m switch on), to ≤1e-4. Inject cosmocnc_jax's `hmf_matrix` into our abundance code: counts to 5e-4.
- Physics comparisons (documented): EH98 against the emulator P(k), and lcdm-v1 against ede-v3. Both are expected at the percent level.
- Sampler: χ² on grids 4–8× finer than the sampling grid, and KS tests of z, ln M and positions within cells. Compare with szsim's `validate_sampler.log`.
- Capacities: size each class at the corners of the SBI prior, and test the overflow flags there.

**M3. Painting.**
- The profile table against high-precision quadrature, with absolute tolerances relative to g(0). Independent references at their own accuracy (1e-5 to 1e-4): `data/template_hankel_arnaud_5R500.npz`, and `verify_real/step1_derivation/proj_shape_table.npz` and `g_of_u_2dfft.npz`.
- The painter against exact type-1 NUFFT synthesis on stored truth tables (`data_real/step3/truth/truth_000.npz`, 71,291 halos): ≤1e-5 of the peak on in-band modes. Also periodicity (a roll by L gives identical results) and flux.
- **Injection test** (painter and template together): isolated clusters at random sub-pixel offsets on all 15 scales, noise-free and with noise. ⟨y0_est/y0_true⟩ = 1 within 1e-3 at the true scale. Cross-check against `data_real/step3/injections`.
- C_ℓ^yy of painted tiles against an analytic Poisson C_ℓ built from the same λ cells, the y0 relation with its e^{2σ²} factor, the same profile tables, and flat-sky ℓ. The errors come from the tiles' own scatter, since the 1-halo term is non-Gaussian. classy_szlite's `cl_yy` is only a documented physics comparison: it is a different model.
- Quote D_3000^tSZ (§3.2 caveat).

**M4. Matched filter, iteration 0.**
- **Identity:** the szsim-compatible mode on exported float64 tiles.
  - Stored intermediates first: `it0_C` to ~1e-12, then σ.
  - Then the catalogue: identical entries except documented near-ties, and |Δq| ≤ 1e-9.
- **Production σ_y0** (continuous template) against our own analytic Hankel σ_y0 with the true covariance (≤1e-6), and against szsim's `data_real/sigmaY_mmf.npz` (≤5e-6; the lattice-versus-integral agreement there is 3.6e-6, `t01_check_a.log`).
- **SZiFi, statistically,** using only runs without bug E4 (`data_real/step3_szifi_real/arbiter/runs/S1z_*`, `verify_real/step3_szifi_real_config/out/runs/S1wg_*`):
  - shift SZiFi's positions by −½ pixel and keep objects ≥ 9′ from edges;
  - ≥ 99.9% matched within 1′;
  - for the production finder, which deliberately differs from SZiFi (periodic convolution, continuous template), mean and rms Δq no worse than szsim's own agreement with SZiFi (Δq = +0.01 ± 0.20–0.25; 85–89% within one scale step).
- **σ_y0 against `analytic_sigma.npz`** (setup S0; covariance per mode = `analytic_shells.npz`, which holds CAMB lensed TT plus SO noise, inverted with Ci[0] = 0; not the emulator's TT): ~1e-5, following a written emulation recipe. Against `it0_sigvec`: scales 1–14, in the 129 isotropic / none / [100, 20000] runs.
- **Float32 against float64:** measure the error budget on exported tiles, then apply the relative tolerances.

**M5. The masking iteration.** Built 2026-11-18 (`LOG.md`, "the masking pass"); one iteration by default (`TileConfig.n_iter = 1`).
- q > 4 discs of radius 3θ_det, built as FFTs of per-scale indicator maps times integer-disc kernels. This reproduces "distance ≤ r" exactly. *Done: identical to a direct construction.*
- Port szsim's whitened masked-covariance estimator. The literal binary estimator is biased (`analysis/realistic/step3/RESULTS.md:93-96`). *Done: on one of our tiles, szsim's own code gives the same mask and σ_j to 3e-8.*
- **Identity** with `data_real/step3/tiles/tile_PID.npz` iteration 1, in the szsim-compatible mode: |Δq| ≤ 1e-9. *Open: needs the szsim-compatible filter mode.*
- **Statistics** in legacy mode over ≥ 400 tiles, in SE units: r(θ) and the noise-only σ ratios. Poisson tests for the false-detection counts. *Open (GPU); 16 CPU tiles per depth so far.*
- **Follow-ups from the GMaster and szifi_jax studies** (`LOG.md`, 2026-11-18), in order:
  - the residual tSZ in σ after one iteration (3.7–7.6% at goal): a minimum disc radius following the beams (2′, 4′), and a second iteration;
  - a many-tile no-cluster test (≥ 80 tiles × several masks, including real first-pass masks) in place of the 2-tile test;
  - the finite-sample correction (n − p − 1)/n per bin, then, optionally, in-band coupling row sums instead of the kept fraction (3 FFTs per tile).
  - No NaMaster and no full coupling matrix for the periodic tiles.

**M6. Data vector, legacy reproduction, timing.**
- The production reduction (§3.4) with unit tests: merging, confirmation, edges, overflow, lensing. *Lensing done 2026-11-18 (`LOG.md`, "stacked CMB lensing"): σ_κ from the tSZ-deprojected SO curves (deproj1), which reproduce the paper's baseline table to 0.3% with no free factor.*
- **Legacy credit rule**, the lean version (candidates within 30′; exact lattice sums only for near-ties), on a fixed-capacity cell grid.
  - Route (a), first: the counting step on exported tiles with the stored kernel files (`data_real/step3b/kernel_prod_goal.npz`, `halos_prod_goal.npz`, `arbiter/datavec_pooled_goal.npz`). Credited counts must be identical except documented near-ties, and R_p must agree to round-off.
  - Route (b), later: port the virtual-cluster kernel machinery in legacy mode, and match the SO goal deficit of −2.10% within ~0.25%.
- Measured time and memory per stage and per tile, on A100, H100 and RTX Blackwell.

**M6b. Realistic noise calibrated on MSS2** (§3.5). Data on rusty: `/mnt/ceph/users/bbolliet/mapsim/mss2` (raw MSS2, 32 stored tiles of 107 candidates, test outputs). Prototype and test: `benchmarks/mapsim/mss2_finder_test.py` (`iso_spectra`, `iso_from_spectra`, `true_ell`) and `mss2_datavector_test.py`.
- **Spectra:** re-measure every candidate tile's binned 6 × 6 spectra in true sky multipoles and true pixel area, for the four-split mean and each split, with the 27/39 GHz floor. *Done 2026-11-20* (`mss2_spectra.py`; `mss2/spectra/spectra.npz`, 107 tiles): its check recovers synthetic fields built from the maps' pixel sizes to 0.05% in all entries and sectors, and the row areas to 0.03%; a square-pixel control fails by 21% at dec −35° (`LOG.md`, "per-tile noise spectra").
- **Code tests** (identical inputs, round-off tolerances). *Done 2026-11-20* on CPU (`classy_szlite/sim/noise.py`, `tests/test_sim_noise.py`: 10 fast tests and 2 slow full-tile tests; `LOG.md`, "the noise generator"):
  - the draw equals a numpy reference from the same random numbers, to float32 round-off in every multipole bin;
  - drawn auto and cross spectra match the input with analytic errors (z-scores per bin, ratios per range of multipole and the mean ratio within 5 standard errors), also for today's tube-correlated 2019 noise;
  - other grid sizes, and the clamp beyond the measured range;
  - batched against single draws to float32 round-off (XLA's CPU FFT is not bit-for-bit batch-invariant), and full tiles under vmap with a spectrum and depth per tile: integer outputs identical;
  - halos, tSZ and CMB unchanged when the noise model changes (the keys are split per component);
  - spectrum indices checked on the host (under jit an out-of-range index would silently take the last spectrum).
- **Validation** (`mss2_datavector_test.py`, the generator as one more noise kind, paired with MSS2 on the same skies, 32 tiles × 16 skies), at four-split depth:
  - total counts within 0.3% of MSS2, with an error of at most 0.2%;
  - each q bin within one 40%-sky Poisson error;
  - the 20-bin Δχ² over 40% of the sky, after removing its noise part, at most 3;
  - the selected sample's mean p̄ within 0.5%.
  - Second check: the finder's first-pass noise level per tile within 0.3% of MSS2's.
  - Tag the finder version used: the planned masking follow-ups (M5) move these numbers, so the test is rerun after any finder change.
- **GPU:** one short RTX job: the noise tests on the GPU; GPU against CPU on one tile; `spike.py`'s run and invariance commands with a per-tile noise option (batch sizes 1, 4 and 16, deterministic ops, integer outputs identical); timing and peak memory of the noise path against the forecast path; the peak buffer at the deepest tile. *Done 2026-11-20* (job 7510857, `LOG.md`): all passed; the MSS2 noise costs no extra time or memory; the peak buffer is 64% full at the heaviest corner.
  - Then the GPU generator must reproduce the numpy periodic kind's counts on the 32 tiles. For that comparison it must be fed the same spectra as that kind (`ft.iso_spectra` on the cut-outs, square-pixel multipoles, bins of 2 × 2π / (1680 × 0.5′)), not `spectra.npz`, whose true-sky multipoles, pixel areas and tile levels differ; `spectra.npz` is for production.
- **Decisions pending:** year 1 and the streaks (§3.5).

**M7. Catalogue-level baselines.**
- Rewrite the paper's SO-like simulator.
  - Code tests against cosmocnc_jax with identical inputs, injecting `hmf_matrix`: counts to 5e-4.
  - Per-halo p to 1e-12, with the same cubic σ_κ fit and only for θ500,lens inside the table (or with clamping switched off).
  - Mean p against SO_sim_0 at its true masses.
  - The empirical correlation of the ln q and ln p residuals equals the pinned value.
- Variants: (i) the paper's (analytic σ_y0, dof = 0); (ii) finder-calibrated (σ_finder = r̄(θ)σ_y0 and the control's isolated-cluster kernel).

**M8. SBI and T1–T4,** first on 10% of the sky. The map-level runs use the M6b noise; the library choice and the catalogue-level parts start before M6b ends.
- Neural posterior estimation as in the paper: sbi, a MAF, 21 numbers.
- Coverage tests.
- The control simulator (`sim/control.py`) for T3, and the legacy generator for T4.
- A lower-amplitude variant (A_SZ lowered until D_3000^tSZ matches SPT) to show how the T2–T3 biases scale with the tSZ power.

**M9. Scale-up.**
- 40% of the sky, 10⁴ simulations, capacity classes, one worker per GPU (disBatch or a job array).
- Then the full-sky version: HEALPix painting, tiling with buffers and cores, clustering, masks and foregrounds.
- Real-data geometry (§3.5): tapered cut-outs with overlaps, only interiors counted, data reprojected to square-pixel tiles; the finder learns to take a pixel mask (it takes none today); spectra for the rest of the footprint (dec −63° to −40° and +10° to +25°, the RA 350°–10° strip, partial tiles).

## 7. Budget

Set by the M0 spike (`M0_LOG.md`, 2026-11-18).
- **Per tile, fiducial halo load, one finder pass:** 6.8 ms on an RTX PRO 6000 Blackwell and 14.4 ms on an A100 (batch 16, 8 GiB). Painting is ~70% of that; the FFTs of the maps and the filter take ~2 ms.
- **10⁴ simulations × 165 tiles:** 3.1 RTX GPU-hours at the fiducial load. That rises to ~4.6 GPU-hours with the prior-averaged halo load (×1.05 over the wide box, from 0.42× to 1.94× of the fiducial; ×1.17 over the paper's prior) and the masking pass (×~1.4, M5; on the CPU the finder takes 2.5× longer with it, and the finder is ~2 ms of the 6.8 ms on the GPU, to be measured). Coverage mocks, the 10%-sky runs and compile time (~35 s per program per job) add less than that again.

- **Realistic noise (M6b):** a few extra FFTs per tile next to the 0.6 ms map stage, and per-tile spectra of a few MB. **Simulating tapered cut-outs (M9):** about 1.6× the GPU time, so roughly 7–8 RTX GPU-hours per 10⁴ simulations of 40% of the sky instead of 4.6.

CPU fallback and cross-check: our szsim pipeline needs ~3,000–6,600 CPU-hours for 10⁴ simulations; SZiFi would need ~50,000. The GPU total must also count the coverage mocks, the 10%-sky runs, compile time, development jobs and capacity padding.

## 8. Development on rusty

The loop:
1. Edit in the Mac clone (branch `mapsim`).
2. rsync to `/mnt/home/bbolliet/classy_szlite`, `.git` included.
3. Run inside a job over SSH.
4. Copy small results back, commit on the Mac, and push the branch.

Jobs (each submission approved by the user):
- **Main development job, CPU:** `sbatch -p genx -c 16 --mem=128G -t 12:00:00`, holding the allocation with `sleep`. SCC suggests a modest genx allocation for agents (their example: 16 cores, 256 GB); we request 128 GB. Most of the float64 correctness work in M1–M5 runs here.
- **GPU jobs, only for GPU work** (the M0 spike, float32 parity, timing): `sbatch -p gpu --gpus=1 -C a100-80gb --cpus-per-gpu=16 --mem=128G`, time-limited to the task. No idle GPU holds.
- **Known limitation:** the agent works from the user's Mac over SSH, so work pauses whenever the Flatiron gateway connection (password and one-time code) drops or the Mac sleeps. Jobs keep running meanwhile.

Environment and data:
- `/mnt/home/bbolliet/venvs/classy_szlite` (uv, `jax[cuda12]` 0.12, classy_szlite editable).
- Emulator data in `~/class_sz_data`.
- References on `/mnt/ceph/users/bbolliet/sz-poisson-blending`; outputs on ceph.

## 9. Open questions for the user

1. **Confirmation rule:** keep (M_conf, r_conf) = (5e13 M_sun, 1.5′), and drop unconfirmed detections?
2. **CMB lensing in v1:** settled 2026-11-18: kept, as the data vector's 21st number (`LOG.md`, "stacked CMB lensing").
3. **NPE library:** `sbi` (PyTorch, as in the paper) first, or a JAX-native flow? Now due: M7 and M8 start in parallel with M6b.
4. **Catalogue baseline:** dof = 0 as in the paper's SO-like setup, or also a calibrated dof (k = 2 came closest in step 3b)?
5. **Target depth:** year 1 or five years? Waiting for the colleague's answer on their year-1 maps and for the accuracy calculation (§3.5).
6. **MSS2's streaks:** model them in the generator, apply a calibrated q shift, or neither? Decided with question 5.
