# mapsim results log

Results after the first tile pipeline; earlier results are in `M0_LOG.md`.
Runs on rusty's CPU development job (16 cores, JAX 0.12) unless stated.

## 2026-11-18: cluster counts at goal and baseline depth

Script `benchmarks/mapsim/counts_modes.py`: 32 tiles at the fiducial point,
with the same tile keys at both depths, so halos, CMB and noise seeds are
identical. Output: `/mnt/ceph/users/bbolliet/mapsim/counts/counts_modes.json`.

Confirmed detections with 5 < q < 40 and 0.01 < z < 1.5, per tile and for
40% of the sky (165 tiles):

| model | goal | baseline | goal / baseline |
|---|---|---|---|
| map level (one finder pass, no masking) | 237.2 ± 3.1 → 39,100 | 171.2 ± 2.5 → 28,300 | 1.39 |
| catalogue, analytic σ (CMB + noise), k = 0 | 341.7 → 56,400 | 221.8 → 36,600 | 1.54 |
| catalogue, σ measured by the finder, k = 0 | 232.8 → 38,400 | 163.3 → 26,900 | 1.43 |
| catalogue, σ measured by the finder, k = 1 | 245.5 → 40,500 | 172.6 → 28,500 | 1.42 |
| catalogue, the paper's σ table, k = 0 | 92.9 → 15,300 | | |

The catalogue model is Σ λ_cell ⟨P(5 < q_obs < 40)⟩, with:
- q_obs ~ N(√(q_true² + k), 1) and q_true = y0/σ(θ500);
- log-normal scatter in y0 and M ≥ 5e13;
- σ interpolated in ln θ between the 15 filter scales.

**The finder's own noise is high.** Its σ exceeds the analytic σ by 19–29%
at goal and 14–21% at baseline, peaking at 3.5′–5.7′. The unmasked tSZ of all
halos enters the covariance it estimates from the map. This accounts for
most of the gap between the map-level and analytic catalogue counts. It is
the bias that SZiFi's iterative masking addresses: the earlier project
measured σ_finder/σ_Y = 1.02–1.07 after masking.

**With the finder's own σ, the catalogue model reproduces the map-level
counts:**
- to 2% (goal) and 5% (baseline) with k = 0;
- to 3% and 1% with k = 1.

**The paper's noise table explains its lower count.** The table is
`data/so_sim_sz_mf_noise.npy` in cosmocnc_jax, fitted with a cubic in
log-log as there (fit error ≤ 1.8%; `benchmarks/mapsim/paper_sigma.py`):

| θ | paper σ / our goal σ | paper σ / our baseline σ |
|---|---|---|
| 0.5′ | 4.1× | 3.1× |
| 1′ | 2.5× | 1.9× |
| 2.7′ | 1.9× | 1.5× |
| 15′ | 1.1× | 0.98× |

With the paper's table, our λ cells give 15,500 clusters at q > 5
(0.01 < z < 3, M ≥ 5e13) in 40% of the sky, against the paper's ~16,000.
The halo model therefore agrees with the paper's; its noise model is what
differs. Our analytic σ gives 59,300 (goal) and 38,000 (baseline) with the
same cuts.

**How the paper's table was made.** The cosmocnc paper (arXiv:2403.09589)
describes it:
- SZiFi's standard matched filter, with no foreground deprojection;
- run on 10 Websky tiles holding the CMB, tSZ, kSZ and cosmic infrared
  background;
- SO baseline white noise, the same levels as ours, and no atmospheric
  noise.

Our maps have atmospheric noise but no foregrounds yet (PLAN.md, "simple
problem first"). The likely culprit is the infrared background, which is
brightest at 225 and 278 GHz and on small scales. That fits a ratio that
grows from 1× at 15′ to 3–4× at 0.5′, but the ratio has not been split into
its parts. SO's own forecast, 16,000 clusters at baseline depth
(arXiv:1808.07445), agrees with the paper's count.

For comparison, the earlier project's analytic counts (legacy population,
full sky, q > 5, no blending) give goal / baseline = 128,167 / 81,705 =
1.57. Here the same kind of calculation gives 1.54.

## 2026-11-18: optimisation bias of the finder

Script `benchmarks/mapsim/optbias.py`. Each depth uses 128 tiles of
CMB + noise with 100 isolated clusters each, 12,800 per depth:
- synthesised exactly, on sites 60′ apart, jittered by ±3′, with random
  sub-pixel offsets;
- θ500 log-uniform in 0.5′–6′;
- target q_true = y0/σ(θ500) uniform in 1–12, where σ(θ) is the analytic
  filter noise for the exact template at θ500.

Output: `/mnt/ceph/users/bbolliet/mapsim/optbias/`. Each quantity is
measured twice: with the true (analytic) covariance and with the covariance
estimated from the tile in one pass.

| quantity | goal, true C | baseline, true C | goal, tile C | baseline, tile C |
|---|---|---|---|---|
| ⟨q − q_noiseless⟩ at the true pixel and nearest scale | −0.012 ± 0.009 | +0.014 ± 0.009 | −0.044 ± 0.009 | −0.021 ± 0.009 |
| scatter of the same | 1.000 | 1.002 | 0.976 | 0.974 |
| noiseless peak / q_true (q_true ≥ 4) | 0.981 | 0.983 | 0.958 | 0.957 |
| k in q_max ≈ √(q_noiseless,max² + k) + N(0,1), q_true ≥ 4 | 2.78 | 3.09 | 2.21 | 2.47 |
| k in q_max ≈ √(q_true² + k) + N(0,1), q_true ≥ 4 | 0.91 | 1.43 | −1.73 | −1.57 |
| σ_tile / σ_true | | | 1.015–1.026 | 1.018–1.029 |

Detection completeness P(q_det > 5) against q_true, true covariance, goal:

| q_true | measured | Φ(√(q²+k) − 5), k = 0 / 2 / 3 |
|---|---|---|
| 4–5 | 0.365 | 0.316 / 0.393 / 0.432 |
| 5–6 | 0.749 | 0.685 / 0.744 / 0.770 |
| 6–7 | 0.945 | 0.925 / 0.944 / 0.952 |

**The search over position and size has the expected effect.** Measured
against the finder's own noiseless peak, the maximum over position and the
15 sizes behaves as √(q² + k) with k = 2.8 (goal) and 3.1 (baseline). That
is three fitted parameters: two for position and one for size.

**The grid has a cost.** The 0.5′ pixels and 15 sizes lose 1.7–1.9% of q
against the continuous template at the true position, which catalogue
models use.

**Net effect against the continuous q_true:**
- k ≈ 1–1.4: about +0.2 in q at q ≈ 5, nothing at q ≈ 8, and −2% of q for
  bright clusters;
- completeness at q > 5 close to the k = 2 curve, as in the earlier project
  ("between N(q, 1) and N(√(q² + 3), 1); the search adds about 0.2 to q").

**Estimating the covariance from a tile that holds the clusters biases q
low.** Here the 100 injected clusters raise σ by 2–3% and shift q by −0.02
to −0.04. In full simulated tiles the unmasked tSZ raises σ by 14–29%
(previous section). This is what the masking iteration has to remove.

**False peaks on these tiles:** 43–48 kept peaks with q > 4 per 100 deg²
(true covariance) and 29–34 (tile covariance) with no injection within
1.5′. The earlier project found 0.80–0.85 per 100 deg² at q > 5.

## 2026-11-18: halo mass function against cosmocnc_jax and colossus

Script `benchmarks/mapsim/hmf_compare.py`; tables in
`/mnt/ceph/users/bbolliet/mapsim/hmf/report.txt`. classy_szlite is the
reference, with the production conventions: Tinker08 at 500c, linear in Δ,
and no neutrinos in ρ_m or Ω_m(z). Each code uses its own P(k):
- cosmocnc_jax: CAMB backend, one 0.06 eV neutrino, P_tot normalised to σ8;
- colossus 1.4: EH98, no massive neutrinos.

Halos per 10° × 10° tile, M ≥ 5e13, 0.01 < z < 3:

| point (σ8, ω_c, h) | classy_szlite | cosmocnc_jax | colossus | classy_szlite defaults |
|---|---|---|---|---|
| fiducial (0.811, 0.121, 0.674) | 4,398 | 1.0002 | 0.981 | 1.007 |
| low corner (0.7, 0.0804, 0.704) | 695 | 1.0005 | 0.995 | 1.011 |
| high corner (0.9, 0.204, 0.644) | 16,652 | 1.0002 | 0.973 | 1.000 |

The paper's five redshift bins agree as well: cosmocnc_jax to within 0.1%,
colossus 0.966–1.003.

**dn/dlnM, fiducial point:**
- cosmocnc_jax agrees to within 0.2% for M ≤ 1e14. Above that it differs by
  0.3–1.5% at 1e15 and 1–4.5% at 3e15, where there are under one halo per
  tile.
- Those differences multiply out of:
  - the emulator's P(k) against CLASS (within 0.2% for 0.01 < k < 2 /Mpc);
  - the neutrino set-up;
  - CAMB against CLASS.
- Conventions and numerics add about 0.1%.
- colossus differs by −2% to +9% at z ≤ 1 (the +9% at 3e15) and up to
  −10% at 1e14 and z = 3. All of it is its EH98 σ(M).

**Other checks:**
- **Tinker f(σ)** at identical σ, z and integer Δ: the three codes agree to
  7e-16.
- **σ(M) and its slope from a shared P(k)**, against direct quadrature:
  - classy_szlite is within 2.7e-4. The residual is a ripple from
    finite-differencing a piecewise-linear σ(M); using the FFTLog dσ²/dR
    would remove it.
  - cosmocnc_jax is within 7e-4 to 1.1e-3.
  - Neither meets the 1e-4 code-test tolerance in the plan. The physics
    agreement is at the percent level, as expected.
- **ω_c = 0.204,** outside the emulators' box (0.0801–0.2000): the emulator
  stays as close to CLASS there as at the fiducial point (≤ 0.1% in
  dn/dlnM up to 3e14, ≤ 1% at 1e15). The extrapolation is harmless.
- **classy_szlite's defaults against production:**
  - interpolating in log10 Δ gives −0.3% to −2.3% at 1e13;
  - including neutrinos in ρ_m and Ω_m(z) gives +0.6% to +1.9%;
  - net, up to +1.8% in counts.

## 2026-11-18: priors: halo load at the edges of narrower boxes

Script `benchmarks/mapsim/prior_corners.py`; output
`/mnt/ceph/users/bbolliet/mapsim/priors/prior_corners.json`. These numbers use
the mass function interpolated in log10 Δ, the production setting from today
(next section).

The proposed boxes put each edge a fixed number of the paper's posterior
widths from the fiducial value. The widths are the likelihood standard
deviations in Table 1 of arXiv:2504.10230: 0.0051 in σ8, 0.0060 in ω_c and
0.028 in A_SZ.
- **wide: 8 widths, adopted for all map-level runs.** Posteriors at 10% of
  the sky are about 1.3× the paper's, with a quarter of our clusters.
- **narrow: 6 widths, an option for the 40%-sky runs.** That is 3 for where
  the posterior can land from mock to mock, 2 for the 95% half-width, and 1
  of margin.
- **Edges and other parameters.** ω_c is clipped at the paper's lower edge.
  α_SZ and the Gaussian priors are unchanged.

h is at ±3σ of N(0.674, 0.01). Halos are per 10° tile.

| box | σ8 | ω_c | A_SZ | halos ≥ 1e13, relative to the fiducial 73,030 | average over the prior | halos ≥ 5e13 | ln(10¹⁰A_s) |
|---|---|---|---|---|---|---|---|
| paper | 0.700–0.900 | 0.0804–0.204 | −4.41 to −4.21 | 0.26–2.68 | 1.17 | 683–16,477 | 2.12–3.81 |
| wide | 0.770–0.852 | 0.0804–0.169 | −4.53 to −4.08 | 0.42–1.94 | 1.05 | 1,386–10,516 | 2.51–3.70 |
| narrow | 0.780–0.842 | 0.085–0.157 | −4.48 to −4.13 | 0.48–1.73 | 1.01 | 1,649–9,069 | 2.62–3.60 |

The averages come from 256 random draws from each prior, with the Gaussian
priors truncated at ±4σ.

Four goal-depth tiles were run at each box's heaviest corner (high σ8, high
ω_c, low h), with the most peaks: A_SZ at the top of the box, α_SZ = 1 and
σ_lnq = 0.323 (+3σ).

| box | peaks with q > 4 (buffer 2,048) | detections in the data vector (fiducial 237) | buffer overflows | CPU s per tile |
|---|---|---|---|---|
| paper | 1,625–1,701 | 807–878 | confirmation buffer (16,384 halos ≥ 5e13) | 7.5 |
| wide | 1,454–1,542 | 797–833 | none | 5.9 |
| narrow | 1,210–1,335 | 634–736 | none | 5.2 |

- **ω_c stays inside the emulators' training range** (0.0801–0.2000) in
  both new boxes, so the extrapolation question disappears.
- **ln(10¹⁰A_s) still leaves the emulators' range** (2.5–3.5) at the top.
  The linear rescaling handles that exactly, as before.
- **The heaviest load falls** from 2.7× the fiducial to 1.9× (wide). The
  average falls from 1.17× to 1.05×.

## 2026-11-18: decisions

- **Priors:** the wide box above for all map-level runs, recorded in PLAN.md.
  - Narrowing to the 6-width box for the 40%-sky runs waits for the first
    map-level posteriors.
  - Catalogue-level analyses of map-level mocks keep the paper's priors, with
    A_SZ widened to −4.53 to −4.08.
- **Mass function:** Tinker08 is now interpolated in log10 Δ
  (`sim/halos.corner_density`).
  - Against linear in Δ, it lowers the fiducial halo count by 0.53%
    (M ≥ 1e13) and 0.63% (M ≥ 5e13).
  - cosmocnc_jax's linear default now gives more halos above 5e13: 0.65% at
    the fiducial point, 1.25% at the paper's low corner and 0.73% at its high
    corner (`hmf_compare.py` data). In dn/dlnM the difference reaches 2.3% at
    1e13. That difference is accepted.
  - Numbers earlier in this log used linear in Δ.
  - The simulator tests pass (37 passed).
- **Stacked lensing:** at SO goal depth, to match the tSZ maps. σ_κ(θ500)
  will come from the goal lensing noise curve with our own filter, checked
  against cosmocnc_jax's baseline table.

## 2026-11-18: the masking pass in the finder

What it does (`sim/mmf.py`: `cluster_mask`, `masked_cov`; `sim/simulator.find_clusters`):

- After the first pass, every peak with q > 4 is masked with a disc of radius
  3θ_j, where θ_j is the filter size at which the peak was found. The discs
  are built as FFTs of per-size peak maps times integer-disc kernels, which
  reproduces "distance ≤ r" exactly. Peaks are masked before merging and
  confirmation.
- The covariance is re-estimated outside the discs with the earlier project's
  whitened estimator. Each mode is whitened with the first-pass covariance of
  its bin. The whitened maps are masked, and their binned spectra are divided
  by the kept fraction and un-whitened.
- The filter then runs again on the unmasked maps. There is one iteration by
  default (`TileConfig.n_iter = 1`, as in SZiFi). Merging and confirmation use
  the final pass.
- The tile output now also carries the first-pass σ_j (`sigma0`), the masked
  fraction and the number of discs. The peak-overflow flag covers all passes.

Checks (`tests/test_sim_mask.py`; 54 tests pass over all simulator test files):

- **Discs:** the FFT discs equal a direct construction, for 300 random peaks
  of all sizes, some at the tile edges.
- **Nothing masked:** the estimator returns the input covariance to 9e-7 (in
  units of that covariance; float32 FFTs).
- **No clusters:** on CMB + noise tiles with ~60 random discs (18–19% masked),
  σ_j from the masked estimate is 0.992–0.995 (goal) and 0.993–0.996
  (baseline) of the analytic σ_j. The unmasked estimate gives 0.995–0.996.
  The earlier project measured 0.990–0.995.
- **Against the earlier project's own code** (szsim, float64): on one of our
  fiducial goal tiles with 487 first-pass peaks, the mask is the same pixel
  for pixel (28.0% masked). The masked covariance agrees to 1.3e-6 in
  whitened units, and σ_j to 3e-8. The reference is
  `$MAPSIM_REF/szsim_mask.npz`, from `benchmarks/mapsim/refgen_szsim_mask.py`.

16 fiducial tiles per depth (`benchmarks/mapsim/mask_check.py tiles`). The
noise columns give σ_finder/σ_analytic over the 15 sizes, 0.5′ to 15′:

| | first pass | after one iteration | after two | masked fraction, one iteration | data vector per tile: first pass, one, two |
|---|---|---|---|---|---|
| goal | 1.184–1.276 | 1.037–1.076 | 1.028–1.064 | 24.1% (19.5–30.0%) | 240.6, 323.2, 329.4 |
| baseline | 1.137–1.201 | 1.026–1.056 | 1.018–1.047 | 21.2% (18.0–26.5%) | 173.2, 222.5, 227.2 |
| earlier project, goal (324 tiles) | 1.160–1.242 | 1.035–1.069 | | 23.9% (15.3–34.7%) | |
| earlier project, baseline | 1.118–1.176 | 1.024–1.051 | | 21.2% (13.4–31.4%) | |

- After one iteration our noise is within 0.01 of the earlier project's at
  every size. That holds even though our first pass is 0.02–0.03 higher (more
  tSZ from our halo model). The masked fractions match.
- **Masking raises the data vector** by 34% at goal and 28% at baseline over
  the first pass. Goal over baseline is now 1.45.
- **One iteration is not converged.**
  - A second iteration lowers σ_j by a further 0.8–1.5% and raises the data
    vector by 1.9% (goal) and 2.1% (baseline).
  - It masks 37% (goal) and 33% (baseline) of the tile, because more peaks
    pass q > 4.
  - We keep one iteration, as SZiFi and the paper's pipeline do. The
    simulator applies the same finder that will be applied to the data, so
    the stopping point is part of the forward model, not a bias.
- **CPU time of the finder on one tile:** 0.15 s for one pass and 0.38 s
  with the masking iteration (×2.5). `spike.py` now times a "finder" stage
  for the GPU measurement.

Counts and the scatter of q, with masking:

- **Counts against the catalogue model** (`counts_modes.py`, 32 tiles,
  masking on by default):
  - 326.1 ± 3.4 per tile at goal and 221.7 ± 2.8 at baseline, or 53,800 and
    36,600 clusters in 40% of the sky. Goal over baseline is 1.47.
  - Against the catalogue model with the finder's own σ, simulation over
    model is 1.029 (goal) and 1.050 (baseline) for k = 1, and 0.974 and 0.989
    for k = 2. Before masking, the k = 1 values were 0.966 and 0.992.
- **Scatter of q away from clusters** (`mask_check.py scatter`, 8 tiles,
  1.4826 × MAD over the pixels outside the mask):
  - First pass: 0.84–0.90 (goal) and 0.88–0.92 (baseline). The first pass's
    noise estimate holds the bright clusters' power, so q is
    under-normalised.
  - After masking: 0.989–0.997 and 0.994–1.000, the unit scatter that the
    catalogue model assumes.
  - The shift in simulation over model goes the way this predicts. The
    remaining +3–5% at k = 1 is not yet explained. Candidates are blending,
    fragments of large clusters and chance alignments with massive halos.
- **Progress page:** updated with a masking figure, the masked discs on the
  patch and the new checks (artifact version 2; data from
  `diagnostics_data.py`).

The peak buffer with masking:

- **Masking raises the final-pass peak count.** At the adopted box's heaviest
  corner, the first pass gives 1,454–1,542 peaks with q > 4. After masking
  the final pass gives 2,745–2,859, which overflowed the 2,048 buffer in all
  three boxes (`prior_corners.py`).
- **Fix 1: the final pass keeps only q > 5** (`TileConfig.q_final`, the data
  vector's lower edge). This is exact.
  - A peak is kept unless a neighbour within 1.5′ has a higher q, so peaks
    with q ≤ 5 never remove a q > 5 peak. Confirmation works peak by peak,
    and the histogram counts q > 5 only.
  - Checked by a unit test on a dense synthetic catalogue, and on 4 fiducial
    tiles, where the data vectors are bit-identical. The final-pass peaks drop
    from ~640 to ~360 per tile.
  - Passes that build a mask still keep every q > 4 peak.
  - `n_peaks` in the tile output now counts the final pass's q > 5 peaks.
- **Fix 2: the buffer is raised to 3,072.** Heaviest corners with masking:
  - adopted box: 1,819–1,890 final-pass peaks, no overflow, the same data
    vectors as with 2,048;
  - 6-width box: 1,463–1,545;
  - the paper's box: 2,039–2,142, with its confirmation buffer (16,384 halos
    ≥ 5e13) still overflowing, as before.
- `priors/prior_corners.json` now holds the masked runs. The first-pass-only
  version is kept as `prior_corners_first_pass_only.json`.
- 55 tests pass.

## 2026-11-18: NaMaster, GMaster and szifi_jax

- **NaMaster is not used** anywhere in mapsim or in the earlier finder. It is
  pinned only in the separate environment that runs SZiFi itself.
- **Coupling corrections are not needed for the periodic-tile finder.** A
  Monte Carlo of our whitened masked estimator ran on 80 CMB-plus-noise tiles
  × 4 random-disc masks (16–20% masked, SO goal), with our own scratch code
  from the standard flat-sky formulas.
  - The mask lowers σ_j by 0.09% (0.5′) to 0.21% (15′), relative to the
    unmasked estimate of the same tiles.
  - Dividing the whitened pseudo-spectra by each bin's in-band coupling row
    sum instead of the kept fraction (3 more FFTs per tile) removes only ~0.02
    points of that. That part is the leakage at the band edges: 2–9% in the
    first and last bins, ~0.1% elsewhere.
  - A full per-tile coupling inversion has the same mean as the row sums, but
    it adds scatter and makes the σ deficit larger. It can also break the
    per-bin Cholesky, because the inverse coupling has negative off-diagonal
    terms. And it would make the finder ~7–12× slower.
  - The rest is finite-sample scatter from whitening with a noisy per-bin
    estimate. This is the same effect that puts the unmasked estimate 0.4–0.5%
    below the analytic one. A zero-cost fix to test: scale each bin's inverse
    covariance by (n − p − 1)/n, with n modes per bin and p = 6.
  - All of these are at least 20× smaller than the 3.7–7.6% σ excess that
    unmasked cluster signal leaves after one masking iteration. They also act
    identically on data and simulations.
- **GMaster** (a GPU re-implementation of NaMaster in JAX + CUDA; no licence):
  - Its strength is fast float32 HEALPix transforms and curved-sky coupling,
    12–22× faster than NaMaster on 96 cores.
  - Its flat-sky path is an interface port that cannot run at 1200² as
    written.
  - Nothing to take now. For the full-sky stage we would write our own:
    - fixed-window coupling per tile geometry, still with a per-tile
      normalisation for the cluster discs, which multiply the fixed window;
    - curved-sky quadrature, to validate simulated spectra.
- **szifi_jax** (a near-literal JAX port of SZiFi; no licence):
  - It inherits SZiFi's known problems: the masking iteration never runs a
    second pass, positions are shifted by half a pixel, the inpaint-and-blend
    noise estimate is biased, and out-of-band round-off is amplified. Some
    merge modes are broken, and it still needs NaMaster.
  - Our finder already has its speed-ups and is roughly 20–75× faster per
    tile.
  - Ideas to write ourselves:
    - the CIB-deprojected filter, for the foreground stage (two more per-bin
      dot products in the weights; quantify its signal-to-noise cost first);
    - greedy highest-q-first merging, as a candidate fix for merge chains
      (time it on the GPU first);
    - a minimum disc radius that follows the beams (SZiFi's
      min_radius_mask_arcmin).
- **Progress page:** the masked discs are now drawn as shaded vector shapes
  with a toggle; markers sit on pixel centres (artifact version 5).

## 2026-11-18: GPU tile 0 — halos now keyed by cell

- **Symptom:** in the 400-tile GPU masking run, tiles 1–15 matched the CPU
  run exactly, but tile 0 had 7 more first-pass peaks and 368 detections
  against 348.
- **Cause, found by a stage-by-stage comparison** (`diag_tile.py`; GPU with
  default and deterministic ops):
  - The GPU drew one more halo (73,830 against 73,829).
  - In cell (z ≈ 0.6, M ≈ 1.3e13), with λ = 22.5, the CPU drew 17 and the GPU
    18. For λ ≥ 10, JAX's Poisson sampler uses rejection with log-gamma
    functions, whose last bits differ between CPU and GPU, so an accept
    decision occasionally flips.
  - Halo attributes were drawn per buffer slot. So that one extra count moved
    every later slot to a different cell, and the rest of the tile became a
    different realisation.
  - Statistically harmless, but not reproducible across hardware, and it
    stopped seeds being reused smoothly across θ.
- **Fix** (`sim/halos.sample_halos`): each halo's random numbers (position,
  offsets, z, ln M, scatter) come from a key folded with its cell and its
  index within the cell. A changed count now adds or removes halos in that
  cell only.
  - New test `test_halos_keyed_by_cell`: doubling one cell's λ (39 → 72
    halos) leaves all 72,399 other halos identical.
  - The Poisson and bilinear-sampler tests pass, and so do all 56 tests.
  - 32 goal tiles: 324.9 ± 3.3 detections per tile (326.1 ± 3.4 before) and
    73,056 halos (λ = 73,030).
  - Realisations differ from earlier runs, which stay statistically valid.
- **Confirmed on the GPU** (job 7496298, snapshot 6073cc5, default and
  deterministic ops):
  - Tile 0 still draws one more halo on the GPU: M = 1.2e13, z = 0.62, the
    same cell as before. Nothing else differs.
  - The maps differ in 4 pixels around it, σ_j by ≤ 2e-6, the first-pass
    peaks are equal (503) and the data vector is equal (361).
  - Tile 1 agrees to round-off.

## 2026-11-18: stacked CMB lensing (the 21st number)

- **Model** (`sim/lensing.py`; the paper's catalogue-level model):
  - For each data-vector detection, from its matched halo:
    p̄ = κ(r_s; 0.92 M500, z)/σ_κ(θ500(0.92 M500)), with log-normal
    scatter 0.22, then unit Gaussian noise.
  - The tile returns Σ p and the count, and the summary is the mean.
  - `summaries.confirm` now returns the matched halo's slot.
  - `Theta` gains `beta_lens` = 0.92 and `sigma_lnp` = 0.22.
  - κ(r_s) = 4π(G/c²) Σ(r_s) D_l (1 − χ_l/χ*), for an NFW halo with
    c500 = 3 truncated in 3D at 5 R500 (Zubeldia & Challinor 2019, Eqs.
    18–20). Σ(r_s) = 0.66259 ρ0 r_s, and χ* = 13,836 Mpc at the fiducial
    point.
- **σ_κ(θ500)** (`benchmarks/mapsim/lens_table.py` → `sim/lens_tables.py`):
  - Our own matched filter: that template normalised at the scale radius,
    the SO v3.1.1 minimum-variance curve, reconstruction noise only,
    2 ≤ ℓ ≤ 3000.
  - On the baseline curve it reproduces the shape of cosmocnc_jax's table to
    1.2% over 0.74′–5.87′.
  - But the table is larger by a constant factor 1.450 that no variant
    explains. Tried: truncation in projection, other estimator columns,
    adding the lensing power spectrum to the noise (much worse), other
    multipole ranges.
  - Both tables carry that constant, to stay consistent with the paper's
    mocks. At goal depth σ_κ is 13–14% lower than at baseline.
  - The constant is still to be explained, by asking the paper's authors.
- **Checks:**
  - `tests/test_sim_lensing.py`: the NFW constants against quadrature;
    κ(r_s) against an independent route via Σ_crit, to 2e-5; the table
    against the reference to 1.5%; the moments of p; the stacked selection
    equal to the histogram count. 62 tests pass.
  - 32 fiducial tiles (`benchmarks/mapsim/lens_check.py`): stacked p = 0.236
    ± 0.011 at goal and 0.235 ± 0.014 at baseline. The catalogue model with
    the finder's own noise gives 0.241 and 0.236.
  - Over 40% of the sky the error on the mean would be ±0.004–0.005 (2%).
- **Correction, same day:** the 1.450 factor is explained. The reference
  table was made with the tSZ-deprojected lensing noise curve (deproj1,
  v3.1.1), over the file's full range 2 ≤ ℓ ≤ 5000.
  - With that curve, our filter with the documented normalisation (κ at the
    scale radius) reproduces cosmocnc_jax's baseline table to 0.3% in level
    and 0.3% in shape, with no free factor.
  - The paper's text cites the deproj0 file name. deproj1 is the sensible
    choice for clusters, because their own tSZ biases a standard
    reconstruction. It is 2.1–2.5× noisier in power at ℓ = 500–2000.
  - Production now uses deproj1, SENS1 for baseline and SENS2 for goal; the
    fudge factor is gone. At goal, σ_κ is 18% below baseline.
  - 32 tiles: stacked p = 0.250 ± 0.011 at goal (catalogue model 0.256) and
    0.236 ± 0.014 at baseline (0.237).
  - The lensing tests pass, with the table matching the reference to 1%.
  - Ruled out along the way: deproj2 (CIB deprojected), v3.1.0, the TT and
    polarisation-only columns, and the lensing power spectrum in the noise.

## 2026-11-18: progress page, version 6

- Added: the data vector marked complete; the measured GPU time (9.4 ms per
  tile with masking); a "Since the last update" section; five new checks
  (GPU against CPU, 400-tile masking, lensing table, stacked lensing, MSS2
  depth); a lensing-filter figure; an MSS2 depth figure.
- Next steps updated: the MSS2-calibrated noise generator, the real
  footprint and mask, SZiFi on identical maps, inference.
- Data: `diagnostics_data.py` now also reads `lensing/lens_check.json`, the
  lensing tables and the MSS2 check (`mss2/mss2_noise_check.json`).

## 2026-11-19: MSS2 noise through the finder

- **Data at Flatiron.** The MSS2 noise maps (4 splits × 6 bands), the six coadd
  inverse-variance maps, 60 mnms simulations (sims 0–4 of each split and tube)
  and Kaper's 70% Galactic mask: 91 files, 142.6 GB, in
  `/mnt/ceph/users/bbolliet/mapsim/mss2/raw`. Copied from NERSC through the
  data-transfer node at 2–3 MB/s; every file's size matches its FITS header,
  and rsync checked every file.
- **Tiles** (`benchmarks/mapsim/mss2_tiles.py`, 12 minutes on 16 cores):
  - 32 tiles of 10° at dec −40° to +10°, fully observed in all bands and ≥ 90%
    inside the mask, stored with a 2° margin, for the finder test; 64 tiles
    for the statistics of the maps.
  - The 2′ cut-outs now come from the 0.5′ rows with integer arithmetic.
    Rounding the tile's mean declination, which sits on a 2′ pixel edge, had
    put them one 2′ row south in four of the five declination bands.
- **The maps' non-Gaussianity** (90 and 150 GHz, each map against a Gaussian
  copy of itself, after removing scales above ~30′; mean over 64 tiles):
  - mnms matches the MSS2 band power to 0.3–0.8% at ℓ = 300–5000.
  - With the depth changes included, MSS2's excess kurtosis at 150 GHz and
    ℓ = 1000–3000 is +0.54 ± 0.07; mnms reproduces +0.27 ± 0.03. At 90 GHz:
    +0.16 against +0.11.
  - After dividing by the local noise level: +0.055 ± 0.006 at 150 GHz and
    ℓ = 1000–3000, with twice the Gaussian rate of 4σ excursions; mnms has
    none (−0.005 ± 0.001). Smaller at 90 GHz (+0.013 ± 0.003), none at
    ℓ = 3000–5000. No skewness anywhere.
  - The splits differ: at 150 GHz split 3 has 0.26–0.72× the band power of
    split 0, and mnms copies this.
- **Streaks.** The maps with the most false peaks have one narrow streak of
  2–3.7× the median noise, in one split only, at the same place at 90 and
  150 GHz (one tube), along the scan direction, and absent from the coadd
  depth map: a single noisy observation that the map-maker did not
  down-weight. In the three worst maps, 83–97% of the false peaks above
  q = 5 sit where the noise is above 1.3× its median.
- **Finder test** (`benchmarks/mapsim/mss2_finder_test.py`; 32 tiles, about
  25 minutes per run on 16 cores):
  - Five kinds of noise, at one-split depth and at four-split depth (the plain
    mean of the splits: a proxy for the coadd, about 20% noisier than the MSS2
    coadd at 150 GHz because the splits differ):
    - MSS2 itself;
    - mnms, upsampled from 2′ after apodising its margin. Above ℓ = 4000–4800
      on the sky it crosses over to a Gaussian copy of the matching MSS2 map's
      own noise, because mnms stops at ℓ = 5400 and the finder uses 90–280 GHz
      up to ℓ ≈ 8000. Those multipoles are therefore an oracle, not a model;
    - Gaussian noise with MSS2's 2D spectra (phases randomised on the cut-out
      with its margin, then cropped);
    - the same with MSS2's depth pattern (coadd inverse variance, 8′
      smoothing);
    - isotropic Gaussian noise with MSS2's azimuthally averaged spectra and
      band correlations: the simulator's kind of noise.
  - Every map gets its tile's lensed-CMB realisation and the finder's 30′
    taper. The finder runs as in production (masking pass, merging within
    1.5′). 64 clusters per tile (θ500 = 0.5′–6′, q = 2–10) are injected at
    four-split depth.
  - Noise peaks per 64 deg² above q = 4 / 4.5 / 5, on the simulator's q scale
    (next item), with the masking at the equivalent threshold:
    - one split: MSS2 37.2 / 7.7 / 1.27. As fractions of MSS2: mnms 0.93 /
      0.82 / 0.66; Gaussian with the depth pattern 0.84 / 0.68 / 0.48;
      Gaussian with the 2D spectra 0.75 / 0.57 / 0.34; isotropic Gaussian 0.74
      / 0.56 / 0.35 (errors 0.02–0.08).
    - four splits: MSS2 41.5 / 8.1 / 1.59. Fractions: mnms 0.88 / 0.72 /
      0.53; depth pattern 0.84 / 0.66 / 0.48; 2D spectra 0.72 / 0.54 / 0.34;
      isotropic 0.72 / 0.53 / 0.27 (errors 0.04–0.2).
    - The excess is largest at dec +5°, where the depth changes fast across
      the tile, and at −15°. By cluster size, mnms matches MSS2's false peaks
      for θ500 ≥ 1′ (0.96–1.02 at one-split depth) and gives 0.8 below, where
      its high multipoles are the stitched copy.
  - The finder's noise level σ_j:
    - Before the masking pass, every model is within 0.2% of MSS2.
    - After it, MSS2's σ_j is 0.2–0.7% lower than the Gaussian models' (0.4–0.9%
      when the pass masks twice as much), because the pass removes its noisiest
      patches. mnms is within 0.1–0.6%.
    - The same injected clusters get q 0.2% higher in MSS2 noise (0.4% with
      the heavier masking).
  - Completeness of the injected clusters (detected with q > 5, in bins of
    noiseless q) agrees between all kinds to within 0.03. The noise part of q
    at the clusters has an rms of 1.08–1.11 for every kind.
- **The q scale of tapered cut-outs** (the same isotropic Gaussian noise as a
  tapered cut-out and as a periodic tile, same CMB, 32 tiles):
  - σ_j is 7.6% lower on the cut-out for θ500 ≤ 2′, 7.3% at 5.7′ and
    5.2–5.5% at 15′. The finder estimates the noise from the tapered map, whose
    mean square is 0.879 of the interior's.
  - Interior q has an rms of 1.08 on the cut-out, against 1.00–1.01 on the
    periodic tile.
  - Noise peaks above q = 4 / 4.5 / 5 are 2.6 / 3.5–3.7 / 5.2–5.7× more
    frequent, and the masked fraction rises from 0.09–0.10 to 0.16.
  - The same injected clusters get q 7.0–7.2% higher.
  - All the noise-model comparisons share the cut-out geometry. Their counts
    are converted with the σ_j ratio per scale and a threshold calibrated on
    the isotropic noise.
  - For the real data, the finder's noise estimate must be corrected for the
    taper, or the simulator must make the same tapered cut-outs.
- **Review.** Five independent reviews, each checked by a sceptic, covered
  the tile geometry, the statistics, how the noise kinds are built, how the
  finder is used, and the test's design. They changed the test:
  - The Gaussian copies were first made from the untapered tile, which spread
    the edge jumps' power over the whole tile (σ 1.4–2× too high). They are
    now made from the cut-out with its margin, tapered there, then cropped.
  - Their random phases come from a real white field, so the rfft symmetry
    holds.
  - The mnms stitch is now in true ℓ: on CAR pixels mnms's limit along RA is
    5400 cos(dec), 4420 at dec −35°, which had left a hole. It now goes to
    the split's own high-ℓ noise: at 150 GHz MSS2 is not white at
    ℓ = 5000–8000, and white noise at the ℓ = 8000–14000 level overshot by
    1.3–1.7×. It is applied before cropping, which removed 10% extra power at
    the join.
  - Other changes: peaks merged as in production, CMB in every map, the
    four-split kinds, errors paired by tile, the isotropic kind and the
    geometry run.
  - The maps' statistics now use a local high-pass and a Gaussian copy of
    each map.
- **What it means for the noise generator.**
  - For the clusters, Gaussian noise with MSS2's spectra is nearly enough:
    the finder's noise level comes out within 0.2–0.9% and the completeness
    exactly.
  - The residual is the masking pass's response to MSS2's streaks: cluster q
    0.2–0.4% higher. That is probably 0.5–1% in counts above a threshold,
    near the Poisson error over 40% of the sky. It needs either streaks in
    the generator or a calibrated correction.
  - False peaks are 3× the Gaussian rate but few: about 1.3–1.6 per 64 deg²
    above q = 5. They enter the data vector only by chance alignment with a
    massive halo.
  - mnms reproduces half to two thirds of the excess false peaks and roughly
    halves the σ_j offset, because it is fitted to MSS2 itself. It stops at
    ℓ = 5400.
- **Next:**
  - Build our own GPU generator: Gaussian noise with MSS2's 2D spectra per
    declination band, the band correlations and the depth pattern. Validate
    it with this test.
  - Measure what the residual does to the data vector: the same simulated
    sky with MSS2 noise and with the generator's noise.
  - Decide on streaks from that number: a scattering-transform model, or a
    streak injector calibrated on MSS2.
  - Add the taper correction for real data.
- **Progress page, version 7:** the MSS2 test (false peaks per noise model,
  the streak picture), five new checks, an updated "Since the last update",
  and the generator and taper correction in "Next".

## 2026-11-20: MSS2 noise and the data vector

- **Test** (`benchmarks/mapsim/mss2_datavector_test.py`; 32 MSS2 tiles × 16
  sky realisations; about 3.5 hours on 16 cores):
  - The simulator's fiducial sky (halos above 10¹³ M☉, tSZ painted by the
    production painter, lensed CMB) goes into every noise map of a tile.
  - The production pipeline gives the data vector: the finder with masking,
    merging, confirmation, the 4 × 5 counts and the stacked-lensing sample.
    The lensing part is the noiseless mean p̄, so it measures selection only.
  - The same sky goes into every noise map of a tile, so the comparison is
    paired.
  - Noise kinds:
    - MSS2;
    - mnms;
    - Gaussian with each tile's own 2D spectra and depth pattern;
    - isotropic Gaussian with each tile's own spectra;
    - a practical generator (next item);
    - a periodic control (two items down).
  - The first four kinds copy each tile's own spectra and level, so they
    test non-Gaussianity with perfect two-point statistics.
  - The generator is isotropic Gaussian noise with the depth-normalised
    spectra averaged over the other tiles of the same declination band,
    times the tile's own absolute depth map. It runs at four-split depth
    only, because the tiles store only the coadd inverse variance.
  - The periodic control is isotropic noise on a periodic tile with no
    taper, run with production's masking threshold and no rescaling.
  - Depths: one split, about 1.35× the simulator's baseline noise in the
    finder (somewhat deeper than a year-1 rescaling of MSS2); the plain mean
    of the four splits, about goal depth (MSS2's own coadd is deeper).
  - On the cut-outs, masking runs at q = 4.3 and each peak's q is divided by
    the geometry run's ratio.
- **Results**, detections relative to MSS2 (bootstrap errors over tiles):
  - One split (MSS2: 77.0 detections per 64 deg²): mnms −0.8% ± 0.2%;
    Gaussian with the 2D spectra and depth pattern −1.0% ± 0.2%; isotropic
    Gaussian −1.2% ± 0.3%.
    - Most of the deficit is in the lowest q bin (5 to 8.4: isotropic −1.5%,
      1.8 times the 40%-sky Poisson error).
    - It is largest at dec +5° (isotropic −3.1%) and −5° (−1.6%), and absent
      at −35° and −25°.
    - 20-bin Δχ² over 40% of the sky, after removing its noise part: about
      3 (isotropic), 2 (depth pattern), 1 (mnms).
  - Four splits (MSS2: 188.4 per 64 deg²): mnms −0.3% ± 0.2%, depth pattern
    −0.3% ± 0.2%, isotropic −0.1% ± 0.2%. Δχ² is consistent with noise.
  - The generator at four splits: −0.9% ± 0.65%. Its error is dominated by
    tile-to-tile differences (0.64% of its 0.67%): it gets each tile's noise
    level only approximately. More tiles would sharpen it (107 candidate
    tiles exist; 32 are stored).
  - Stacked lensing: the selected sample's mean p̄ moves by at most 0.5%,
    against a 2% measurement error.
  - Mechanism at one split: MSS2's noise gets more of its patches masked
    (0.153 of the tile against 0.146 for isotropic noise), so its σ_j after
    masking is 0.27% lower. That gives q 0.27% higher and about 1% more
    detections. At four splits the masked fractions agree (0.178).
- **The q scale of the cut-outs with a cluster sky:** the periodic control
  gives 0.8% (one split) and 0.55% (four splits) more detections than the
  same isotropic noise on the rescaled cut-out. The conversion from tapered
  cut-outs, calibrated on noise alone, is therefore 0.5–0.8% low in counts
  with clusters in the map. This is common to all kinds, so the ratios hold,
  but the taper correction for real data must be calibrated with a cluster
  sky.
- **Review.** Three independent reviews, each checked by a sceptic,
  checked the test before the results were read. Their points:
  - Precision is about ±0.2% per depth. The three kinds' ratios to MSS2 at
    four splits are 75–85% correlated (one MSS2 map per tile), so their
    agreement counts as one measurement.
  - The per-tile Gaussian kinds are oracles, hence the generator kind.
  - The cut-out q conversion needed a check with a cluster sky, hence the
    periodic control.
  - The depth labels were corrected (one split is not quite year 1; four
    splits are about goal).
  - The explanation of the 4.3 masking threshold was wrong for noise alone,
    and is corrected in the scripts.
- **What it means:**
  - At goal-like depth, Gaussian noise with MSS2's spectra and depth pattern
    reproduces MSS2's data vector to 0.1–0.3%, inside the 40%-sky Poisson
    error (0.45%).
  - At about year-1 depth, MSS2's streaks add about 1% to the detections,
    mostly near threshold. That is about 1.5 Poisson errors over 40% of the
    sky. The shift is nearly uniform in q, so most of it would be absorbed
    by the mass calibration, but it biases that calibration.
  - mnms closes only a fifth of this gap.
  - For a year-1 analysis the generator needs either streaks or a calibrated
    shift of about 0.3% in q. The shift can be measured this way, with its
    error.
- **Next:**
  - Ask how the colleagues' year-1 maps were made (in progress).
  - Run the generator on more tiles to pin it below 0.3%.
  - Calibrate the year-1 shift, or model the streaks.
  - Add the taper correction for real data, calibrated with a cluster sky.
- **Progress page, version 8:** a figure of detections per q bin relative to
  MSS2 for each noise model, at both depths, against the 40%-sky Poisson
  error; four new checks (goal depth, year-1 depth, the generator, the taper
  conversion with clusters); an updated "Since the last update" and "Next";
  the masking-threshold caption corrected. `diagnostics_data.py` reads
  `datavector_test/summary.json`.

## 2026-11-20: per-tile noise spectra in true sky multipoles

- **Measurement** (`benchmarks/mapsim/mss2_spectra.py`, 7 minutes on 16
  cores): binned 6 × 6 auto and cross spectra for all 107 candidate tiles,
  for the four-split mean and for each split. They are what the simulator's
  noise generator will draw from: `mss2/spectra/spectra.npz` (47 MB), with
  white levels and positions in `spectra.json`.
  - **Multipoles:** true sky multipoles, hypot(k_dec, k_RA / cos δ). The
    maps' RA pixels are cos δ narrower than their dec pixels; earlier tests
    treated them as square.
  - **Units:** per-mode power on the simulator's square 0.5′ pixels, which
    is the measured power times cos δ. Each row is weighted by
    sqrt(cos δ_row / cos δ_centre).
  - **Shape and level:** the shape comes from the cut-out with its 120′
    margin. Each channel's level comes from the tile itself under the
    finder's 30′ taper, over ℓ = 6000–20000 (3000–4500 at 27/39 GHz).
  - **27 and 39 GHz:** shifted 1.5 fine rows, because a 2′ row k is centred
    on 0.5′ row 4k + 1.5. Above ℓ = 4500 their autos are floored at their
    3000–4500 level and their crosses set to 0.
- **Check** (`--check`): synthetic fields built from the maps' physical
  pixel sizes at dec −35°, −15° and +5°, through the functions that write the
  spectra.
  - All 36 auto and cross entries come back within 0.01–0.05%, and the RA-
    and dec-dominated sectors within 0.05%.
  - White noise on rows whose pixel area follows their own cos δ: the white
    level comes back to 0.9997–1.0002 with the row weights, against
    1.0012–1.0032 without them.
  - A square-pixel control misses by 21% at dec −35°, as it must.
  - A planted square-pixel multipole mapping is caught (autos 0.988, RA
    sector 0.980).
- **Findings:**
  - **White levels** of the four-split mean (ℓ = 8000–12000, true sky units,
    median over tiles): f030 57.1, f040 37.7, f090 6.9, f150 7.9, f230 20.2,
    f290 53.5 μK-arcmin. At 90 and 150 GHz that is between SO's goal (5.8,
    6.3) and baseline (8.0, 10), and MSS2's own weighted coadd is deeper
    still.
  - The Perlmutter check's "coadd at baseline" reading (f150 10.6) took its
    level over ℓ = 12000–16000, which sits on a bump. The finder-based
    reading (four splits ≈ goal) stands.
  - **Bumps in MSS2's spectra:** at ℓ ≈ 15000, 1.8× the surrounding power at
    150 GHz, 1.5–1.6× at 230/280 GHz and 1.1× at 90 GHz. At 150 GHz there is
    also 1.2× at ℓ ≈ 8000 and a dip to 0.88–0.90 at ℓ = 5000–6000. They are
    probably from the scanning or the map-making, and the generator
    reproduces them.
  - **Tile against cut-out level:** 1.1–1.7% rms over tiles, in power
    (four-split mean).
  - **27/39 GHz offset in the earlier tests:** their maps sat 0.75′ off the
    90–280 GHz maps. That changes only their cross spectra (by up to ~2% at
    ℓ ≈ 1000), which does not touch the earlier conclusions.
- **Review.** Two independent reviews, each checked by a sceptic, ran
  before the measurement was kept. They found:
  - the margin mixed into each tile's level;
  - one centre cos δ biasing the power 0.1–0.35% high, cancelling the margin
    error on average, so both are fixed together;
  - a check that never ran the production binning, nor tested the multipole
    mapping;
  - the 27/39 GHz row offset;
  - output issues: a mislabelled depth field, and partial runs overwriting
    the full file.

  All are fixed. The first run is kept as `spectra_v1_unfixed.*` for
  reference.
- **Next:** port the draw to the GPU with per-tile inputs from
  `spectra.npz` (M6b).

## 2026-11-20: the noise generator (M6b, step 2)

- **Module:** `classy_szlite/sim/noise.py`.
  - Each tile's noise is stationary, isotropic and Gaussian, drawn from that
    tile's binned 6 × 6 spectra: X_c(k) = depth Σ_d F[s, b(k), c, d] W_d(k).
    W is the rfft of unit white fields, F F^T = C per multipole bin (an
    eigendecomposition with negative eigenvalues clipped), s is the tile's
    spectrum index, and depth is one amplitude per tile.
  - The draw uses one gather per entry, so there is no 6 × 6 matrix per mode.
    It works on any grid size whose bin map is built from that grid; the
    per-mode power assumes 0.5′ pixels.
  - `load_mss2` reads `mss2_spectra.py`'s spectra (a chosen set: the
    four-split mean or a split). `noise_input` and `spectrum_rows` check
    spectrum indices on the host, because under jit an out-of-range index
    would silently take the last spectrum.
- **Simulator:** `channel_maps`, `tile_maps` and `simulate_tile` take an
  optional `NoiseInput` (tables, spectrum index, depth), vmapped per tile.
  - The default is unchanged: the 2019 forecast noise.
  - The CMB and halos use the same keys either way.
  - With the MSS2 noise, `tile_static`'s mode still sets the lensing table
    (σ_κ), which the per-tile depth does not reach.
- **Tests:** `tests/test_sim_noise.py`, 12 tests passing on rusty (CPU), plus
  the 29 earlier simulator tests.
  - **Draw:** equals a float64 numpy reference from the same random
    numbers, to 1e-5 of the local rms in every multipole bin.
  - **Spectra:** drawn auto and cross spectra match the input with analytic
    errors: under 1.5% of bin entries beyond 3σ, and ratios per range of
    multipole and the mean ratio within 5σ.
    - The same test runs on the 2019 forecast noise of `channel_maps`, never
      tested this way before. Its bin-averaged truth needed the full-plane
      mode weights, because the atmospheric noise is steep within a bin.
  - **Other grids:** a 256² grid with the clamp beyond the last bin.
  - **Batching:** batched draws match single ones to round-off. XLA's CPU FFT
    is not bit-for-bit batch-invariant (differences of order 1e-5 relative on
    a few modes), so bitwise invariance is left to the GPU and the integer
    data vector.
  - **Depth scale:** exact.
  - **Host checks and loader:** both tested.
  - **Sky unchanged:** the tSZ and CMB terms are the same with either noise
    model, also with the CMB off.
  - **Full tiles (slow):** `tile_maps` gives the same halos with either noise
    model, and its maps differ only by the noise draw. `simulate_tile` under
    vmap with a different spectrum and depth per tile gives the same integer
    outputs as single-tile runs.
- **Review.** Two independent reviews, each checked by a sceptic, found no
  bug in the module (units, factors, bins, Hermitian symmetry checked
  independently). They found weak tests, all fixed:
  - bias bounds 26σ wide;
  - a sky test that could not see the tSZ;
  - a tolerance set by the largest low-ℓ modes;
  - no full-tile test;
  - a draw tied to the 1200² grid;
  - unchecked indices;
  - a GPU check in the plan that would have compared differently defined
    spectra.
- **Next:** the GPU step (M6b). It needs a per-tile noise option in
  `spike.py`, then one short RTX job.

## 2026-11-20: the noise generator on the GPU (M6b)

Job 7510857: one RTX PRO 6000 Blackwell, snapshot 09ec5fb
(`benchmarks/mapsim/noise_gpu.sh`), 19 minutes.

- **Noise tests on the GPU:** 12 of 12 pass, including the two full-tile tests
  (deterministic ops).
- **GPU against CPU**, 4 tiles, each with its own MSS2 spectrum and depths 1.0,
  1.3, 0.8 and 1.0:
  - data vectors, peaks, kept and confirmed clusters identical on 4 of 4 tiles;
  - halos identical on 3 of 4. Tile 0 has 73,830 against 73,829: the known
    last-bit difference of the Poisson sampler for λ ≥ 10, which changes only
    that cell's halos (`halos.sample_halos`);
  - σ within 8×10⁻⁷.
- **Batch invariance** with deterministic ops, batch sizes 1, 4 and 16 over 16
  tiles, each with its own spectrum and depth: every integer output identical;
  σ within 9×10⁻⁹.
- **Speed and memory** at batch 16 (whole tile):

  | | per tile | noise stage | peak memory |
  |---|---|---|---|
  | 2019 forecast noise | 9.34 ms | 0.62 ms | 9.18 GiB |
  | MSS2 noise | 9.17 ms | 0.50 ms | 9.20 GiB |

  - The MSS2 noise costs no extra time or memory. Compiling its noise stage
    takes 7 s instead of 0.8 s, once per process.
  - The whole tile takes 9.2–9.3 ms, against 6.8 ms in the M0 spike, because
    the finder now includes the masking pass (4.5 ms).
- **Peak buffer** at the deepest MSS2 tile (lowest 150 GHz white level), depth 1:
  - at most 494 peaks at the fiducial point, and 1,976 at the adopted box's
    heaviest corner, of a buffer of 3,072 (64%); no overflow;
  - a depth below 1 (deeper than the four-split mean) at that corner would
    need a new check.
