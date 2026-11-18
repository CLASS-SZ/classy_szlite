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
