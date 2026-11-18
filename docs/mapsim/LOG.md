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
colossus 0.97–1.003.

**dn/dlnM, fiducial point:**
- cosmocnc_jax agrees to within 0.2% for M ≤ 1e14. Above that it differs by
  0.3–1.5% at 1e15 and 1–4.5% at 3e15, where there are under one halo per
  tile.
- Those differences multiply out of:
  - the emulator's P(k) against CLASS (within 0.2% for 0.01 < k < 2 /Mpc);
  - the neutrino set-up;
  - CAMB against CLASS.
- Conventions and numerics add about 0.1%.
- colossus differs by −1.5% to +4% at z ≤ 1, up to −10% at 1e14 and z = 3.
  All of it is its EH98 σ(M).

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
