# Cluster counts from maps

`classy_szlite.sim` runs simulation-based inference of Simons Observatory (SO) cluster counts with a
forward model at the level of maps. Each simulation draws halos, adds their thermal SZ signal to the lensed
CMB and the instrument noise in six frequency channels, then runs the same matched-filter cluster finder
that would run on the data. Its output is the 21-number summary of
[Zubeldia et al. (2025)](https://arxiv.org/abs/2504.10230): counts in 4 bins of signal-to-noise $q$ and
5 of redshift $z$, and the mean CMB-lensing signal-to-noise $\bar p$ of the clusters with $q > 5$. Neural
posterior estimation trained on these simulations infers the cosmological and scaling-relation parameters.
To our knowledge, this is the first inference of cluster counts in which every training simulation runs the
cluster finder on simulated multi-frequency maps.

The package also simulates catalogue-level models in the form of the observation model of
Zubeldia et al. (2025), so that they can be compared with map-level inference on the same skies. Such models
are calibrated on the maps in practice; here we calibrate them on map-level simulations, at one point or
across the prior ([Calibration](calibration.md)). As a reference we also keep the uncalibrated catalogue
model: the same model with a noise table for the instrument and the CMB only, as used in forecasts.

## The result

Map-level inference works at SO scale. One survey simulation takes 0.42 s on one GPU at 10% of the sky and
1.6 s at 40.7%, so a training set of 10,000 surveys takes 70 min and 4.6 h. On 100 map-level mocks at known
parameters, the map-level posterior is within 0.11 posterior widths of the truth at 40.7% of the sky (0.12
at 10%). On 500 skies drawn from the prior, its 90% intervals contain the truth for 88–91% of them at 40.7%
(86–90% at 10%), and its 50% intervals for 46–51% at 40.7% (44–49% at 10%). At each size the map-level posterior
is an equal mixture of five networks trained with different seeds, and the result is robust to the seed.

A catalogue model calibrated on map-level simulations is the realistic comparison. The calibration has to
carry what the finder does on the maps: confusion noise (it raises the noise by 4–8%), the search lift, neighbours,
and an upward tail of its errors ([The finder](finder.md)). Calibrated at one point (the finder's noise,
lift, scatter and losses fitted at the true parameters), the model is within about 0.2 widths in
$\sigma_8$, $\omega_c$ and $A_\mathrm{SZ}$ there, but $\alpha_\mathrm{SZ}$ is 0.50 widths high (0.52 at
40.7%), and its intervals are too narrow or off-centre across the prior. Calibrated at 25 points of the prior, through the
finder's noise level, it has close to correct coverage in $\sigma_8$, $\omega_c$ and $A_\mathrm{SZ}$, but the slope
stays 0.6–0.7 widths high and its intervals too narrow. We attribute this to the upward, tSZ-dependent tail
of the finder's errors, which a per-cluster Gaussian calibration cannot carry.

For reference, the uncalibrated model is off on our skies by up to 5.3 widths at 10% of the sky and 8.3 at
40.7%. This shows how large the finder's effects are at goal-depth forecast noise, with a flat prior on the
slope ([Scope](results.md#scope)); it is not a test of calibrated analyses.

| Posterior (10% of the sky) | $\sigma_8$ | $\omega_c$ | $A_\mathrm{SZ}$ | $\alpha_\mathrm{SZ}$ | 90% coverage of $\sigma_8$ over the prior |
|---|---|---|---|---|---|
| catalogue, uncalibrated | +1.86 | −0.77 | −1.42 | −5.28 | not shown |
| catalogue, calibrated at one point | +0.13 | +0.09 | −0.06 | +0.50 | 65% |
| catalogue, calibrated through the noise level | −0.07 | +0.11 | +0.09 | +0.62 | 90% |
| map level (5 networks) | −0.05 | +0.10 | −0.05 | +0.08 | 90% |

Bias = mean over the mocks of (posterior mean − truth) / posterior standard deviation, errors 0.06–0.11.
All numbers on these pages use the public 2019 SO forecast noise. [Results](results.md) has the figures,
both survey sizes and the coverage tests.

A write-up of this work as a note in MNRAS format: [Cluster counts from maps (PDF)](https://github.com/CLASS-SZ/classy_szlite/blob/main/note/main.pdf),
with its LaTeX source in [`note/`](https://github.com/CLASS-SZ/classy_szlite/tree/main/note).

## How these pages are organised

| Page | What it covers |
|---|---|
| [Forward models](forward-models.md) | The survey, the prior, the map-level simulator step by step, and the catalogue model, with equations |
| [The finder](finder.md) | The multi-frequency matched filter, its masking pass, confirmation, and how its $q$ differs from the catalogue model's |
| [Calibration](calibration.md) | Calibrating the catalogue model on map-level simulations: at one point, across the prior, and why it falls short |
| [Inference](inference.md) | Neural posterior estimation and the checks (bias, coverage, TARP, ensembles) |
| [Results](results.md) | The map-level posterior, the calibrated models and the uncalibrated reference, at 10% and 40.7% of the sky: bias and coverage |
| [Pipeline](pipeline.md) | The `mapsim` command, its outputs, Slurm templates, and the time and memory each step takes |
| [Credits](credits.md) | What the code takes from SZiFi, cosmocnc, mcfit and earlier code, and the software it uses |
| Tutorials | [One tile](tutorials/one-tile.md), [catalogue models](tutorials/catalogues.md), [a posterior](tutorials/posterior.md): notebooks that run on a laptop |

## What it costs

| Step | Hardware | Time | Memory |
|---|---|---|---|
| One tile through every stage | laptop CPU (Apple M1 Pro) | 48 s the first time (28 s of it builds the paint tables, then cached), a few seconds after | 3.4 GB |
| One survey simulation, 41 tiles (10% of the sky) | 1 GPU (NVIDIA RTX PRO 6000 Blackwell) | 0.42 s | fits a 96 GB card |
| 10,000 training simulations, 41 tiles | 1 GPU | 70 min | |
| 10,000 training simulations, 168 tiles (40.7%) | 1 GPU | 4.6 h | |
| 10,000 catalogue-level simulations | 1 GPU | 1.3–2 min | |
| Training a posterior on 10,000 simulations | CPU, 2–16 cores | 2–8 min | < 2 GB |

The whole 10%-sky comparison (four posteriors, one to five networks each) costs about 2 GPU hours and 1 CPU hour;
the 40.7% one about 6 GPU hours and 4 CPU hours. [Pipeline](pipeline.md) breaks this down per step.

## Limitations

The simulator places halos at random positions (no spatial clustering), uses Gaussian noise with the
same power spectrum everywhere on a tile, has no foregrounds (CIB, radio sources), and confirms detections
by matching them to simulated halos with exact redshifts. These do not affect the comparison between the
inferences, which share the same skies, but they would matter for real data. [Results](results.md#limitations)
discusses each.
