# Cluster counts from maps

`classy_szlite.sim` simulates Simons Observatory (SO) cluster counts at the level of maps: halos, their
thermal SZ signal, the lensed CMB and the instrument noise in six frequency channels, then the same
matched-filter cluster finder that would run on the data. Its output is the 21-number summary of
[Zubeldia et al. (2025)](https://arxiv.org/abs/2504.10230): counts in 4 bins of signal-to-noise $q$ and
5 of redshift $z$, and the mean CMB-lensing signal $\bar p$ of the clusters with $q > 5$. Neural posterior
estimation trained on these simulations infers the cosmological and scaling-relation parameters.

The same package also simulates the catalogue-level model that the paper used, so the two can be
compared on the same skies.

## The result

On map-level mocks at known parameters (100 per survey size), the posterior trained on the paper's
catalogue model is biased by up to 5.3 posterior widths at 10% of the sky and 8.3 at 40%. Calibrating
that model on map-level simulations removes most of the bias where it is calibrated, but not across the
prior, and the slope $\alpha_\mathrm{SZ}$ stays off by about 0.6 widths in every version. The posterior
trained on map-level simulations is unbiased (within 0.1 width) and calibrated over the whole prior.

| Posterior (10% of the sky) | $\sigma_8$ | $\omega_c$ | $A_\mathrm{SZ}$ | $\alpha_\mathrm{SZ}$ | 90% coverage of $\sigma_8$ over the prior |
|---|---|---|---|---|---|
| catalogue, as published | +1.86 | −0.77 | −1.42 | −5.28 | not calibrated |
| catalogue, calibrated at one point | +0.13 | +0.09 | −0.06 | +0.50 | 65% |
| catalogue, calibrated through the noise level | −0.07 | +0.11 | +0.09 | +0.62 | 90% |
| map level (5 networks) | −0.05 | +0.10 | −0.05 | +0.08 | 90% |

Bias = mean over the mocks of (posterior mean − truth) / posterior standard deviation, errors 0.06–0.11.
All numbers on these pages use the public 2019 SO forecast noise. [Results](results.md) has the figures,
both survey sizes and the coverage tests.

## How these pages are organised

| Page | What it covers |
|---|---|
| [Forward models](forward-models.md) | The survey, the prior, the map-level simulator step by step, and the catalogue model, with equations |
| [The finder](finder.md) | The multi-frequency matched filter, its masking pass, confirmation, and how its $q$ differs from the catalogue model's |
| [Calibration](calibration.md) | Calibrating the catalogue model on map-level simulations: at one point, across the prior, and why it falls short |
| [Inference](inference.md) | Neural posterior estimation and the checks (bias, coverage, TARP, ensembles) |
| [Results](results.md) | The bias test at 10% and 40% of the sky |
| [Pipeline](pipeline.md) | The `mapsim` command, its outputs, Slurm templates, and the time and memory each step takes |
| Tutorials | [One tile](tutorials/one-tile.md), [catalogue models](tutorials/catalogues.md), [a posterior](tutorials/posterior.md): notebooks that run on a laptop |

## What it costs

| Step | Hardware | Time | Memory |
|---|---|---|---|
| One tile through every stage | laptop CPU (Apple M1 Pro) | 48 s the first time (28 s of it builds the paint tables, then cached), a few seconds after | 3.4 GB |
| One survey simulation, 41 tiles (10% of the sky) | 1 GPU (NVIDIA RTX PRO 6000 Blackwell) | 0.42 s | fits a 96 GB card |
| 10,000 training simulations, 41 tiles | 1 GPU | 70 min | |
| 10,000 training simulations, 168 tiles (40.7%) | 1 GPU | 4.6 h | |
| 10,000 catalogue-level simulations | 1 GPU | 1.5–2 min | |
| Training a posterior on 10,000 simulations | CPU, 2–16 cores | 2–8 min | < 2 GB |

The whole 10%-sky comparison (four posteriors, 5 networks each) costs about 2 GPU hours and 1 CPU hour;
the 40% one about 6 GPU hours and 4 CPU hours. [Pipeline](pipeline.md) breaks this down per step.

## Limitations

The simulator places halos at random positions (no spatial clustering), uses Gaussian noise with the
same power spectrum everywhere on a tile, has no foregrounds (CIB, radio sources), and confirms detections
by matching them to simulated halos with exact redshifts. These do not affect the comparison between the
two inferences, which share the same skies, but they would matter for real data. [Results](results.md#limitations)
discusses each.
