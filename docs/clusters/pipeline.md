# Pipeline

## Install

```bash
pip install "classy_szlite[sbi]"        # sbi and torch, for training posteriors
classy-szlite-get-data                   # the CosmoPower emulator weights, about 115 MB (public)
```

For a GPU, install the CUDA build of jax (`pip install "jax[cuda12]"`). Everything else is computed
locally the first time and cached: the profile and paint tables (about 1 min on a laptop) in
`$CLASSY_SZLITE_CACHE` (default `~/.cache/classy_szlite`). All inputs are public: the CMB comes from the
package's emulator, the noise from the 2019 SO forecast, and the noise tables of Zubeldia et al. (2025) ship
with the package (from cosmocnc, MIT).

## The `mapsim` command

| Step | Command | Writes |
|---|---|---|
| one tile, every stage | `mapsim tile --out OUT` | `OUT/tile_m0_t0.npz` |
| map-level sets | `mapsim maps --out OUT --set truth\|heldout\|train [--tiles 41]` | `OUT/maps_t41/{set}.npz` |
| calibration at the true point | `mapsim calibrate --out OUT point` | `OUT/calibration/point.npz` |
| calibration across the prior | `mapsim calibrate --out OUT grid`, then `fit-noise` or `fit-theta` | `OUT/calibration/grid/`, `noise.npz`, `theta.npz` |
| catalogue-level sets | `mapsim catalogues --out OUT --model M --set S` | `OUT/catalogues/{M}_t41/{S}.npz` |
| train and apply a posterior | `mapsim train --out OUT --data maps\|M --seed 11` | `OUT/npe/{data}_t41/seed11*.npz/json` |
| compare every posterior | `mapsim compare --out OUT` | `OUT/compare_t41.json` |

Catalogue models `M`: `published` (the [uncalibrated model](forward-models.md#the-catalogue-model)), `calibrated-point`, `calibrated-noise`,
`calibrated-theta` (and `paper`, the noise tables of Zubeldia et al. (2025) for their filter and 40% of the sky). Sets: `truth` (100 simulations at
the true parameters, the mocks), `heldout` (500 prior draws, for coverage), `train` (10,000 prior draws).
`--tiles 168` gives the 40.7%-sky survey; its first 41 tiles are the 41-tile survey's skies.

Every step writes as it goes and resumes where it stopped. `mapsim train` applies the posterior to the
map-level mocks (`mocks`), the map-level prior draws (`prior`) and its own model's prior draws (`own`), and
prints the bias and 90% coverage of the four main parameters. `mapsim compare` combines the networks of
each model into an ensemble.

## On a cluster

`slurm/mapsim.sh` runs one step as a Slurm job, configured with environment variables (`MAPSIM_OUT`,
`MAPSIM_PYTHON`, and optionally `JAX_COMPILATION_CACHE_DIR` to reuse compiled programs across jobs):

```bash
export MAPSIM_OUT=/scratch/mapsim MAPSIM_PYTHON=$HOME/venvs/classy_szlite/bin/python
sbatch -p gpu --gpus=1 --cpus-per-gpu=16 --mem=96G -t 04:00:00 slurm/mapsim.sh maps --set train
sbatch -p cpu -c 16 --mem=32G -t 02:00:00 slurm/mapsim.sh train --data maps --seed 11
```

`slurm/paper1_10pct.sh` submits the whole 10%-sky test as a chain of jobs with dependencies.

## Time and resources

Measured on one NVIDIA RTX PRO 6000 Blackwell (96 GB) with 16 CPU cores, CPU jobs on AMD EPYC (rome)
nodes, and an Apple M1 Pro laptop (10 cores, 16 GB).

| Step | Hardware | 41 tiles (10%) | 168 tiles (40.7%) | Memory |
|---|---|---|---|---|
| one tile, every stage | laptop | 48 s first time (28 s paint tables), seconds after | | 3.4 GB |
| one survey simulation | GPU | 0.42 s (2.4 per second) | 1.6 s (0.61 per second) | under 96 GB |
| `maps --set truth` (100) | GPU | 2 min | 3 min | |
| `maps --set heldout` (500) | GPU | 6.5 min | 15 min | |
| `maps --set train` (10,000) | GPU | 70 min | 4.6 h | |
| `calibrate point` (20 mocks) | GPU + CPU | about 1 min + 8 min of fits | | |
| `calibrate grid` (25 points × 5 mocks) | GPU + CPU | about 7 min including the fits (15 min with `--size`) | | |
| `catalogues` (10,000) | GPU | 1.3–1.5 min | 2 min | |
| one catalogue-level simulation | CPU, 16 cores | | 1.1 s (at 40%) | |
| `train` (10,000 simulations) | CPU | 2–8 min + 1–15 min applying | the same | < 2 GB |
| the end-to-end test (2 tiles, 12 + 12 simulations) | laptop | 4 min 40 s | | |

Compilation adds 1–3 min to the first call of a map-level step (reused across jobs with
`JAX_COMPILATION_CACHE_DIR`). The map-level steps need a GPU in practice: one 41-tile survey simulation on a
laptop CPU takes a few minutes.

## Research scripts

`benchmarks/mapsim/` keeps the research scripts behind the runs on these pages, including one-off checks;
the pipeline above reproduces their results.
