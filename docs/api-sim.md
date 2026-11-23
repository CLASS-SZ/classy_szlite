# Simulator API

`classy_szlite.sim` simulates the cluster counts of an SO-like survey from maps and from catalogues,
and fits the posterior of the parameters from them. A simulated survey is a set of 10-degree tiles,
each reduced to the same data vector of 21 numbers: the counts $N_{ij}$ in four bins of the
signal-to-noise $q$ and five bins of redshift $z$, and the mean CMB-lensing signal $\bar p$ of the
selected clusters,

$$
\mathbf{x} = \big(N_{11}, \dots, N_{45},\ \bar p\big),
\qquad q\ \text{edges} = 5 \times 8^{\,k/4}\ (k = 0, \dots, 4),
\qquad z\ \text{edges} = 0.01, 0.308, \dots, 1.5 .
$$

The pipeline is also a command-line tool, `mapsim <command>` (see [Command line](#command-line)).
For the method and results, see [Cluster counts from maps](clusters/index.md).

| Module | What it does | Typical cost |
| --- | --- | --- |
| `priors` | The ten parameters, their flat and Gaussian priors, and the derived $\Omega_m$ and $S_8$. | none |
| `setup` | Everything a tile needs before it can be simulated: paint tables, CMB spectrum, static arrays. | paint tables 1-3 min the first time (about 17 MB, then cached) |
| `tile` | One tile through every stage, keeping the intermediate maps for figures and checks. | 3-6 min on a laptop CPU the first time, then a few seconds; 1-2 GB of memory |
| `survey` | Sets of map-level and catalogue-level simulations of a survey of tiles, saved as they go. | per set, see the [pipeline](clusters/pipeline.md) |
| `catalogue` | Catalogue-level simulator: halos, observed $q$ and $p$, selection and data vector, without maps. | |
| `calibration` | Fits the catalogue model's observation model to the finder's behaviour on map-level simulations. | one GPU, 41 tiles: 20 mocks about 1 min; a 25-point grid about 7 min plus fits (about 1 min per point on CPU) |
| `npe` | Neural posterior estimation of the ten parameters from the 21 numbers, with bias and coverage checks. | 10,000 simulations train in 2-7 min on 2-16 CPU cores |
| `cli` | The `mapsim` command line: tile, maps, calibrate, catalogues, train, compare. | |
| `simulator` | One tile end to end: cosmology, halos, maps, matched filter, peaks, masking pass, histogram. | |
| `halos` | Poisson halos on one periodic tile, in fixed-capacity buffers. | up to about 130,000 halos per tile at the heaviest prior corner |
| `paint` | Band-limited painting of the Compton-$y$ map of the halos. | |
| `instrument` | SO LAT channels (temperature): frequencies, beams, tSZ spectrum, forecast noise, CMB spectrum. | |
| `mmf` | Multi-frequency matched filter: covariance from the maps, $q$ maps per scale, peaks, masking. | |
| `summaries` | Detections to data vector: merging, confirmation against halos, $(q, z)$ histogram. | |
| `lensing` | Stacked CMB-lensing observable $p$ of each detection, from its matched halo. | |
| `flatsky` | Flat periodic tile geometry and its Fourier grid. | |

## Parameters and priors

::: classy_szlite.sim.priors
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Tile set-up

::: classy_szlite.sim.setup
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## One tile, every stage

::: classy_szlite.sim.tile
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Survey sets

::: classy_szlite.sim.survey
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Catalogue-level simulator

::: classy_szlite.sim.catalogue
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Calibration on maps

::: classy_szlite.sim.calibration
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Posterior estimation

::: classy_szlite.sim.npe
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Command line

::: classy_szlite.sim.cli
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source
      group_by_category: false

## Building blocks

The modules below are the stages of one tile. Most users call them through `tile`, `survey` and
`catalogue`.

### One tile, end to end

::: classy_szlite.sim.simulator
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Halos

::: classy_szlite.sim.halos
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Painting the y map

::: classy_szlite.sim.paint
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Instrument

::: classy_szlite.sim.instrument
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Matched filter

::: classy_szlite.sim.mmf
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Summaries

::: classy_szlite.sim.summaries
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### CMB lensing

::: classy_szlite.sim.lensing
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false

### Tile geometry

::: classy_szlite.sim.flatsky
    options:
      show_root_heading: false
      heading_level: 4
      members_order: source
      group_by_category: false
