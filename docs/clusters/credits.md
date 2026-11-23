# Credits

What the cluster-count code (`classy_szlite.sim`) takes from other codes, checked file by file against
them (November 2026).

## Codes

| Code | What we take | How |
|---|---|---|
| [SZiFi](https://github.com/inigozubeldia/szifi) (Zubeldia et al. 2023a,b) | the finder's conventions: peak rule, masking threshold and radius, covariance bins, filter sizes, profile cut | rewritten in JAX; no SZiFi code. SZiFi was run, from its own source, only to check our filter ([differences](finder.md#how-it-differs-from-szifi)) |
| [cosmocnc](https://github.com/inigozubeldia/cosmocnc) (Zubeldia & Bolliet 2024) | the form of the catalogue model, the SO-like survey's fiducial values; two noise tables | the tables are copied unchanged under its MIT License (`sim/data/NOTICE`) and used only by the `paper` model and checks; no code ([differences](forward-models.md#how-it-differs-from-cosmocnc)) |
| [cosmocnc_jax](https://github.com/inigozubeldia/cosmocnc_jax) | code tests of our observables and mass function | run from a separate copy; one helper in `benchmarks/mapsim/hmf_compare.py` follows its `_camb_tophat_sigma` (credited there) |
| [mcfit](https://github.com/eelregit/mcfit) (Yin Li) | the FFTLog (Hamilton 2000) behind $\sigma(R)$ | `classy_szlite/_fftlog.py` is adapted from it and is distributed under the GNU GPL v3, as mcfit is |
| szsim ([sz-poisson-blending](https://github.com/borisbolliet/sz-poisson-blending)) | the instrument model, the halo painter, the whitened covariance estimate of the masking pass | adapted from this earlier code by the same author |
| class_sz notebooks (Bolliet et al. 2023) | the Tinker et al. (2008) mass function | follows their function |

## Software and data

JAX, NumPy, SciPy, Matplotlib; sbi 0.27 with PyTorch and nflows (posteriors); CAMB and colossus (the CMB
spectrum of the runs on these pages); the ede-v3 CosmoPower emulators (cosmology); the 2019 SO forecast noise
and SO's lensing-noise curves v3.1.1. The runs used the Flatiron Institute's rusty cluster.

## Licences

classy_szlite is MIT, except `classy_szlite/_fftlog.py` (GNU GPL v3, from mcfit); the two cosmocnc tables are
MIT. See [LICENSE](https://github.com/CLASS-SZ/classy_szlite/blob/main/LICENSE).
