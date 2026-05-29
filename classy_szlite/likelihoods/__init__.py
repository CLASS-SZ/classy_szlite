"""Pure-JAX CMB bandpower likelihoods compatible with classy_szlite.

These wrappers reproduce four cobaya CMB likelihoods bit-for-bit (Δχ² < 0.1
at chain best-fit) but evaluate in pure JAX so they are jit/grad-friendly for
NumPyro NUTS samplers:

  * :func:`chi2_lowTT`     — Planck 2018 low-ℓ TT (Commander, spline)
  * :func:`chi2_sroll2`    — Planck 2018 low-ℓ EE (sroll2 lookup, interpolated)
  * :func:`chi2_plac`      — Planck plik_lite v22 with per-spectrum ℓ-cuts
  * :func:`chi2_mflike`    — ACT DR6 multifrequency (fixed foreground at the
                             best-fit point — fastest, cosmology-only HMC)
  * :func:`chi2_mflike_v2` — ACT DR6 with sampled foreground amplitudes via
                             :mod:`classy_szlite.likelihoods.foreground`

  * :func:`total_chi2`     — full pipeline (cosmology → Cls → all four χ²)

The cached data tables (bandpowers, covariance, window functions, best-fit
foreground template) live in :class:`~classy_szlite.likelihoods.data.LikelihoodData`
and are extracted from a working ``cobaya`` install via the script
``classy_szlite/likelihoods/extract_data.py``. Run that once on a machine
where ``cobaya``, ``act_dr6_mflike`` and the data packages are installed,
then point :envvar:`CLASSY_SZLITE_LIKELIHOOD_DATA` at the resulting
``.npz`` file.
"""
from .core import (
    chi2_lowTT,
    chi2_sroll2,
    chi2_plac,
    chi2_mflike,
    chi2_mflike_v2,
    total_chi2,
)
from .foreground import fg_totals_jax

__all__ = [
    "chi2_lowTT", "chi2_sroll2", "chi2_plac",
    "chi2_mflike", "chi2_mflike_v2",
    "fg_totals_jax", "total_chi2",
]
