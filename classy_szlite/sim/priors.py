"""Parameters and priors of the cluster-count inference.

Ten parameters, in this order everywhere (data files, posterior samples, figures):

    sigma8, omega_cdm, A_SZ, alpha_SZ, h, omega_b, n_s, sigma_lnq, beta_lens, sigma_lnp

The first four have flat priors (the counts constrain them); the last six have the Gaussian priors of
Zubeldia et al. (2025, Table 1), with the two scatters truncated at zero (redrawn).

Two boxes for the flat priors:

* ``PAPER``: the paper's Table 1 (omega_cdm clipped at 0.200, the emulators' training edge).
* ``MAP``: the narrower box of the map-level simulations, chosen so that the heaviest corner stays
  affordable (about 130,000 halos per 10-degree tile).

The hydrostatic mass bias is fixed, 1 - b = 0.8. Omega_m and S8 = sigma8 sqrt(Omega_m / 0.3) are
derived, with one massive neutrino of 0.02 eV x 3 as in the emulators (omega_nu = 0.06 / 93.14).
"""
from __future__ import annotations

import numpy as np

from .simulator import THETA_FID

NAMES = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ", "h", "omega_b", "n_s", "sigma_lnq", "beta_lens", "sigma_lnp")
LABELS = ("σ8", "ω_c", "A_SZ", "α_SZ", "h", "ω_b", "n_s", "σ_SZ", "β_lens", "σ_lens")

PAPER = {"sigma8": (0.7, 0.9), "omega_cdm": (0.0804, 0.200), "A_SZ": (-4.41, -4.21), "alpha_SZ": (1.0, 1.24)}
MAP = {"sigma8": (0.770, 0.852), "omega_cdm": (0.0804, 0.169), "A_SZ": (-4.53, -4.08), "alpha_SZ": (1.0, 1.24)}
BOXES = {"paper": PAPER, "map": MAP}
GAUSS = {"h": (0.674, 0.01), "omega_b": (0.0222, 0.00015), "n_s": (0.96, 0.0042), "sigma_lnq": (0.173, 0.05),
         "beta_lens": (0.92, 0.01), "sigma_lnp": (0.22, 0.05)}
POSITIVE = ("sigma_lnq", "sigma_lnp")
OMEGA_NU_H2 = 3 * 0.02 / 93.14

# numpy seeds of the parameter draws and jax seeds of the simulations, per set (the values used for
# the published runs; a set's simulation i uses fold_in(key(seed), i))
SEEDS_MAP = {"train": 20261124, "heldout": 20261125, "truth": 20261126}
SEEDS_CATALOGUE = {"train": 20261121, "heldout": 20261122, "truth": 20261123}


def _box(box):
    return BOXES[box] if isinstance(box, str) else box


def fiducial():
    """The true parameters of the mocks (simulator.THETA_FID) as a vector in NAMES order."""
    return np.array([getattr(THETA_FID, k) for k in NAMES])


def draw_prior(n, rng, box="map"):
    """n parameter vectors (n, 10) from the prior with the flat box ``box`` ("map", "paper" or a dict)."""
    flat = _box(box)
    th = np.empty((n, len(NAMES)))
    for j, name in enumerate(NAMES):
        if name in flat:
            th[:, j] = rng.uniform(*flat[name], n)
        else:
            mu, sd = GAUSS[name]
            v = rng.normal(mu, sd, n)
            if name in POSITIVE:
                while np.any(v <= 0):
                    bad = v <= 0
                    v[bad] = rng.normal(mu, sd, bad.sum())
            th[:, j] = v
    return th


def draws(set_name, n=None, box="map", seeds=None):
    """The parameter vectors of a named set: 'truth' repeats the fiducial point; 'train' and
    'heldout' are prior draws with the set's seed. Default sizes 10,000, 500 and 100."""
    seeds = seeds or (SEEDS_MAP if box == "map" else SEEDS_CATALOGUE)
    n = n or {"train": 10000, "heldout": 500, "truth": 100}[set_name]
    if set_name == "truth":
        return np.tile(fiducial(), (n, 1))
    return draw_prior(n, np.random.default_rng(seeds[set_name]), box)


def to_theta(row):
    """A simulator.Theta from a vector in NAMES order (other fields at their fiducial values)."""
    return THETA_FID._replace(**dict(zip(NAMES, (float(v) for v in row))))


def derived(samples):
    """(Omega_m, S8) from samples (n, 10) in NAMES order."""
    s = np.asarray(samples)
    om = (s[:, 1] + s[:, 5] + OMEGA_NU_H2) / s[:, 4] ** 2
    return om, s[:, 0] * np.sqrt(om / 0.3)


def sbi_prior(box="map"):
    """The prior as a torch distribution for sbi (needs the optional ``sbi`` extra)."""
    import torch
    from sbi.utils import MultipleIndependent
    from torch.distributions import Normal, Uniform
    flat = _box(box)
    d = []
    for n in NAMES:
        if n in flat:
            lo, hi = flat[n]
            d.append(Uniform(torch.tensor([lo], dtype=torch.float32), torch.tensor([hi], dtype=torch.float32)))
        else:
            mu, sd = GAUSS[n]
            d.append(Normal(torch.tensor([mu], dtype=torch.float32), torch.tensor([sd], dtype=torch.float32)))
    return MultipleIndependent(d, validate_args=False)
