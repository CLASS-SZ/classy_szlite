"""The public pipeline (classy_szlite.sim: priors, setup, survey, calibration, npe, cli)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import priors, calibration as cal, npe, survey, catalogue as ct  # noqa: E402


def test_priors_reproduce_the_published_draws():
    """The first parameter draws of the published sets (numpy default_rng of the set seeds)."""
    th = priors.draws("heldout", 3, box="map", seeds=priors.SEEDS_MAP)
    rng = np.random.default_rng(20261125)
    assert np.allclose(th[0, 0], rng.uniform(0.770, 0.852, 3)[0])
    assert th.shape == (3, 10) and np.all(th[:, 7] > 0) and np.all(th[:, 9] > 0)
    assert np.array_equal(priors.draws("truth", 2), np.tile(priors.fiducial(), (2, 1)))
    lo = np.array([priors.MAP[k][0] for k in priors.NAMES[:4]])
    hi = np.array([priors.MAP[k][1] for k in priors.NAMES[:4]])
    big = priors.draws("train", 2000)
    assert np.all(big[:, :4] >= lo) and np.all(big[:, :4] <= hi)


def test_derived_parameters():
    om, s8 = priors.derived(priors.fiducial()[None])
    assert abs(om[0] - (0.121 + 0.0222 + 3 * 0.02 / 93.14) / 0.674 ** 2) < 1e-12
    assert abs(s8[0] - 0.811 * np.sqrt(om[0] / 0.3)) < 1e-12


def test_censored_fit_recovers_a_known_relation():
    rng = np.random.default_rng(0)
    q = np.exp(rng.uniform(np.log(2.0), np.log(40.0), 60000))
    qo = np.sqrt((0.95 * q) ** 2 + 1.5) + 1.1 * rng.standard_normal(q.size)
    qo = np.where((qo > 4) & (rng.random(q.size) > 0.02), qo, -np.inf)
    r = cal.censored_fit(q, qo)
    assert abs(r["s"] - 0.95) < 0.005 and abs(r["k"] - 1.5) < 0.3 and abs(r["w"] - 1.1) < 0.02 and abs(r["f"] - 0.02) < 0.004


def test_grid_points_are_the_published_ones():
    P = cal.grid_points(24)
    assert np.allclose(P[0], [0.811, 0.121, -4.3054, 1.1233])
    assert np.allclose(P[1], [0.8368, 0.1423, -4.4237, 1.066], atol=1e-4)
    for j, k in enumerate(cal.PARAMS):
        assert P[1:, j].min() >= priors.MAP[k][0] and P[1:, j].max() <= priors.MAP[k][1]


def test_ensemble_of_identical_members_is_the_member():
    rng = np.random.default_rng(1)
    s = dict(theta=rng.normal(size=(5, 10)), mu=rng.normal(size=(5, 10)), sd=rng.uniform(1, 2, (5, 10)),
             rank=rng.uniform(size=(5, 10)), mu_derived=rng.normal(size=(5, 2)), sd_derived=rng.uniform(1, 2, (5, 2)))
    e = npe.ensemble([s, s, s])
    for k in ("mu", "sd", "rank"):
        assert np.allclose(e[k], s[k])
    cov = npe.coverage(dict(rank=np.linspace(0.0005, 0.9995, 1000)[:, None] * np.ones((1, 10))))
    assert np.allclose(cov[:, 0], npe.LEVELS, atol=0.002)


def test_published_catalogue_model_simulates():
    """The published-style catalogue model of the 41-tile survey (public inputs only), a few catalogues
    at the fiducial point, against the expectation."""
    sz, le, cfg = survey.catalogue_model("published", n_tiles=41)
    assert abs(cfg.f_sky - 0.0994) < 2e-4
    cells = ct.survey_cells()
    th = priors.to_theta(priors.fiducial())
    ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, sz, le, cfg))(th)
    f = jax.jit(lambda k: ct.simulate(k, th, cells, sz, le, cfg))
    ns = np.array([int(f(jax.random.key(i)).n_sel) for i in range(8)], float)
    assert abs(ns.mean() - float(en)) / np.sqrt(float(en) / 8) < 4.0


@pytest.mark.slow
def test_cli_tile(tmp_path):
    """mapsim tile on the CPU (minutes the first time: paint tables and compilation)."""
    from classy_szlite.sim import cli
    cli.main(["tile", "--out", str(tmp_path)])
    d = np.load(tmp_path / "tile_m0_t0.npz")
    assert d["q_final"].shape == (1200, 1200) and np.sum(d["peaks_confirmed"] & (d["peaks_q"] > 5)) > 100


@pytest.mark.slow
def test_cli_small_survey(tmp_path):
    """The whole chain on a tiny survey: 2 tiles, 12 map-level and 12 catalogue simulations, compare."""
    from classy_szlite.sim import cli
    out = str(tmp_path)
    for s in ("truth", "heldout"):
        cli.main(["maps", "--out", out, "--set", s, "--tiles", "2", "--n", "6", "--chunk", "2"])
        cli.main(["catalogues", "--out", out, "--model", "published", "--set", s, "--tiles", "2", "--n", "6"])
    d = np.load(os.path.join(out, "maps_t2", "truth.npz"))
    assert d["done"].all() and not d["overflow"].any() and np.all(d["x"][:, :20].sum(1) > 50)
    c = np.load(os.path.join(out, "catalogues", "published_t2", "truth.npz"))
    assert c["done"].all()
    # the catalogue model selects more clusters than the finder finds (no confusion noise)
    assert c["x"][:, :20].sum(1).mean() > d["x"][:, :20].sum(1).mean()
