"""Write the three tutorial notebooks (examples/notebooks/*.ipynb). Execute and convert them for the docs with

    python examples/notebooks/make_notebooks.py
    for nb in one_tile catalogue_models posterior; do
      jupyter nbconvert --to notebook --execute --inplace examples/notebooks/$nb.ipynb --ExecutePreprocessor.timeout=3600
      jupyter nbconvert --to markdown examples/notebooks/$nb.ipynb --output-dir docs/clusters/tutorials
    done
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    n["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


ONE_TILE = [
    """md:# One tile, stage by stage

One 10° × 10° tile of the survey at the true parameters, through every stage of the map-level simulator
and the cluster finder, with the public 2019 SO forecast noise. Runs on a laptop CPU: about 1 min the first
time (the paint tables are built and cached, the code is compiled), a few seconds after; about 3.5 GB of
memory.""",
    """import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
from classy_szlite.sim import tile, priors

t0 = time.time()
o = tile.stages()          # the first tile of the first mock of the published 'truth' set
print(f"{time.time() - t0:.0f} s")""",
    """md:## The maps

`o` holds the Compton-y map, the 145 GHz map and its three parts, and the finder's signal-to-noise before
and after its masking pass, each 1200 × 1200 pixels of 0.5′.""",
    """down = lambda a: a.reshape(600, 2, 600, 2).mean((1, 3))
panels = [("y", "Compton-y × 10⁵", "magma", 0, 3), ("tsz145", "145 GHz: tSZ [μK]", "RdBu_r", -200, 200),
          ("cmb145", "145 GHz: CMB [μK]", "RdBu_r", -200, 200), ("noise145", "145 GHz: noise [μK]", "RdBu_r", -200, 200),
          ("m145", "145 GHz: sum [μK]", "RdBu_r", -200, 200), ("q_first", "S/N, first pass", "viridis", -3, 8),
          ("mask", "masking-pass discs", "gray", 0, 1), ("q_final", "S/N, after masking", "viridis", -3, 8)]
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for a, (k, t, cm, lo, hi) in zip(ax.ravel(), panels):
    m = o[k] * (1e5 if k == "y" else 1)
    a.imshow(down(m), cmap=cm, vmin=lo, vmax=hi, origin="lower", extent=[0, 10, 0, 10])
    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
plt.tight_layout()""",
    """md:## The catalogue

Halos above $5\\times10^{13}\\,M_\\odot$, and the finder's kept peaks with $q > 4$; a peak is confirmed if
a halo above that mass lies within 1.5′.""",
    """p, h = o["peaks"], o["halos"]
det = p["confirmed"] & (p["q"] > 5)
print(f"halos above 5e13 Msun: {len(h['z'])}")
print(f"kept peaks with q > 4: {len(p['q'])}; confirmed with q > 5: {det.sum()}; not confirmed with q > 5: {np.sum(~p['confirmed'] & (p['q'] > 5))}")
order = np.argsort(-p["q"])[:8]
print("brightest detections: q, size [arcmin], the halo's z, M500 [Msun] and catalogue-model q")
for i in order:
    j = p["halo"][i]
    print(f"  {p['q'][i]:6.1f}  {p['size'][i]:4.1f}'  " + (f"z {h['z'][j]:.2f}  M {h['M500'][j]:.1e}  q_model {h['q_model'][j]:.1f}" if j >= 0 else "no halo"))""",
    """md:## The finder's noise

The finder estimates its noise from the maps, which contain the tSZ of every cluster. Before masking, the
bright clusters raise it by 20–30%; after masking, the faint ones still add 4–8% over the instrument and the
CMB: the confusion noise that the catalogue model leaves out.""",
    """th = o["thetas"]
plt.figure(figsize=(5, 3.2))
plt.plot(th, o["sigma_first"] / o["sigma_model"], label="first pass")
plt.plot(th, o["sigma_final"] / o["sigma_model"], label="after masking")
plt.axhline(1, color="gray", ls=":")
plt.xscale("log"); plt.xlabel("filter size θ [arcmin]"); plt.ylabel("σ finder / σ instrument + CMB"); plt.legend()
print("after masking, at 2': ", round(float(o["sigma_final"][7] / o["sigma_model"][7]), 3))""",
    """md:## Detected q against the catalogue model's q

For every confirmed detection, the finder's $q$ against the catalogue model's $q$ of its halo. Above
$q \\approx 6$ the finder's is lower (its noise is higher); near the threshold the search over positions and
sizes lifts it.""",
    """qm = h["q_model"][p["halo"][p["confirmed"]]]
qo = p["q"][p["confirmed"]]
plt.figure(figsize=(4.2, 4))
plt.loglog(qm, qo, ".", ms=3, alpha=0.6)
plt.plot([3, 80], [3, 80], "k--", lw=1, label="catalogue model: q_obs = q")
plt.xlim(3, 80); plt.ylim(4, 80); plt.xlabel("catalogue model's q"); plt.ylabel("finder's q"); plt.legend()""",
]

CATALOGUES = [
    """md:# Catalogue models

The catalogue-level model of Zubeldia et al. (2025) for the 41-tile survey (10% of the sky), as published and
calibrated on map-level simulations. Runs on a laptop CPU in a few minutes.""",
    """import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
from classy_szlite.sim import survey, catalogue as ct, priors, summaries as sm

sz, le, cfg = survey.catalogue_model("published", n_tiles=41)
cells = ct.survey_cells()
th = priors.to_theta(priors.fiducial())
print(f"f_sky = {cfg.f_sky:.4f}")""",
    """md:## The noise table

The published model's $q$ is $\\bar y_0 / \\sigma_{y_0}(\\theta_{500})$, with $\\sigma_{y_0}$ the matched
filter's noise for the instrument and the CMB: here computed analytically for our filter and the 2019 SO
forecast noise. The paper's own table (for its filter and 40% of the sky) is shown for comparison.""",
    """import jax.numpy as jnp
paper_sz, _ = ct.paper_tables()
tt = np.geomspace(0.5, 15, 50)
plt.figure(figsize=(5, 3.2))
plt.loglog(tt, np.asarray(ct.noise_sigma(sz, jnp.asarray(tt))), label="our filter, forecast noise")
plt.loglog(tt, np.asarray(ct.noise_sigma(paper_sz, jnp.asarray(tt))), "--", label="the paper's table")
plt.xlabel("θ500 [arcmin]"); plt.ylabel("σ_y0"); plt.legend()""",
    """md:## Expected counts and simulations

`catalogue.expected` integrates the model; `catalogue.simulate` draws catalogues (the halos, their scatter and
noise). The two agree.""",
    """ec, en, ep = jax.jit(lambda t: ct.expected(t, cells, sz, le, cfg))(th)
sim = jax.jit(lambda k: ct.simulate(k, th, cells, sz, le, cfg))
t0 = time.time()
cats = [sim(jax.random.key(i)) for i in range(20)]
print(f"20 catalogues in {time.time() - t0:.1f} s (including compilation)")
C = np.array([np.asarray(c.counts) for c in cats])
print("expected counts (rows: q 5-8.4, 8.4-14, 14-24, 24-40; columns: z bins of 0.3):")
print(np.round(np.asarray(ec)).astype(int))
print(f"selected (q > 5): expected {float(en):.0f}, simulated {np.mean([int(c.n_sel) for c in cats]):.0f}; mean p {float(ep / en):.4f}")""",
    """md:## Calibrated on map-level simulations

A calibration replaces the instrument-plus-CMB noise by the finder's own noise and, per redshift bin, draws
$q_\\mathrm{obs} = \\sqrt{(s q)^2 + k} + w\\,\\mathcal N(0,1)$, with a loss rate $f$. Here the values measured on 20
map-level mocks at the true parameters (the table on the Calibration page), with the finder's noise 7.6% above
the analytic one at 2′ (and the measured ratio at each filter size).""",
    """# the finder's noise over the analytic one at the filter's 15 sizes (0.5' to 15'), mean over 41 tiles x 5 mocks
ratio = np.array([1.059, 1.062, 1.064, 1.067, 1.070, 1.073, 1.075, 1.076, 1.076, 1.074, 1.070, 1.064, 1.056, 1.047, 1.038])
sz_cal = sz._replace(lnsig=sz.lnsig + np.log(ratio)[None])
qcal = ((0.31, 0.991, 1.32, 1.035, 0.029), (0.61, 0.989, 1.51, 1.043, 0.013), (0.90, 0.988, 1.50, 1.079, 0.006),
        (1.20, 0.991, 1.29, 1.132, 0.003), (1.50, 0.990, 1.07, 1.177, 0.002), (99.0, 1.002, 0.07, 1.317, 0.0))
cfg_cal = cfg._replace(qcal=qcal)
ec_cal, en_cal, _ = jax.jit(lambda t: ct.expected(t, cells, sz_cal, le, cfg_cal))(th)
r = np.asarray(ec_cal) / np.asarray(ec)
plt.figure(figsize=(5.5, 3))
plt.imshow(r, cmap="RdBu", vmin=0.8, vmax=1.2, origin="lower")
for i in range(4):
    for j in range(5):
        plt.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.xlabel("redshift bin"); plt.ylabel("q bin"); plt.title("calibrated / published expected counts", fontsize=9); plt.colorbar()
print(f"selected: published {float(en):.0f}, calibrated {float(en_cal):.0f}")""",
]

POSTERIOR = [
    """md:# A posterior, and how to check it

Neural posterior estimation on a small set of catalogue-level simulations (the published model, 10% of the
sky), its posterior for one mock, and its coverage. Needs the `sbi` extra. Runs on a laptop in about 10 min;
the published runs use 10,000 simulations and five networks.""",
    """import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
from classy_szlite.sim import survey, catalogue as ct, priors, npe, simulator

sz, le, cfg = survey.catalogue_model("published", n_tiles=41)
cfg = cfg._replace(capacity=2 ** 20)
cells = ct.survey_cells()

def simulate(theta, seed):
    key = jax.random.key(seed)
    one = jax.jit(jax.vmap(lambda i, *v: ct.data_vector(ct.simulate(
        jax.random.fold_in(key, i), simulator.THETA_FID._replace(**dict(zip(priors.NAMES, v))), cells, sz, le, cfg))))
    out = []
    for a in range(0, len(theta), 16):
        b = theta[a:a + 16]
        pad = np.concatenate([b, np.repeat(b[-1:], 16 - len(b), 0)])
        x = one(jax.numpy.arange(a, a + 16, dtype=jax.numpy.uint32), *[jax.numpy.asarray(pad[:, j]) for j in range(10)])
        out.append(np.asarray(x)[:len(b)])
    return np.concatenate(out)

t0 = time.time()
th_train = priors.draw_prior(2000, np.random.default_rng(1))
x_train = simulate(th_train, 1)
print(f"2,000 training catalogues in {time.time() - t0:.0f} s")""",
    """md:## Training""",
    """post, info = npe.train(th_train, x_train, box="map", seed=11)
info""",
    """md:## One mock

A catalogue at the true parameters, and the posterior of the four parameters the counts constrain.""",
    """th_mock = priors.draws("truth", 1)
x_mock = simulate(th_mock, 2)
s = npe.sample(post, x_mock[0], 4000)
fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
for j, a in enumerate(ax):
    a.hist(s[:, j], 50, histtype="step", density=True)
    a.axvline(th_mock[0, j], color="k", ls=":")
    a.set_xlabel(priors.LABELS[j])
plt.tight_layout()""",
    """md:## Coverage

On 100 prior draws: the fraction whose true value lies in each central credible interval. A calibrated
posterior follows the diagonal (with 100 draws, expect deviations of about ±0.05).""",
    """th_test = priors.draw_prior(100, np.random.default_rng(3))
x_test = simulate(th_test, 3)
t0 = time.time()
summ = npe.summarise(post, th_test, x_test, n_samples=1000)
cov = npe.coverage(summ)
print(f"{time.time() - t0:.0f} s; bias over the prior draws:", np.round(npe.bias(summ)["bias"][:4], 2))
plt.figure(figsize=(4, 4))
plt.plot([0, 1], [0, 1], "k:")
for j in range(4):
    plt.plot(np.r_[0, npe.LEVELS, 1], np.r_[0, cov[:, j], 1], label=priors.LABELS[j])
plt.xlabel("credible level"); plt.ylabel("coverage"); plt.legend()""",
]

for name, cells in (("one_tile", ONE_TILE), ("catalogue_models", CATALOGUES), ("posterior", POSTERIOR)):
    nbf.write(nb(cells), os.path.join(HERE, f"{name}.ipynb"))
    print("wrote", name)
