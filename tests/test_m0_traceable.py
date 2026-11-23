"""M0: the cosmology stage is jit- and vmap-able, with explicit shapes.

- jit(vmap(θ → ln A_s → CosmoGrids → HaloGrids)) equals the eager result to
  ≤ 1e-12 (elementwise relative) on the domain the simulator uses, in a fresh
  process with the jitted calls first, at batch sizes 1 and 7; batch 1
  equals batch 7 to the same tolerance. (σ(R) on R = 0.5-60 Mpc, dn/dlnM
  where ≥ 1e-12 of its maximum: see ``_m0_jit_first.domain_mask``.)
- Emulators are pytrees, usable as jit arguments; ``predict`` is always 2-D.
- σ(R) from the pure-JAX FFTLog against two independent references: mcfit
  with its numpy backend (≤ 1e-10), and direct quadrature (≤ 1e-4 where the
  HMF uses it).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))


def _run_script(name, *args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([sys.executable, os.path.join(HERE, name), *args],
                       capture_output=True, text=True, env=e, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_jit_vmap_pipeline_equals_eager_fresh_process():
    res = _run_script("_m0_jit_first.py")
    for B, fields in res["jit_vs_eager"].items():
        bad = {k: d for k, d in fields.items() if not d <= 1e-12}
        assert not bad, f"batch {B}: jit vs eager above 1e-12: {bad}"
    bad = {k: d for k, d in res["batch1_vs_batch7"].items() if not d <= 1e-12}
    assert not bad, f"batch 1 vs batch 7 above 1e-12: {bad}"


# ---- emulator shapes and pytree -------------------------------------------

def _der_inputs(em, **over):
    from classy_szlite._registry import DEFAULT_COSMO
    full = dict(DEFAULT_COSMO, **over)
    return {k: full[k] for k in em.parameters}


def test_predict_is_always_2d():
    from classy_szlite._registry import get_emulator, DEFAULT_COSMO
    der = get_emulator("der")
    out = der.predict(_der_inputs(der))
    assert out.ndim == 2 and out.shape[0] == 1
    pkl = get_emulator("pkl")
    base = {k: DEFAULT_COSMO[k] for k in pkl.parameters if k != "z_pk_save_nonclass"}
    for n in (1, 3):
        o = pkl.predict(dict(base, z_pk_save_nonclass=np.linspace(0.1, 1.0, n)))
        assert o.shape == (n, len(pkl.modes))


def test_build_with_a_single_redshift():
    from classy_szlite.cosmology import build, K_GRID
    cg = build({}, z_grid=[0.5])
    assert cg.pk.shape == (1, len(K_GRID))
    assert cg.sigma.shape == (1, len(cg.R))
    assert cg.Hz.shape == cg.chi.shape == cg.Da.shape == (1,)


def test_emulator_is_a_pytree_and_a_jit_argument():
    from classy_szlite._registry import get_emulator
    der = get_emulator("der")
    leaves = jax.tree_util.tree_leaves(der)
    assert leaves and all(isinstance(x, jax.Array) for x in leaves)
    x = _der_inputs(der)
    f = jax.jit(lambda em, p: em.predict(p))
    np.testing.assert_allclose(np.asarray(f(der, x)), np.asarray(der.predict(x)),
                               rtol=1e-13, atol=0)


# ---- σ(R): pure-JAX FFTLog against two independent references --------------

def _fiducial_pk():
    from classy_szlite.cosmology import build
    cg = build({}, z_grid=[0.01, 1.0, 3.0])
    return cg


def test_sigma_R_matches_mcfit_numpy_backend():
    from mcfit import TophatVar
    cg = _fiducial_pk()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tv = TophatVar(np.asarray(cg.k), lowring=True, backend="numpy")
    R_ref, var_ref = tv(np.asarray(cg.pk), extrap=True)
    np.testing.assert_allclose(np.asarray(cg.R), R_ref, rtol=1e-13, atol=0)
    # the HMF uses R ≈ 0.5-40 Mpc (M500c ≈ 1e11-5e16 M_sun); check a wide margin
    sel = (R_ref > 0.05) & (R_ref < 400.0)
    np.testing.assert_allclose(np.asarray(cg.sigma)[:, sel] ** 2, var_ref[:, sel],
                               rtol=1e-10, atol=0)


def test_sigma_R_matches_direct_quadrature():
    from scipy.interpolate import CubicSpline
    from scipy.integrate import simpson
    cg = _fiducial_pk()
    k, pk = np.asarray(cg.k), np.asarray(cg.pk)
    lnk_f = np.linspace(np.log(k[0]), np.log(k[-1]), 2 ** 17 + 1)
    kf = np.exp(lnk_f)
    R_test = np.geomspace(3.0, 25.0, 9)                # M500c ~ 1e13-5e15 M_sun
    for iz in range(pk.shape[0]):
        pkf = np.exp(CubicSpline(np.log(k), np.log(pk[iz]))(lnk_f))
        x = kf[None, :] * R_test[:, None]
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        var_q = simpson(kf ** 3 * pkf * W ** 2, x=lnk_f, axis=-1) / (2.0 * np.pi ** 2)
        var_f = np.interp(np.log(R_test), np.log(np.asarray(cg.R)),
                          np.log(np.asarray(cg.sigma)[iz] ** 2))
        np.testing.assert_allclose(np.exp(var_f), var_q, rtol=1e-4)


@pytest.mark.gpu
def test_cpu_gpu_parity(tmp_path):
    """The cosmology stage on CPU and GPU agrees to ≤ 1e-10 on the used domain."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_m0", os.path.join(HERE, "_m0_jit_first.py"))
    m0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m0)
    out = {}
    for plat in ("cpu", "cuda"):
        path = str(tmp_path / f"{plat}.npz")
        res = _run_script("_m0_jit_first.py", "--save", path, env={"JAX_PLATFORMS": plat})
        assert res["backend"] == ("gpu" if plat == "cuda" else "cpu")
        out[plat] = dict(np.load(path))
    worst = {}
    for i in range(out["cpu"]["thetas"].shape[0]):
        a = {k: v[i] for k, v in out["cuda"].items() if k != "thetas"}
        b = {k: v[i] for k, v in out["cpu"].items() if k != "thetas"}
        used, _ = m0.compare(a, b)
        for k, d in used.items():
            worst[k] = max(worst.get(k, 0.0), d)
    print("CPU vs GPU, used domain:", {k: f"{v:.1e}" for k, v in worst.items()})
    bad = {k: d for k, d in worst.items() if not d <= 1e-10}
    assert not bad, f"CPU vs GPU above 1e-10: {bad}"
