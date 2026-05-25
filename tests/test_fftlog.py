"""Test classy_szlite._fftlog: pure-JAX path vs mcfit, dispatch shim, dtype switch."""
from __future__ import annotations
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp


# Forcing the pure-JAX path via env var — must be done before _fftlog import.
def _force_pure_jax_module():
    """Reload classy_szlite._fftlog with the env var set, so TophatVar /
    SphericalBessel return our FFTLog instead of the mcfit shim."""
    import importlib
    os.environ["CLASSY_SZLITE_FORCE_PURE_FFTLOG"] = "1"
    import classy_szlite._fftlog as fft
    importlib.reload(fft)
    return fft


def _restore_default_module():
    """Drop the env override and reload, so other tests see the normal dispatch."""
    import importlib
    os.environ.pop("CLASSY_SZLITE_FORCE_PURE_FFTLOG", None)
    import classy_szlite._fftlog as fft
    importlib.reload(fft)


# ---- Pure-JAX path agrees with mcfit reference at f64 -------------------

def test_pure_jax_tophatvar_matches_mcfit():
    fft = _force_pure_jax_module()
    try:
        from mcfit import TophatVar as McfitTV
        k = np.geomspace(1e-4, 100.0, 512)
        pk = (k / 0.05) ** 0.96 * np.exp(-(k / 5.0) ** 2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="use backend='jax' if desired")
            mtv = McfitTV(k, lowring=True, backend="jax")
        R_ref, var_ref = mtv(pk, extrap=True)

        my = fft.TophatVar(k, dtype=np.float64)
        R_mine, var_mine = my(pk)

        np.testing.assert_allclose(np.asarray(R_mine), np.asarray(R_ref),
                                    rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.asarray(var_mine), np.asarray(var_ref),
                                    rtol=1e-10, atol=1e-18)
    finally:
        _restore_default_module()


def test_pure_jax_sphericalbessel_matches_mcfit():
    fft = _force_pure_jax_module()
    try:
        from mcfit import SphericalBessel as McfitSB
        u = np.geomspace(1e-5, 100.0, 256)
        f = u ** (-0.3092) * (1.0 + u ** 1.062) ** ((0.3092 - 5.48) / 1.062)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="use backend='jax' if desired")
            msb = McfitSB(u, nu=0, lowring=True, backend="jax")
        y_ref, g_ref = msb(f, extrap=True)

        my = fft.SphericalBessel(u, nu=0, dtype=np.float64)
        y_mine, g_mine = my(f)

        np.testing.assert_allclose(np.asarray(y_mine), np.asarray(y_ref),
                                    rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.asarray(g_mine), np.asarray(g_ref),
                                    rtol=1e-9, atol=1e-25)
    finally:
        _restore_default_module()


def test_pure_jax_tophatvar_batched():
    """Batched (n_z, n_k) input — what cosmology.py actually feeds it."""
    fft = _force_pure_jax_module()
    try:
        from mcfit import TophatVar as McfitTV
        k = np.geomspace(1e-4, 100.0, 512)
        pks = np.stack([(k / 0.05) ** 0.96 * np.exp(-(k / 5.0) ** 2)
                        * (1.0 + 0.1 * i)
                        for i in range(5)], axis=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="use backend='jax' if desired")
            mtv = McfitTV(k, lowring=True, backend="jax")
        R_ref, var_ref = mtv(pks, extrap=True)

        my = fft.TophatVar(k, dtype=np.float64)
        _, var_mine = my(pks)

        np.testing.assert_allclose(np.asarray(var_mine), np.asarray(var_ref),
                                    rtol=1e-10, atol=1e-18)
    finally:
        _restore_default_module()


# ---- Dispatch / shim behaviour ------------------------------------------

def test_default_dispatch_uses_mcfit_on_cpu():
    """On CPU, the default TophatVar() / SphericalBessel() should return the
    mcfit shim (which we keep on CPU because it's faster than our pure-JAX
    FFTLog: 3.5 ms vs 8.4 ms median per call). Override happens automatically
    on TPU."""
    # Make sure no env override is leaking in
    os.environ.pop("CLASSY_SZLITE_FORCE_PURE_FFTLOG", None)
    import importlib
    import classy_szlite._fftlog as fft
    importlib.reload(fft)
    k = np.geomspace(1e-4, 100.0, 256)
    tv = fft.TophatVar(k, lowring=True)
    # On CPU the dispatcher should return _McfitShim, not FFTLog
    if jax.default_backend() == "cpu":
        assert isinstance(tv, fft._McfitShim)
    # On TPU it should be FFTLog (no easy way to assert without a TPU)


def test_mcfit_shim_call_signature():
    """The shim wraps mcfit's (F, extrap=True) → (y, G) to our (F) → (y, G)."""
    os.environ.pop("CLASSY_SZLITE_FORCE_PURE_FFTLOG", None)
    import importlib
    import classy_szlite._fftlog as fft
    importlib.reload(fft)
    k = np.geomspace(1e-4, 100.0, 256)
    pk = (k / 0.05) ** 0.96
    tv = fft.TophatVar(k, lowring=True)
    R, var = tv(pk)
    # Shim should expose .y and .dtype like our FFTLog
    assert hasattr(tv, "y")
    assert hasattr(tv, "dtype")
    assert R.shape == tv.y.shape == (256,)
    assert var.shape == (256,)


def test_use_pure_jax_env_override():
    """$CLASSY_SZLITE_FORCE_PURE_FFTLOG=1 should flip the dispatcher."""
    fft = _force_pure_jax_module()
    try:
        assert fft._use_pure_jax() is True
        k = np.geomspace(1e-4, 100.0, 64)
        tv = fft.TophatVar(k, lowring=True)
        assert isinstance(tv, fft.FFTLog)
    finally:
        _restore_default_module()


def test_sphericalbessel_only_nu0_implemented():
    """The pure-JAX SphericalBessel only ships nu=0 for now."""
    fft = _force_pure_jax_module()
    try:
        import pytest
        u = np.geomspace(1e-5, 100.0, 64)
        with pytest.raises(NotImplementedError):
            fft.SphericalBessel(u, nu=1)
    finally:
        _restore_default_module()
