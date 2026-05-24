"""Validate classy_szlite._fftlog against mcfit on CPU at f64 (should be ~machine precision)."""
from __future__ import annotations
import warnings

import numpy as np
import jax.numpy as jnp


def test_tophatvar():
    from mcfit import TophatVar as McfitTV
    from classy_szlite._fftlog import TophatVar as MyTV

    k = np.geomspace(1e-4, 100.0, 512)
    pk = (k / 0.05) ** 0.96 * np.exp(-(k / 5.0) ** 2)  # simple model

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="use backend='jax' if desired")
        mtv = McfitTV(k, lowring=True, backend="jax")
    R_ref, var_ref = mtv(pk, extrap=True)

    my = MyTV(k, dtype=np.float64)
    R_mine, var_mine = my(pk)

    print(f"TophatVar: y-grid maxdev = {np.max(np.abs(np.asarray(R_mine) - np.asarray(R_ref))):.2e}")
    print(f"TophatVar: var maxrel = {np.max(np.abs(np.asarray(var_mine) - np.asarray(var_ref)) / np.abs(np.asarray(var_ref))):.2e}")
    np.testing.assert_allclose(np.asarray(R_mine), np.asarray(R_ref), rtol=1e-12, atol=0)
    # tails of var(R) cancel down to ~1e-9; abs diff 3e-19 is f64 round-off
    np.testing.assert_allclose(np.asarray(var_mine), np.asarray(var_ref), rtol=1e-10, atol=1e-18)
    print("TophatVar: OK")


def test_sphericalbessel():
    from mcfit import SphericalBessel as McfitSB
    from classy_szlite._fftlog import SphericalBessel as MySB

    u = np.geomspace(1e-5, 100.0, 256)
    # toy profile (Arnaud-10-like kernel shape)
    f = u ** (-0.3092) * (1.0 + u ** 1.062) ** ((0.3092 - 5.48) / 1.062)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="use backend='jax' if desired")
        msb = McfitSB(u, nu=0, lowring=True, backend="jax")
    y_ref, g_ref = msb(f, extrap=True)

    my = MySB(u, nu=0, dtype=np.float64)
    y_mine, g_mine = my(f)

    print(f"SphericalBessel: y-grid maxdev = {np.max(np.abs(np.asarray(y_mine) - np.asarray(y_ref))):.2e}")
    print(f"SphericalBessel: g maxrel = {np.max(np.abs(np.asarray(g_mine) - np.asarray(g_ref)) / np.maximum(np.abs(np.asarray(g_ref)), 1e-30)):.2e}")
    np.testing.assert_allclose(np.asarray(y_mine), np.asarray(y_ref), rtol=1e-12, atol=0)
    # Tails of g can be tiny (jaggy positive/negative numerical noise); use atol there.
    np.testing.assert_allclose(np.asarray(g_mine), np.asarray(g_ref), rtol=1e-9, atol=1e-25)
    print("SphericalBessel: OK")


def test_tophatvar_batched():
    """Batched (n_z, n_k) input — what cosmology.py actually feeds it."""
    from mcfit import TophatVar as McfitTV
    from classy_szlite._fftlog import TophatVar as MyTV

    k = np.geomspace(1e-4, 100.0, 512)
    pks = np.stack([(k / 0.05) ** 0.96 * np.exp(-(k / 5.0) ** 2) * (1.0 + 0.1 * i)
                    for i in range(5)], axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="use backend='jax' if desired")
        mtv = McfitTV(k, lowring=True, backend="jax")
    R_ref, var_ref = mtv(pks, extrap=True)

    my = MyTV(k, dtype=np.float64)
    R_mine, var_mine = my(pks)

    print(f"Batched TophatVar: var maxrel = {np.max(np.abs(np.asarray(var_mine) - np.asarray(var_ref)) / np.abs(np.asarray(var_ref))):.2e}")
    np.testing.assert_allclose(np.asarray(var_mine), np.asarray(var_ref), rtol=1e-10, atol=1e-18)
    print("Batched TophatVar: OK")


if __name__ == "__main__":
    test_tophatvar()
    test_sphericalbessel()
    test_tophatvar_batched()
    print("\nAll FFTLog validations PASSED.")
