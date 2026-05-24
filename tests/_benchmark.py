"""CPU vs TPU benchmark + plots for classy_szlite cl_yy.

Outputs:
  - cl_yy_compare.png   — cl_yy at baseline + lows8, CPU overlay vs TPU + residuals
  - timing.png          — bar chart of warm wall times for {cl_yy_factory(p), vmap×N}
  - timing.txt          — same numbers in plaintext

Reads JAX_PLATFORMS to pick backend (cpu or tpu). Run once per backend, then
plot reads both .npz files to overlay.
"""
from __future__ import annotations
import os
import time
import numpy as np
import jax
import jax.numpy as jnp


def measure(fn, n=5):
    """Measure warm wall-clock time of fn(). Returns median over n runs."""
    out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def main():
    import classy_szlite as csl
    backend = jax.default_backend()
    print(f"Backend: {backend}  Devices: {jax.devices()}")

    ell_bp = jnp.geomspace(100.0, 5000.0, 8)
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
    cosmologies = {
        "baseline": csl.CosmoParams(),
        "lows8":    csl.CosmoParams(ln10_10_As=2.910, H0=67.14),
    }

    # ----- cl_yy values per cosmology -----
    results = {"ell_bp": np.asarray(ell_bp), "backend": backend}
    for name, cosmo in cosmologies.items():
        cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell_bp)
        jax.block_until_ready(cl_1h)
        results[f"cl_1h_{name}"] = np.asarray(cl_1h)
        results[f"cl_2h_{name}"] = np.asarray(cl_2h)
        print(f"{name:8s}  cl_1h[0]={float(cl_1h[0]):.4e}  cl_2h[0]={float(cl_2h[0]):.4e}")

    # ----- timing benchmarks (use baseline cosmology) -----
    cosmo = cosmologies["baseline"]
    ev = csl.cl_yy_factory(cosmo, ell_bp)

    # 1: single fast-path evaluation
    t1, _ = measure(lambda: ev(profile), n=5)
    print(f"\ncl_yy_factory(profile)         : {t1*1e3:8.2f} ms  (warm, median of 5)")
    results["t_single_ms"] = t1 * 1e3

    # 2: full cosmology pipeline
    t2, _ = measure(lambda: csl.cl_yy(cosmo, profile, ell_bp), n=3)
    print(f"cl_yy(cosmo, profile, ell)     : {t2*1e3:8.2f} ms  (warm, median of 3)")
    results["t_full_ms"] = t2 * 1e3

    # 3: vmap'd batches of profiles  (the TPU's actual use case)
    P0s = jnp.linspace(4.0, 12.0, 32)
    betas = jnp.linspace(4.5, 6.5, 32)
    profiles_batch = jax.vmap(lambda p, b: csl.ProfileParamsA10(P0=p, beta=b, B=1.25))(P0s, betas)
    ev_batched = jax.jit(jax.vmap(ev))

    # warm up
    out = ev_batched(profiles_batch); jax.block_until_ready(out)
    t3, _ = measure(lambda: ev_batched(profiles_batch), n=5)
    print(f"vmap×32 cl_yy_factory(p_batch) : {t3*1e3:8.2f} ms total → {t3/32*1e3:6.2f} ms/eval")
    results["t_vmap32_ms"] = t3 * 1e3
    results["t_vmap32_per_eval_ms"] = t3 / 32 * 1e3

    # 4: bigger vmap if TPU has memory headroom
    try:
        P0s = jnp.linspace(4.0, 12.0, 1024)
        betas = jnp.linspace(4.5, 6.5, 1024)
        profiles_batch = jax.vmap(lambda p, b: csl.ProfileParamsA10(P0=p, beta=b, B=1.25))(P0s, betas)
        out = ev_batched(profiles_batch); jax.block_until_ready(out)
        t4, _ = measure(lambda: ev_batched(profiles_batch), n=3)
        print(f"vmap×1024 cl_yy_factory(p_b) : {t4*1e3:8.2f} ms total → {t4/1024*1e3:6.3f} ms/eval")
        results["t_vmap1024_ms"] = t4 * 1e3
        results["t_vmap1024_per_eval_ms"] = t4 / 1024 * 1e3
    except Exception as e:
        print(f"vmap×1024 skipped: {type(e).__name__}: {e}")
        results["t_vmap1024_ms"] = float("nan")
        results["t_vmap1024_per_eval_ms"] = float("nan")

    out_path = os.path.join(os.path.dirname(__file__), f"_bench_{backend}.npz")
    np.savez(out_path, **results)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
