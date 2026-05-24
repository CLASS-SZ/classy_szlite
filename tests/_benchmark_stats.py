"""Detailed timing benchmark for cl_yy_factory with mean, std, median, min, max.

Runs N evaluations after warmup; reports full distribution.
"""
from __future__ import annotations
import os, time, gc
import numpy as np
import jax
import jax.numpy as jnp
import classy_szlite as csl


def bench(label, fn, n=200, warmup=5):
    # warmup (caches, compile, allocator)
    for _ in range(warmup):
        out = fn(); jax.block_until_ready(out)
    gc.collect()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        out = fn(); jax.block_until_ready(out)
        ts[i] = time.perf_counter() - t0
    ts_ms = ts * 1000
    return {
        "label":   label,
        "n":       n,
        "mean":    float(ts_ms.mean()),
        "std":     float(ts_ms.std()),
        "median":  float(np.median(ts_ms)),
        "p10":     float(np.percentile(ts_ms, 10)),
        "p90":     float(np.percentile(ts_ms, 90)),
        "min":     float(ts_ms.min()),
        "max":     float(ts_ms.max()),
        "samples": ts_ms,
    }


def main():
    backend = jax.default_backend()
    print(f"Backend: {backend}   Devices: {jax.devices()}")
    print(f"jax {jax.__version__}, jaxlib {jax.lib.__version__ if hasattr(jax.lib, '__version__') else 'n/a'}\n")

    cosmo = csl.CosmoParams()
    ell   = jnp.geomspace(100.0, 5000.0, 8)
    p_ref = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

    # 1) cl_yy_factory closure (auto-JIT'd after our patch)
    ev = csl.cl_yy_factory(cosmo, ell)
    r1 = bench("cl_yy_factory(profile)         ", lambda: ev(p_ref), n=200)

    # 2) Vary inputs each call so we hit the actual MCMC pattern (constant cost; just
    #    confirms we aren't caching outputs accidentally).
    profiles = [csl.ProfileParamsA10(P0=8.0 + 0.01*i, beta=5.5 - 0.005*i, B=1.25)
                for i in range(200)]
    counter = {"i": 0}
    def vary():
        p = profiles[counter["i"] % 200]
        counter["i"] += 1
        return ev(p)
    r2 = bench("cl_yy_factory(varied profile)  ", vary, n=200)

    # 3) Full pipeline (cosmology + halo build + integration; not just the closure)
    r3 = bench("cl_yy(cosmo, profile, ell)     ",
               lambda: csl.cl_yy(cosmo, p_ref, ell), n=50)

    # 4) Gradient via jax.grad
    def loss(P0, beta):
        c1, c2 = ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
        return jnp.sum(c1 + c2)
    grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    P0_jax   = jnp.asarray(8.13)
    beta_jax = jnp.asarray(5.48)
    r4 = bench("jax.grad(loss)(P0, beta)       ", lambda: grad(P0_jax, beta_jax), n=200)

    # ---- Print table ----
    rows = [r1, r2, r3, r4]
    print(f"{'workload':<32} {'n':>4}  {'mean':>7} {'std':>6}  {'med':>7}  "
          f"{'p10':>6} {'p90':>6}  {'min':>6} {'max':>7}")
    print("-" * 95)
    for r in rows:
        print(f"{r['label']:<32} {r['n']:>4}  {r['mean']:7.3f} {r['std']:6.3f}  "
              f"{r['median']:7.3f}  {r['p10']:6.3f} {r['p90']:6.3f}  "
              f"{r['min']:6.3f} {r['max']:7.3f}")
    print("\n(All times in ms; std is sample stdev across the n runs.)")

    here = os.path.dirname(__file__)
    np.savez(os.path.join(here, f"_bench_stats_{backend}.npz"),
             **{r["label"].strip(): r["samples"] for r in rows})
    print(f"\nRaw distributions saved to _bench_stats_{backend}.npz")


if __name__ == "__main__":
    main()
