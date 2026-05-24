"""Profile classy_szlite cl_yy_factory closure: where do the 23 ms come from?

Tests three hypotheses for why our factory is slower than the paper's ~5 ms:
  H1: per-call dispatch overhead (un-JIT'd closure)
  H2: tracing cost for dict-keyed profile_params
  H3: a slow inner op (FFTLog rebuild, interp, trapezoid)
"""
from __future__ import annotations
import os, time, gc
import numpy as np
import jax
import jax.numpy as jnp
import classy_szlite as csl

print(f"Backend: {jax.default_backend()}")
print(f"jax {jax.__version__}, jaxlib {jax.lib.__version__ if hasattr(jax.lib, '__version__') else '?'}")


def measure(fn, n=10):
    # warm
    out = fn(); jax.block_until_ready(out)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(); jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000, float(np.min(times)) * 1000


cosmo = csl.CosmoParams()
profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
ell = jnp.geomspace(100.0, 5000.0, 8)
ev = csl.cl_yy_factory(cosmo, ell)

# --- Baseline: the closure as-is ---
t_med, t_min = measure(lambda: ev(profile))
print(f"\nun-jitted closure ev(profile):                 {t_med:7.2f} ms  (min {t_min:.2f})")

# --- JIT-wrap the closure ---
ev_jit = jax.jit(ev)
out = ev_jit(profile); jax.block_until_ready(out)   # compile pass
t_med, t_min = measure(lambda: ev_jit(profile))
print(f"jax.jit(ev)(profile):                          {t_med:7.2f} ms  (min {t_min:.2f})")

# --- JIT with arrays (skip the named-tuple repacking) ---
P0 = jnp.asarray(profile.P0)
beta = jnp.asarray(profile.beta)
def ev_arrays(P0, beta):
    return ev(csl.ProfileParamsA10(P0=P0, beta=beta, B=1.25))
ev_arr_jit = jax.jit(ev_arrays)
out = ev_arr_jit(P0, beta); jax.block_until_ready(out)
t_med, t_min = measure(lambda: ev_arr_jit(P0, beta))
print(f"jax.jit on (P0, beta) scalars:                 {t_med:7.2f} ms  (min {t_min:.2f})")

# --- Direct call to cl_yy_1h_2h, JIT'd, no closure ---
from classy_szlite.api import cosmo_to_dict
from classy_szlite.cosmology import build as build_cg
from classy_szlite.hmf import build_halo_grids
from classy_szlite.power_spectrum import cl_yy_1h_2h

cosmo_dict = cosmo_to_dict(cosmo)
z_grid = jnp.geomspace(0.005, 3.0, 100)
cg = build_cg(cosmo_dict, z_grid=z_grid)
hg = build_halo_grids(cg, cosmo_dict, delta_crit=500.0, m_min=1e10, m_max=3.5e15, n_m=200)

# Treat dict via static profile_params for JIT
def core(P0, beta):
    pp = profile._replace(P0=P0, beta=beta)._asdict()
    return cl_yy_1h_2h(ell, cg, hg, cosmo_dict, profile='arnaud10',
                       profile_params=pp)
core_jit = jax.jit(core)
out = core_jit(P0, beta); jax.block_until_ready(out)
t_med, t_min = measure(lambda: core_jit(P0, beta))
print(f"jax.jit(cl_yy_1h_2h_direct):                   {t_med:7.2f} ms  (min {t_min:.2f})")

# --- Microbench: just the dispatch overhead of a tiny op ---
@jax.jit
def tiny(x):
    return x + 1
out = tiny(P0); jax.block_until_ready(out)
t_med, t_min = measure(lambda: tiny(P0))
print(f"jax.jit(tiny add) — dispatch floor:            {t_med:7.4f} ms (min {t_min:.4f})")
