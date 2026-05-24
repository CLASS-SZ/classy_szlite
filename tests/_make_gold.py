"""Capture CPU gold reference for TPU-branch correctness checks.

Saves cl_yy 1h/2h arrays + d(logL)/dprofile at baseline and lows8 cosmologies
to tests/_gold_cpu.npz. The TPU branch must reproduce these to within tight
tolerances.

Run from repo root with JAX_PLATFORMS=cpu.
"""
from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
import classy_szlite as csl


def main():
    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())

    ell_bp = jnp.geomspace(100.0, 5000.0, 8)
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

    baseline = csl.CosmoParams()
    lows8 = csl.CosmoParams(ln10_10_As=2.910, H0=67.14)

    out = {"ell_bp": np.asarray(ell_bp)}

    for name, cosmo in [("baseline", baseline), ("lows8", lows8)]:
        cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell_bp)
        out[f"cl_1h_{name}"] = np.asarray(cl_1h)
        out[f"cl_2h_{name}"] = np.asarray(cl_2h)
        print(f"{name:8s}  cl_1h[0]={float(cl_1h[0]):.6e}  cl_2h[0]={float(cl_2h[0]):.6e}")

        def total_at(p):
            c1, c2 = csl.cl_yy(cosmo, p, ell_bp)
            return jnp.sum(c1 + c2)

        grad = jax.grad(total_at)(profile)
        out[f"grad_P0_{name}"] = float(grad.P0)
        out[f"grad_beta_{name}"] = float(grad.beta)
        print(f"          dL/dP0={float(grad.P0):.6e}  dL/dbeta={float(grad.beta):.6e}")

    out_path = os.path.join(os.path.dirname(__file__), "_gold_cpu.npz")
    np.savez(out_path, **out)
    print(f"\nWrote {out_path} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
