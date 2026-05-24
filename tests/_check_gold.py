"""Re-evaluate cl_yy + gradients and diff against tests/_gold_cpu.npz."""
from __future__ import annotations
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import classy_szlite as csl


def main():
    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())

    gold = np.load(os.path.join(os.path.dirname(__file__), "_gold_cpu.npz"))
    ell_bp = jnp.asarray(gold["ell_bp"])
    profile = csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)

    cosmologies = {
        "baseline": csl.CosmoParams(),
        "lows8":    csl.CosmoParams(ln10_10_As=2.910, H0=67.14),
    }

    rtol = float(os.environ.get("GOLD_RTOL", "1e-5"))
    print(f"\nUsing rtol={rtol:.0e} for cl_yy and gradients")

    max_rel_seen = 0.0
    for name, cosmo in cosmologies.items():
        cl_1h, cl_2h = csl.cl_yy(cosmo, profile, ell_bp)
        ref_1h = gold[f"cl_1h_{name}"]
        ref_2h = gold[f"cl_2h_{name}"]
        d1 = np.max(np.abs(np.asarray(cl_1h) - ref_1h) / np.abs(ref_1h))
        d2 = np.max(np.abs(np.asarray(cl_2h) - ref_2h) / np.abs(ref_2h))
        print(f"{name:8s}  cl_1h maxrel={d1:.2e}  cl_2h maxrel={d2:.2e}")
        max_rel_seen = max(max_rel_seen, d1, d2)

        def total_at(p):
            c1, c2 = csl.cl_yy(cosmo, p, ell_bp)
            return jnp.sum(c1 + c2)

        grad = jax.grad(total_at)(profile)
        gp = float(grad.P0)
        gb = float(grad.beta)
        gp_ref = float(gold[f"grad_P0_{name}"])
        gb_ref = float(gold[f"grad_beta_{name}"])
        rp = abs(gp - gp_ref) / abs(gp_ref)
        rb = abs(gb - gb_ref) / abs(gb_ref)
        print(f"          dL/dP0={gp:.4e} (ref {gp_ref:.4e}, rel {rp:.2e})")
        print(f"          dL/dbeta={gb:.4e} (ref {gb_ref:.4e}, rel {rb:.2e})")
        max_rel_seen = max(max_rel_seen, rp, rb)

    print(f"\nWorst relative deviation vs CPU gold: {max_rel_seen:.2e}")
    if max_rel_seen > rtol:
        print(f"FAIL: exceeds rtol={rtol:.0e}")
        sys.exit(1)
    print(f"PASS: within rtol={rtol:.0e}")


if __name__ == "__main__":
    main()
