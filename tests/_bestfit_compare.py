"""Compare bestfit-search methods: L-BFGS-B vs Adam vs vanilla GD.

Same problem (Gaussian likelihood on Cl^yy bandpowers), same initial point,
same forward / gradient via jax.grad of cl_yy_factory. Record
chi^2 vs iteration number and final wall.

Outputs: tests/bestfit_compare.png
"""
from __future__ import annotations
import os, time
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize as so
import optax
import classy_szlite as csl

DATA_DIR = os.path.expanduser("~/class-sz-plugin-tests/data")


def load_data():
    ell  = np.loadtxt(os.path.join(DATA_DIR, "ls_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    bp   = np.loadtxt(os.path.join(DATA_DIR, "data_ps-ell-y2-erry2_total-act-26_lmin2000_lmax600.txt"))
    cov  = np.loadtxt(os.path.join(DATA_DIR, "cov_standard_newer3_bp_CIbbp_newcovlmin2000_plancklmax600_test.txt"))
    return jnp.asarray(ell), jnp.asarray(bp[:, 1]), jnp.asarray(cov)


def main():
    print(f"Backend: {jax.default_backend()}")
    ell, Dell_data, cov = load_data()
    cov_inv = jnp.linalg.inv(cov)
    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)
    ev = csl.cl_yy_factory(cosmo, ell)
    dl_fac = ell * (ell + 1.0) / (2.0 * jnp.pi) * 1e12

    def chi2(x):
        prof = csl.ProfileParamsA10(P0=x[0], beta=x[1], B=1.25)
        c1, c2 = ev(prof)
        Dell = dl_fac * (c1 + c2)
        r = Dell - Dell_data
        return r @ cov_inv @ r

    chi2_jit  = jax.jit(chi2)
    grad_chi2 = jax.jit(jax.grad(chi2))
    # warmup compile
    x0 = jnp.array([8.13, 5.48], dtype=jnp.float64)
    _ = chi2_jit(x0); jax.block_until_ready(_)
    _ = grad_chi2(x0); jax.block_until_ready(_)

    histories = {}

    # 1) L-BFGS-B (scipy)  -----------------------------------------
    print("\n-- L-BFGS-B --")
    hist = {"x": [], "chi2": []}
    def f(x):
        v = float(chi2_jit(jnp.asarray(x)))
        hist["x"].append(np.asarray(x).copy()); hist["chi2"].append(v)
        return v
    def g(x):
        return np.asarray(grad_chi2(jnp.asarray(x)))
    t0 = time.perf_counter()
    res_lbfgs = so.minimize(f, np.asarray(x0), jac=g, method="L-BFGS-B",
                             bounds=((0.1, 20.0), (0.5, 10.0)))
    t_lbfgs = time.perf_counter() - t0
    histories["L-BFGS-B"] = {"chi2": np.asarray(hist["chi2"]),
                              "x": np.asarray(hist["x"]),
                              "wall": t_lbfgs,
                              "n_evals": len(hist["chi2"])}
    print(f"  bestfit:  P0={res_lbfgs.x[0]:.3f}, beta={res_lbfgs.x[1]:.3f}, chi2={res_lbfgs.fun:.3f}")
    print(f"  evals:    {len(hist['chi2'])}   wall: {t_lbfgs*1000:.1f} ms")

    # 2) Adam (optax)  -----------------------------------------
    print("\n-- Adam --")
    optimizer = optax.adam(learning_rate=0.05)
    x = x0.copy(); state = optimizer.init(x)
    hist = {"chi2": []}
    t0 = time.perf_counter()
    for _ in range(200):
        g_ = grad_chi2(x)
        updates, state = optimizer.update(g_, state)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, jnp.array([0.1, 0.5]), jnp.array([20.0, 10.0]))
        hist["chi2"].append(float(chi2_jit(x)))
    t_adam = time.perf_counter() - t0
    histories["Adam"] = {"chi2": np.asarray(hist["chi2"]),
                          "wall": t_adam,
                          "n_evals": len(hist["chi2"])}
    print(f"  bestfit:  P0={float(x[0]):.3f}, beta={float(x[1]):.3f}, chi2={float(chi2_jit(x)):.3f}")
    print(f"  evals:    {len(hist['chi2'])}   wall: {t_adam*1000:.1f} ms")

    # 3) Vanilla gradient descent  -----------------------------------------
    print("\n-- Vanilla GD --")
    x = x0.copy()
    hist = {"chi2": []}
    lr = 0.001                                                  # crude — large LR diverges
    t0 = time.perf_counter()
    for _ in range(200):
        g_ = grad_chi2(x)
        x = x - lr * g_
        x = jnp.clip(x, jnp.array([0.1, 0.5]), jnp.array([20.0, 10.0]))
        hist["chi2"].append(float(chi2_jit(x)))
    t_gd = time.perf_counter() - t0
    histories["Vanilla GD"] = {"chi2": np.asarray(hist["chi2"]),
                                "wall": t_gd,
                                "n_evals": len(hist["chi2"])}
    print(f"  bestfit:  P0={float(x[0]):.3f}, beta={float(x[1]):.3f}, chi2={float(chi2_jit(x)):.3f}")
    print(f"  evals:    {len(hist['chi2'])}   wall: {t_gd*1000:.1f} ms")

    # ---- Plot: chi2 vs forward-evaluation count ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    chi2_inf = float(res_lbfgs.fun)
    for name, color in [("L-BFGS-B", "C3"), ("Adam", "C0"), ("Vanilla GD", "C1")]:
        h = histories[name]
        # δχ² above L-BFGS-B minimum (assumed to be true minimum)
        delta = h["chi2"] - chi2_inf + 1e-3
        ax.semilogy(np.arange(1, len(delta)+1), delta, label=f"{name}  ({h['wall']*1000:.0f} ms total)",
                    color=color, lw=1.5)
    ax.axhline(1e-3, color='k', ls=':', alpha=0.5, label="floor (L-BFGS min)")
    ax.set_xlabel("forward evaluation count")
    ax.set_ylabel(r"$\chi^2 - \chi^2_\mathrm{min}$  (+ 10$^{-3}$ offset)")
    ax.set_title("MAP search: L-BFGS-B vs Adam vs vanilla GD (exact jax.grad)")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "bestfit_compare.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
