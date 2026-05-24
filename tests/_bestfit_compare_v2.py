"""Tuned optimizer comparison: L-BFGS-B vs Adam (with LR schedule) vs Newton.

Outputs two figures:
  - bestfit_loss_curves.png  — chi^2 vs eval count (loss curve)
  - bestfit_paths.png        — optimizer trajectories overlaid on chi^2 surface
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

    chi2_jit   = jax.jit(chi2)
    grad_chi2  = jax.jit(jax.grad(chi2))
    hess_chi2  = jax.jit(jax.hessian(chi2))

    x0_np = np.array([8.13, 5.48], dtype=np.float64)
    x0 = jnp.asarray(x0_np)
    # warm-up compile
    _  = chi2_jit(x0); _ = grad_chi2(x0); _ = hess_chi2(x0)
    jax.block_until_ready(_)

    P0_BOUNDS   = (0.1, 20.0)
    BETA_BOUNDS = (0.5, 10.0)
    LO = jnp.array([P0_BOUNDS[0], BETA_BOUNDS[0]])
    HI = jnp.array([P0_BOUNDS[1], BETA_BOUNDS[1]])

    histories = {}

    # 1) L-BFGS-B (scipy with exact jax grad)  ---------------------------
    print("\n-- L-BFGS-B --")
    hist = {"x": [], "chi2": []}
    def f(x):
        v = float(chi2_jit(jnp.asarray(x)))
        hist["x"].append(np.asarray(x).copy()); hist["chi2"].append(v)
        return v
    def g(x):
        return np.asarray(grad_chi2(jnp.asarray(x)))
    t0 = time.perf_counter()
    res = so.minimize(f, x0_np, jac=g, method="L-BFGS-B",
                       bounds=(P0_BOUNDS, BETA_BOUNDS))
    t_lbfgs = time.perf_counter() - t0
    histories["L-BFGS-B"] = {"x": np.asarray(hist["x"]),
                              "chi2": np.asarray(hist["chi2"]),
                              "wall": t_lbfgs, "color": "C3"}
    print(f"  bestfit: P0={res.x[0]:.3f}, beta={res.x[1]:.3f}, chi2={res.fun:.3f}")
    print(f"  evals = {len(hist['chi2'])}   wall = {t_lbfgs*1000:.0f} ms")
    chi2_min = float(res.fun)
    bf = res.x

    # 2) Adam with cosine LR schedule (tuned)  ---------------------------
    print("\n-- Adam (cosine LR, warmup) --")
    N_ADAM_STEPS = 200
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-3, peak_value=0.2,
        warmup_steps=20, decay_steps=N_ADAM_STEPS, end_value=1e-3,
    )
    optimizer = optax.chain(
        optax.adam(learning_rate=schedule),
    )
    x = x0
    state = optimizer.init(x)
    xs, chs = [np.asarray(x).copy()], [float(chi2_jit(x))]
    t0 = time.perf_counter()
    for k in range(N_ADAM_STEPS):
        g_ = grad_chi2(x)
        updates, state = optimizer.update(g_, state)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, LO, HI)
        xs.append(np.asarray(x).copy()); chs.append(float(chi2_jit(x)))
    t_adam = time.perf_counter() - t0
    histories["Adam (tuned)"] = {"x": np.asarray(xs), "chi2": np.asarray(chs),
                                  "wall": t_adam, "color": "C0"}
    print(f"  bestfit: P0={float(x[0]):.3f}, beta={float(x[1]):.3f}, chi2={chs[-1]:.3f}")
    print(f"  evals = {len(chs)}   wall = {t_adam*1000:.0f} ms")

    # 3) Newton (full Hessian via jax.hessian)  ---------------------------
    print("\n-- Newton (full Hessian) --")
    N_NEWT = 25
    x = x0
    xs, chs = [np.asarray(x).copy()], [float(chi2_jit(x))]
    t0 = time.perf_counter()
    for k in range(N_NEWT):
        g_ = grad_chi2(x)
        H_ = hess_chi2(x)
        # Levenberg–Marquardt-ish damping for numerical safety
        damping = 1e-6
        H_d = H_ + damping * jnp.eye(2)
        dx = jnp.linalg.solve(H_d, g_)
        x = x - dx
        x = jnp.clip(x, LO, HI)
        xs.append(np.asarray(x).copy()); chs.append(float(chi2_jit(x)))
        if jnp.linalg.norm(dx) < 1e-9:
            break
    t_newt = time.perf_counter() - t0
    histories["Newton (jax.hessian)"] = {"x": np.asarray(xs), "chi2": np.asarray(chs),
                                          "wall": t_newt, "color": "C2"}
    print(f"  bestfit: P0={float(x[0]):.3f}, beta={float(x[1]):.3f}, chi2={chs[-1]:.3f}")
    print(f"  evals = {len(chs)}   wall = {t_newt*1000:.0f} ms")

    # ---- Plot A: loss curves -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, h in histories.items():
        delta = h["chi2"] - chi2_min + 1e-3
        ax.semilogy(np.arange(1, len(delta)+1), delta, '-',
                    color=h["color"], lw=1.5,
                    label=f"{name}  ({len(delta)} evals, {h['wall']*1000:.0f} ms)")
    ax.axhline(1e-3, color='k', ls=':', alpha=0.4, label="floor")
    ax.set_xlabel("optimizer iteration (= forward+grad evals)")
    ax.set_ylabel(r"$\chi^2 - \chi^2_\mathrm{min}$  (+ $10^{-3}$ offset)")
    ax.set_title("MAP search: convergence vs evaluation count")
    ax.set_xlim(0, 50)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), "bestfit_loss_curves.png")
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    print(f"\nSaved {out1}")

    # ---- Plot B: trajectories on chi^2 surface -------------------------
    print("\nBuilding chi^2 surface for the trajectory plot ...")
    nx = 60
    P0_grid   = np.linspace(0.5, 8.0, nx)
    beta_grid = np.linspace(1.5, 7.0, nx)
    X, Y = np.meshgrid(P0_grid, beta_grid)
    chi2_v = jax.jit(jax.vmap(chi2))
    pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], axis=-1)
    Z = np.asarray(chi2_v(pts)).reshape(nx, nx)

    fig, ax = plt.subplots(figsize=(8, 6))
    log_levels = np.logspace(np.log10(chi2_min + 0.5),
                              np.log10(chi2_min + 200), 12)
    cs = ax.contourf(X, Y, Z, levels=log_levels, cmap='viridis_r')
    plt.colorbar(cs, label=r"$\chi^2$")
    ax.contour(X, Y, Z, levels=log_levels, colors='k', linewidths=0.3, alpha=0.4)

    for name, h in histories.items():
        xs = h["x"]
        ax.plot(xs[:, 0], xs[:, 1], 'o-', color=h["color"], ms=3, lw=1.2,
                label=f"{name} ({len(xs)} steps)")
        ax.plot(xs[0, 0], xs[0, 1], 'k*', ms=12, mfc='none', mew=1.5)
    ax.plot(bf[0], bf[1], 'r*', ms=14, mec='k', label='MAP')
    ax.set_xlabel(r"$P_0$"); ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Optimizer trajectories on $\chi^2(P_0, \beta)$, baseline cosmology")
    ax.legend(loc='upper right', fontsize=9)
    out2 = os.path.join(os.path.dirname(__file__), "bestfit_paths.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
