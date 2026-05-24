"""Plot CPU vs TPU comparison and timing benchmark from _bench_{backend}.npz files."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__)
    cpu = np.load(os.path.join(here, "_bench_cpu.npz"))
    tpu = np.load(os.path.join(here, "_bench_tpu.npz"))

    ell = cpu["ell_bp"]
    # ---------- Plot 1: cl_yy CPU vs TPU + residual ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 6),
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05),
                             sharex='col')
    for j, cos in enumerate(["baseline", "lows8"]):
        ax_top = axs[0, j]
        ax_bot = axs[1, j]
        cl_cpu = cpu[f"cl_1h_{cos}"] + cpu[f"cl_2h_{cos}"]
        cl_tpu = tpu[f"cl_1h_{cos}"] + tpu[f"cl_2h_{cos}"]
        Dell_cpu = ell * (ell + 1) / (2 * np.pi) * cl_cpu * 1e12
        Dell_tpu = ell * (ell + 1) / (2 * np.pi) * cl_tpu * 1e12

        ax_top.loglog(ell, Dell_cpu, 'C0o-', label='CPU (f64)', lw=1.5)
        ax_top.loglog(ell, Dell_tpu, 'C1x--', label='TPU (f64+f32 FFT)', lw=1.5)
        ax_top.set_ylabel(r'$D_\ell^{yy} \times 10^{12}$')
        ax_top.set_title(f"{cos}: 1h + 2h")
        ax_top.legend(loc='best', fontsize=9)
        ax_top.grid(True, alpha=0.3, which='both')

        resid = (cl_tpu - cl_cpu) / cl_cpu
        ax_bot.semilogx(ell, resid * 1e4, 'C3o-')
        ax_bot.axhline(0, color='k', lw=0.5)
        ax_bot.set_xlabel(r'$\ell$')
        ax_bot.set_ylabel(r'(TPU$-$CPU)/CPU  [$\times 10^{-4}$]')
        ax_bot.grid(True, alpha=0.3, which='both')

    plt.suptitle("classy_szlite Cl^yy: CPU vs TPU agreement", fontweight='bold')
    plt.tight_layout()
    out1 = os.path.join(here, "cl_yy_compare.png")
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    print(f"Saved {out1}")

    # ---------- Plot 2: timing bars ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ['cl_yy_factory(p)\n(single)',
              'cl_yy(cosmo,p,ell)\n(full pipeline)',
              'vmap×32 factory\n(per eval)',
              'vmap×1024 factory\n(per eval)']
    cpu_t = [cpu['t_single_ms'], cpu['t_full_ms'],
             cpu['t_vmap32_per_eval_ms'], cpu['t_vmap1024_per_eval_ms']]
    tpu_t = [tpu['t_single_ms'], tpu['t_full_ms'],
             tpu['t_vmap32_per_eval_ms'], tpu['t_vmap1024_per_eval_ms']]
    x = np.arange(len(labels))
    w = 0.38
    bcpu = ax.bar(x - w/2, cpu_t, w, label='CPU', color='C0')
    btpu = ax.bar(x + w/2, tpu_t, w, label='TPU v6e', color='C1')
    ax.set_yscale('log')
    ax.set_ylabel('Wall time per evaluation (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', which='both')
    for bar, t in zip(bcpu, cpu_t):
        if not np.isnan(t):
            ax.text(bar.get_x() + bar.get_width()/2, t * 1.1,
                    f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, t in zip(btpu, tpu_t):
        if not np.isnan(t):
            ax.text(bar.get_x() + bar.get_width()/2, t * 1.1,
                    f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('classy_szlite warm-eval wall time: CPU vs TPU v6e', fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(here, "timing.png")
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    print(f"Saved {out2}")

    # ---------- Plain-text summary ----------
    txt = os.path.join(here, "timing.txt")
    with open(txt, "w") as f:
        f.write("classy_szlite Cl^yy warm-eval timing (median ms)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'workload':<32} {'CPU':>10} {'TPU v6e':>10} {'speedup':>10}\n")
        for lab, c, t in zip(labels, cpu_t, tpu_t):
            su = c / t if t > 0 and not np.isnan(t) else float('nan')
            short = lab.replace('\n', ' ')
            f.write(f"{short:<32} {c:10.2f} {t:10.2f} {su:10.2f}\n")
        f.write("-" * 60 + "\n")
        f.write("Max |cl_yy TPU - CPU|/|CPU|:\n")
        for cos in ["baseline", "lows8"]:
            cl_cpu = cpu[f"cl_1h_{cos}"] + cpu[f"cl_2h_{cos}"]
            cl_tpu = tpu[f"cl_1h_{cos}"] + tpu[f"cl_2h_{cos}"]
            r = np.max(np.abs(cl_tpu - cl_cpu) / np.abs(cl_cpu))
            f.write(f"  {cos}:  {r:.2e}\n")
    print(f"Saved {txt}")


if __name__ == "__main__":
    main()
