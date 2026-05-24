"""Plot wall vs ESS / R-hat from the NUTS budget sweep."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__)
    sw = np.load(os.path.join(here, "_nuts_sweep.npz"))["sweep"]
    # cols: warm, samp, chains, wall, ESS_P0, ESS_b, Rhat_P0, Rhat_b
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axs[0]
    ax.plot(sw[:, 3], sw[:, 4], 'o-', color='C0', lw=1.5, label="ESS $P_0$")
    ax.plot(sw[:, 3], sw[:, 5], 's-', color='C3', lw=1.5, label=r"ESS $\beta$")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("effective sample size")
    ax.set_title("NUTS budget sweep: ESS")
    ax.axvline(10, color='k', ls='--', alpha=0.5, label='10 s')
    ax.legend(); ax.grid(True, alpha=0.3)

    for i, row in enumerate(sw):
        label = f"{int(row[0])}+{int(row[1])}×{int(row[2])}"
        ax.annotate(label, (row[3], row[4]), fontsize=7,
                     xytext=(3, -10), textcoords='offset points', color='C0')

    ax = axs[1]
    ax.plot(sw[:, 3], (sw[:, 6] - 1), 'o-', color='C0', lw=1.5, label="R-hat$-$1, $P_0$")
    ax.plot(sw[:, 3], (sw[:, 7] - 1), 's-', color='C3', lw=1.5, label=r"R-hat$-$1, $\beta$")
    ax.axhline(0.05, color='k', ls=':',  alpha=0.7, label='loose (R-1 < 0.05)')
    ax.axhline(0.01, color='k', ls='--', alpha=0.7, label='tight (R-1 < 0.01)')
    ax.axvline(10, color='k', ls='--', alpha=0.5)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("R-hat $-$ 1")
    ax.set_yscale("log")
    ax.set_title("NUTS budget sweep: convergence")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    plt.suptitle("Sub-10s converged posteriors with NUTS on baseline $C_\\ell^{yy}$",
                 fontweight='bold')
    plt.tight_layout()
    out = os.path.join(here, "nuts_sweep.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
