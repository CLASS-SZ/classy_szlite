"""Pipeline schematic for README — matplotlib equivalent of paper Fig 2
(classy_szlite TikZ schematic). Renders to docs/_static/pipeline.png so
it can be embedded in the GitHub README via raw-content URL."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(-0.4, 17.4)
    ax.set_ylim(-3.6, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Style palette — match the paper's TikZ colours roughly
    styles = {
        "inp":   dict(fc="#dfe4f3", ec="#1d3a8a", w=1.8, h=1.3),   # cosmology / output
        "emu":   dict(fc="#bcc7e6", ec="#1d3a8a", w=3.1, h=1.3),
        "fft":   dict(fc="#c8e5c9", ec="#1f7a1f", w=3.2, h=1.3),
        "halo":  dict(fc="#ffd5a8", ec="#a35a00", w=2.7, h=1.3),
        "intg":  dict(fc="#f8c4c4", ec="#a01b1b", w=2.7, h=1.3),
    }

    def box(x, y, kind, title, subtitle):
        s = styles[kind]
        rect = FancyBboxPatch(
            (x - s["w"] / 2, y - s["h"] / 2), s["w"], s["h"],
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.5, edgecolor=s["ec"], facecolor=s["fc"], zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.25, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=s["ec"], zorder=3)
        ax.text(x, y - 0.25, subtitle, ha="center", va="center",
                fontsize=8.5, color=s["ec"], zorder=3)

    def arrow(p1, p2):
        ax.annotate(
            "", xy=p2, xycoords="data", xytext=p1, textcoords="data",
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.4,
                             shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0.0"),
            zorder=1,
        )

    # Coordinates
    P_IN   = (0.0,   0.0)
    P_EMU  = (3.4,   0.0)
    P_SIG  = (7.3,   1.4)
    P_PROF = (7.3,  -1.4)
    P_HALO = (10.8,  1.4)
    P_INT  = (13.6,  0.0)
    P_OUT  = (16.5,  0.0)

    box(*P_IN,   "inp",  "cosmology",       r"$\theta_\mathrm{cos}$")
    box(*P_EMU,  "emu",  "neural network",  r"emulators: $P_\mathrm{lin}(k,z),\ H(z),\ \chi(z)$")
    box(*P_SIG,  "fft",  "FFTLog",          r"$P_\mathrm{lin}(k)\to\sigma(R,z)$")
    box(*P_PROF, "fft",  "FFTLog",          r"$\mathbb{P}(r)\to\tilde u(k)$ lookup")
    box(*P_HALO, "halo", "halo grids",      "HMF + bias")
    box(*P_INT,  "intg", "1h + 2h",         r"$\int\!dz\,\int\!d\ln M$")
    box(*P_OUT,  "inp",  r"$C_\ell^{yy}$",   r"$\sim$5 ms warm")

    # Arrows (forward, gray)
    arrow((P_IN[0] + styles["inp"]["w"] / 2,  0), (P_EMU[0] - styles["emu"]["w"] / 2,  0))
    arrow((P_EMU[0],  styles["emu"]["h"] / 2), (P_EMU[0],  P_SIG[1]  - 0.65))
    arrow((P_EMU[0], -styles["emu"]["h"] / 2), (P_EMU[0],  P_PROF[1] + 0.65))
    arrow((P_EMU[0],  P_SIG[1]  - 0.65), (P_SIG[0]  - styles["fft"]["w"] / 2,  P_SIG[1]))
    arrow((P_EMU[0],  P_PROF[1] + 0.65), (P_PROF[0] - styles["fft"]["w"] / 2,  P_PROF[1]))
    arrow((P_SIG[0] + styles["fft"]["w"] / 2,  P_SIG[1]), (P_HALO[0] - styles["halo"]["w"] / 2, P_HALO[1]))
    arrow((P_HALO[0] + styles["halo"]["w"] / 2, P_HALO[1]), (P_INT[0], P_INT[1] + styles["intg"]["h"] / 2))
    arrow((P_PROF[0] + styles["fft"]["w"] / 2, P_PROF[1]), (P_INT[0], P_INT[1] - styles["intg"]["h"] / 2))
    arrow((P_INT[0] + styles["intg"]["w"] / 2, P_INT[1]), (P_OUT[0] - styles["inp"]["w"] / 2, P_OUT[1]))

    # Backward gradient arrow — output → input via a shallow curved arc
    # below the boxes. Label placed below the arc.
    grad = FancyArrowPatch(
        (P_OUT[0] - 0.2, -0.7), (P_IN[0] + 0.2, -0.7),
        connectionstyle="arc3,rad=-0.18",
        arrowstyle="-|>", mutation_scale=18,
        color="#c1272d", linewidth=1.8, linestyle="--", zorder=1,
    )
    ax.add_patch(grad)
    ax.text(8.25, -3.30,
            r"jax.grad / backprop: $\partial C_\ell^{yy}/\partial\theta$ in one back-pass",
            ha="center", va="center", fontsize=10.5, style="italic", color="#c1272d")

    plt.tight_layout()
    out = os.path.expanduser("~/classy_szlite/docs/_static/pipeline.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
