"""Plot D_ell^yy at the fiducial A10 profile (P0=8.13, c500=1.156,
gamma=0.3292, alpha=1.062, beta=5.4807) — the standard tSZ power-spectrum
prediction, for sanity-checking against literature values.

Overlays the lower-amplitude bestfit-recovered profile (P0=1.20, β=2.74)
for context.
"""
from __future__ import annotations
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import classy_szlite as csl


def main():
    cosmo = csl.CosmoParams(omega_b=0.0226, omega_cdm=0.118, H0=68.22,
                            tau_reio=0.0561, ln10_10_As=3.06, n_s=0.9743)

    fiducial_a10 = csl.ProfileParamsA10(
        P0=8.130, c500=1.156, gamma=0.3292, alpha=1.062, beta=5.4807, B=1.25,
    )
    bestfit = csl.ProfileParamsA10(
        P0=1.20, c500=1.156, gamma=0.3292, alpha=1.062, beta=2.74, B=1.25,
    )

    ell = jnp.geomspace(50.0, 10000.0, 80)
    pref = ell * (ell + 1) / (2 * jnp.pi) * 1e12

    c1_a, c2_a = csl.cl_yy(cosmo, fiducial_a10, ell)
    c1_b, c2_b = csl.cl_yy(cosmo, bestfit, ell)
    Dell_a = pref * (c1_a + c2_a)
    Dell_b = pref * (c1_b + c2_b)
    Dell_a_1h = pref * c1_a
    Dell_a_2h = pref * c2_a

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ell, Dell_a, 'C3-', lw=2.0,
              label=r"A10 fiducial: $P_0\!=\!8.13$, $\beta\!=\!5.48$ (total)")
    ax.loglog(ell, Dell_a_1h, 'C3--', lw=1.0, alpha=0.7, label="    1-halo only")
    ax.loglog(ell, Dell_a_2h, 'C3:',  lw=1.0, alpha=0.7, label="    2-halo only")
    ax.loglog(ell, Dell_b, 'C0-', lw=1.8,
              label=r"recovered bestfit: $P_0\!=\!1.20$, $\beta\!=\!2.74$")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12} = \ell(\ell+1)\,C_\ell^{yy} / 2\pi \times 10^{12}$")
    ax.set_title(r"tSZ $D_\ell^{yy}$ — A10 fiducial vs recovered bestfit "
                  r"(baseline cosmology, B=1.25)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlim(50, 1e4)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "clyy_fiducial.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')

    print(f"\nA10 fiducial D_ell × 1e12 at sample ells:")
    sample_ells = [100, 1000, 3000, 6000]
    for el in sample_ells:
        i = int(np.argmin(np.abs(np.asarray(ell) - el)))
        print(f"  ell={int(np.asarray(ell)[i]):5d}:  total={float(Dell_a[i]):.3f}, "
              f"1h={float(Dell_a_1h[i]):.3f}, 2h={float(Dell_a_2h[i]):.3f}")
    i3000 = int(np.argmin(np.abs(np.asarray(ell) - 3000)))
    ratio = float(Dell_a[i3000]) / float(Dell_b[i3000])
    print(f"\nratio (A10 total / bestfit total) at ell=3000: {ratio:.1f}×")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
