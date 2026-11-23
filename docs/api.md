# API reference

All functions are importable from the top-level `classy_szlite` namespace.
For runnable code + plots, see [Tutorials & examples](examples.md).

## Parameter containers

::: classy_szlite.params.CosmoParams
    options:
      heading_level: 3

::: classy_szlite.params.ProfileParamsA10
    options:
      heading_level: 3

## Derived parameters

::: classy_szlite.api.derived
    options:
      heading_level: 3

The `der_full` array contains 17 entries from the ede-v3 DER emulator
(index 0 = $h$, index 1 = $\sigma_8$, index 2 = $\Omega_m$, etc.).

## CMB angular power spectra

::: classy_szlite.api.cl_TTTEEE
    options:
      heading_level: 3

The default output is $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$ in units of
$T_{\rm CMB}^2$ — multiply by $T_{\rm CMB}^2 = (2.7255\times10^6\,\mu{\rm K})^2$
to get $\mu{\rm K}^2$.

## Matter power spectrum

::: classy_szlite.api.Pk
    options:
      heading_level: 3

::: classy_szlite.api.Pnl
    options:
      heading_level: 3

## Distances

::: classy_szlite.api.distances
    options:
      heading_level: 3

Returns `Hz / c` in $1/\mathrm{Mpc}$ (multiply by $c=299\,792.458$ km/s
for $H(z)$ in km/s/Mpc), $\chi(z)$ in Mpc, and $D_A(z)=\chi(z)/(1+z)$ in Mpc.

## Halo-model tSZ Cl^yy

::: classy_szlite.api.cl_yy
    options:
      heading_level: 3

::: classy_szlite.api.cl_yy_factory
    options:
      heading_level: 3

::: classy_szlite.api.cl_yy_trispectrum
    options:
      heading_level: 3

::: classy_szlite.api.cl_yy_covariance
    options:
      heading_level: 3

See the [convergence study](convergence.md) for the dependence on
$n_z$, $n_M$, $M_{\rm min}$, $M_{\rm max}$, and the
[Tutorials & examples](examples.md) page for an end-to-end example
using `cl_yy_covariance` to generate synthetic data via Cholesky.

## Utility

::: classy_szlite.api.cosmo_to_dict
    options:
      heading_level: 3
