# API reference

All functions are importable from the top-level `classy_szlite` namespace.
For runnable code + plots, see [Tutorials & examples](examples.md).

## Parameter containers

```{eval-rst}
.. autoclass:: classy_szlite.CosmoParams
   :members:
```

```{eval-rst}
.. autoclass:: classy_szlite.ProfileParamsA10
   :members:
```

## Derived parameters

```{eval-rst}
.. autofunction:: classy_szlite.derived
```

The ``der_full`` array contains 17 entries from the ede-v2 DER emulator
(index 0 = $h$, index 1 = $\sigma_8$, index 2 = $\Omega_m$, etc.).

## CMB angular power spectra

```{eval-rst}
.. autofunction:: classy_szlite.cl_TTTEEE
```

The default output is $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$ in units of
$T_{\rm CMB}^2$ — multiply by $T_{\rm CMB}^2 = (2.7255\times10^6\,\mu{\rm K})^2$
to get $\mu{\rm K}^2$.

## Matter power spectrum

```{eval-rst}
.. autofunction:: classy_szlite.Pk
```

```{eval-rst}
.. autofunction:: classy_szlite.Pnl
```

## Distances

```{eval-rst}
.. autofunction:: classy_szlite.distances
```

Returns ``Hz / c`` in $1/\mathrm{Mpc}$ (multiply by $c=299\,792.458$ km/s
for $H(z)$ in km/s/Mpc), $\chi(z)$ in Mpc, and $D_A(z)=\chi(z)/(1+z)$ in Mpc.

## Halo-model tSZ Cl^yy

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy
```

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy_factory
```

See the [convergence study](convergence.md) for the dependence on
$n_z$, $n_M$, $M_{\rm min}$, $M_{\rm max}$.

## Utility

```{eval-rst}
.. autofunction:: classy_szlite.cosmo_to_dict
```
