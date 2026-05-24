# API reference

All functions are importable from the top-level `classy_szlite` namespace.

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

The ``der_full`` array contains 17 entries from the ede-v2 DER emulator:
position 0 is h, position 1 is $\sigma_8$, position 2 is $\Omega_m$, the
remaining 14 entries cover background, sound-horizon and CMB summaries.

### Linear growth

The linear growth factor $\sigma_8(z)$ is derived from the linear matter
power spectrum via
$\sigma_8(z)^2 = \sigma_8(0)^2 \int k^2 P(k,z)\,\mathrm{d}k / \int k^2 P(k,0)\,\mathrm{d}k$
(equivalent to fixed-shape growth in the linear regime):

![Linear growth σ8(z)](_static/growth.png)

## CMB angular power spectra

```{eval-rst}
.. autofunction:: classy_szlite.cl_TTTEEE
```

The default output is $D_\ell = \ell(\ell+1)C_\ell/(2\pi)$ in units of
the squared CMB temperature; multiply by $T_{\rm CMB}^2 = (2.7255\times10^6\,\mu{\rm K})^2$
to get $\mu{\rm K}^2$.

![CMB TT, TE, EE](_static/cmb_ttteee.png)

## Matter power spectrum

```{eval-rst}
.. autofunction:: classy_szlite.Pk
```

```{eval-rst}
.. autofunction:: classy_szlite.Pnl
```

Linear ``Pk`` and non-linear ``Pnl`` (HMcode) at four redshifts:

![Matter power spectrum (linear + nonlinear)](_static/pk.png)

## Distances

```{eval-rst}
.. autofunction:: classy_szlite.distances
```

Returns ``Hz / c`` in $1/\mathrm{Mpc}$ (multiply by $c=299\,792.458$ km/s for
$H(z)$ in km/s/Mpc), $\chi(z)$ in Mpc, and $D_A(z) = \chi(z)/(1+z)$ in Mpc:

![Hubble rate, comoving + angular-diameter distance](_static/distances.png)

## Halo-model tSZ Cl^yy

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy
```

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy_factory
```

The 1-halo term dominates above $\ell \gtrsim 200$; the 2-halo term carries
the large-scale ($\ell \lesssim 200$) signal:

![Halo-model tSZ Cl^yy: 1h + 2h decomposition](_static/cl_yy.png)

See the [convergence study](convergence.md) for the dependence on
$n_z$, $n_M$, $M_{\rm min}$, $M_{\rm max}$.

## Utility

```{eval-rst}
.. autofunction:: classy_szlite.cosmo_to_dict
```
