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

## CMB angular power spectra

```{eval-rst}
.. autofunction:: classy_szlite.cl_TTTEEE
```

![CMB TT power spectrum](_static/cl_tt.png)

## Matter power spectrum

```{eval-rst}
.. autofunction:: classy_szlite.Pk
```

```{eval-rst}
.. autofunction:: classy_szlite.Pnl
```

![Matter power spectrum](_static/pk.png)

## Distances

```{eval-rst}
.. autofunction:: classy_szlite.distances
```

![Hubble rate and distances](_static/distances.png)

## Halo-model tSZ Cl^yy

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy
```

```{eval-rst}
.. autofunction:: classy_szlite.cl_yy_factory
```

![Halo-model tSZ Cl^yy](_static/cl_yy.png)

## Utility

```{eval-rst}
.. autofunction:: classy_szlite.cosmo_to_dict
```
