"""Small traceable helpers shared by the simulator modules."""
from __future__ import annotations

import jax.numpy as jnp


def compact(mask, size: int, fill_value: int = 0):
    """Indices of the True entries of the 1-D ``mask``, ascending, padded to
    ``size`` with ``fill_value``, and their total count.

    Same result as ``jnp.nonzero(mask, size=size, fill_value=fill_value)[0]``,
    but by a prefix sum and a binary search per output slot, without the
    scatter (``bincount``) that ``jnp.nonzero`` uses: on GPUs that scatter is
    heavily contended for sparse masks (one kernel took 19 ms per tile on an
    A100 for the 21.6M-element peak mask). Deterministic.
    """
    csum = jnp.cumsum(mask.astype(jnp.int32))
    count = csum[-1]
    k = jnp.arange(1, size + 1, dtype=jnp.int32)
    idx = jnp.searchsorted(csum, k, side="left").astype(jnp.int32)
    return jnp.where(k <= count, idx, jnp.int32(fill_value)), count
