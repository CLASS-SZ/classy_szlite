"""Pure-JAX CosmoPower emulator forward pass.

Loads ``.npz`` files produced by the CosmoPower training pipeline and
exposes a JAX-traceable ``predict`` function. Supports two architectures:

1. **Plain NN** (TT, DER, HZ, DAZ, PKL, PKNL, S8Z, etc.):
   output = feature_train_std * NN(p_in) + feature_train_mean

2. **NN + PCA** (TE and similar that decompress PCA coefficients):
   output = ((NN(p_in) @ pca_transform_matrix) + pca_mean) * pca_std
            * feature_train_std + feature_train_mean
   (exact ordering inferred from cosmopower/classy_szfast convention)

The NN itself uses the "α-β sigmoid" activation introduced in
Spurio Mancini, Piras, Alsing et al. (2022, MNRAS 511, 1771) — a
trainable smooth interpolation between identity and sigmoid:

    h_out = (β + sigmoid(α * z) * (1 - β)) * z

No keras / tensorflow dependency: the .npz files are read with numpy,
and the forward pass is a handful of jnp ops.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# .npz loading — plain (no pickle) format
# ---------------------------------------------------------------------------

_PLAIN_SEP = "."     # cosmopower field names contain `_` but never `.`


def _load_emulator_dict(path: str) -> dict:
    """Return the cosmopower-style dict for a ``*_plain.npz`` file."""
    saved = np.load(path, allow_pickle=False)
    if "weights_.n" not in saved.files:
        # Defensive: a few users still have legacy pickled `_v2.npz` files
        # in their data dir.  Try that path, which requires tensorflow.
        saved.close()
        raw = np.load(path, allow_pickle=True)
        if "arr_0" not in raw.files:
            raise ValueError(
                f"classy_szlite: {path} is not a recognised emulator file."
            )
        return raw["arr_0"].item()                                   # may raise ModuleNotFoundError
    return _unflatten_plain(saved)


def _unflatten_plain(saved: np.lib.npyio.NpzFile) -> dict:
    """Inverse of the converter's flatten step.

    ``foo.0, foo.1, ...`` + ``foo.n`` (count) become ``foo: [...]``.
    0-D numeric arrays become Python scalars; 0-D Unicode arrays become str.
    """
    out: dict = {}
    list_groups: dict = {}
    for k in saved.files:
        if _PLAIN_SEP in k:
            base, idx = k.rsplit(_PLAIN_SEP, 1)
            if idx == "n":
                continue
            list_groups.setdefault(base, {})[int(idx)] = saved[k]
        else:
            v = saved[k]
            if v.ndim == 0 and v.dtype.kind in "iuf":
                out[k] = v.item()
            elif v.ndim == 0 and v.dtype.kind == "U":
                out[k] = str(v)
            else:
                out[k] = v
    for base, idx_map in list_groups.items():
        out[base] = [idx_map[i] for i in sorted(idx_map)]
    return out


class EmulatorWeights(NamedTuple):
    """Network weights and standardisation arrays of one emulator.

    A NamedTuple, hence a JAX pytree: it can be passed as an argument to
    ``jax.jit``/``jax.vmap`` functions instead of being baked in as constants.
    ``pca_*`` are ``None`` for plain (non-PCA) emulators.
    """
    weights:      tuple            # ((W, b), ...) per layer
    alphas:       tuple            # α per hidden layer
    betas:        tuple            # β per hidden layer
    param_mean:   jax.Array
    param_std:    jax.Array
    feature_mean: jax.Array
    feature_std:  jax.Array
    pca_matrix:   jax.Array | None = None
    pca_mean:     jax.Array | None = None
    pca_std:      jax.Array | None = None


def nn_forward(w: EmulatorWeights, p: jax.Array) -> jax.Array:
    """Forward pass. ``p`` has shape ``(B, n_params)``; returns ``(B, n_out)``.

    Explicit shapes: no squeeze, so B = 1 and n_out = 1 keep their axes.
    """
    h = (p - w.param_mean) / w.param_std
    # CosmoPower stores W as (in_dim, out_dim): the matmul is h @ W.
    for (W, b), a, beta in zip(w.weights[:-1], w.alphas, w.betas):
        z = h @ W + b
        h = (beta + _sigmoid(a * z) * (1.0 - beta)) * z
    W, b = w.weights[-1]
    preds = h @ W + b                                        # (B, n_out_nn)
    if w.pca_matrix is not None:
        preds = preds * w.pca_std + w.pca_mean               # (B, n_pca)
        preds = preds @ w.pca_matrix                         # (B, n_modes)
    return preds * w.feature_std + w.feature_mean


class _Static:
    """Holder compared and hashed by identity, for pytree metadata such as
    the ``modes`` array (numpy arrays cannot be compared as aux data)."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


@dataclass(frozen=True)
class Emulator:
    """JAX-traceable CosmoPower emulator.

    A JAX pytree: the weights (``w``) are its leaves, and the parameter
    names and output modes are static metadata. An ``Emulator`` can
    therefore be passed straight into ``jax.jit``/``jax.vmap`` functions.

    Attributes
    ----------
    w : EmulatorWeights
        Network weights and standardisation arrays.
    parameters : tuple[str, ...]
        Input parameter names, in training order.
    modes : np.ndarray
        Output mode labels (ell, k-index, etc.; emulator-specific), of
        length n_out after PCA decompression.
    """

    w:          EmulatorWeights = field(repr=False)
    parameters: tuple[str, ...] = ()
    _modes:     _Static = field(default=None, repr=False)

    @property
    def modes(self) -> np.ndarray:
        return None if self._modes is None else self._modes.value

    @classmethod
    def from_npz(cls, path: str) -> "Emulator":
        """Load an emulator from a CosmoPower ``*_plain.npz`` file.

        The plain format is produced by ``np.savez(**flat_dict)`` with the
        list-valued fields (``weights_``, ``biases_``, ``alphas_``, ``betas_``)
        flattened to keys like ``weights_.0, weights_.1, ..., weights_.n``.
        Loads with ``allow_pickle=False``: no tensorflow / cosmopower import.

        The arrays are created under ``jax.ensure_compile_time_eval()``, so
        they are concrete even if the first load happens inside a ``jit``
        trace (a cached emulator must never hold tracers).
        """
        d = _load_emulator_dict(path)
        f64 = jnp.float64
        with jax.ensure_compile_time_eval():
            # weights_ and biases_ are separate lists, one entry per layer.
            weights = tuple((jnp.asarray(W, dtype=f64), jnp.asarray(b, dtype=f64))
                            for W, b in zip(d["weights_"], d["biases_"]))
            # alphas_/betas_: one per hidden layer (the last layer is linear).
            alphas = tuple(jnp.asarray(a, dtype=f64) for a in d["alphas_"])
            betas = tuple(jnp.asarray(b, dtype=f64) for b in d["betas_"])
            has_pca = "pca_transform_matrix" in d
            w = EmulatorWeights(
                weights=weights, alphas=alphas, betas=betas,
                param_mean=jnp.asarray(d["param_train_mean"], dtype=f64),
                param_std=jnp.asarray(d["param_train_std"], dtype=f64),
                feature_mean=jnp.asarray(d["feature_train_mean"], dtype=f64),
                feature_std=jnp.asarray(d["feature_train_std"], dtype=f64),
                pca_matrix=jnp.asarray(d["pca_transform_matrix"], dtype=f64) if has_pca else None,
                pca_mean=jnp.asarray(d["pca_mean"], dtype=f64) if has_pca else None,
                pca_std=jnp.asarray(d["pca_std"], dtype=f64) if has_pca else None,
            )
        return cls(w=w, parameters=tuple(str(p) for p in d["parameters"]),
                   _modes=_Static(np.asarray(d["modes"])))

    def stack_inputs(self, params_dict: dict) -> jax.Array:
        """Stack ``params_dict`` into the ``(B, n_params)`` input matrix.

        Each value is a scalar or a 1-D array of length B; scalars are
        broadcast to B.
        """
        cols = [jnp.atleast_1d(jnp.asarray(params_dict[name], dtype=jnp.float64))
                for name in self.parameters]
        return jnp.stack(jnp.broadcast_arrays(*cols), axis=-1)

    def predict(self, params_dict: dict) -> jax.Array:
        """Forward pass. Returns shape ``(B, n_out)`` (always 2-D).

        ``params_dict`` maps each name in ``self.parameters`` to a scalar or
        a 1-D array of length B.
        """
        return nn_forward(self.w, self.stack_inputs(params_dict))


jax.tree_util.register_pytree_node(
    Emulator,
    lambda e: ((e.w,), (e.parameters, e._modes)),
    lambda aux, children: Emulator(w=children[0], parameters=aux[0], _modes=aux[1]),
)


# ---------------------------------------------------------------------------
# Default data-directory resolution
# ---------------------------------------------------------------------------

def default_data_dir() -> str:
    """Resolve the CosmoPower emulator data directory.

    Resolution order:
      1. ``$CLASSY_SZLITE_DATA_DIR`` environment variable
      2. ``~/class_sz_data`` (the recommended new location)
      3. ``~/class_sz_data_directory`` (legacy classy_szfast path)
    """
    env = os.environ.get("CLASSY_SZLITE_DATA_DIR")
    if env:
        return os.path.expanduser(env)
    for candidate in ("~/class_sz_data", "~/class_sz_data_directory"):
        path = os.path.expanduser(candidate)
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        "classy_szlite: cannot find CosmoPower emulator data. Set the "
        "$CLASSY_SZLITE_DATA_DIR environment variable, or place the emulator "
        "directory tree at ~/class_sz_data (or the legacy "
        "~/class_sz_data_directory). Required subdirs per cosmo_model: "
        "PK/, growth-and-distances/, TTTEEE/, derived-parameters/."
    )
