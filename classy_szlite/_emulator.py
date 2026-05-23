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
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


@dataclass(frozen=True)
class Emulator:
    """JAX-traceable CosmoPower emulator.

    Hold the static weights as jnp arrays; ``predict(params_dict)`` runs
    the forward pass and returns the output (already de-standardised, and
    PCA-decompressed if applicable).

    Attributes
    ----------
    parameters : list[str]
        Input parameter names, in training order.
    modes : np.ndarray
        Output mode labels (ell, k-index, etc. — interpretation is
        emulator-specific). Length = output dimension AFTER PCA decompression.
    is_log : bool
        If True, the emulator output is log10 of the physical quantity.
        Caller can apply ``10**`` if needed; we leave that to the caller
        because the convention varies (e.g. TE doesn't apply log10 since it
        can go negative).
    """

    # JAX-side state
    weights:        tuple = field(repr=False)             # tuple of (W, b) per layer
    alphas:         tuple = field(repr=False)             # α per hidden layer
    betas:          tuple = field(repr=False)             # β per hidden layer
    param_mean:     jnp.ndarray = field(repr=False)
    param_std:      jnp.ndarray = field(repr=False)
    feature_mean:   jnp.ndarray = field(repr=False)
    feature_std:    jnp.ndarray = field(repr=False)
    pca_matrix:     jnp.ndarray | None = field(repr=False, default=None)
    pca_mean:       jnp.ndarray | None = field(repr=False, default=None)
    pca_std:        jnp.ndarray | None = field(repr=False, default=None)

    # Metadata
    parameters:     tuple[str, ...] = ()
    modes:          np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_npz(cls, path: str) -> "Emulator":
        """Load an emulator from a CosmoPower-style ``.npz`` file."""
        raw = np.load(path, allow_pickle=True)
        # CosmoPower saves a single object array containing a dict-like
        d = raw["arr_0"].item()

        # weights_ and biases_ are SEPARATE lists (one entry per layer);
        # pair them up so the forward pass can iterate over (W, b) tuples.
        weights = tuple(
            (jnp.asarray(W, dtype=jnp.float64),
             jnp.asarray(b, dtype=jnp.float64))
            for W, b in zip(d["weights_"], d["biases_"])
        )
        # alphas_ / betas_: per hidden layer (one entry per hidden layer,
        # i.e. len = n_layers - 1; final linear layer has no α-β activation)
        alphas = tuple(jnp.asarray(a, dtype=jnp.float64) for a in d["alphas_"])
        betas  = tuple(jnp.asarray(b, dtype=jnp.float64) for b in d["betas_"])

        param_mean   = jnp.asarray(d["param_train_mean"],   dtype=jnp.float64)
        param_std    = jnp.asarray(d["param_train_std"],    dtype=jnp.float64)
        feature_mean = jnp.asarray(d["feature_train_mean"], dtype=jnp.float64)
        feature_std  = jnp.asarray(d["feature_train_std"],  dtype=jnp.float64)

        # PCA fields are present only for PCA emulators (e.g. TE)
        has_pca = "pca_transform_matrix" in d
        pca_matrix = jnp.asarray(d["pca_transform_matrix"], dtype=jnp.float64) if has_pca else None
        pca_mean   = jnp.asarray(d["pca_mean"],   dtype=jnp.float64) if has_pca else None
        pca_std    = jnp.asarray(d["pca_std"],    dtype=jnp.float64) if has_pca else None

        parameters = tuple(d["parameters"])
        modes      = np.asarray(d["modes"])

        return cls(
            weights=weights, alphas=alphas, betas=betas,
            param_mean=param_mean, param_std=param_std,
            feature_mean=feature_mean, feature_std=feature_std,
            pca_matrix=pca_matrix, pca_mean=pca_mean, pca_std=pca_std,
            parameters=parameters, modes=modes,
        )

    def predict(self, params_dict: dict) -> jnp.ndarray:
        """Forward pass on a single or batched input.

        ``params_dict``: maps each ``self.parameters`` name to a 1-D array-like
        of length B (batch size); the output has shape ``(B, n_modes)`` (or
        ``(n_modes,)`` if B=1).
        """
        # Stack inputs in the canonical order
        p = jnp.stack([jnp.asarray(params_dict[name], dtype=jnp.float64)
                       for name in self.parameters], axis=-1)   # (B, n_params)
        # Standardise
        h = (p - self.param_mean) / self.param_std

        # Hidden layers with α-β sigmoid activation.
        # CosmoPower .npz weights are stored as (in_dim, out_dim) so the
        # matmul is `h @ W` (no transpose).
        for (W, b), a, beta in zip(self.weights[:-1], self.alphas, self.betas):
            z = h @ W + b
            h = (beta + _sigmoid(a * z) * (1.0 - beta)) * z

        # Final linear layer (no activation)
        W, b = self.weights[-1]
        preds = h @ W + b                                            # (B, n_out)

        # PCA decompression (if applicable)
        if self.pca_matrix is not None:
            # PCA-encoded: undo PCA standardisation, project back, then
            # undo feature standardisation
            preds = preds * self.pca_std + self.pca_mean             # (B, n_pca)
            preds = preds @ self.pca_matrix                          # (B, n_modes)

        # Undo feature standardisation
        preds = preds * self.feature_std + self.feature_mean
        return preds.squeeze()


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
