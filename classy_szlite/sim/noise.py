"""Per-tile noise calibrated on SO's MSS2 noise simulation (plan M6b, section 3.5).

Each tile's noise is stationary, isotropic and Gaussian, with binned 6 x 6 auto and cross spectra
measured from MSS2 by ``benchmarks/mapsim/mss2_spectra.py``: true sky multipoles, per-mode power on
the simulator's square 0.5' pixels (a white field of pixel variance s^2 has per-mode power s^2),
bins of width ``dl`` in multipole. One draw, in the simulator's rfft units
(X(k) = sum_p x(p) e^{-i k.p}, so E|X_c(k)|^2 = N^2 depth^2 C_cc):

    X_c(k) = depth * sum_d F[s, b(k), c, d] W_d(k),    W_d = rfft2(unit white field d),

with F F^T = C[s, b] for every bin b (from an eigendecomposition, negative eigenvalues of a
noisy estimate clipped to 0), b(k) the bin of mode k's multipole on the simulator's grid, s the
tile's spectrum index and depth a per-tile amplitude scale (1 = the measured level; year 1 or
another depth is one number per tile). The mean mode gets no noise, and modes beyond the measured
range take the last bin.

The 2019 SO forecast model (white plus atmospheric noise, correlated within a tube) stays in
``simulator.channel_maps``; :func:`forecast_spectra` gives its per-mode spectra binned the same
way, for comparisons.

Not modelled (plan section 3.5): depth changes within a tile, 2D anisotropy (both changed the
counts by < 0.2% in the MSS2 tests) and MSS2's streaks (about 1% more detections at one-split
depth; a decision pending).

Grids: the draw works on any n x n grid whose bin map was built from that grid's multipoles
(``tables_from_spectra(..., L=grid(n).L)``). The per-mode power assumes 0.5' pixels; on pixels of
side p it scales by (0.5' / p)^2. Spectrum indices are checked on the host (:func:`noise_input`,
:func:`spectrum_rows`): under jit an out-of-range index would silently take the last spectrum.
"""
from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .flatsky import grid


class NoiseTables(NamedTuple):
    """Static noise arrays (jit arguments)."""
    F: jax.Array          # (n_spec, n_bins + 1, 6, 6) f32: factors, F F^T = C; bin n_bins is zero (the mean)
    bin_idx: jax.Array    # (n, n/2+1) int32: the bin of each mode of the grid (flatsky's n = 1200 by default)


class NoiseInput(NamedTuple):
    """One tile's choice: spectrum index and amplitude scale (both traced; vmap over tiles)."""
    tables: NoiseTables
    spec: jax.Array       # () int32
    depth: jax.Array      # () f32


def factors(C):
    """F with F F^T = C for symmetric C (..., 6, 6), from an eigendecomposition with negative
    eigenvalues clipped to 0 (float64)."""
    C = 0.5 * (np.asarray(C, dtype=np.float64) + np.swapaxes(np.asarray(C, dtype=np.float64), -1, -2))
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None))[..., None, :]


def mode_weights(L):
    """Full-plane weight of each rfft half-plane mode of an n x n grid (L: (n, n//2+1)): 2 inside,
    1 on the k_x = 0 column and, for even n, on the Nyquist column."""
    w = np.full(L.shape, 2.0)
    w[:, 0] = 1.0
    if L.shape[0] % 2 == 0:
        w[:, -1] = 1.0
    return w


def mode_bins(dl, n_bins, L=None):
    """Bin of each mode of the simulator's grid (multipole L, default flatsky's), clamped to the
    last measured bin; the mean mode goes to the zero bin n_bins."""
    L = grid().L if L is None else L
    b = np.minimum(np.floor(L / dl).astype(np.int64), n_bins - 1)
    b[0, 0] = n_bins
    return b.astype(np.int32)


def tables_from_spectra(C, dl, L=None):
    """NoiseTables from spectra C (n_spec, n_bins, 6, 6): per-mode power on square 0.5' pixels in
    bins of width dl in multipole, for the grid with multipoles L (default: flatsky's)."""
    C = np.asarray(C, dtype=np.float64)
    n_spec, n_bins = C.shape[:2]
    F = np.zeros((n_spec, n_bins + 1, 6, 6))
    F[:, :n_bins] = factors(C)
    return NoiseTables(F=jnp.asarray(F.astype(np.float32)), bin_idx=jnp.asarray(mode_bins(dl, n_bins, L)))


def noise_input(nt: NoiseTables, spec, depth=1.0):
    """NoiseInput for one tile (scalars) or a batch (arrays), with the spectrum indices checked."""
    s = np.asarray(spec)
    if not np.issubdtype(s.dtype, np.integer) or np.any(s < 0) or np.any(s >= nt.F.shape[0]):
        raise ValueError(f"spectrum indices must be integers in [0, {nt.F.shape[0]}): {s}")
    return NoiseInput(tables=nt, spec=jnp.asarray(s, dtype=jnp.int32), depth=jnp.asarray(depth, dtype=jnp.float32))


def spectrum_rows(info, tile_ids):
    """Rows of the loaded tables for MSS2 tile numbers (``info`` from :func:`load_mss2`)."""
    row = {t: s for s, t in enumerate(info["tile"])}
    missing = [t for t in np.atleast_1d(tile_ids).tolist() if t not in row]
    if missing:
        raise KeyError(f"MSS2 tiles not in the loaded spectra: {missing}")
    return np.array([row[t] for t in np.atleast_1d(tile_ids).tolist()], dtype=np.int32)


def load_mss2(path, set_name="mean", L=None):
    """NoiseTables for every tile of ``mss2_spectra.py``'s spectra.npz, for one set ("mean", the
    plain mean of the four splits, or "split0".."split3"), and the tiles' info: spectrum index s
    -> MSS2 tile number, RA and dec of its centre [deg], white levels [uK-arcmin]."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    k = meta["sets"].index(set_name)
    nt = tables_from_spectra(d["C"][:, k], float(meta["dl"]), L)
    info = dict(tile=d["index"].tolist(), ra=d["ra"].tolist(), dec=d["dec"].tolist(),
                white=d["white"][:, k].tolist(), bands=meta["bands"], set=set_name)
    return nt, info


def forecast_spectra(white_pix, atm_amp, tubes, r, dl, n_bins, L=None):
    """The 2019 forecast model's per-mode spectra (n_bins, 6, 6), binned like the MSS2 ones: white
    power white_pix^2 on the diagonal plus atmospheric power atm_amp_i atm_amp_j (r within a
    tube, 1 on the diagonal), averaged over each bin's modes of the full plane (rfft columns inside
    the half-plane count twice; atm_amp: (6, N, N/2+1))."""
    L = grid().L if L is None else L
    b = np.minimum(np.floor(L / dl).astype(np.int64), n_bins - 1).ravel()
    b[0] = -1
    ok = b >= 0
    w = mode_weights(L).ravel()[ok]
    cnt = np.bincount(b[ok], weights=w, minlength=n_bins)
    A = np.asarray(atm_amp, dtype=np.float64).reshape(6, -1)[:, ok] * np.sqrt(w)[None]
    rho = np.eye(6)
    for i, j in tubes:
        rho[i, j] = rho[j, i] = r
    C = np.zeros((n_bins, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            if rho[i, j]:
                C[:, i, j] = C[:, j, i] = rho[i, j] * np.bincount(b[ok], weights=A[i] * A[j], minlength=n_bins) / np.maximum(cnt, 1)
    C[:, range(6), range(6)] += np.asarray(white_pix, dtype=np.float64)[None] ** 2
    return C


def draw(key, nt: NoiseTables, spec, depth=1.0):
    """(6, n, n/2+1) complex64 noise modes of one tile with spectrum index ``spec`` (n from the
    tables' bin map)."""
    f32 = jnp.float32
    n = nt.bin_idx.shape[0]
    assert nt.bin_idx.shape == (n, n // 2 + 1), nt.bin_idx.shape
    W = jnp.fft.rfft2(jax.random.normal(key, (6, n, n), dtype=f32))
    Fs = nt.F[spec]                                           # (n_bins + 1, 6, 6)
    out = []
    for c in range(6):
        acc = Fs[:, c, 0][nt.bin_idx] * W[0]
        for d in range(1, 6):
            acc = acc + Fs[:, c, d][nt.bin_idx] * W[d]
        out.append(acc)
    return jnp.asarray(depth, dtype=f32) * jnp.stack(out)
