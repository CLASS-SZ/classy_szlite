"""Per-tile noise calibrated on MSS2 (noise.py; plan M6b).

Fast tests: the factorisation; the draw against a numpy reference from the same random numbers,
to float32 round-off in every multipole bin; drawn auto and cross spectra against the input with
analytic per-bin errors (z-scores, ranges of multipole, the mean ratio), for the MSS2-like noise
and for the 2019 forecast noise of simulator.channel_maps (never tested this way before); other
grid sizes and the clamp beyond the measured range; batch against single draws and the depth
scale; the host-side index checks; the loader; the sky (tSZ and CMB) unchanged by the noise model.

Slow tests (``-m slow``, with the reference data): a full tile through tile_maps with and without
the MSS2 noise (halos identical, maps differing by the noise draw only), and simulate_tile under
vmap with a different spectrum and depth per tile against single-tile runs (integer outputs
identical).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import noise as nz, simulator as simu, instrument as ins, paint as pt  # noqa: E402
from classy_szlite.sim.flatsky import N, grid  # noqa: E402

DL = 50.0
NB = int(np.ceil(21600 * np.sqrt(2) / DL))
M_DRAWS = 24
RANGES = ((100.0, 1000.0), (1000.0, 5000.0), (5000.0, 20000.0))


def synthetic_spectra(n_spec=3, seed=0, n_bins=NB, dl=DL):
    """Smooth, correlated 6 x 6 per-mode spectra with a bump, a white floor and one band whose
    crosses vanish at high ell, like the 27/39 GHz floor."""
    rng = np.random.default_rng(seed)
    ell = (np.arange(n_bins) + 0.5) * dl
    C = np.zeros((n_spec, n_bins, 6, 6))
    for s in range(n_spec):
        amp = rng.uniform(0.5, 3.0, 6)
        knee = rng.uniform(800, 3000, 6)
        corr = np.full((6, 6), 0.15) + np.eye(6) * 0.85
        corr[0, 1] = corr[1, 0] = corr[2, 3] = corr[3, 2] = corr[4, 5] = corr[5, 4] = 0.6
        red = (knee[None] / np.maximum(ell[:, None], 50.0)) ** 2.5
        bump = 1.0 + 0.8 * np.exp(-0.5 * ((ell - 15000) / 600) ** 2)
        sd = amp[None] * np.sqrt(1.0 + red) * bump[:, None]                  # (n_bins, 6)
        C[s] = sd[:, :, None] * sd[:, None, :] * corr[None]
        hi = ell >= 4500
        C[s, hi, 0, 1:] = C[s, hi, 1:, 0] = 0.0
    return C


@pytest.fixture(scope="module")
def tables():
    C = synthetic_spectra()
    return C, nz.tables_from_spectra(C, DL)


def bin_of_modes(L, dl=DL, n_bins=NB):
    b = np.minimum(np.floor(L / dl).astype(np.int64), n_bins - 1).ravel()
    b[0] = -1
    return b


def binned(X, L, dl=DL, n_bins=NB):
    """Binned per-mode power (n_bins, 6, 6) of rfft modes X (6, n, n/2+1), full-plane weights."""
    b, w = bin_of_modes(L, dl, n_bins), nz.mode_weights(L).ravel()
    ok = b >= 0
    sw = np.bincount(b[ok], weights=w[ok], minlength=n_bins)
    Xf = np.asarray(X).reshape(6, -1)[:, ok]
    n = L.shape[0]
    out = np.zeros((n_bins, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            p = np.real(Xf[i] * np.conj(Xf[j])) / n ** 2
            out[:, i, j] = out[:, j, i] = np.bincount(b[ok], weights=w[ok] * p, minlength=n_bins) / np.maximum(sw, 1e-30)
    return out


def analytic(Pk, L, M, dl=DL, n_bins=NB):
    """Expected binned spectra (n_bins, 6, 6), the variance of their M-draw mean, and the bins'
    summed full-plane weights, for per-mode power Pk (6, 6, n, n/2+1). Each rfft entry inside the
    half-plane stands for a mode and its conjugate (weight 2, variance weight 4); on the k_x = 0 and
    Nyquist columns conjugate pairs appear as two entries of weight 1, perfectly correlated
    (variance weight 2 each). Var Re(X_i X_j*)/n^2 = (P_ii P_jj + P_ij^2) / 2 per complex mode."""
    b, w = bin_of_modes(L, dl, n_bins), nz.mode_weights(L).ravel()
    c = np.where(w == 2.0, 4.0, 2.0)
    ok = b >= 0
    sw = np.bincount(b[ok], weights=w[ok], minlength=n_bins)
    sc = np.zeros((n_bins, 6, 6))
    mean = np.zeros((n_bins, 6, 6))
    P = Pk.reshape(6, 6, -1)[:, :, ok]
    for i in range(6):
        for j in range(i, 6):
            mean[:, i, j] = mean[:, j, i] = np.bincount(b[ok], weights=w[ok] * P[i, j], minlength=n_bins) / np.maximum(sw, 1e-30)
            v = (P[i, i] * P[j, j] + P[i, j] ** 2) / 2.0
            sc[:, i, j] = sc[:, j, i] = np.bincount(b[ok], weights=c[ok] * v, minlength=n_bins)
    var = sc / np.maximum(sw, 1e-30)[:, None, None] ** 2 / M
    return mean, var, sw


def assert_spectra(m, truth, var, sw, label, min_weight=50.0):
    """z-scores per (bin, entry); mode-weighted auto ratios per range of multipole; the mean auto
    ratio per bin; all against analytic errors. Entries with no variance but a nonzero input fail."""
    ell = (np.arange(NB) + 0.5) * DL
    use = (sw >= min_weight) & (ell >= 100)
    iu = np.triu_indices(6)
    t, v = truth[use][:, iu[0], iu[1]], var[use][:, iu[0], iu[1]]
    z = (m[use][:, iu[0], iu[1]] - t) / np.sqrt(np.maximum(v, 1e-300))
    keep = (t != 0) | (v > 0)
    frac = float(np.mean(np.abs(z[keep]) > 3))
    assert frac < 0.015, f"{label}: {100 * frac:.2f}% of bin entries beyond 3 sigma"
    for lo, hi in RANGES:                                            # ell 100-250 pooled into the first range
        s = (ell >= lo) & (ell < hi)
        for i in range(6):
            num = np.sum(sw[s] * m[s, i, i])
            den = np.sum(sw[s] * truth[s, i, i])
            se = np.sqrt(np.sum(sw[s] ** 2 * var[s, i, i])) / den
            assert abs(num / den - 1) < 5 * se, f"{label}: channel {i}, ell {lo:.0f}-{hi:.0f}: {num / den:.5f} +- {se:.5f}"
    for i in range(6):
        r = m[use, i, i] / truth[use, i, i]
        se = np.sqrt(np.sum(var[use, i, i] / truth[use, i, i] ** 2)) / use.sum()
        assert abs(r.mean() - 1) < 5 * se, f"{label}: channel {i}, mean ratio {r.mean():.5f} +- {se:.5f}"


def test_factors_reproduce_spectra():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 6, 6))
    C = A @ np.swapaxes(A, -1, -2)                                   # positive definite
    C[10, 0, :] = C[10, :, 0] = 0.0                                  # a zero row
    F = nz.factors(C)
    np.testing.assert_allclose(F @ np.swapaxes(F, -1, -2), C, rtol=1e-10, atol=1e-10)
    w, V = np.linalg.eigh(C[3])                                      # a slightly negative eigenvalue is clipped
    Fn = nz.factors((V * np.r_[-1e-3, w[1:]]) @ V.T)
    np.testing.assert_allclose(Fn @ Fn.T, (V * np.r_[0.0, w[1:]]) @ V.T, rtol=1e-10, atol=1e-10)


def test_mode_bins_and_weights():
    G = grid()
    b = nz.mode_bins(DL, NB)
    assert b[0, 0] == NB and b.max() == NB                           # the mean: the zero bin
    assert np.all(b[1:] <= NB - 1) and np.all(b[0, 1:] <= NB - 1)
    assert b[5, 7] == min(int(np.floor(G.L[5, 7] / DL)), NB - 1)
    np.testing.assert_array_equal(nz.mode_weights(G.L), G.W)          # flatsky's full-plane weights


def test_draw_matches_numpy_reference(tables):
    """The JAX draw equals a float64 numpy application of the same factors to the same random
    numbers, to float32 round-off in every multipole bin (not only relative to the largest,
    low-ell modes)."""
    C, nt = tables
    key = jax.random.key(7)
    X = np.asarray(jax.jit(nz.draw)(key, nt, jnp.int32(2), jnp.float32(1.0)))
    W = np.fft.rfft2(np.asarray(jax.random.normal(key, (6, N, N), dtype=jnp.float32), dtype=np.float64))
    F = np.asarray(nt.F[2], dtype=np.float64)
    b = np.asarray(nt.bin_idx)
    ref = np.einsum("...cd,d...->c...", F[b], W)
    bb = b.ravel()
    for c in range(6):
        err = np.zeros(NB + 1)
        np.maximum.at(err, bb, np.abs(X[c] - ref[c]).ravel())
        cnt = np.bincount(bb, minlength=NB + 1)
        rms = np.sqrt(np.bincount(bb, weights=np.abs(ref[c]).ravel() ** 2, minlength=NB + 1) / np.maximum(cnt, 1))
        ok = (cnt > 0) & (rms > 0)
        assert np.all(err[ok] <= 1e-5 * rms[ok]), (c, float(np.max(err[ok] / rms[ok])))
    assert np.all(X[:, 0, 0] == 0)                                    # no noise in the mean


def test_drawn_spectra_match_input(tables):
    C, nt = tables
    G = grid()
    f = jax.jit(nz.draw)
    m = np.mean([binned(np.asarray(f(jax.random.key(100 + r), nt, jnp.int32(1), jnp.float32(1.0))), G.L)
                 for r in range(M_DRAWS)], axis=0)
    Cz = np.concatenate([C[1], np.zeros((1, 6, 6))])                 # the zero bin of the mean
    Pk = np.moveaxis(Cz[nz.mode_bins(DL, NB)], (2, 3), (0, 1))        # (6, 6, N, N/2+1)
    truth, var, sw = analytic(Pk, G.L, M_DRAWS)
    assert_spectra(m, truth, var, sw, "MSS2-like")


def test_forecast_noise_spectra():
    """simulator.channel_maps's 2019 forecast noise (white plus atmospheric, correlated within a
    tube): drawn spectra against noise.forecast_spectra, with the same analytic errors."""
    G = grid()
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(ins.cmb_cl_emulator(), tb, mode="baseline")
    st = st._replace(cmb_amp=jnp.zeros_like(st.cmb_amp))              # noise only
    yk = jnp.zeros((N, N // 2 + 1), dtype=jnp.complex64)
    f = jax.jit(lambda k: simu.channel_maps(k, yk, st))
    m = np.mean([binned(np.asarray(f(jax.random.key(300 + r))), G.L) for r in range(M_DRAWS)], axis=0)
    wp, A = np.asarray(st.white_pix, dtype=np.float64), np.asarray(st.atm_amp, dtype=np.float64)
    rho = np.eye(6)
    for i, j in ins.SO_LAT["tubes"]:
        rho[i, j] = rho[j, i] = ins.SO_LAT["atm_corr"]
    Pk = rho[:, :, None, None] * A[:, None] * A[None, :] + np.diag(wp ** 2)[:, :, None, None]
    truth, var, sw = analytic(Pk, G.L, M_DRAWS)
    np.testing.assert_allclose(nz.forecast_spectra(wp, A, ins.SO_LAT["tubes"], ins.SO_LAT["atm_corr"], DL, NB, G.L),
                               truth, rtol=1e-10, atol=0)
    assert_spectra(m, truth, var, sw, "2019 forecast")


def test_other_grid_and_clamp():
    """On a 256^2 grid with fewer bins than its corner needs, the draw has that grid's shape and
    the modes beyond the last bin carry the last bin's spectrum."""
    G = grid(n=256)
    dl, nb = 400.0, 20                                                # the corner needs 77 bins of 400
    C = synthetic_spectra(n_spec=1, n_bins=nb, dl=dl)
    nt = nz.tables_from_spectra(C, dl, G.L)
    f = jax.jit(nz.draw)
    X = np.stack([np.asarray(f(jax.random.key(500 + r), nt, jnp.int32(0), jnp.float32(1.0))) for r in range(64)])
    assert X.shape == (64, 6, 256, 129)
    beyond = G.L >= nb * dl
    n_eff = 64 * beyond.sum() / 2.0
    for i in range(6):
        p = np.mean(np.abs(X[:, i][:, beyond]) ** 2) / 256 ** 2
        assert abs(p / C[0, -1, i, i] - 1) < 5 / np.sqrt(n_eff), (i, p / C[0, -1, i, i])


def test_batch_matches_single_and_depth(tables):
    """vmap over tiles with different spectra and depths equals the single draws to float32
    round-off (XLA's CPU FFT is not bit-for-bit batch-invariant; the integer data vector's batch
    invariance is the slow test below, and bitwise on the GPU), and the depth scales the draw exactly."""
    C, nt = tables
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(5), i))(jnp.arange(3, dtype=jnp.uint32))
    ni = nz.noise_input(nt, np.array([2, 0, 1]), np.array([1.0, 0.5, 2.0]))
    f = jax.jit(nz.draw)
    batch = np.asarray(jax.jit(jax.vmap(nz.draw, in_axes=(0, None, 0, 0)))(keys, nt, ni.spec, ni.depth))
    for i in range(3):
        one = np.asarray(f(keys[i], nt, ni.spec[i], ni.depth[i]))
        np.testing.assert_allclose(batch[i], one, rtol=0, atol=2e-6 * np.abs(one).max())
    a = np.asarray(f(keys[0], nt, ni.spec[0], jnp.float32(1.0)))
    b = np.asarray(f(keys[0], nt, ni.spec[0], jnp.float32(2.0)))
    np.testing.assert_array_equal(b, np.float32(2.0) * a)


def test_host_checks(tables):
    C, nt = tables
    for bad in (-1, 3, np.array([0, 3]), 1.5):
        with pytest.raises(ValueError):
            nz.noise_input(nt, bad)
    info = dict(tile=[3, 9, 12])
    np.testing.assert_array_equal(nz.spectrum_rows(info, [12, 3]), [2, 0])
    with pytest.raises(KeyError):
        nz.spectrum_rows(info, [4])


def test_noise_model_leaves_the_sky_unchanged(tables):
    """The tSZ signal and the CMB use the same keys and terms whichever noise model is used: with
    both noise models off the maps agree mode by mode (also the tSZ-only modes, with the CMB off),
    and with the MSS2 noise on, the maps minus its draw are the noise-free maps."""
    C, nt = tables
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(ins.cmb_cl_emulator(), tb)
    rng = np.random.default_rng(3)
    yk = jnp.asarray((rng.normal(size=(N, N // 2 + 1)) + 1j * rng.normal(size=(N, N // 2 + 1))).astype(np.complex64) * 1e-3)
    st0 = st._replace(white_pix=jnp.zeros_like(st.white_pix), atm_amp=jnp.zeros_like(st.atm_amp))
    zero = nz.noise_input(nt._replace(F=jnp.zeros_like(nt.F)), 0)
    key = jax.random.key(11)
    for st_ in (st0, st0._replace(cmb_amp=jnp.zeros_like(st0.cmb_amp))):    # with and without the CMB
        a = np.asarray(jax.jit(lambda k: simu.channel_maps(k, yk, st_))(key))
        b = np.asarray(jax.jit(lambda k: simu.channel_maps(k, yk, st_, zero))(key))
        assert np.abs(a).max() > 0
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-9 * np.abs(a).max())
    real = nz.noise_input(nt, 1)
    a = np.asarray(jax.jit(lambda k: simu.channel_maps(k, yk, st0))(key))
    c = np.asarray(jax.jit(lambda k: simu.channel_maps(k, yk, st, real))(key))
    k_w = jax.random.split(key, 3)[1]
    n = np.asarray(jax.jit(nz.draw)(k_w, nt, real.spec, real.depth))
    np.testing.assert_allclose(c - n, a, rtol=0, atol=2e-6 * np.abs(c).max())


def test_load_mss2(tmp_path):
    """The loader picks the set, keeps the tiles' order and info, and builds matching tables."""
    C = synthetic_spectra(n_spec=2)
    sets = ["mean", "split0", "split1", "split2", "split3"]
    Cs = np.stack([C * (1.0 + 0.1 * k) for k in range(5)], axis=1)          # (2, 5, NB, 6, 6)
    p = tmp_path / "spectra.npz"
    np.savez(p, C=Cs.astype(np.float32), index=np.array([3, 9]), ra=np.array([10.0, 20.0]),
             dec=np.array([-35.0, 5.0]), white=np.ones((2, 5, 6)),
             meta=json.dumps(dict(sets=sets, bands=["f030", "f040", "f090", "f150", "f230", "f290"], dl=DL)))
    nt, info = nz.load_mss2(p, "split1")
    assert nt.F.shape == (2, NB + 1, 6, 6) and info["tile"] == [3, 9] and info["set"] == "split1"
    F = np.asarray(nt.F[1, :NB], dtype=np.float64)
    np.testing.assert_allclose(F @ np.swapaxes(F, -1, -2), Cs[1, 2], rtol=2e-5, atol=1e-6 * Cs[1, 2].max())


# ---- full tiles (slow, reference data) ----------------------------------------------------

@pytest.fixture(scope="module")
def full_tile():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "mapsim"))
    import spike
    cells, tb, cs, cfg, st, info = spike.setup(simu.THETA_FID)
    return cells, tb, cs, cfg, st, nz.tables_from_spectra(synthetic_spectra(), DL)


@pytest.mark.slow
@pytest.mark.refdata
def test_tile_maps_with_mss2_noise(full_tile):
    """tile_maps with the MSS2 noise: the halos are those of the forecast-noise run, and the maps
    minus the noise draw equal the noise-free maps."""
    cells, tb, cs, cfg, st, nt = full_tile
    th = simu.THETA_FID
    key = jax.random.key(21)
    st0 = st._replace(white_pix=jnp.zeros_like(st.white_pix), atm_amp=jnp.zeros_like(st.atm_amp))
    run = jax.jit(lambda k, s, noise: simu.tile_maps(k, cs, th, s, tb, cells, cfg, noise))
    h0, X0, _ = run(key, st0, None)
    h1, X1, _ = run(key, st, nz.noise_input(nt, 1, 1.5))
    for f in h0._fields:
        np.testing.assert_array_equal(np.asarray(getattr(h0, f)), np.asarray(getattr(h1, f)), err_msg=f)
    k_m = jax.random.split(key)[1]
    k_w = jax.random.split(k_m, 3)[1]
    n = np.asarray(jax.jit(nz.draw)(k_w, nt, jnp.int32(1), jnp.float32(1.5)))
    X0, X1 = np.asarray(X0), np.asarray(X1)
    np.testing.assert_allclose(X1 - n, X0, rtol=0, atol=2e-6 * np.abs(X1).max())


@pytest.mark.slow
@pytest.mark.refdata
def test_simulate_tile_batch_with_mss2_noise(full_tile):
    """simulate_tile under vmap with a different spectrum and depth per tile gives the same
    integer outputs as single-tile runs."""
    cells, tb, cs, cfg, st, nt = full_tile
    th = simu.THETA_FID
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(31), i))(jnp.arange(3, dtype=jnp.uint32))
    ni = nz.noise_input(nt, np.array([0, 2, 1]), np.array([1.0, 1.3, 0.8]))

    def one(k, s, d):
        return simu.simulate_tile(k, cs, th, st, tb, cells, cfg, nz.NoiseInput(nt, s, d))
    many = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))
    single = jax.jit(one)
    ob = jax.device_get(many(keys, ni.spec, ni.depth))
    for i in range(3):
        o1 = jax.device_get(single(keys[i], ni.spec[i], ni.depth[i]))
        for k in ("counts", "n_halos", "n_peaks", "n_kept", "n_conf"):
            np.testing.assert_array_equal(np.asarray(ob[k])[i], np.asarray(o1[k]), err_msg=f"tile {i}, {k}")
