"""Masking iteration of the matched filter (mmf.cluster_mask, mmf.masked_cov).

* The FFT-built disc mask equals a direct "distance ≤ r" construction.
* With nothing masked, the whitened estimate returns the input covariance.
* On CMB + noise tiles (no halos) with random discs masked, σ_j from the
  masked estimate agrees with the unmasked estimate and the analytic σ_j.
* Against szsim (sz-poisson-blending), on one simulated tile with halos: the
  same mask and the same masked covariance (reference dumped by
  benchmarks/mapsim/refgen_szsim_mask.py; skipped if absent).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import flatsky as fsk  # noqa: E402
from classy_szlite.sim import mmf, paint as pt, instrument as ins, simulator as simu  # noqa: E402
from classy_szlite.sim.flatsky import N, PIX  # noqa: E402

REFDATA = os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending")
MASK_REF = os.path.join(os.environ.get("MAPSIM_REF", "/mnt/ceph/users/bbolliet/mapsim/ref"),
                        "szsim_mask.npz")


def direct_disc_mask(j, iy, ix, thetas=mmf.THETAS15, radius=mmf.MASK_RADIUS):
    """1 = kept; 0 where dy² + dx² ≤ (radius θ_j / PIX)² for some peak (periodic)."""
    m = np.ones((N, N), np.float32)
    for jj, y, x in zip(j, iy, ix):
        R = radius * float(thetas[jj]) / PIX
        Ri = int(np.ceil(R))
        dy, dx = np.meshgrid(np.arange(-Ri, Ri + 1), np.arange(-Ri, Ri + 1), indexing="ij")
        inside = dy * dy + dx * dx <= R * R
        m[(y + dy[inside]) % N, (x + dx[inside]) % N] = 0.0
    return m


def make_peaks(j, iy, ix, q, capacity):
    n = len(j)
    pad = lambda a, dt, fill: jnp.asarray(np.concatenate([np.asarray(a), np.full(capacity - n, fill)]).astype(dt))
    return mmf.Peaks(j=pad(j, np.int32, 0), iy=pad(iy, np.int32, 0), ix=pad(ix, np.int32, 0),
                     q=pad(q, np.float32, -np.inf), valid=jnp.arange(capacity) < n,
                     count=jnp.int32(n), overflow=jnp.bool_(False))


@pytest.fixture(scope="module")
def fs():
    return mmf.filter_static()


@pytest.fixture(scope="module")
def noise_setup(fs):
    path = os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz")
    cmb = ins.cmb_cl_from_table(path) if os.path.exists(path) else ins.cmb_cl_emulator()
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    G = fsk.grid()
    out = {}
    for mode in ("goal", "baseline"):
        st = simu.tile_static(cmb, tb, mode=mode)
        B = ins.beams(G.m_L)
        C = ins.noise_cov(G.m_L, mode) + B[:, None, :] * B[None, :, :] * cmb(G.m_L)[None, None, :]
        C *= G.Npx / G.Om
        Cb = np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[i, j], minlength=G.n_bins)
                                 / G.bin_count for j in range(6)], -1) for i in range(6)], 1)
        out[mode] = (st, jnp.asarray(Cb))
    return out


def test_disc_kernels_are_real_and_integer():
    K = mmf.disc_kernels()
    for j in (0, 7, 14):
        R = mmf.MASK_RADIUS * mmf.THETAS15[j] / PIX
        disc = np.fft.irfft2(K[j], s=(N, N))
        np.testing.assert_allclose(disc, np.round(disc), atol=1e-9)
        d = np.fft.fftfreq(N, 1.0 / N)
        assert int(np.round(disc.sum())) == int(((d[:, None] ** 2 + d[None, :] ** 2) <= R * R).sum())


def test_cluster_mask_matches_direct(fs):
    """Random peaks over all scales, some at the tile edges (periodic), some
    below q_mask, and padding slots: identical to the direct construction."""
    rng = np.random.default_rng(7)
    n = 300
    j = rng.integers(0, 15, n)
    iy = rng.integers(0, N, n)
    ix = rng.integers(0, N, n)
    iy[:10], ix[10:20] = rng.integers(0, 3, 10), N - 1 - rng.integers(0, 3, 10)
    q = rng.uniform(3.0, 30.0, n)
    pk = make_peaks(j, iy, ix, q, 512)
    mask, n_disc = jax.jit(lambda p: mmf.cluster_mask(p, fs))(pk)
    sel = q > mmf.Q_MASK
    ref = direct_disc_mask(j[sel], iy[sel], ix[sel])
    assert int(n_disc) == int(sel.sum())
    np.testing.assert_array_equal(np.asarray(mask), ref)
    print(f"masked fraction {1 - ref.mean():.3f} from {sel.sum()} discs")


def test_cluster_mask_empty(fs):
    pk = make_peaks([], [], [], [], 64)
    mask, n_disc = mmf.cluster_mask(pk, fs)
    assert int(n_disc) == 0 and float(np.asarray(mask).min()) == 1.0


def _noise_tile(st, seed):
    return jax.jit(lambda k: simu.channel_maps(k, jnp.zeros(fsk.grid().L.shape, jnp.complex64), st))(
        jax.random.key(seed))


def _sigma(C, fs):
    _, aCa = mmf.weights(C, fs)
    return np.asarray(1.0 / jnp.sqrt(jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)))


def test_masked_cov_no_mask_is_identity(fs, noise_setup):
    st, _ = noise_setup["goal"]
    X = _noise_tile(st, 11)
    C0 = mmf.binned_cov(X.reshape(6, -1)[:, fs.m_idx], fs)
    C1, f = jax.jit(lambda X, C0: mmf.masked_cov(X, jnp.ones((N, N), jnp.float32), C0, fs))(X, C0)
    assert float(f) == 1.0
    Wm, _ = mmf.whitening(C0)
    D = np.asarray(jnp.einsum("bij,bjk,bkl->bil", Wm, C1 - C0, Wm))   # in units of C0
    print(f"max |C0^-1/2 (C1 - C0) C0^-1/2| = {np.abs(D).max():.2e}")
    assert np.abs(D).max() < 1e-5


@pytest.mark.parametrize("mode", ["goal", "baseline"])
def test_masked_cov_noise_only(fs, noise_setup, mode):
    """No halos, ~60 random discs masked (masked fraction ~15%): σ_j from the
    whitened masked estimate agrees with the unmasked estimate and with the
    analytic covariance's σ_j."""
    st, Cb = noise_setup[mode]
    rng = np.random.default_rng(5 if mode == "goal" else 6)
    n = 60
    pk = make_peaks(rng.integers(0, 15, n), rng.integers(0, N, n), rng.integers(0, N, n),
                    np.full(n, 10.0), 64)
    sig_an = _sigma(Cb, fs)
    r0, r1 = [], []
    for seed in (21, 22):
        X = _noise_tile(st, seed)
        C0 = mmf.binned_cov(X.reshape(6, -1)[:, fs.m_idx], fs)
        mask, _ = mmf.cluster_mask(pk, fs)
        C1, f = jax.jit(lambda X, m, C0: mmf.masked_cov(X, m, C0, fs))(X, mask, C0)
        r0.append(_sigma(C0, fs) / sig_an)
        r1.append(_sigma(C1, fs) / sig_an)
    r0, r1 = np.array(r0), np.array(r1)
    print(f"{mode}: masked fraction {1 - float(f):.3f}")
    print("  σ(unmasked est.)/σ_an:", np.round(r0.mean(0), 4))
    print("  σ(masked est.)/σ_an:  ", np.round(r1.mean(0), 4))
    assert 0.05 < 1 - float(f) < 0.3
    assert np.all(np.abs(r1 / r0 - 1) < 0.01), r1 / r0
    assert np.all(np.abs(r1 - 1) < 0.02), r1


@pytest.mark.skipif(not os.path.exists(MASK_REF), reason="szsim mask reference not generated")
def test_masked_cov_matches_szsim(fs):
    """One fiducial SO-goal tile with halos and its first-pass peaks: szsim's
    disc_mask and masked_cov(method='whitened') in float64 against ours
    (float32 maps and FFTs)."""
    ref = np.load(MASK_REF)
    np.testing.assert_allclose(ref["thetas"], mmf.THETAS15, rtol=1e-15)
    X = jnp.asarray(ref["X"].astype(np.complex64))
    pk = make_peaks(ref["j"], ref["iy"], ref["ix"], ref["q"], int(ref["j"].size) + 64)
    mask, n_disc = mmf.cluster_mask(pk, fs)
    np.testing.assert_array_equal(np.asarray(mask), ref["mask"])
    G = fsk.grid()
    C0 = mmf.binned_cov(X.reshape(6, -1)[:, fs.m_idx], fs)
    C1, f = mmf.masked_cov(X, mask, C0, fs)
    assert abs(float(f) - float(ref["f"])) < 1e-15
    C0r, C1r = ref["C0"][G.bins_used], ref["C1"][G.bins_used]
    Wm, _ = mmf.whitening(jnp.asarray(C0r))
    for name, A, B in (("C0", C0, C0r), ("C1", C1, C1r)):
        D = np.asarray(jnp.einsum("bij,bjk,bkl->bil", Wm, jnp.asarray(A) - B, Wm))
        s = _sigma(jnp.asarray(A), fs) / _sigma(jnp.asarray(B), fs) - 1
        print(f"{name}: max whitened difference {np.abs(D).max():.2e}, max |Δσ/σ| {np.abs(s).max():.2e}")
        assert np.abs(D).max() < 1e-5
        assert np.abs(s).max() < 1e-6
