"""The matched filter on a subset of the SO channels (mmf.filter_static(channels=...)).

* The default equals channels=range(6) (the same static arrays and S/N cube), and its channel
  vector is the committed construction. Bit-identity of the six-channel finder with the code
  before channel subsets is checked on real maps by benchmarks/mapsim/szifi_compare2.sh (a run of
  the earlier snapshot in the same job).
* Exact synthesis on a subset: a noise-free cluster synthesised in the subset's channels gives
  y0_hat = y0 at its position and scale, with the analytic covariance restricted to the subset.
* With fewer channels the analytic sigma_j can only grow (the full set's filter is optimal:
  per mode a^T C^-1 a over a superset is at least the subset's).
* On CMB + noise tiles (no halos), the subset's q maps from the estimated covariance have std
  ~ 1 at every scale (the estimated covariance is consistent with the maps' cross-spectra), and
  sigma_j agrees with the analytic one to 5%.
* The whitened masked covariance with nothing masked returns the input covariance.
* Maps whose channel count differs from the filter's, and repeated, empty or out-of-range
  channel lists, raise an error.
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
from classy_szlite.sim.flatsky import N  # noqa: E402

REFDATA = os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending")
ALL = (0, 1, 2, 3, 4, 5)
HF = (2, 3, 4, 5)                  # 93, 145, 225, 280 GHz
SUBSETS = [HF, (3,)]


@pytest.fixture(scope="module")
def setup():
    path = os.path.join(REFDATA, "data_real", "cmb_lensed_TT_camb.npz")
    cmb = ins.cmb_cl_from_table(path) if os.path.exists(path) else ins.cmb_cl_emulator()
    G = fsk.grid()
    B = ins.beams(G.m_L)
    C = ins.noise_cov(G.m_L, "goal") + B[:, None, :] * B[None, :, :] * cmb(G.m_L)[None, None, :]
    C *= G.Npx / G.Om
    Cb = np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[i, j], minlength=G.n_bins)
                             / G.bin_count for j in range(6)], -1) for i in range(6)], 1)
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(cmb, tb)
    fss = {ch: mmf.filter_static(channels=ch) for ch in [ALL] + SUBSETS}
    return Cb, st, fss


def _sub(Cb, ch):
    ch = list(ch)
    return jnp.asarray(Cb[:, ch][:, :, ch])


def _noise_tile(st, seed):
    return jax.jit(lambda k: simu.channel_maps(k, jnp.zeros(fsk.grid().L.shape, jnp.complex64), st))(
        jax.random.key(seed))


def _one_cluster(iyc, ixc, y0, theta):
    """A single halo at the centre of coarse pixel (iyc, ixc) (as test_sim_basic)."""
    return dict(iy=np.array([2 * iyc], np.int32), ix=np.array([2 * ixc], np.int32),
                oy=np.zeros(1, np.float32), ox=np.zeros(1, np.float32),
                y0=np.array([y0]), theta500=np.array([theta]))


def _sigma(C, fs):
    _, aCa = mmf.weights(C, fs)
    return np.asarray(1.0 / jnp.sqrt(jnp.sum(fs.m_W[None, :] * fs.tau_m ** 2 * aCa[None, :], axis=1)))


def test_default_channels_unchanged(setup):
    _, st, fss = setup
    fs = mmf.filter_static()
    G = fsk.grid()
    a_committed = (ins.beams(G.m_L) * ins.tsz_f_uK(ins.SO_LAT["freqs"])[:, None]).T     # no column indexing
    assert np.array_equal(np.asarray(fs.a), a_committed)
    for name in fs._fields:
        assert np.array_equal(np.asarray(getattr(fs, name)), np.asarray(getattr(fss[ALL], name))), name
    X = _noise_tile(st, 5)
    q, s, C = jax.jit(lambda X: mmf.q_cube(X, fs))(X)
    q6, s6, C6 = jax.jit(lambda X: mmf.q_cube(X, fss[ALL]))(X)
    assert np.array_equal(np.asarray(q), np.asarray(q6))
    assert np.array_equal(np.asarray(s), np.asarray(s6))
    assert np.array_equal(np.asarray(C), np.asarray(C6))


@pytest.mark.parametrize("ch", SUBSETS)
def test_subset_exact_synthesis(setup, ch):
    Cb, _, fss = setup
    fs = fss[ch]
    L = fsk.grid().L
    a = jnp.asarray((ins.beams(L) * ins.tsz_f_uK(ins.SO_LAT["freqs"])[:, None, None])[list(ch)])
    T_tab = pt.T_uniform_table()
    for j in (0, 5, 10, 14):
        th = float(mmf.THETAS15[j])
        yk = pt.synth_direct(_one_cluster(300, 700, 1e-4, th), T_tab, jnp.float64)
        q, sig, _ = mmf.q_cube(a * yk[None], fs, C=_sub(Cb, ch), dtype=jnp.float64)
        y0_hat = float(q[j, 300, 700] * sig[j])
        print(f"channels {ch}, scale {j}: y0_hat/y0 - 1 = {y0_hat / 1e-4 - 1:+.2e}")
        assert abs(y0_hat / 1e-4 - 1.0) < 2e-6, (ch, j, y0_hat)


def test_fewer_channels_more_noise(setup):
    Cb, _, fss = setup
    s = {ch: _sigma(_sub(Cb, ch), fss[ch]) for ch in [ALL] + SUBSETS}
    print({ch: np.round(v / s[ALL], 4).tolist() for ch, v in s.items()})
    assert np.all(s[HF] >= s[ALL] * (1 - 1e-10))
    assert np.all(s[(3,)] >= s[HF] * (1 - 1e-10))


@pytest.mark.parametrize("ch", SUBSETS)
def test_subset_noise_only_q_statistics(setup, ch):
    Cb, st, fss = setup
    fs = fss[ch]
    Xs = _noise_tile(st, 3)[jnp.asarray(ch)]
    q, sig, _ = jax.jit(lambda X: mmf.q_cube(X, fs))(Xs)
    _, sig_an, _ = jax.jit(lambda X: mmf.q_cube(X, fs, C=_sub(Cb, ch)))(Xs)
    q = np.asarray(q)
    for j in range(q.shape[0]):
        assert abs(q[j].std() - 1.0) < 0.05, (ch, j, q[j].std())
    r = np.asarray(sig) / np.asarray(sig_an)
    print(f"channels {ch}: sigma estimated / analytic {np.round(r, 4).tolist()}")
    assert np.all(np.abs(r - 1.0) < 0.05), r


@pytest.mark.parametrize("ch", SUBSETS)
def test_subset_masked_cov_no_mask_is_identity(setup, ch):
    _, st, fss = setup
    fs = fss[ch]
    Xs = _noise_tile(st, 11)[jnp.asarray(ch)]
    C0 = mmf.binned_cov(Xs.reshape(len(ch), -1)[:, fs.m_idx], fs)
    C1, f = jax.jit(lambda X, C0: mmf.masked_cov(X, jnp.ones((N, N), jnp.float32), C0, fs))(Xs, C0)
    assert float(f) == 1.0
    Wm, _ = mmf.whitening(C0)
    D = np.asarray(jnp.einsum("bij,bjk,bkl->bil", Wm, C1 - C0, Wm))   # in units of C0
    print(f"channels {ch}: max |C0^-1/2 (C1 - C0) C0^-1/2| = {np.abs(D).max():.2e}")
    assert np.abs(D).max() < 1e-5


@pytest.mark.parametrize("bad", [(2, 2, 3), (), (6,), (-1, 2)])
def test_bad_channel_lists_raise(bad):
    with pytest.raises(ValueError):
        mmf.filter_static(channels=bad)


def test_channel_mismatch_raises(setup):
    _, _, fss = setup
    fs = fss[HF]
    X = jnp.zeros((6, N, N // 2 + 1), jnp.complex64)
    with pytest.raises(ValueError):
        mmf.q_cube(X, fs)
    with pytest.raises(ValueError):
        mmf.masked_cov(X, jnp.ones((N, N), jnp.float32), jnp.zeros((fs.bin_count.shape[0], 4, 4)), fs)
    with pytest.raises(ValueError):
        mmf.binned_cov(X.reshape(6, -1)[:, fs.m_idx], fs)
