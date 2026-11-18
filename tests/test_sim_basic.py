"""mapsim unit tests (CPU): grid bookkeeping, summaries, halo counts, filter
statistics and the noise-free injection test (painter + template + filter).
"""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szlite.sim import flatsky as fsk  # noqa: E402
from classy_szlite.sim import mmf, paint as pt, summaries as sm, instrument as ins  # noqa: E402
from classy_szlite.sim import halos as hl, simulator as simu  # noqa: E402


# ---- grid -------------------------------------------------------------------------

def test_grid_bookkeeping():
    G = fsk.grid()
    assert abs(fsk.DK - 36.0) < 1e-12
    assert G.W.sum() == G.n ** 2                             # full plane
    # in-band modes sorted by bin, bins contiguous, padded layout consistent
    assert np.all(np.diff(G.m_bin) >= 0)
    for b in (0, G.n_bins // 2, G.n_bins - 1):
        pos = G.bin_pad[b][G.bin_pad[b] >= 0]
        assert np.all(G.m_binpos[pos] == b)
        assert abs(G.m_W[pos].sum() - G.bin_count[b]) < 1e-9
    assert (G.bin_pad >= 0).sum() == G.n_modes
    # band edges: |k| in [100, 20000]
    assert G.m_L.min() >= 100.0 and G.m_L.max() <= 20000.0


# ---- summaries ----------------------------------------------------------------------

def _peaks(iy, ix, q):
    K = 8
    pad = K - len(q)
    return mmf.Peaks(j=jnp.zeros(K, jnp.int32),
                     iy=jnp.asarray(list(iy) + [0] * pad, jnp.int32),
                     ix=jnp.asarray(list(ix) + [0] * pad, jnp.int32),
                     q=jnp.asarray(list(q) + [-np.inf] * pad, jnp.float32),
                     valid=jnp.arange(K) < len(q), count=jnp.int32(len(q)),
                     overflow=jnp.bool_(False))


def test_merge_local_max():
    # A (q 10) and B (q 6) 1′ apart (2 px), C (q 8) 1.5′ from B but 2.5′ from A,
    # D far away; E straddles the periodic edge with F
    pk = _peaks([100, 102, 105, 500, 0, 1199], [100, 100, 100, 500, 7, 7],
                [10., 6., 8., 5., 7., 9.])
    keep = np.asarray(sm.merge_local_max(pk, 1.5))
    # B is dropped (A is higher within 1.5′); C is kept (only B, lower, is near);
    # E is dropped (F, higher, is 0.5′ away across the edge)
    assert keep.tolist() == [True, False, True, True, False, True, False, False]


def test_histogram_open_intervals():
    q = jnp.asarray([5.0, 5.01, 39.99, 40.0, 7.0, 7.0, 7.0], jnp.float32)
    z = jnp.asarray([0.5, 0.5, 1.0, 0.5, 0.01, 1.5, 0.3], jnp.float32)
    ok = jnp.ones(7, bool)
    c = np.asarray(sm.histogram(q, z, ok))
    # q = 5 and q = 40 are dropped (open intervals), as are z = 0.01 and z = 1.5
    assert c.sum() == 3
    assert c[0].sum() == 2 and c[3].sum() == 1
    iz = np.searchsorted(sm.Z_EDGES, [0.5, 1.0, 0.3]) - 1
    assert c[0, iz[0]] == 1 and c[3, iz[1]] == 1 and c[0, iz[2]] == 1


def test_confirm_takes_largest_y0():
    pk = _peaks([200, 800], [200, 800], [9.0, 6.0])
    keep = pk.valid
    C = 4
    halos = hl.Halos(
        iy=jnp.asarray([400, 401, 1600, 0], jnp.int32), ix=jnp.asarray([400, 402, 1600, 0], jnp.int32),
        oy=jnp.zeros(C, jnp.float32), ox=jnp.zeros(C, jnp.float32),
        z=jnp.asarray([0.3, 0.7, 0.5, 0.1], jnp.float32),
        lnM=jnp.log(jnp.asarray([6e13, 8e13, 1e13, 1e15], jnp.float32)),
        y0=jnp.asarray([1e-5, 3e-5, 1e-5, 1e-4], jnp.float32),
        theta500=jnp.ones(C, jnp.float32), valid=jnp.asarray([True, True, True, False]),
        count=jnp.int32(3), overflow=jnp.bool_(False))
    cf = sm.confirm(pk, keep, halos, 5e13, 1.5, 8)
    ok, z = np.asarray(cf.ok), np.asarray(cf.z)
    # peak 0 at (100′, 100′): halos 0 and 1 within 1.5′ (fine pixels 400, 401/402);
    # it takes halo 1's z (larger y0). Peak 1 at (400′, 400′): only a 1e13 halo there.
    assert ok[0] and abs(z[0] - 0.7) < 1e-7
    assert not ok[1]


# ---- halo counts ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fid_lambda():
    cells = hl.default_cells()
    lam, cg = jax.jit(lambda th: simu.cosmology_stage(th, cells))(simu.THETA_FID)
    return cells, lam, cg


def test_halo_counts_are_poisson(fid_lambda):
    """Halo draws against the cells' λ: total counts, the redshift distribution
    (6 coarse bins, exact cell overlaps since z is uniform inside a cell) and
    the 150 mass cells, over 16 tiles. A fixed seed makes the test
    deterministic; the thresholds are p ≲ 1e-4 for a correct sampler (checked
    with four other seeds, 8 and 64 tiles: χ²_z = 1.3-8.5)."""
    cells, lam, cg = fid_lambda
    th = simu.THETA_FID
    lam_np = np.asarray(lam)
    lam_tot = float(lam_np.sum())
    cap = int(lam_tot + 8 * np.sqrt(lam_tot) + 1000)
    draw = jax.jit(jax.vmap(lambda k: hl.sample_halos(
        k, lam, cells, cap, cg, th.h, th.A_SZ, th.alpha_SZ, th.sigma_lnq, th.one_minus_b)))
    R = 16
    zs, ms, n = [], [], 0
    for r0 in range(0, R, 8):
        h = draw(jax.random.split(jax.random.key(2026 + r0), 8))
        assert not np.asarray(h.overflow).any()
        v = np.asarray(h.valid)
        zs.append(np.asarray(h.z)[v])
        ms.append(np.asarray(h.lnM)[v])
        n += int(np.asarray(h.count).sum())
    zs, ms = np.concatenate(zs), np.concatenate(ms)
    assert abs(n - R * lam_tot) < 4 * np.sqrt(R * lam_tot)
    edges = np.array([0.01, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    ze = cells.z_edges
    lz = lam_np.sum(1)
    expect = R * np.array([np.sum(lz * np.clip((np.minimum(ze[1:], b) - np.maximum(ze[:-1], a))
                                               / np.diff(ze), 0.0, 1.0))
                           for a, b in zip(edges[:-1], edges[1:])])
    zsc = (np.histogram(zs, edges)[0] - expect) / np.sqrt(expect)
    assert np.all(np.abs(zsc) < 5.0) and np.sum(zsc ** 2) < 30.0, zsc
    mexp = R * lam_np.sum(0)
    mgot = np.histogram(ms, cells.lnM_edges)[0]
    keep = mexp > 5
    chi2 = np.sum((mgot[keep] - mexp[keep]) ** 2 / mexp[keep])
    assert chi2 < keep.sum() + 6 * np.sqrt(2 * keep.sum()), (chi2, keep.sum())


# ---- filter statistics -------------------------------------------------------------------

@pytest.fixture(scope="module")
def static():
    import os
    path = os.path.join(os.environ.get("MAPSIM_REFDATA", "/mnt/ceph/users/bbolliet/sz-poisson-blending"),
                        "data_real", "cmb_lensed_TT_camb.npz")
    cmb = ins.cmb_cl_from_table(path) if os.path.exists(path) else ins.cmb_cl_emulator()
    G = fsk.grid()
    fs = mmf.filter_static()
    # analytic binned covariance (rfft units)
    B = ins.beams(G.m_L)
    C = ins.noise_cov(G.m_L, "goal") + B[:, None, :] * B[None, :, :] * cmb(G.m_L)[None, None, :]
    C *= G.Npx / G.Om
    Cb = np.stack([np.stack([np.bincount(G.m_binpos, weights=G.m_W * C[i, j], minlength=G.n_bins)
                             / G.bin_count for j in range(6)], -1) for i in range(6)], 1)
    return cmb, fs, jnp.asarray(Cb)


def test_noise_only_q_statistics(static):
    """CMB + noise, no halos: with the covariance estimated from the tile, the
    q maps have mean ≈ 0 and std ≈ 1 at every scale, and σ_j agrees with the
    analytic covariance's to a few per cent."""
    cmb, fs, Cb = static
    tb = pt.PaintTables(lnth0=0.0, dln=0.02, rP=np.ones(2), tab=np.zeros((2, 4)), th_split=6.0)
    st = simu.tile_static(cmb, tb)
    X = jax.jit(lambda k: simu.channel_maps(k, jnp.zeros(fsk.grid().L.shape, jnp.complex64), st))(
        jax.random.key(3))
    q, sig, C_est = jax.jit(lambda X: mmf.q_cube(X, fs))(X)
    _, sig_an, _ = jax.jit(lambda X: mmf.q_cube(X, fs, C=Cb))(X)
    q = np.asarray(q)
    for j in range(q.shape[0]):
        assert abs(q[j].mean()) < 0.05 * (1 + j)            # large scales: few independent modes
        assert abs(q[j].std() - 1.0) < 0.05
    r = np.asarray(sig) / np.asarray(sig_an)
    assert np.all(np.abs(r - 1.0) < 0.05), r


def _one_cluster(iyc, ixc, y0, theta):
    """A single halo at the centre of coarse pixel (iyc, ixc)."""
    return dict(iy=np.array([2 * iyc], np.int32), ix=np.array([2 * ixc], np.int32),
                oy=np.zeros(1, np.float32), ox=np.zeros(1, np.float32),
                y0=np.array([y0]), theta500=np.array([theta]))


def test_injection_noise_free_exact_synthesis(static):
    """ŷ0 = y0 at the true position and scale, to round-off, when the halo is
    synthesised exactly (template and filter consistent by construction)."""
    _, fs, Cb = static
    T_tab = pt.T_uniform_table()
    a = jnp.asarray((ins.beams(fsk.grid().L) * ins.tsz_f_uK(ins.SO_LAT["freqs"])[:, None, None]))
    for j in (0, 5, 10, 14):
        th = float(mmf.THETAS15[j])
        yk = pt.synth_direct(_one_cluster(300, 700, 1e-4, th), T_tab, jnp.float64)
        q, sig, _ = mmf.q_cube(a * yk[None], fs, C=Cb, dtype=jnp.float64)
        y0_hat = float(q[j, 300, 700] * sig[j])
        print(f"exact synthesis, scale {j}: y0_hat/y0 - 1 = {y0_hat / 1e-4 - 1:+.2e}")
        assert abs(y0_hat / 1e-4 - 1.0) < 2e-6, (j, y0_hat)


def test_injection_noise_free_painter(static):
    """The same with the scatter-add painter (tables with ln θ nodes 0.005
    apart around the scale): ŷ0 = y0 to ~1e-5."""
    _, fs, Cb = static
    a = jnp.asarray((ins.beams(fsk.grid().L) * ins.tsz_f_uK(ins.SO_LAT["freqs"])[:, None, None]))
    for j in (3, 8):
        th = float(mmf.THETAS15[j])
        tb = pt.build_tables(th_min=th / 1.05, th_split=th * 1.05, dln=0.005)
        h1 = _one_cluster(301, 699, 1e-4, th)
        C = 64
        halos = hl.Halos(iy=jnp.zeros(C, jnp.int32).at[0].set(h1["iy"][0]),
                         ix=jnp.zeros(C, jnp.int32).at[0].set(h1["ix"][0]),
                         oy=jnp.zeros(C, jnp.float32), ox=jnp.zeros(C, jnp.float32),
                         z=jnp.zeros(C, jnp.float32), lnM=jnp.zeros(C, jnp.float32),
                         y0=jnp.zeros(C, jnp.float32).at[0].set(1e-4),
                         theta500=jnp.full(C, th, jnp.float32), valid=jnp.arange(C) < 1,
                         count=jnp.int32(1), overflow=jnp.bool_(False))
        cfg = pt.make_config(tb, [1.0] * len(pt.SIDES), cap_big=32, margin=1.0, extra=0)
        yk, ovf = pt.paint_y(halos, tb, cfg, pt.T_uniform_table())
        assert not np.asarray(ovf).any()
        q, sig, _ = mmf.q_cube(a * yk.astype(jnp.complex128)[None], fs, C=Cb, dtype=jnp.float64)
        y0_hat = float(q[j, 301, 699] * sig[j])
        print(f"painter, scale {j}: y0_hat/y0 - 1 = {y0_hat / 1e-4 - 1:+.2e}")
        assert abs(y0_hat / 1e-4 - 1.0) < 3e-5, (j, y0_hat)
