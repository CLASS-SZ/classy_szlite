# Diagnosis of the slope bias (LOG 2026-11-22). Usage: python qcal_diag_tail.py DIR [points], DIR as for qcal_diag_counts.py.
"""Gaussian against Gaussian-plus-exponential (exponentially modified Gaussian) scatter for the finder's q:
censored fits per point and z bin, then counts predicted from each halo against counts found."""
import sys, numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
D = sys.argv[1]
QE = np.logspace(np.log10(5.0), np.log10(40.0), 5); ZE = np.linspace(0.01, 1.5, 6); ZB = [0.01, 0.31, 0.61, 0.90, 1.20, 1.50, 99.0]
LO = 4.0

def nll_g(p, q, qo, det):
    s, k, w, f = p
    if not (0 <= f < 1 and w > 0.05): return np.inf
    mu = np.sqrt(np.maximum((s * q) ** 2 + k, 1e-6))
    return -np.sum(np.where(det, np.log1p(-f) + norm.logpdf(qo, mu, w), np.logaddexp(np.log(max(f, 1e-300)), np.log1p(-f) + norm.logcdf((LO - mu) / w))))

def emg_logpdf(x, mu, w, t):
    lam = 1.0 / t
    return np.log(lam) + lam * (mu - x) + 0.5 * (lam * w) ** 2 + log_ndtr((x - mu) / w - lam * w)


def emg_cdf(x, mu, w, t):
    lam = 1.0 / t
    return norm.cdf((x - mu) / w) - np.exp(lam * (mu - x) + 0.5 * (lam * w) ** 2 + log_ndtr((x - mu) / w - lam * w))


def nll_e(p, q, qo, det):
    s, k, w, t, f = p
    if not (0 <= f < 1 and w > 0.05 and t > 1e-3): return np.inf
    mu = np.sqrt(np.maximum((s * q) ** 2 + k, 1e-6))
    c = np.clip(emg_cdf(LO, mu, w, t), 1e-300, 1.0)
    return -np.sum(np.where(det, np.log1p(-f) + emg_logpdf(qo, mu, w, t), np.logaddexp(np.log(max(f, 1e-300)), np.log1p(-f) + np.log(c))))

def fit(args):
    q, qo = args
    det = np.isfinite(qo) & (qo > LO)
    qo = np.where(det, qo, 0.0)
    rg = minimize(nll_g, [1.0, 1.0, 1.0, 0.01], args=(q, qo, det), method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-5, fatol=1e-3))
    s, k, w, f = rg.x
    best = None
    for t0 in (0.1, 0.3, 0.6):
        re = minimize(nll_e, [s, k - 2 * t0 * s * 6, max(w - 0.3 * t0, 0.3), t0, f], args=(q, qo, det), method="Nelder-Mead",
                      options=dict(maxiter=5000, xatol=1e-5, fatol=1e-3))
        if best is None or re.fun < best.fun: best = re
    return rg.x, rg.fun, best.x, best.fun

def hist(q, z):
    c = np.zeros((4, 5)); k = np.isfinite(q) & (q > QE[0]) & (q < QE[-1]) & (z > ZE[0]) & (z < ZE[-1])
    np.add.at(c, (np.searchsorted(QE[1:-1], q[k]), np.searchsorted(ZE[1:-1], z[k])), 1); return c

def cdf(kind, x, mu, w, t):
    return norm.cdf((x - mu) / w) if kind == "g" else emg_cdf(x, mu, w, t)

def expect(kind, mu, w, t, f, z):
    c = np.zeros((4, 5)); v = np.zeros((4, 5)); k = (z > ZE[0]) & (z < ZE[-1]); iz = np.searchsorted(ZE[1:-1], z[k])
    for b in range(4):
        p = (1 - f[k]) * (cdf(kind, QE[b + 1], mu[k], w[k], t[k]) - cdf(kind, QE[b], mu[k], w[k], t[k]))
        np.add.at(c[b], iz, p); np.add.at(v[b], iz, p * (1 - p))
    return c, v

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(25))
    print("point | Delta ln L (exp. tail better if > 0) per z bin | tail mean t per z bin | one-per-halo counts / predicted per q bin: Gaussian | with tail | chi2/20 Gaussian, tail")
    for i in pts:
        h = np.load(f"{D}/points/point_{i:02d}.npz")
        qf, qo, z = h["q_fsig"].astype(float), h["q_obs"].astype(float), h["z"].astype(float)
        jobs = []
        for a, b in zip(ZB[:-1], ZB[1:]):
            m = np.flatnonzero((z > a) & (z <= b) & (qf > 2))
            if m.size > 30000: m = rng.choice(m, 30000, replace=False)
            jobs.append((qf[m], qo[m]))
        res = [fit(j) for j in jobs]
        ib = np.clip(np.searchsorted(ZB[1:], z, side="left"), 0, 5)
        G = np.array([r[0] for r in res]); E = np.array([r[2] for r in res])
        mu_g = np.sqrt(np.maximum((G[ib, 0] * qf) ** 2 + G[ib, 1], 0)); mu_e = np.sqrt(np.maximum((E[ib, 0] * qf) ** 2 + E[ib, 1], 0))
        obs = hist(qo, z)
        pg, vg = expect("g", mu_g, G[ib, 2], np.ones_like(qf), G[ib, 3], z)
        pe, ve = expect("e", mu_e, E[ib, 2], E[ib, 3], E[ib, 4], z)
        c2 = lambda p, v: np.sum((obs - p) ** 2 / np.maximum(v, 1)) / 20
        dl = [r[1] - r[3] for r in res]
        print(f"{i:3d} | " + " ".join(f"{v:7.0f}" for v in dl) + " | " + " ".join(f"{v:.2f}" for v in E[:, 3]) +
              f" | {np.round(obs.sum(1) / pg.sum(1), 3)} | {np.round(obs.sum(1) / pe.sum(1), 3)} | {c2(pg, vg):.1f}, {c2(pe, ve):.1f}", flush=True)
