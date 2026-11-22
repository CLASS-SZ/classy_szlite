# Diagnosis of the slope bias (LOG 2026-11-22). Usage: python qcal_diag_counts.py DIR, where DIR holds points/point_XX.npz,
# calib_grid.npz (qcal_grid_forecast/) and maps_truth.npz, cat_truth_conf.npz, cat_truth_plain.npz (truth sets).
"""1. At the fiducial point: the 21 numbers of the calibrated catalogue model against the map-level mocks.
2. At each of the 25 points: counts predicted from each halo's own q with that point's measured calibration
   (no interpolation) against the counts the finder found on the same halos."""
import sys, numpy as np
from scipy.stats import norm
D = sys.argv[1]
np.set_printoptions(precision=3, suppress=True, linewidth=150)
QE = np.logspace(np.log10(5.0), np.log10(40.0), 5); ZE = np.linspace(0.01, 1.5, 6); ZB = [0.01, 0.31, 0.61, 0.90, 1.20, 1.50, 99.0]
m = np.load(f"{D}/maps_truth.npz")["x"]; cc = np.load(f"{D}/cat_truth_conf.npz")["x"]; cp = np.load(f"{D}/cat_truth_plain.npz")["x"]
print("1. fiducial point, 100 mocks each: mean counts, maps / calibrated catalogue (rows q bins, columns z bins)")
r = m[:, :20].mean(0) / cc[:, :20].mean(0)
se = np.sqrt(m[:, :20].var(0) / 100 + cc[:, :20].var(0) / 100) / cc[:, :20].mean(0)
print(r.reshape(4, 5)); print("z-scores\n", ((r - 1) / se).reshape(4, 5).round(1))
print(f"   mean p: maps {m[:, 20].mean():.4f} +- {m[:, 20].std() / 10:.4f}; calibrated catalogue {cc[:, 20].mean():.4f} +- {cc[:, 20].std() / 10:.4f};"
      f" plain catalogue {cp[:, 20].mean():.4f}")
print(f"   totals: maps {m[:, :20].sum(1).mean():.0f}, calibrated {cc[:, :20].sum(1).mean():.0f}")

def hist(q, z):
    c = np.zeros((4, 5)); k = np.isfinite(q) & (q > QE[0]) & (q < QE[-1]) & (z > ZE[0]) & (z < ZE[-1])
    np.add.at(c, (np.searchsorted(QE[1:-1], q[k]), np.searchsorted(ZE[1:-1], z[k])), 1); return c

def expect(mu, w, f, z):
    c = np.zeros((4, 5)); v = np.zeros((4, 5)); k = (z > ZE[0]) & (z < ZE[-1]); iz = np.searchsorted(ZE[1:-1], z[k])
    for b in range(4):
        p = (1 - f[k]) * (norm.cdf((QE[b + 1] - mu[k]) / w[k]) - norm.cdf((QE[b] - mu[k]) / w[k]))
        np.add.at(c[b], iz, p); np.add.at(v[b], iz, p * (1 - p))
    return c, v

g = np.load(f"{D}/calib_grid.npz"); rows = g["rows"]; P = g["theta"]
print("\n2. per point: one-per-halo counts found / predicted from each halo's q_fsig with that point's own z-bin calibration")
print("   point  sigma8 omega_c  A_SZ  alpha | total ratio | ratio per q bin (all z) | chi2/20 | mean residual (q_obs - mu)/w at q_fsig 8-12, 12-20, 20-40 | skew")
allr = []
for i in range(25):
    h = np.load(f"{D}/points/point_{i:02d}.npz")
    qf, qo, z = h["q_fsig"].astype(float), h["q_obs"].astype(float), h["z"].astype(float)
    ib = np.clip(np.searchsorted(ZB[1:], z, side="left"), 0, 5)
    s, k, w, f = (rows[i][ib, c] for c in range(4))
    mu = np.sqrt(np.maximum((s * qf) ** 2 + k, 0))
    obs = hist(qo, z); pred, var = expect(mu, w, f, z)
    zsc = (obs - pred) / np.sqrt(np.maximum(var, 1))
    det = np.isfinite(qo) & (qo > 4)
    res = (qo - mu) / w
    rr = [res[det & (qf > a) & (qf <= b)].mean() for a, b in ((8, 12), (12, 20), (20, 40))]
    hi = det & (qf > 8)
    sk = np.mean(((res[hi] - res[hi].mean()) / res[hi].std()) ** 3)
    print(f"   {i:3d}  {P[i][0]:.3f} {P[i][1]:.4f} {P[i][2]:.3f} {P[i][3]:.3f} | {obs.sum() / pred.sum():.4f} | {(obs.sum(1) / pred.sum(1)).round(3)} | "
          f"{np.sum(zsc ** 2) / 20:6.1f} | {rr[0]:+.3f} {rr[1]:+.3f} {rr[2]:+.3f} | {sk:+.2f}")
    allr.append(np.r_[obs.sum(1) / pred.sum(1), rr, sk])
allr = np.array(allr)
x = (P - P[0]) / np.array([0.082, 0.0886, 0.45, 0.24])
X = np.c_[np.ones(25), x]
for j, nm in enumerate(["ratio q5-8.4", "ratio 8.4-14", "ratio 14-24", "ratio 24-40", "res 8-12", "res 12-20", "res 20-40", "skew"]):
    c, *_ = np.linalg.lstsq(X, allr[:, j], rcond=None)
    print(f"   {nm:13s}: mean {allr[:, j].mean():+.4f}, spread {allr[:, j].std():.4f}; trend per prior width (s8, wc, A, alpha) " + " ".join(f"{v:+.4f}" for v in c[1:]))
