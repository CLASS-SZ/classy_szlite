"""Ensemble of neural posteriors (catalogue_npe.py runs that differ only in seed), on the sets each was
applied to (npe_{tag}_applied_{name}.npz).

    python benchmarks/mapsim/npe_ensemble.py DIR --tags e0,e1,e2,e3,e4 --single n5k --sets mocks,held

Mixture of equal weights: mean = average of the members' means, variance = average variance plus the
variance of the means, and the rank of the truth = average of the members' ranks (exact for equal sample
numbers). Prints, per member, the mixture and --single (e.g. fewer training simulations): the bias on
'mocks' (posterior sd units, with the error of the mean) and the 50% / 90% coverage on 'held' for the four
parameters the counts constrain, and the members' spread.
"""
import argparse
import os
import numpy as np

P4 = ("sigma8", "omega_cdm", "A_SZ", "alpha_SZ")


def stats(theta, mu, sd, rank, kind):
    if kind == "mocks":
        b = (mu - theta) / sd
        return " ".join(f"{b[:, j].mean():+.3f}+-{b[:, j].std(ddof=1) / np.sqrt(len(b)):.3f}" for j in range(4))
    cov = lambda lv: np.mean(np.abs(rank - 0.5) < lv / 2, axis=0)
    c5, c9 = cov(0.5), cov(0.9)
    return " ".join(f"{c5[j]:.2f}/{c9[j]:.2f}" for j in range(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--single", default="")
    ap.add_argument("--sets", default="mocks,held")
    a = ap.parse_args()
    tags = a.tags.split(",")
    for s in a.sets.split(","):
        print(f"\n{s}: " + ("bias (sigma8, omega_cdm, A_SZ, alpha_SZ)" if s == "mocks" else "coverage 50%/90% (sigma8, omega_cdm, A_SZ, alpha_SZ)"))
        D = [np.load(os.path.join(a.dir, f"npe_{t}_applied_{s}.npz")) for t in tags]
        th = D[0]["theta"][:, :4]
        for t, d in zip(tags, D):
            print(f"   {t:8s} {stats(th, d['mu'][:, :4], d['sd'][:, :4], d['rank'][:, :4], s)}")
        mus = np.stack([d["mu"][:, :4] for d in D]); sds = np.stack([d["sd"][:, :4] for d in D])
        rk = np.stack([d["rank"][:, :4] for d in D]).mean(0)
        mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
        print(f"   {'mixture':8s} {stats(th, mu, sd, rk, s)}")
        print(f"   members' spread of the posterior mean, in posterior sd: " + " ".join(f"{(mus.std(0) / sds.mean(0))[:, j].mean():.3f}" for j in range(4)))
        if a.single:
            d = np.load(os.path.join(a.dir, f"npe_{a.single}_applied_{s}.npz"))
            print(f"   {a.single:8s} {stats(th, d['mu'][:, :4], d['sd'][:, :4], d['rank'][:, :4], s)}")


if __name__ == "__main__":
    main()
