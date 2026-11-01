"""Independent re-implementation of the exact Markov-DS count (prereg 3.1-3.2, Amendment A1/A2/A5).

Written from the specification only; does not read or import spectral_utils/markov_ds_count.py or
results/markov_ds_count_v1/run.py. Uses label-free fit outputs only (no labels, no benchmark data).

log J_d(tau) is computed through the parabolic cylinder function:
    J_d(tau) = Gamma(d) * exp(tau^2/4) * D_{-d}(-tau)      (scipy.special.pbdv)
and is cross-checked against direct scipy.integrate.quad of the log-shifted integrand.
"""
import json, sys, math, time
from pathlib import Path
import numpy as np
from scipy.special import pbdv, gammaln
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
CELL = sys.argv[1] if len(sys.argv) > 1 else "socratic__qwen_prm_7b__TELE16"
FIT = ROOT / "results/markov_ds_count_v1/fits"
OUT = Path(__file__).resolve().parent / "PARITY.json"
NMAX = 16


def logJ_pcf(d, tau):
    D, _ = pbdv(-float(d), -np.asarray(tau, float))
    return gammaln(d) + np.asarray(tau, float) ** 2 / 4.0 + np.log(D)


def logJ_quad(d, tau):
    # integrand r^{d-1} exp(-r^2/2 + r tau); shift by its max on r>=0 for numerical safety
    rstar = (tau + math.sqrt(tau * tau + 4 * (d - 1))) / 2 if d > 1 else max(tau, 0.0)
    def lg(r):
        return (d - 1) * math.log(r) - r * r / 2 + r * tau if r > 0 else (-np.inf if d > 1 else 0.0)
    c = lg(rstar) if rstar > 0 else lg(1e-300) if d > 1 else 0.0
    f = lambda r: math.exp(lg(r) - c) if r > 0 else (1.0 * math.exp(-c) if d == 1 else 0.0)
    pts = [rstar] if rstar > 0 else None
    hi = max(rstar, 0) + 40
    v, _ = quad(f, 0, hi, points=pts, limit=400, epsabs=0, epsrel=1e-13)
    return c + math.log(v)


def check_logJ(rng):
    errs = []
    for d in range(1, 16):
        for tau in np.concatenate([np.linspace(-20, 20, 41), rng.uniform(-20, 20, 20)]):
            errs.append(abs(float(logJ_pcf(d, tau)) - logJ_quad(d, float(tau))))
    return float(np.max(errs)), len(errs)


_bits = {}
def configs(n):
    if n not in _bits:
        m = np.arange(1, 2 ** n, dtype=np.int64)
        B = ((m[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)  # bit t <-> step t
        _bits[n] = B
    return _bits[n]


def log_prior(B, e, rho):
    B = B.astype(np.int64)
    y1 = B[:, 0]
    prev, cur = B[:, :-1], B[:, 1:]
    n01 = ((prev == 0) & (cur == 1)).sum(1); n00 = ((prev == 0) & (cur == 0)).sum(1)
    n11 = ((prev == 1) & (cur == 1)).sum(1); n10 = ((prev == 1) & (cur == 0)).sum(1)
    return (y1 * math.log(e) + (1 - y1) * math.log(1 - e) + n01 * math.log(e) + n00 * math.log(1 - e)
            + n11 * math.log(rho) + n10 * math.log(1 - rho))


def answer_posterior(X, delta, e, rho):
    n, K = X.shape
    B = configs(n)
    k = B.sum(1).astype(float)
    L = log_prior(B, e, rho)
    ndeg = 0
    for g in range(K):
        z = X[:, g] - X[:, g].mean()
        nz = np.linalg.norm(z)
        if not np.isfinite(nz) or nz <= 1e-12 * max(1.0, np.abs(X[:, g]).max()):
            ndeg += 1
            continue
        u = z / nz
        tau = delta[g] * (B @ u)
        L = L - delta[g] ** 2 * k * (n - k) / (2 * n) + logJ_pcf(n - 1, tau)
    w = np.exp(L - L.max()); w /= w.sum()
    pk = np.bincount(B.sum(1), weights=w, minlength=n + 1)
    marg = w @ B
    cdf = np.cumsum(pk)
    kh = int(np.argmax(cdf >= 0.5))
    kh = min(max(kh, 1), math.ceil(n / 2))
    return kh, pk, cdf, marg, ndeg


def main():
    t0 = time.time()
    Z = np.load(FIT / f"{CELL}.npz")
    P = json.load(open(FIT / f"{CELL}.json"))["MDS"]
    assert P["valid"]
    delta = np.asarray(P["delta"], float); e = float(P["e"]); rho = float(P["rho"])
    off, V, kst, mst = Z["off"], Z["Vz"], Z["k_MDS"], Z["marg_MDS"]
    rng = np.random.default_rng(0)
    jerr, jn = check_logJ(rng)
    ncmp = nmatch = 0; mism = []; maxd = 0.0; ndeg_tot = 0
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]; n = b - a
        if n < 3 or n > NMAX:
            continue
        kh, pk, cdf, marg, ndeg = answer_posterior(V[a:b], delta, e, rho)
        ndeg_tot += ndeg
        ncmp += 1
        if kh == int(kst[i]):
            nmatch += 1
        else:
            mism.append(dict(answer=i, n=int(n), mine=kh, stored=int(kst[i]),
                             cdf={str(j): float(cdf[j]) for j in range(max(0, kh - 2), min(n, kh + 2) + 1)}))
        maxd = max(maxd, float(np.nanmax(np.abs(marg - mst[a:b]))))
    res = dict(cell=CELL, n_compared=ncmp, n_match=nmatch, mismatches=mism,
               max_abs_marginal_diff=maxd, degenerate_voter_answer_pairs=ndeg_tot,
               params=dict(delta=delta.tolist(), e=e, rho=rho),
               J_method="log J_d(tau) = lgamma(d) + tau^2/4 + log D_{-d}(-tau) via scipy.special.pbdv; "
                        f"cross-checked against scipy.integrate.quad (max |diff| {jerr:.2e} over {jn} points, "
                        "d=1..15, tau in [-20,20])",
               J_check_max_abs_diff=jerr,
               enumeration="all 2^n-1 configurations with >=1 error, 3<=n<=16; k-hat = smallest k with CDF>=0.5, clipped to [1, ceil(n/2)]",
               script=str(Path(__file__).resolve()), seconds=round(time.time() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))
    print("mismatches:", mism[:20])


if __name__ == "__main__":
    main()
