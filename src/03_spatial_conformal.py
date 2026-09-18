#!/usr/bin/env python3
"""
03_spatial_conformal.py - spatially weighted conformal prediction (Mao, Martin & Reich, 2024, JASA 119(546):904-914)

Implementation follows Mao, Martin & Reich (2024, JASA 119(546):904-914):
    non-conformity score  delta_i = |standardised Kriging residual|, analytic leave-one-out form delta_i = |[K^-1 y]_i| / sqrt([K^-1]_ii)
    Gaussian kernel weights  w_i = exp(-d_i^2/(2 eta^2)) / (1 + sum_j exp(-d_j^2/(2 eta^2))), including the self term w_{n+1}=1/(1+sum_j ...)
    weighted plausibility  p_w(y) = sum_i w_i * 1{delta_i >= delta_*(y)}, delta_*(y)=|y-mu_hat(s*)|/sigma_hat(s*)
    prediction interval   {y : p_w(y) >= alpha}, inverted to h = sigma_hat(s*) * delta_(widest feasible threshold)

Known differences from the original paper (declared in the Methods section):
    1. delta_i is computed plug-in from the full fit, rather than by refitting with each point removed (the exact version in the paper);
    2. the interval is inverted by sorting and suffix weighted sums rather than by the paper's closed-form quadratic solution (same interval shape, different computational route);
    3. an exponential covariance is used (Matern kappa=0.5) with parameters estimated by marginal likelihood; the paper mainly uses Matern kappa=0.7 and others.
    => The fidelity of this implementation is therefore backed by a self-test on synthetic data: if it reproduces near-nominal coverage under the paper's simulation settings,
         it may be regarded as usable. If the self-test fails, the implementation must be corrected before use on real data.

Usage
    python3 code/03_spatial_conformal.py            # run the synthetic-data self-test
"""
import numpy as np
from scipy.optimize import minimize

ALPHA = 0.10

# ---------------- Gaussian process (exponential kernel + noise); parameters by marginal likelihood ----------------
def _exp_kernel(D, sig2, phi):
    return sig2 * np.exp(-D / phi)

def _nll(params, D, y):
    sig2, phi, tau2 = np.exp(params)
    K = _exp_kernel(D, sig2, phi) + tau2 * np.eye(len(y))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    a = np.linalg.solve(L, y)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return 0.5 * (a @ a + logdet + len(y) * np.log(2 * np.pi))

class SpatialConformal:
    def __init__(self, eta=None, M=None, alpha=ALPHA):
        self.eta, self.M, self.alpha = eta, M, alpha

    def fit(self, S, y):
        """S: (n,2) coordinates scaled to [0,1]^2; y: (n,)"""
        self.S, self.y = np.asarray(S, float), np.asarray(y, float)
        D = np.sqrt(((self.S[:, None, :] - self.S[None, :, :]) ** 2).sum(-1))
        self.D = D
        # initial values: variance from y, correlation length from mean nearest-neighbour distance, noise at 10% of variance
        nn = np.sort(D + np.eye(len(y)) * 1e9, axis=1)[:, 0].mean()
        p0 = np.log([max(y.var(), 1e-6), max(nn, 1e-3), 0.1 * max(y.var(), 1e-6)])
        r = minimize(_nll, p0, args=(D, y), method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
        self.sig2, self.phi, self.tau2 = np.exp(r.x)
        K = _exp_kernel(D, self.sig2, self.phi) + self.tau2 * np.eye(len(y))
        self.Kinv = np.linalg.inv(K)
        self.alpha_vec = self.Kinv @ y
        # analytic leave-one-out standardised residuals
        self.delta = np.abs(self.alpha_vec) / np.sqrt(np.diag(self.Kinv))
        # default eta: twice the correlation length; M: nearest max(20, 10% of n) points
        if self.eta is None: self.eta = 2.0 * self.phi
        if self.M is None: self.M = int(max(20, 0.10 * len(y)))
        return self

    def _pred(self, s):
        d = np.sqrt(((self.S - s) ** 2).sum(1))
        k = _exp_kernel(d, self.sig2, self.phi)
        mu = float(k @ self.alpha_vec)
        var = self.sig2 + self.tau2 - float(k @ (self.Kinv @ k))
        return mu, float(np.sqrt(max(var, 1e-12))), d

    def interval(self, s, n_grid=4001):
        """Return (lo, hi); with uniform weights this reduces to GSCP (standard full-data conformal)."""
        mu, sd, d = self._pred(s)
        # take the M nearest points
        idx = np.argsort(d)[:self.M]
        dM = d[idx]
        w = np.exp(-dM ** 2 / (2 * self.eta ** 2))
        Z = 1.0 + w.sum()                       # includes the self term (Eq. 12 in the paper)
        w_self = 1.0 / Z
        w_cal = w / Z
        order = np.argsort(self.delta[idx])
        ds, ws = self.delta[idx][order], w_cal[order]
        # the self term w_self always enters p_w, since delta_{n+1} >= delta_{n+1} holds trivially
        tail = w_self + np.cumsum(ws[::-1])[::-1]   # tail[j] = w_self + mass(delta >= ds[j])
        ok = np.where(tail >= self.alpha)[0]
        if not len(ok):
            return mu, mu                            # even the self term is below alpha: return a degenerate interval
        j = ok[-1]                                   # take the largest feasible threshold, not the smallest
        h = sd * ds[j]
        return mu - h, mu + h

def synth(n_side=20, seed=0, scenario="gauss"):
    """Synthetic data as in the paper: GP(Z) + white noise; 'gauss' corresponds to Scenario 1."""
    rng = np.random.RandomState(seed)
    g = np.linspace(0, 1, n_side)
    S = np.array([(a, b) for a in g for b in g])
    D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    sig2, phi = 3.0, 0.1
    K = sig2 * np.exp(-D / phi) + 1e-8 * np.eye(len(S))
    L = np.linalg.cholesky(K)
    Z = L @ rng.randn(len(S))
    E = rng.randn(len(S))
    if scenario == "gauss":
        y = Z + E
    elif scenario == "cube":
        y = Z ** 3 + E
    else:
        raise ValueError(scenario)
    return S, y

def run_selftest(seed=0, n_side=20, scenario="gauss", alpha=ALPHA, n_test=100):
    S, y = synth(n_side, seed, scenario)
    n = len(y)
    rng = np.random.RandomState(1234)
    cov_g = cov_s = cov_k = 0.0; w_g = w_s = w_k = 0.0
    for t in range(n_test):
        te = rng.choice(n, 1, replace=False)
        tr = np.setdiff1d(np.arange(n), te)
        m = SpatialConformal(alpha=alpha).fit(S[tr], y[tr])
        s_star = S[te[0]]
        for tag in ("s", "g"):
            mm = m if tag == "s" else SpatialConformal(eta=1e6, M=len(tr), alpha=alpha).fit(S[tr], y[tr])
            lo, hi = mm.interval(s_star)
            hit = (lo <= y[te[0]] <= hi)
            if tag == "s": cov_s += hit; w_s += (hi - lo)
            else:          cov_g += hit; w_g += (hi - lo)
        mu, sd, _ = m._pred(s_star)
        lo, hi = mu - 1.6449 * sd, mu + 1.6449 * sd
        cov_k += (lo <= y[te[0]] <= hi); w_k += (hi - lo)
    print(f"  scenario={scenario}  grid={n_side}x{n_side}={n}  replicates={n_test}  nominal={1-alpha:.0%}")
    print(f"    sLSCP (spatially weighted)  coverage {cov_s/n_test:6.1%}  mean width {w_s/n_test:6.2f}")
    print(f"    GSCP (equal weight)        coverage {cov_g/n_test:6.1%}  mean width {w_g/n_test:6.2f}")
    print(f"    Kriging normal interval    coverage {cov_k/n_test:6.1%}  mean width {w_k/n_test:6.2f}")

if __name__ == "__main__":
    print("=== Synthetic-data self-test (fidelity check; not a study result) ===")
    run_selftest(seed=0, n_side=15, scenario="gauss", n_test=60)
    print()
    run_selftest(seed=1, n_side=15, scenario="cube",  n_test=60)
