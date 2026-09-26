"""
Empirical Variational Bayesian Matrix Factorization (EVBMF).
Global analytic solution of Nakajima et al. (2013), used by Kim et al. (2016,
ICLR) for automatic rank selection in Tucker decomposition.

References
----------
Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization."Journal of Machine Learning Research 14.Jan (2013): 1-37.
https://github.com/jacobgil/pytorch-tensor-decompositions/blob/master/VBMF/VBMF.py
"""
import numpy as np
from scipy.optimize import minimize_scalar


def EVBMF(Y, sigma2=None, H=None):
    """Empirical VBMF.  Y: (L, M). Returns U, S, V, post; rank = S.shape[0]."""
    L, M = Y.shape
    if L > M:                       # ensure L <= M
        return EVBMF(Y.T, sigma2, H)
    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    U, s, V = np.linalg.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    residual = 0.0
    if H < L:
        residual = np.sum(np.sum(Y ** 2) - np.sum(s ** 2))

    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (np.sum(s ** 2) + residual) / (L * M)
        lower_bound = np.max([s[eH_ub + 1] ** 2 / (M * xubar),
                              np.mean(s[eH_ub + 1:] ** 2) / M])

        sigma2_opt = minimize_scalar(
            EVBsigma2, args=(L, M, s, residual, xubar),
            bounds=[lower_bound, upper_bound], method='Bounded')
        sigma2 = sigma2_opt.x

    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    pos = np.sum(s > threshold)

    d = (s[:pos] / 2) * (1 - (L + M) * sigma2 / s[:pos] ** 2 +
         np.sqrt((1 - (L + M) * sigma2 / s[:pos] ** 2) ** 2
                 - 4 * L * M * sigma2 ** 2 / s[:pos] ** 4))

    post = {}
    return U[:, :pos], np.diag(d), V[:, :pos], post


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)
    alpha = L / M
    x = s ** 2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = _tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
    term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))

    obj = (term1 + term2 + term3 + term4
           + residual / (M * sigma2) + (L - H) * np.log(sigma2))
    return obj


def _tau(x, alpha):
    return 0.5 * (x - (1 + alpha)
                  + np.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))
