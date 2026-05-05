"""
sabr_model.py
=============
Hagan et al. (2002) SABR implied-volatility approximation.

Parameters
----------
alpha : initial (instantaneous) volatility  (α > 0)
beta  : CEV exponent, fixed externally       (0 ≤ β ≤ 1)
rho   : forward-vol correlation              (-1 < ρ < 1)
nu    : volatility of volatility             (ν > 0)
"""

import numpy as np


def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float,
             eps: float = 1e-7) -> float:
    """
    Single-strike SABR implied vol (Black-Scholes basis).
    Uses the ATM-limit formula when |F - K| < eps * F.
    """
    if T <= 0:
        return alpha

    # ── ATM branch ───────────────────────────────────────────────────────────
    if abs(F - K) < eps * F:
        FK_beta = F ** (1.0 - beta)
        A = (1 - beta) ** 2 * alpha ** 2 / (24 * F ** (2 - 2 * beta))
        B = rho * beta * nu * alpha   / (4  * F ** (1 - beta))
        C = (2 - 3 * rho ** 2) * nu ** 2 / 24
        return (alpha / FK_beta) * (1 + (A + B + C) * T)

    # ── OTM / ITM branch ─────────────────────────────────────────────────────
    FK      = F * K
    FK_beta = FK ** ((1 - beta) / 2.0)
    ln_FK   = np.log(F / K)

    z   = (nu / alpha) * FK_beta * ln_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    zx  = z / x_z if abs(x_z) > eps else 1.0

    denom = FK_beta * (
        1
        + (1 - beta) ** 2 / 24    * ln_FK ** 2
        + (1 - beta) ** 4 / 1920  * ln_FK ** 4
    )

    A = (1 - beta) ** 2 * alpha ** 2 / (24 * FK ** (1 - beta))
    B = rho * beta * nu * alpha       / (4  * FK_beta)
    C = (2 - 3 * rho ** 2) * nu ** 2 / 24

    return (alpha / denom) * zx * (1 + (A + B + C) * T)


def sabr_vol_vec(F: float, strikes: np.ndarray, T: float,
                 alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorised wrapper over an array of strikes."""
    return np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])