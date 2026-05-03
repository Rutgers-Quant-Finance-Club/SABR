"""
sabr_model.py
=============
Core SABR implied-volatility formula (Hagan et al. 2002).

Implements the closed-form asymptotic expansion for implied Black-Scholes
volatility given the four SABR parameters:
    alpha  – initial (instantaneous) volatility
    beta   – CEV exponent  ∈ [0, 1]
    rho    – correlation between forward and vol processes
    nu     – volatility-of-volatility (volvol)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Main SABR formula
# ─────────────────────────────────────────────────────────────────────────────

def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float,
             eps: float = 1e-7) -> float:
    """
    Hagan et al. (2002) SABR implied-volatility approximation.

    Parameters
    ----------
    F     : forward price / rate
    K     : strike
    T     : time-to-expiry (years)
    alpha : initial vol  (α > 0)
    beta  : CEV exponent  (0 ≤ β ≤ 1)
    rho   : correlation   (−1 < ρ < 1)
    nu    : volvol         (ν > 0)
    eps   : threshold below which F≈K (ATM branch)

    Returns
    -------
    Implied Black-Scholes vol (float)
    """
    if T <= 0:
        return alpha  # degenerate case – return alpha directly

    # ── ATM branch (K → F) ───────────────────────────────────────────────────
    if abs(F - K) < eps * F:
        FK_mid = F ** (1.0 - beta)
        term1  = alpha / FK_mid

        A  = (1 - beta) ** 2 * alpha ** 2 / (24 * F ** (2 - 2 * beta))
        B  = rho * beta * nu * alpha / (4 * F ** (1 - beta))
        C  = (2 - 3 * rho ** 2) * nu ** 2 / 24

        return term1 * (1 + (A + B + C) * T)

    # ── OTM / ITM branch ─────────────────────────────────────────────────────
    FK  = F * K
    FK_beta = FK ** ((1 - beta) / 2.0)
    ln_FK   = np.log(F / K)

    # z and x(z)
    z   = (nu / alpha) * FK_beta * ln_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    if abs(x_z) < eps:
        zx = 1.0
    else:
        zx = z / x_z

    # Leading term
    num_lead = alpha
    den_lead = FK_beta * (
        1
        + (1 - beta) ** 2 / 24 * ln_FK ** 2
        + (1 - beta) ** 4 / 1920 * ln_FK ** 4
    )

    # Correction factor
    A = (1 - beta) ** 2 * alpha ** 2 / (24 * FK ** (1 - beta))
    B = rho * beta * nu * alpha / (4 * FK_beta)
    C = (2 - 3 * rho ** 2) * nu ** 2 / 24

    correction = 1 + (A + B + C) * T

    return (num_lead / den_lead) * zx * correction


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised wrapper
# ─────────────────────────────────────────────────────────────────────────────

def sabr_vol_vec(F: float, strikes: np.ndarray, T: float,
                 alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorised SABR vol over an array of strikes."""
    return np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])


# ─────────────────────────────────────────────────────────────────────────────
# Greeks
# ─────────────────────────────────────────────────────────────────────────────

def sabr_vega(F: float, K: float, T: float,
              alpha: float, beta: float, rho: float, nu: float,
              dAlpha: float = 1e-5) -> float:
    """
    SABR Vega  ∂σ/∂α  (sensitivity of implied vol to alpha).
    Computed via central finite difference on alpha.
    """
    up   = sabr_vol(F, K, T, alpha + dAlpha, beta, rho, nu)
    down = sabr_vol(F, K, T, alpha - dAlpha, beta, rho, nu)
    return (up - down) / (2 * dAlpha)


def sabr_delta(F: float, K: float, T: float,
               alpha: float, beta: float, rho: float, nu: float,
               dF: float = 1e-3) -> float:
    """
    SABR Delta  ∂σ/∂F  (sensitivity of implied vol to forward price).
    This is the vol-surface delta, not the option delta.
    Computed via central finite difference on F.
    """
    up   = sabr_vol(F + dF, K, T, alpha, beta, rho, nu)
    down = sabr_vol(F - dF, K, T, alpha, beta, rho, nu)
    return (up - down) / (2 * dF)