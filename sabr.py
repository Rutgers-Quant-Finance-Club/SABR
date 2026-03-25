"""
SABR Model - Implied Volatility Formula
Implements the Hagan et al. (2002) closed-form asymptotic expansion
for SABR implied volatility (Black vol), equations 2.17a-c and 2.18.

Model dynamics (under forward measure):
    dF = alpha * F^beta * dW1
    d(alpha) = nu * alpha * dW2
    dW1 dW2 = rho * dt

Parameters
----------
F     : forward price
K     : strike price
T     : time to expiration (years)
alpha : initial stochastic volatility (alpha > 0)
beta  : CEV elasticity exponent (0 <= beta <= 1)
rho   : correlation between forward and vol (-1 < rho < 1)
nu    : vol of vol (nu >= 0)
"""

import numpy as np


class SABRModel:
    """
    SABR implied volatility calculator.

    Parameters are validated at construction time. Call `vol(F, K, T)`
    to compute the SABR Black implied volatility for a given forward,
    strike, and time to expiry.
    """

    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu}")

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def vol(self, F: float, K: float, T: float) -> float:
        """
        Compute SABR Black implied volatility.

        Uses the ATM formula (Eq. 2.18) when K ≈ F to avoid 0/0,
        and the general formula (Eq. 2.17a-c) otherwise.

        Parameters
        ----------
        F : forward price (> 0)
        K : strike price (> 0)
        T : time to expiry in years (> 0)

        Returns
        -------
        float : SABR implied Black volatility
        """
        return _sabr_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu)

    def __repr__(self):
        return (
            f"SABRModel(alpha={self.alpha}, beta={self.beta}, "
            f"rho={self.rho}, nu={self.nu})"
        )


def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Functional interface to the SABR Black implied volatility formula.

    Equivalent to SABRModel(alpha, beta, rho, nu).vol(F, K, T).
    """
    return _sabr_vol(F, K, T, alpha, beta, rho, nu)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _sabr_vol(F: float, K: float, T: float,
              alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Core SABR implied volatility calculation (Hagan et al. 2002, Eq. 2.17a-c / 2.18).
    """
    if F <= 0 or K <= 0 or T <= 0:
        raise ValueError(f"F, K, T must all be positive; got F={F}, K={K}, T={T}")

    one_minus_beta = 1.0 - beta

    # ------------------------------------------------------------------
    # ATM case  K ≈ F  →  use Eq. 2.18 (avoids 0/0 in log and z/x(z))
    # ------------------------------------------------------------------
    if abs(F - K) < 1e-8 * F:
        F_1mb = F ** one_minus_beta                   # F^{1-β}
        F_2_1mb = F ** (2.0 * one_minus_beta)         # F^{2(1-β)}

        correction = (
            one_minus_beta**2 * alpha**2 / (24.0 * F_2_1mb)
            + rho * beta * nu * alpha / (4.0 * F_1mb)
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        )
        return (alpha / F_1mb) * (1.0 + correction * T)

    # ------------------------------------------------------------------
    # General case  K ≠ F  →  use Eq. 2.17a-c
    # ------------------------------------------------------------------
    log_FK = np.log(F / K)                            # log(f/K)
    FK = F * K                                         # f·K
    FK_half_1mb = FK ** (one_minus_beta / 2.0)        # (fK)^{(1-β)/2}
    FK_1mb = FK ** one_minus_beta                     # (fK)^{1-β}

    # --- z and x(z)  (Eq. 2.17b-c) ---
    z = (nu / alpha) * FK_half_1mb * log_FK

    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))

    # z/x(z): use Taylor expansion near z=0 to avoid cancellation
    if abs(z) < 1e-8:
        zx_ratio = 1.0 - rho * z / 2.0
    else:
        zx_ratio = z / x_z

    # --- Denominator series  (Eq. 2.17a, middle bracket) ---
    log2 = log_FK**2
    denom_series = (
        1.0
        + one_minus_beta**2 / 24.0 * log2
        + one_minus_beta**4 / 1920.0 * log2**2
    )

    # --- Time-correction bracket  (Eq. 2.17a, last bracket) ---
    correction = (
        one_minus_beta**2 * alpha**2 / (24.0 * FK_1mb)
        + rho * beta * nu * alpha / (4.0 * FK_half_1mb)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )

    vol = (alpha / (FK_half_1mb * denom_series)) * zx_ratio * (1.0 + correction * T)
    return vol
