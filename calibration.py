"""
calibration.py
==============
Calibration engine for the SABR model.

Given a slice of market (strike, implied-vol) pairs for a single expiry,
fits (alpha, rho, nu) while keeping beta fixed (common practice: β = 0.5
for equity indices, β = 1.0 for rates).

Returns calibrated parameters + per-slice diagnostics (SSE, RMSE).
"""

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from sabr import sabr_vol_vec


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    expiration: str
    maturity:   float                      # T in years
    alpha:      float
    beta:       float
    rho:        float
    nu:         float
    forward:    float
    sse:        float                      # Sum of Squared Errors
    rmse:       float                      # Root Mean Squared Error
    n_points:   int
    strikes:    np.ndarray = field(repr=False)
    market_vols: np.ndarray = field(repr=False)
    model_vols:  np.ndarray = field(repr=False)

    def __str__(self):
        return (
            f"[{self.expiration}]  T={self.maturity:.4f}y  F={self.forward:.1f}\n"
            f"  α={self.alpha:.6f}  β={self.beta:.2f}  "
            f"ρ={self.rho:.4f}  ν={self.nu:.4f}\n"
            f"  SSE={self.sse:.6f}  RMSE={self.rmse:.6f}  n={self.n_points}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Forward-price estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_forward(strikes: np.ndarray, market_vols: np.ndarray) -> float:
    """
    Naïve ATM-forward estimate: the strike with the minimum implied vol
    (the smile's trough for a standard equity skew).
    """
    idx = np.argmin(market_vols)
    return float(strikes[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _residuals(params, F, strikes, market_vols, T, beta):
    """Vector of (model_vol - market_vol) for least_squares."""
    alpha, rho, nu = params
    if alpha <= 0 or nu <= 0 or rho <= -1 or rho >= 1:
        return np.full(len(strikes), 1e6)
    model = sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu)
    return model - market_vols


def _sse(params, F, strikes, market_vols, T, beta):
    """Scalar SSE for differential_evolution."""
    res = _residuals(params, F, strikes, market_vols, T, beta)
    return float(np.sum(res ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice calibrator
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_slice(
    strikes:     np.ndarray,
    market_vols: np.ndarray,
    T:           float,
    expiration:  str,
    beta:        float = 0.5,
    forward:     Optional[float] = None,
    n_restarts:  int = 5,
) -> CalibrationResult:
    """
    Calibrate (alpha, rho, nu) for a single expiry slice.

    Strategy
    --------
    1. Use differential_evolution (global) for a robust initial guess.
    2. Refine with least_squares (local, Levenberg-Marquardt).
    3. Return the best result across n_restarts of the local step.

    Parameters
    ----------
    strikes      : array of strike prices
    market_vols  : array of market-implied vols (same length)
    T            : time-to-expiry (years)
    expiration   : label string (e.g. "2026-06-20")
    beta         : fixed CEV exponent (default 0.5)
    forward      : ATM forward; estimated from data if None
    n_restarts   : number of random re-starts for local solver
    """
    # ── Data cleaning ────────────────────────────────────────────────────────
    mask = (
        np.isfinite(market_vols) & (market_vols > 1e-4) &
        np.isfinite(strikes)     & (strikes > 0)
    )
    strikes     = strikes[mask]
    market_vols = market_vols[mask]

    if len(strikes) < 3:
        raise ValueError(f"Not enough valid data points for {expiration}: {len(strikes)}")

    if forward is None:
        forward = estimate_forward(strikes, market_vols)

    F = forward

    # ── Bounds & global search ───────────────────────────────────────────────
    bounds_de = [
        (1e-4, 5.0),    # alpha
        (-0.999, 0.999),# rho
        (1e-4, 5.0),    # nu
    ]

    de_result = differential_evolution(
        _sse, bounds_de,
        args=(F, strikes, market_vols, T, beta),
        maxiter=300, tol=1e-8, seed=42, polish=True,
        workers=1, updating='deferred',
    )
    best_x   = de_result.x
    best_sse = de_result.fun

    # ── Local refinement with multiple restarts ───────────────────────────────
    rng = np.random.default_rng(0)
    for _ in range(n_restarts):
        x0 = best_x + rng.normal(0, 0.05, size=3)
        x0[0] = np.clip(x0[0], 1e-4, 5.0)
        x0[1] = np.clip(x0[1], -0.99, 0.99)
        x0[2] = np.clip(x0[2], 1e-4, 5.0)

        try:
            res = least_squares(
                _residuals, x0,
                args=(F, strikes, market_vols, T, beta),
                bounds=([1e-4, -0.999, 1e-4], [5.0, 0.999, 5.0]),
                method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10,
                max_nfev=5000,
            )
            sse = float(np.sum(res.fun ** 2))
            if sse < best_sse:
                best_sse = sse
                best_x   = res.x
        except Exception:
            continue

    alpha, rho, nu = best_x
    model_vols = sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu)
    sse  = float(np.sum((model_vols - market_vols) ** 2))
    rmse = float(np.sqrt(sse / len(strikes)))

    return CalibrationResult(
        expiration=expiration,
        maturity=T,
        alpha=alpha, beta=beta, rho=rho, nu=nu,
        forward=F,
        sse=sse, rmse=rmse,
        n_points=len(strikes),
        strikes=strikes,
        market_vols=market_vols,
        model_vols=model_vols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Surface calibrator
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_surface(
    df,
    beta:      float = 0.5,
    min_points: int  = 5,
    verbose:   bool  = True,
) -> List[CalibrationResult]:
    """
    Loop over all expiration slices in the dataframe and calibrate each.

    Parameters
    ----------
    df         : pandas DataFrame with columns
                 [strike, expiration, maturity_years, impliedVolatility]
    beta       : fixed CEV exponent
    min_points : minimum number of quotes to attempt calibration
    verbose    : print per-slice diagnostics

    Returns
    -------
    List of CalibrationResult, one per expiry (sorted by maturity).
    """
    results = []
    expirations = df.groupby('expiration')

    for exp_label, group in sorted(expirations, key=lambda x: x[1]['maturity_years'].iloc[0]):
        T       = float(group['maturity_years'].iloc[0])
        strikes = group['strike'].values.astype(float)
        ivs     = group['impliedVolatility'].values.astype(float)

        if len(strikes) < min_points:
            if verbose:
                print(f"  SKIP {exp_label}: only {len(strikes)} points")
            continue

        try:
            result = calibrate_slice(strikes, ivs, T, exp_label, beta=beta)
            results.append(result)
            if verbose:
                print(result)
                print()
        except Exception as e:
            if verbose:
                print(f"  ERROR {exp_label}: {e}\n")

    return results