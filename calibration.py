"""
calibration.py  —  Week 4
==========================
Fits (alpha, rho, nu) to a single expiry slice of market IV data,
with beta fixed (typically 0.5 for equity indices).

Strategy
--------
1. Global search via differential_evolution  →  robust initial guess
2. Local refinement via least_squares (TRF)  →  tight convergence
3. Returns a CalibrationResult with SSE / RMSE diagnostics
"""

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from dataclasses import dataclass, field
from typing import Optional, List

from sabr import sabr_vol_vec


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    expiration:  str
    maturity:    float
    forward:     float
    alpha:       float
    beta:        float
    rho:         float
    nu:          float
    sse:         float
    rmse:        float
    n_points:    int
    strikes:     np.ndarray = field(repr=False)
    market_vols: np.ndarray = field(repr=False)
    model_vols:  np.ndarray = field(repr=False)

    def summary(self) -> str:
        return (
            f"{self.expiration:<14}  T={self.maturity:.4f}y  F={self.forward:.1f}  "
            f"α={self.alpha:.5f}  β={self.beta:.2f}  ρ={self.rho:.4f}  ν={self.nu:.4f}  "
            f"RMSE={self.rmse*100:.4f}%  n={self.n_points}"
        )


# ─────────────────────────────────────────────────────────────────────────────

def _residuals(params, F, strikes, market_vols, T, beta):
    alpha, rho, nu = params
    if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
        return np.full(len(strikes), 1e6)
    return sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu) - market_vols


def _sse_scalar(params, F, strikes, market_vols, T, beta):
    return float(np.sum(_residuals(params, F, strikes, market_vols, T, beta) ** 2))


def _estimate_forward(strikes, market_vols):
    """Minimum-IV strike as ATM-forward proxy."""
    return float(strikes[np.argmin(market_vols)])


def calibrate_slice(
    strikes:     np.ndarray,
    market_vols: np.ndarray,
    T:           float,
    expiration:  str,
    beta:        float = 0.5,
    forward:     Optional[float] = None,
) -> CalibrationResult:
    """
    Calibrate one expiry slice.

    Parameters
    ----------
    strikes      : strike prices
    market_vols  : market implied vols (matched length)
    T            : time-to-expiry in years
    expiration   : label string
    beta         : fixed CEV exponent (default 0.5)
    forward      : ATM forward; auto-estimated if None
    """
    # Clean
    mask = (
        np.isfinite(market_vols) & (market_vols > 0.005) &
        np.isfinite(strikes)     & (strikes > 0)
    )
    strikes, market_vols = strikes[mask], market_vols[mask]
    if len(strikes) < 3:
        raise ValueError(f"Insufficient data for {expiration}: {len(strikes)} points")

    F = forward if forward is not None else _estimate_forward(strikes, market_vols)

    # ── Stage 1: global search ───────────────────────────────────────────────
    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]   # alpha, rho, nu
    de = differential_evolution(
        _sse_scalar, bounds,
        args=(F, strikes, market_vols, T, beta),
        maxiter=400, tol=1e-9, seed=42, polish=True,
        workers=1, updating="deferred",
    )
    best_x, best_sse = de.x, de.fun

    # ── Stage 2: local refinement (a few random restarts) ────────────────────
    rng = np.random.default_rng(0)
    for _ in range(8):
        x0 = best_x + rng.normal(0, 0.05, 3)
        x0 = np.clip(x0, [1e-4, -0.999, 1e-4], [5.0, 0.999, 5.0])
        try:
            res = least_squares(
                _residuals, x0,
                args=(F, strikes, market_vols, T, beta),
                bounds=([1e-4, -0.999, 1e-4], [5.0, 0.999, 5.0]),
                method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=10_000,
            )
            s = float(np.sum(res.fun ** 2))
            if s < best_sse:
                best_sse, best_x = s, res.x
        except Exception:
            continue

    alpha, rho, nu = best_x
    model_vols = sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu)
    sse  = float(np.sum((model_vols - market_vols) ** 2))
    rmse = float(np.sqrt(sse / len(strikes)))

    return CalibrationResult(
        expiration=expiration, maturity=T, forward=F,
        alpha=alpha, beta=beta, rho=rho, nu=nu,
        sse=sse, rmse=rmse, n_points=len(strikes),
        strikes=strikes, market_vols=market_vols, model_vols=model_vols,
    )


def calibrate_surface(
    df,
    beta:       float = 0.5,
    min_points: int   = 5,
    verbose:    bool  = True,
) -> List[CalibrationResult]:
    """
    Calibrate all expiry slices in a DataFrame with columns:
      [strike, expiration, maturity_years, impliedVolatility]
    """
    results = []
    groups  = sorted(
        df.groupby("expiration"),
        key=lambda x: x[1]["maturity_years"].iloc[0],
    )
    for exp_label, group in groups:
        T   = float(group["maturity_years"].iloc[0])
        K   = group["strike"].values.astype(float)
        iv  = group["impliedVolatility"].values.astype(float)

        if np.sum((iv > 0.005) & np.isfinite(iv)) < min_points:
            if verbose:
                print(f"  SKIP {exp_label}: too few valid points")
            continue

        try:
            r = calibrate_slice(K, iv, T, exp_label, beta=beta)
            results.append(r)
            if verbose:
                print(r.summary())
        except Exception as e:
            if verbose:
                print(f"  ERROR {exp_label}: {e}")

    return results