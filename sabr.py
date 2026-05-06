"""
sabr.py
=======
Hagan et al. (2002) SABR implied-volatility approximation.

Parameters
----------
alpha : initial (instantaneous) volatility  (α > 0)
beta  : CEV exponent, fixed externally       (0 ≤ β ≤ 1)
rho   : forward-vol correlation              (-1 < ρ < 1)
nu    : volatility of volatility             (ν ≥ 0)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _validate_market_inputs(F: float, K: float, T: float) -> None:
    if F <= 0 or K <= 0 or T <= 0:
        raise ValueError("F, K must be positive and T must be positive for SABR implied vol.")


def _validate_sabr_params(alpha: float, beta: float, rho: float, nu: float) -> None:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1].")
    if not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1.")
    if nu < 0:
        raise ValueError("nu must be non-negative.")


def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float,
             eps: float = 1e-7) -> float:
    """
    Single-strike SABR implied vol (Black-Scholes basis).
    Uses the ATM-limit formula when |F - K| < eps * F.
    """
    _validate_market_inputs(F, K, T)

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

def d1(F,K,sigma,T):
    return (np.log(F/K)+0.5*(sigma**2)*T)/(sigma*np.sqrt(T))

def vega_BS(F: float, K:float, r: float, T: float, sigma:float) -> float:
    d = d1(F,K,sigma,T)
    vega_BS = F * np.exp(-r*T) * norm.pdf(d) * np.sqrt(T)
    return vega_BS
def dsigma_dalpha(F: float, K: float, T: float, alpha: float, beta:float, rho:float, nu:float):
    eps = 1e-4
    sigma_up = sabr_vol(F,K,T,alpha+eps,beta,rho,nu)
    sigma = sabr_vol(F,K,T,alpha,beta,rho,nu)
    dsig_dalp = (sigma_up-sigma)/eps
    return dsig_dalp
def dsigma_dF(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float):
    eps = 1e-4
    sigma_up = sabr_vol(F + eps, K, T, alpha, beta, rho, nu)
    sigma = sabr_vol(F, K, T, alpha, beta, rho, nu)
    return (sigma_up - sigma) / eps

def delta_BS(F: float,K: float,sigma:float,T:float,r:float,option:str):
    d = d1(F,K,sigma,T)
    if option == "call":
        delta = np.exp(-r*T)*norm.cdf(d)
        return delta
    if option == "put":
        delta = np.exp(-r*T)*(norm.cdf(d)-1)
        return delta
    raise ValueError(f"option must be 'call' or 'put', got '{option}'")

class SABRModel:
    """Convenience wrapper holding (α, β, ρ, ν) and exposing `.vol(F, K, T)`."""

    __slots__ = ("alpha", "beta", "rho", "nu")

    def __init__(self, alpha: float, beta: float, rho: float, nu: float) -> None:
        _validate_sabr_params(alpha, beta, rho, nu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.nu = float(nu)

    def vol(self, F: float, K: float, T: float) -> float:
        return sabr_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu)
    
    def vega(self, F: float, K: float, T: float, r:float) -> float:
        sigma = self.vol(F,K,T)
        return vega_BS(F,K,r,T,sigma)*dsigma_dalpha(F,K,T,self.alpha,self.beta,self.rho,self.nu)
        
    def delta(self, F: float, K: float, T: float, r: float, option: str) -> float:
        sigma = self.vol(F, K, T)
        return delta_BS(F, K, sigma, T, r, option)+vega_BS(F,K,r,T,sigma)*dsigma_dF(F,K,T,self.alpha,self.beta,self.rho,self.nu)

    def __repr__(self) -> str:
        return (
            f"SABRModel(alpha={self.alpha}, beta={self.beta}, "
            f"rho={self.rho}, nu={self.nu})"
        )