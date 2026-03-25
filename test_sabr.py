"""
Unit tests for the SABR implied volatility formula (sabr.py).
Run with:  python -m pytest test_sabr.py -v
"""

import math
import pytest
from sabr import sabr_vol, SABRModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx(expected, rel=1e-6):
    """Relative-tolerance wrapper for pytest.approx."""
    return pytest.approx(expected, rel=rel)


# ---------------------------------------------------------------------------
# ATM edge case: K == F  (formula reduces to Eq. 2.18)
# ---------------------------------------------------------------------------

class TestATM:
    """vol(F, K=F, T) must use the ATM formula without hitting 0/0."""

    def test_atm_exact(self):
        # Typical equity-like params (beta=0.5)
        vol = sabr_vol(F=100.0, K=100.0, T=1.0,
                       alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        assert vol > 0

    def test_atm_nearly_equal(self):
        # K very close to F should give the same answer as K==F
        vol_exact = sabr_vol(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4)
        vol_near  = sabr_vol(100.0, 100.0 + 1e-10, 1.0, 0.2, 0.5, -0.3, 0.4)
        assert abs(vol_exact - vol_near) < 1e-8

    def test_atm_beta0(self):
        # beta=0: verify against the ATM formula (Eq. 2.18) directly
        alpha, beta, rho, nu, T, F = 0.2, 0.0, 0.0, 0.3, 1.0, 100.0
        vol = sabr_vol(F, F, T, alpha, beta, rho, nu)
        F_1mb = F ** (1 - beta)
        F_2_1mb = F ** (2 * (1 - beta))
        correction = (
            (1 - beta)**2 * alpha**2 / (24 * F_2_1mb)
            + rho * beta * nu * alpha / (4 * F_1mb)
            + (2 - 3 * rho**2) * nu**2 / 24
        )
        expected = (alpha / F_1mb) * (1 + correction * T)
        assert vol == approx(expected)

    def test_atm_beta1(self):
        # beta=1: stochastic log-normal model; 1-beta terms vanish
        # correction = (2-3*rho^2)*nu^2/24 * T
        alpha, rho, nu, T = 0.25, -0.2, 0.5, 0.5
        vol = sabr_vol(100.0, 100.0, T, alpha, 1.0, rho, nu)
        correction = (rho * 1.0 * nu * alpha) / 4.0 + (2 - 3 * rho**2) * nu**2 / 24.0
        expected = alpha * (1 + correction * T)
        assert vol == approx(expected)

    def test_atm_zero_nu(self):
        # nu=0 and rho=0: reduce to pure CEV; correction = (1-β)²α²/(24 F^{2(1-β)}) * T
        alpha, beta, T, F = 0.3, 0.5, 2.0, 80.0
        vol = sabr_vol(F, F, T, alpha, beta, 0.0, 0.0)
        F_1mb = F ** (1 - beta)
        F_2_1mb = F ** (2 * (1 - beta))
        expected = (alpha / F_1mb) * (1 + (1 - beta)**2 * alpha**2 / (24 * F_2_1mb) * T)
        assert vol == approx(expected)


# ---------------------------------------------------------------------------
# General (off-ATM) cases
# ---------------------------------------------------------------------------

class TestOffATM:
    """Sanity checks for strikes away from the forward."""

    def test_otm_call_positive(self):
        vol = sabr_vol(100.0, 110.0, 1.0, 0.3, 0.5, -0.3, 0.5)
        assert vol > 0

    def test_itm_call_positive(self):
        vol = sabr_vol(100.0, 90.0, 1.0, 0.3, 0.5, -0.3, 0.5)
        assert vol > 0

    def test_smile_convexity_zero_rho(self):
        # With rho=0 the smile should be convex (wings higher than ATM)
        F, alpha, beta, rho, nu, T = 100.0, 0.2, 0.5, 0.0, 0.5, 1.0
        vol_atm  = sabr_vol(F, F,       T, alpha, beta, rho, nu)
        vol_high = sabr_vol(F, F * 1.2, T, alpha, beta, rho, nu)
        vol_low  = sabr_vol(F, F * 0.8, T, alpha, beta, rho, nu)
        assert vol_high > vol_atm
        assert vol_low  > vol_atm

    def test_negative_skew_with_negative_rho(self):
        # Negative rho => downward sloping smile (higher strikes => lower vol)
        F, T = 100.0, 1.0
        params = dict(alpha=0.25, beta=0.5, rho=-0.5, nu=0.4)
        vol_low  = sabr_vol(F, 90.0,  T, **params)
        vol_atm  = sabr_vol(F, 100.0, T, **params)
        vol_high = sabr_vol(F, 110.0, T, **params)
        assert vol_low > vol_atm > vol_high

    def test_increasing_nu_increases_wings(self):
        # Higher vol-of-vol => fatter wings (smile curvature increases)
        F, K_wing, T = 100.0, 115.0, 1.0
        base = dict(alpha=0.2, beta=0.5, rho=0.0, T=T)
        vol_low_nu  = sabr_vol(F, K_wing, T, 0.2, 0.5, 0.0, 0.1)
        vol_high_nu = sabr_vol(F, K_wing, T, 0.2, 0.5, 0.0, 0.8)
        assert vol_high_nu > vol_low_nu

    def test_continuity_near_atm(self):
        # vol should be continuous as K crosses F
        F, T = 100.0, 1.0
        params = (F, T, 0.2, 0.5, -0.3, 0.4)
        eps = 1e-5
        vol_below = sabr_vol(F, F - eps, T, 0.2, 0.5, -0.3, 0.4)
        vol_atm   = sabr_vol(F, F,       T, 0.2, 0.5, -0.3, 0.4)
        vol_above = sabr_vol(F, F + eps, T, 0.2, 0.5, -0.3, 0.4)
        assert abs(vol_below - vol_atm) < 1e-5
        assert abs(vol_above - vol_atm) < 1e-5


# ---------------------------------------------------------------------------
# SABRModel class interface
# ---------------------------------------------------------------------------

class TestSABRModel:

    def test_class_matches_function(self):
        model = SABRModel(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        assert model.vol(100.0, 100.0, 1.0) == sabr_vol(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4)
        assert model.vol(100.0, 110.0, 1.0) == sabr_vol(100.0, 110.0, 1.0, 0.2, 0.5, -0.3, 0.4)

    def test_repr(self):
        model = SABRModel(0.2, 0.5, -0.3, 0.4)
        assert "SABRModel" in repr(model)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:

    @pytest.mark.parametrize("F,K,T", [
        (0.0,  100.0, 1.0),   # F = 0
        (-1.0, 100.0, 1.0),   # F < 0
        (100.0, 0.0,  1.0),   # K = 0
        (100.0, -5.0, 1.0),   # K < 0
        (100.0, 100.0, 0.0),  # T = 0
        (100.0, 100.0, -1.0), # T < 0
    ])
    def test_invalid_market_inputs(self, F, K, T):
        with pytest.raises(ValueError):
            sabr_vol(F, K, T, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.0, beta=0.5, rho=0.0, nu=0.3)

    def test_invalid_beta_low(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.2, beta=-0.1, rho=0.0, nu=0.3)

    def test_invalid_beta_high(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.2, beta=1.1, rho=0.0, nu=0.3)

    def test_invalid_rho_low(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.2, beta=0.5, rho=-1.0, nu=0.3)

    def test_invalid_rho_high(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.2, beta=0.5, rho=1.0, nu=0.3)

    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            SABRModel(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1)
