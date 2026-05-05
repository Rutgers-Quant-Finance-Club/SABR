"""
surface_3d.py  —  Week 6
=========================
Surface Generation:
  - Expand the calibration to loop through multiple expiration dates.
  - Generate the full 3D volatility surface (strike, maturity, implied vol).

Usage
-----
    python surface_3d.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from calibration import calibrate_surface
from sabr import sabr_vol_vec


DATA_PATH = "^SPX_iv_surface.csv"
OUT_DIR = "smile_curves_output"
BETA = 0.5       # fixed for SPX (square-root / Heston-like regime)
MIN_PTS = 6


def load(path: str) -> pd.DataFrame:
    """
    Load and lightly clean the raw implied-vol surface snapshot.

    Expected columns (from data_sourcing.get_clean_iv_surface):
      type, strike, expiration, days_to_expiry, maturity_years,
      impliedVolatility, mid, volume, openInterest
    """
    df = pd.read_csv(path)
    df = df[
        (df["impliedVolatility"] > 0.005)
        & (df["strike"] > 0)
        & (df["maturity_years"] > 0)
    ].copy()
    return df


def build_model_surface(results):
    """
    From a list of CalibrationResult objects, build scattered SABR vol points
    across (strike, maturity) for use in a 3D trisurf.
    """
    Ks = []
    Ts = []
    Vs = []

    for r in results:
        # Use the calibrated slice's strikes and model vols directly.
        K_slice = np.asarray(r.strikes, dtype=float)
        T_slice = float(r.maturity)
        V_slice = np.asarray(r.model_vols, dtype=float)

        # Sanity: ensure finite values only
        mask = np.isfinite(K_slice) & np.isfinite(V_slice) & (K_slice > 0) & (V_slice > 0)
        if not np.any(mask):
            continue

        Ks.append(K_slice[mask])
        Ts.append(np.full(mask.sum(), T_slice))
        Vs.append(V_slice[mask])

    if not Ks:
        raise ValueError("No valid calibration slices available to build surface.")

    K_all = np.concatenate(Ks)
    T_all = np.concatenate(Ts)
    V_all = np.concatenate(Vs)
    return K_all, T_all, V_all


def plot_surface_3d(K_all, T_all, V_all, out_path: str) -> None:
    """
    3D surface: X=strike, Y=maturity (years), Z=implied volatility (%).
    Uses a triangulated surface over the scattered SABR points.
    """
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        K_all,
        T_all,
        V_all * 100.0,
        cmap=cm.viridis,
        linewidth=0.2,
        antialiased=True,
        alpha=0.9,
    )

    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Volatility (%)")
    ax.set_title("SABR-Implied Volatility Surface (β = 0.5)")

    fig.colorbar(surf, shrink=0.5, aspect=12, label="Implied Volatility (%)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 3D surface saved → {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load(DATA_PATH)
    # Week 6 requirement: loop through multiple expirations using the calibration engine.
    results = calibrate_surface(df, beta=BETA, min_points=MIN_PTS, verbose=True)

    if not results:
        print("No slices calibrated — cannot build 3D surface.")
        return

    K_all, T_all, V_all = build_model_surface(results)
    out_path = os.path.join(OUT_DIR, "sabr_surface_3d.png")
    plot_surface_3d(K_all, T_all, V_all, out_path=out_path)


if __name__ == "__main__":
    main()

