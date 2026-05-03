"""
main.py
=======
End-to-end SABR pipeline:
  1. Load & clean the SPX options surface data
  2. Calibrate SABR per expiry slice  (Weeks 4–5)
  3. Generate 2-D smile curves        (Week 5)
  4. Generate 3-D vol surface         (Week 6)
  5. Compute & plot Greeks            (Week 7)
  6. Print performance metrics table  (Week 8)

Usage:  python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time

from calibration    import calibrate_surface
from visualization  import (
    plot_smile_curves,
    plot_surface_3d_mpl,
    plot_surface_3d_plotly,
    plot_greeks_surface,
    plot_calibration_diagnostics,
    plot_parameter_term_structure,
)

OUT_DIR  = "/mnt/user-data/outputs"
DATA_PATH = "/mnt/user-data/uploads/_SPX_iv_surface.csv"
BETA      = 0.5          # fix β = 0.5 (square-root / Bachelier blend for equity)
MIN_PTS   = 6            # min data points per slice to calibrate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop rows with degenerate IVs (yfinance placeholder = 1e-5)
    df = df[df["impliedVolatility"] > 0.005].copy()

    # Drop rows with non-positive strikes or maturities
    df = df[(df["strike"] > 0) & (df["maturity_years"] > 0)].copy()

    # For each (strike, expiration) keep only the most liquid option
    # (highest open interest) to avoid duplicate strikes per slice
    df["oi_vol"] = df["openInterest"] * df["volume"]
    df = (
        df.sort_values("oi_vol", ascending=False)
          .drop_duplicates(subset=["strike", "expiration"], keep="first")
          .sort_values(["expiration", "strike"])
          .reset_index(drop=True)
    )

    print(f"[data] Loaded {len(df):,} rows across "
          f"{df['expiration'].nunique()} expiry slices.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Performance-metrics table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(results):
    header = (
        f"{'Expiration':<14} {'T(y)':>6} {'F':>8} "
        f"{'α':>8} {'β':>5} {'ρ':>8} {'ν':>8} "
        f"{'RMSE(%)':>9} {'SSE':>10} {'n':>5}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.expiration:<14} {r.maturity:>6.3f} {r.forward:>8.1f} "
            f"{r.alpha:>8.5f} {r.beta:>5.2f} {r.rho:>8.4f} {r.nu:>8.4f} "
            f"{r.rmse*100:>9.4f} {r.sse:>10.6f} {r.n_points:>5}"
        )
    print(sep)

    rmses = [r.rmse for r in results]
    sses  = [r.sse  for r in results]
    print(
        f"{'AGGREGATE':<14} {'':>6} {'':>8} "
        f"{'':>8} {'':>5} {'':>8} {'':>8} "
        f"{np.mean(rmses)*100:>9.4f} {np.sum(sses):>10.6f} "
        f"{sum(r.n_points for r in results):>5}"
    )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    df = load_and_clean("^SPX_iv_surface.csv")

    # ── Step 2: Calibrate surface ────────────────────────────────────────────
    print("\n[calibration] Running SABR calibration across all expiry slices …")
    t0 = time.time()
    results = calibrate_surface(df, beta=BETA, min_points=MIN_PTS, verbose=True)
    elapsed = time.time() - t0
    print(f"[calibration] Done in {elapsed:.1f}s  —  {len(results)} slices calibrated.\n")

    if not results:
        print("ERROR: No slices were successfully calibrated.")
        return

    # ── Step 3: Performance metrics ──────────────────────────────────────────
    print_metrics_table(results)

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    plot_smile_curves(
        results,
        max_panels=9,
        output_path=f"{OUT_DIR}/smile_curves.png",
    )

    plot_surface_3d_mpl(
        results,
        output_path=f"{OUT_DIR}/vol_surface_3d.png",
    )

    plot_surface_3d_plotly(
        results,
        output_path=f"{OUT_DIR}/vol_surface_interactive.html",
    )

    plot_greeks_surface(
        results,
        output_path=f"{OUT_DIR}/greeks_surface.png",
    )

    plot_calibration_diagnostics(
        results,
        output_path=f"{OUT_DIR}/calibration_diagnostics.png",
    )

    plot_parameter_term_structure(
        results,
        output_path=f"{OUT_DIR}/parameter_term_structure.png",
    )

    print(f"\n[done] All outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()