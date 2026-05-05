"""
smile_fit.py  —  Week 5
========================
Run the calibration on all expiry slices, then plot each resulting
smile (market quotes vs SABR fit) and print a performance-metrics table.

Usage
-----
    python smile_fit.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from calibration import calibrate_surface
from sabr import sabr_vol_vec

DATA_PATH = "^SPX_iv_surface.csv"
OUT_DIR   = "smile_curves_output"
BETA      = 0.5      # fixed for SPX (square-root / Heston-like regime)
MIN_PTS   = 6

# ── palette ──────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load & clean
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["impliedVolatility"] > 0.005) &
            (df["strike"] > 0) &
            (df["maturity_years"] > 0)].copy()
    # one row per (strike, expiration) — keep highest open interest
    df["oi_vol"] = df["openInterest"] * df["volume"]
    df = (
        df.sort_values("oi_vol", ascending=False)
          .drop_duplicates(subset=["strike", "expiration"], keep="first")
          .sort_values(["expiration", "strike"])
          .reset_index(drop=True)
    )
    print(f"[data] {len(df):,} rows across {df['expiration'].nunique()} expiries")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Performance-metrics table (SSE / RMSE per slice)
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(results):
    hdr = f"{'Expiry':<14} {'T(y)':>6} {'F':>8} {'α':>8} {'β':>5} {'ρ':>8} {'ν':>8} {'RMSE%':>8} {'SSE':>10} {'n':>5}"
    bar = "─" * len(hdr)
    print(f"\n{bar}\n{hdr}\n{bar}")
    for r in results:
        print(
            f"{r.expiration:<14} {r.maturity:>6.3f} {r.forward:>8.1f} "
            f"{r.alpha:>8.5f} {r.beta:>5.2f} {r.rho:>8.4f} {r.nu:>8.4f} "
            f"{r.rmse*100:>8.4f} {r.sse:>10.6f} {r.n_points:>5}"
        )
    agg_rmse = np.mean([r.rmse for r in results])
    agg_sse  = np.sum( [r.sse  for r in results])
    n_total  = sum(r.n_points for r in results)
    print(bar)
    print(
        f"{'AGGREGATE':<14} {'':>6} {'':>8} {'':>8} {'':>5} {'':>8} {'':>8} "
        f"{agg_rmse*100:>8.4f} {agg_sse:>10.6f} {n_total:>5}"
    )
    print(bar + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Smile plots  (Week 5 deliverable)
# ─────────────────────────────────────────────────────────────────────────────

def plot_smiles(results, max_panels=9, out_path="smile_curves.png"):
    """
    Grid of per-expiry panels.
    X-axis: moneyness K/F  —  comparable across all maturities.
    Market quotes as scatter; SABR fit as smooth line.
    """
    n    = min(len(results), max_panels)
    idxs = np.round(np.linspace(0, len(results) - 1, n)).astype(int)
    sel  = [results[i] for i in idxs]

    cols = 3
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(14, rows * 3.9 + 1.2), facecolor=BG)
    fig.suptitle(
        "SPX Implied-Volatility Smiles  —  SABR Calibration  (β = 0.5)",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(
        rows, cols, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    palette = plt.cm.plasma(np.linspace(0.18, 0.88, n))

    for i, (r, clr) in enumerate(zip(sel, palette)):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)

        mono_mkt = r.strikes / r.forward

        # smooth model curve on a fine moneyness grid
        K_lo  = max(r.strikes.min(), r.forward * 0.70)
        K_hi  = min(r.strikes.max(), r.forward * 1.30)
        K_fine = np.linspace(K_lo, K_hi, 300)
        M_fine = K_fine / r.forward
        IV_fine = sabr_vol_vec(r.forward, K_fine, r.maturity,
                               r.alpha, r.beta, r.rho, r.nu) * 100

        ax.scatter(mono_mkt, r.market_vols * 100,
                   s=20, color=clr, alpha=0.85, zorder=3, label="Market")
        ax.plot(M_fine, IV_fine,
                color=clr, linewidth=1.8, alpha=0.95, label="SABR")
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":")

        T_lbl = f"{r.maturity:.2f}y" if r.maturity < 1 else f"{r.maturity:.1f}y"
        ax.set_title(
            f"{r.expiration}  ({T_lbl})   RMSE = {r.rmse*100:.3f}%",
            fontsize=8.5, fontweight="bold", pad=5,
        )
        ax.set_xlabel("Moneyness  K / F", fontsize=8)
        ax.set_ylabel("Impl. Vol  (%)",   fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
                  labelcolor=TEXT, markerscale=1.0)

    for j in range(n, rows * cols):
        fig.add_subplot(gs[j // cols, j % cols]).set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[plot] Smile grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df      = load(DATA_PATH)
    results = calibrate_surface(df, beta=BETA, min_points=MIN_PTS, verbose=True)

    if not results:
        print("No slices calibrated — check data.")
        return

    print_metrics(results)
    plot_smiles(results, max_panels=9,
                out_path=os.path.join(OUT_DIR, "smile_curves.png"))


if __name__ == "__main__":
    main()