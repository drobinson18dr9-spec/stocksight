"""
StockSight visualizations. Generates PNG charts from the latest scorecard so
you can SEE what the screen is doing:

  1. score_distribution.png  - composite score histogram with GOOD/BAD bands
  2. risk_return.png         - annualized return vs volatility, colored by verdict
  3. picks_vs_spy.png        - top picks' trailing-1y price path vs SPY (normalized)
  4. portfolio_weights.png   - the optimized portfolio's weights

Usage: python src/visualize.py [--universe-limit N]
Outputs land in reports/charts/.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # headless backend (works in the cloud too)
import matplotlib.pyplot as plt
import numpy as np

import scorecard as sc

CHART_DIR = Path(__file__).resolve().parents[1] / "reports" / "charts"
COLORS = {"GOOD": "#1a9850", "NEUTRAL": "#999999", "BAD": "#d73027"}


def main(universe_limit=None):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    bars = sc.get_bars(universe_limit=universe_limit)
    metrics = sc.compute_metrics(bars)
    investable = sc.filter_investable(metrics)
    scored = sc.score(investable)
    portfolio = sc.build_portfolio(scored, bars, top_n=12, max_weight=0.25)
    asof = bars["timestamp"].max().strftime("%Y-%m-%d")

    # 1) Composite score distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for v in ["BAD", "NEUTRAL", "GOOD"]:
        s = scored.loc[scored["verdict"] == v, "composite"]
        ax.hist(s, bins=60, alpha=0.7, color=COLORS[v], label=f"{v} ({len(s)})")
    ax.set_title(f"Composite score distribution, {len(scored)} investable names ({asof})")
    ax.set_xlabel("Composite score (robust z blend)")
    ax.set_ylabel("Number of stocks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CHART_DIR / "score_distribution.png", dpi=110); plt.close(fig)

    # 2) Risk-return scatter ---------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for v in ["NEUTRAL", "BAD", "GOOD"]:
        d = scored[scored["verdict"] == v]
        ax.scatter(d["ann_vol"] * 100, d["cagr"] * 100, s=10, alpha=0.5,
                   color=COLORS[v], label=v)
    spy = scored[scored["ticker"] == sc.BENCHMARK]
    if len(spy):
        ax.scatter(spy["ann_vol"] * 100, spy["cagr"] * 100, s=200, marker="*",
                   color="black", label="SPY", zorder=5)
    # annotate the portfolio names
    for _, r in portfolio.head(8).iterrows():
        m = scored[scored["ticker"] == r["ticker"]]
        if len(m):
            ax.annotate(r["ticker"], (m["ann_vol"].iloc[0]*100, m["cagr"].iloc[0]*100),
                        fontsize=8, fontweight="bold")
    ax.set_title(f"Risk vs return, trailing 1y ({asof})")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return / CAGR (%)")
    ax.set_ylim(top=min(scored["cagr"].quantile(0.99)*100, 400))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CHART_DIR / "risk_return.png", dpi=110); plt.close(fig)

    # 3) Top picks vs SPY, normalized ------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    picks = portfolio.head(6)["ticker"].tolist() or scored.head(6)["ticker"].tolist()
    for t in picks + [sc.BENCHMARK]:
        d = bars[bars["ticker"] == t].sort_values("timestamp").tail(252)
        if len(d) < 50:
            continue
        norm = d["close"] / d["close"].iloc[0] * 100
        lw = 2.5 if t == sc.BENCHMARK else 1.4
        style = "--" if t == sc.BENCHMARK else "-"
        ax.plot(d["timestamp"], norm, style, linewidth=lw, label=t)
    ax.set_title(f"Portfolio picks vs SPY, trailing 1y (start = 100) ({asof})")
    ax.set_ylabel("Growth of 100")
    ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CHART_DIR / "picks_vs_spy.png", dpi=110); plt.close(fig)

    # 4) Portfolio weights -----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    p = portfolio[portfolio["weight"] > 0.005].sort_values("weight")
    ax.barh(p["ticker"], p["weight"] * 100, color="#2c7fb8")
    for i, (_, r) in enumerate(p.iterrows()):
        ax.text(r["weight"]*100 + 0.3, i, f"{r['weight']*100:.1f}%", va="center", fontsize=8)
    ax.set_title(f"Optimized max-Sharpe portfolio weights ({asof})")
    ax.set_xlabel("Weight (%)")
    fig.tight_layout(); fig.savefig(CHART_DIR / "portfolio_weights.png", dpi=110); plt.close(fig)

    print(f"Saved 4 charts to {CHART_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-limit", type=int, default=None)
    args = ap.parse_args()
    main(universe_limit=args.universe_limit)
