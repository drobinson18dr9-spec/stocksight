"""
StockSight out-of-sample backtest.

The honest test of the screen: does ranking stocks by our signal actually
separate future winners from losers? At each monthly rebalance we compute the
signal using ONLY data available up to that date (strictly point-in-time, no
lookahead), buy the top decile, and measure the forward 1-month return against
the bottom decile and against SPY. Chaining those forward returns gives an
out-of-sample equity curve.

Key honesty caveat: the universe is currently-listed tickers, so delisted
names are missing (survivorship bias). This inflates ABSOLUTE returns AND
biases the GOOD-minus-BAD spread: names that delisted (often the worst) never
enter the BAD bucket, so the spread is NOT survivorship-neutral. Treat all
figures as upper bounds; a point-in-time universe with delisted-name returns
would be required to remove the bias.

Usage: python src/backtest.py [--years 5] [--universe-limit N]
Outputs reports/charts/backtest.png and reports/backtest_summary.md
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scorecard as sc

CHART_DIR = Path(__file__).resolve().parents[1] / "reports" / "charts"
REPORTS = Path(__file__).resolve().parents[1] / "reports"

HOLD = 21                      # ~1 month holding / rebalance step
WINDOW = sc.TRADING_DAYS       # 252-day trailing signal window
REBAL_COST = 0.001             # 10 bps per rebalance (turnover proxy)


def annualized_stats(monthly_rets: pd.Series, periods_per_year: float):
    r = monthly_rets.dropna()
    if len(r) < 3:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    equity = (1 + r).cumprod()
    n_years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year - sc.RISK_FREE_RATE) / vol if vol > 0 else np.nan
    maxdd = (equity / equity.cummax() - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=maxdd)


def main(years=5, universe_limit=None):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    bars = sc.get_bars(universe_limit=universe_limit, lookback_days=int(years * 365) + 400)

    close = bars.pivot_table(index="timestamp", columns="ticker", values="close").sort_index()
    rets = close.pct_change()

    # Point-in-time signal panels (every cell uses only past data) -------
    rf_d = sc.RF_DAILY
    sharpe_panel = (rets.rolling(WINDOW).mean() - rf_d) / rets.rolling(WINDOW).std() * np.sqrt(WINDOW)
    mom_panel = close.shift(HOLD) / close.shift(WINDOW) - 1
    # Sanitize: near-zero-vol names produce inf Sharpe; drop those cells.
    sharpe_panel = sharpe_panel.replace([np.inf, -np.inf], np.nan)
    mom_panel = mom_panel.replace([np.inf, -np.inf], np.nan)
    eligible = (close >= sc.MIN_PRICE) & sharpe_panel.notna() & mom_panel.notna()

    dates = close.index
    start_i = WINDOW + 1
    rebal = list(range(start_i, len(dates) - HOLD, HOLD))
    periods_per_year = 252 / HOLD

    good, bad, spy = [], [], []
    n_held = []
    for i in rebal:
        sh, mo, el = sharpe_panel.iloc[i], mom_panel.iloc[i], eligible.iloc[i]
        valid = el & sh.notna() & mo.notna()
        valid[sc.BENCHMARK] = False            # SPY is the benchmark, not a pick (audit fix)
        sh, mo = sh[valid], mo[valid]
        if len(sh) < 30:
            continue
        # Rank-based composite: immune to inf/outliers (cross-sectional percentile).
        comp = 0.5 * sh.rank(pct=True) + 0.5 * mo.rank(pct=True)
        k = max(10, int(0.10 * len(comp)))
        top = comp.nlargest(k).index
        bot = comp.nsmallest(k).index

        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1     # forward 1-month return
        g = fwd.reindex(top).dropna().mean() - REBAL_COST
        b = fwd.reindex(bot).dropna().mean() - REBAL_COST
        s = (close.iloc[i + HOLD][sc.BENCHMARK] / close.iloc[i][sc.BENCHMARK] - 1
             if sc.BENCHMARK in close.columns else np.nan)
        good.append(g); bad.append(b); spy.append(s); n_held.append(k)

    idx = [dates[i] for i in rebal[:len(good)]]
    good = pd.Series(good, index=idx); bad = pd.Series(bad, index=idx); spy = pd.Series(spy, index=idx)

    gs = annualized_stats(good, periods_per_year)
    bs = annualized_stats(bad, periods_per_year)
    ss = annualized_stats(spy, periods_per_year)
    hm = good.notna() & spy.notna()            # mask NaN benchmark months (audit fix)
    hit = float((good[hm] > spy[hm]).mean()) if hm.any() else float("nan")
    spread_ann = (good.mean() - bad.mean()) * periods_per_year

    # Equity curves ------------------------------------------------------
    eq_good = (1 + good).cumprod() * 100
    eq_bad = (1 + bad).cumprod() * 100
    eq_spy = (1 + spy).cumprod() * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq_good.index, eq_good, color="#1a9850", lw=2, label=f"GOOD top decile (CAGR {gs['cagr']*100:.0f}%)")
    ax.plot(eq_spy.index, eq_spy, color="black", lw=2, ls="--", label=f"SPY (CAGR {ss['cagr']*100:.0f}%)")
    ax.plot(eq_bad.index, eq_bad, color="#d73027", lw=2, label=f"BAD bottom decile (CAGR {bs['cagr']*100:.0f}%)")
    ax.set_yscale("log")
    ax.set_title(f"Out-of-sample backtest: top vs bottom decile vs SPY ({years}y, monthly rebalance)")
    ax.set_ylabel("Growth of $100 (log scale)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(CHART_DIR / "backtest.png", dpi=110); plt.close(fig)

    # Save equity curves for the interactive dashboard
    pd.DataFrame({"GOOD": eq_good, "SPY": eq_spy, "BAD": eq_bad}).to_csv(
        REPORTS / "backtest_equity.csv")

    # Summary ------------------------------------------------------------
    def row(name, st):
        return (f"| {name} | {st['cagr']*100:.1f}% | {st['vol']*100:.1f}% | "
                f"{st['sharpe']:.2f} | {st['maxdd']*100:.1f}% |")
    lines = [
        f"# StockSight backtest summary ({years}y, {len(good)} monthly rebalances)",
        "",
        f"Avg names held per side: {int(np.mean(n_held))}. Cost: {REBAL_COST*1e4:.0f} bps/rebalance.",
        "",
        "| Bucket | CAGR | Vol | Sharpe | Max DD |",
        "|--------|------|-----|--------|--------|",
        row("GOOD (top decile)", gs),
        row("SPY (benchmark)", ss),
        row("BAD (bottom decile)", bs),
        "",
        f"- GOOD beat SPY in {hit*100:.0f}% of months.",
        f"- GOOD-minus-BAD spread: {spread_ann*100:.1f}% annualized (the signal's separating power).",
        "",
        "Reading it: if GOOD > BAD by a wide margin out-of-sample, the ranking has",
        "real separating power, BUT survivorship bias (delisted names absent)",
        "inflates both absolute returns and the GOOD-minus-BAD spread; treat as upper bounds.",
    ]
    (REPORTS / "backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved chart to {CHART_DIR / 'backtest.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--universe-limit", type=int, default=None)
    args = ap.parse_args()
    main(years=args.years, universe_limit=args.universe_limit)
