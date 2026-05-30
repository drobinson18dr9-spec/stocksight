"""
StockSight Stage 1: quant-grade cross-sectional alpha engine.

What real quant desks do (and what this implements):
  - Predict CROSS-SECTIONAL relative forward returns, not price levels.
  - Factor features: momentum (12-1), short reversal (1m), low-volatility,
    trailing Sharpe, trend (vs 200d), max drawdown, volume trend.
  - Model: gradient-boosted trees (XGBoost) ranking forward returns.
  - Leakage-safe evaluation: expanding walk-forward with an EMBARGO equal to
    the label horizon, so no training label overlaps the test window
    (Lopez de Prado, Advances in Financial ML).
  - Anti-overfit statistics computed on the out-of-sample strategy returns:
      * Probabilistic Sharpe Ratio (PSR)         Bailey & Lopez de Prado (2012)
      * Deflated Sharpe Ratio (DSR)              Bailey & Lopez de Prado (2014)
  - Verdict: does the ML-ranked top quintile generate ALPHA vs SPY, out of
    sample, after the deflation that punishes multiple-testing?

This is the engine that answers "is it actually good." Honest result printed,
whatever it is.

Usage: python src/quant_alpha.py --years 6 --horizon 21 --trials 10
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

import scorecard as sc

REPORTS = Path(__file__).resolve().parents[1] / "reports"
CHART_DIR = REPORTS / "charts"

EULER = 0.5772156649015329


# ──────────────────────────────────────────────────────────────────────
# Feature panel (every feature is point-in-time, no lookahead)
# ──────────────────────────────────────────────────────────────────────
def build_panel(bars: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = bars.pivot_table(index="timestamp", columns="ticker", values="close").sort_index()
    vol = bars.pivot_table(index="timestamp", columns="ticker", values="volume").sort_index()
    rets = close.pct_change()
    rf_d = sc.RF_DAILY

    feats = {
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "reversal_1m": close / close.shift(21) - 1,
        "vol_60": rets.rolling(60).std() * np.sqrt(252),
        "sharpe_252": (rets.rolling(252).mean() - rf_d) / rets.rolling(252).std() * np.sqrt(252),
        "trend_200": close / close.rolling(200).mean() - 1,
        "dd_252": (close / close.rolling(252).max() - 1),
        "vol_trend": (vol.rolling(20).mean() / vol.rolling(60).mean() - 1),
    }
    # Eligibility: real, liquid names only
    dollar_vol = (close * vol).rolling(60).median()
    eligible = (close >= sc.MIN_PRICE) & (dollar_vol >= sc.MIN_MEDIAN_DOLLAR_VOL)

    fwd = close.shift(-horizon) / close - 1     # label: forward return

    # Long format
    frames = []
    for name, panel in feats.items():
        frames.append(panel.stack().rename(name))
    X = pd.concat(frames, axis=1)
    X["fwd_ret"] = fwd.stack()
    X["eligible"] = eligible.stack()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(subset=list(feats.keys()))        # features must be finite
    X = X[X["eligible"]].drop(columns="eligible")
    X.index.set_names(["date", "ticker"], inplace=True)
    return X.reset_index(), close


# ──────────────────────────────────────────────────────────────────────
# Anti-overfit statistics (real formulas)
# ──────────────────────────────────────────────────────────────────────
def probabilistic_sharpe(sr: float, n: int, sk: float, ku: float, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true Sharpe exceeds sr_benchmark, adjusting for
    skew/kurtosis of returns (Bailey & Lopez de Prado 2012). sr is per-period."""
    denom = np.sqrt(1 - sk * sr + ((ku - 1) / 4) * sr ** 2)
    if denom <= 0 or n < 2:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(sr: float, n: int, sk: float, ku: float, n_trials: int,
                    sr_trials_std: float) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the expected-maximum
    Sharpe under the null of zero skill across n_trials independent trials."""
    if n_trials < 1 or sr_trials_std <= 0:
        return np.nan
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = sr_trials_std * ((1 - EULER) * z1 + EULER * z2)   # expected max SR under null
    return probabilistic_sharpe(sr, n, sk, ku, sr_benchmark=sr0)


# ──────────────────────────────────────────────────────────────────────
# Walk-forward with embargo, XGBoost cross-sectional ranker
# ──────────────────────────────────────────────────────────────────────
def run(years: int = 6, horizon: int = 21, trials: int = 10, top_q: float = 0.2):
    import xgboost as xgb
    bars = sc.get_bars(lookback_days=2225)   # reuse the existing ~6y cache
    panel, close = build_panel(bars, horizon)
    feat_cols = ["mom_12_1", "reversal_1m", "vol_60", "sharpe_252", "trend_200", "dd_252", "vol_trend"]

    dates = np.sort(panel["date"].unique())
    # monthly rebalance dates with >= 1y of training history
    rebal = [d for i, d in enumerate(dates) if i >= 252 and i % horizon == 0]

    strat_rets, spy_rets, dates_used = [], [], []
    spy_close = close[sc.BENCHMARK] if sc.BENCHMARK in close.columns else None

    model = None
    for k, t in enumerate(rebal):
        # Embargo: training labels must be realized at least `horizon` days
        # before the test date, so no label window overlaps the test.
        embargo_cut = pd.Timestamp(t) - pd.Timedelta(days=int(horizon * 1.6) + 7)
        train = panel[panel["date"] <= embargo_cut]
        train = train[np.isfinite(train["fwd_ret"].values)]    # labels must be finite
        test = panel[panel["date"] == t]
        if len(train) < 5000 or len(test) < 30:
            continue
        # Refit every 3 rebalances for speed (model persists between)
        if model is None or k % 3 == 0:
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=1.0, n_jobs=4, verbosity=0)
            model.fit(train[feat_cols].values, train["fwd_ret"].values)

        pred = model.predict(test[feat_cols].values)
        test = test.assign(pred=pred)
        cut = test["pred"].quantile(1 - top_q)
        picks = test[test["pred"] >= cut]
        fwd_mean = picks["fwd_ret"].dropna().mean()
        if not np.isfinite(fwd_mean):
            continue
        strat_rets.append(float(fwd_mean))
        if spy_close is not None:
            loc = spy_close.index.get_indexer([pd.Timestamp(t)], method="nearest")[0]
            if loc + horizon < len(spy_close):
                spy_rets.append(float(spy_close.iloc[loc + horizon] / spy_close.iloc[loc] - 1))
            else:
                spy_rets.append(np.nan)
        dates_used.append(pd.Timestamp(t))

    s = pd.Series(strat_rets, index=dates_used)
    b = pd.Series(spy_rets, index=dates_used)
    m = s.notna() & b.notna()
    s, b = s[m], b[m]
    ppy = 252 / horizon
    active = s - b

    def ann(series):
        return (1 + series.mean()) ** ppy - 1

    sr_period = s.mean() / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
    sr_ann = sr_period * np.sqrt(ppy)
    te = active.std(ddof=1) * np.sqrt(ppy)
    info_ratio = (ann(s) - ann(b)) / te if te > 0 else np.nan
    psr = probabilistic_sharpe(sr_period, len(s), float(skew(s)), float(kurtosis(s, fisher=False)))
    # Deflate for the number of model/feature configurations we effectively tried
    dsr = deflated_sharpe(sr_period, len(s), float(skew(s)), float(kurtosis(s, fisher=False)),
                          n_trials=trials, sr_trials_std=max(s.std(ddof=1) / np.sqrt(len(s)), 1e-6))

    out = {
        "rebalances": int(len(s)),
        "horizon_days": horizon,
        "strategy_CAGR_%": round(ann(s) * 100, 2),
        "spy_CAGR_%": round(ann(b) * 100, 2),
        "annual_alpha_%": round((ann(s) - ann(b)) * 100, 2),
        "strategy_Sharpe_ann": round(sr_ann, 3),
        "information_ratio": round(float(info_ratio), 3),
        "hit_rate_vs_spy_%": round(float((s > b).mean()) * 100, 1),
        "PSR_vs_0": round(float(psr), 3),
        "Deflated_Sharpe": round(float(dsr), 3),
        "n_trials_assumed": trials,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "quant_alpha_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Stage 1: cross-sectional XGBoost alpha (out-of-sample, embargoed) ===")
    for k_, v_ in out.items():
        print(f"  {k_:24} {v_}")
    print("\nReading it:")
    print("  - annual_alpha_% > 0 means the ML ranker beat SPY out of sample.")
    print("  - Deflated_Sharpe is the honest test: > 0.95 means the result is")
    print("    unlikely to be a fluke of multiple testing; < 0.95 means treat")
    print("    the alpha as not proven.")
    _plot(s, b)
    return out


def _plot(s, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    eq_s = (1 + s).cumprod() * 100
    eq_b = (1 + b).cumprod() * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq_s.index, eq_s, color="#1a9850", lw=2.5, label="ML top-quintile (OOS)")
    ax.plot(eq_b.index, eq_b, color="black", lw=2.5, ls="--", label="SPY")
    ax.set_yscale("log")
    ax.set_title("Stage 1 alpha engine: out-of-sample ML ranker vs SPY")
    ax.set_ylabel("Growth of $100 (log)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(CHART_DIR / "quant_alpha.png", dpi=110); plt.close(fig)
    print(f"Saved chart: quant_alpha.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    run(years=args.years, horizon=args.horizon, trials=args.trials)
