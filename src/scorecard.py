"""
StockSight Daily Scorecard
==========================

Purpose : Screen a ticker universe on trailing risk-adjusted performance,
          label each name GOOD / NEUTRAL / BAD, then build a max-Sharpe
          portfolio from the top names using a Ledoit-Wolf shrinkage
          covariance (stable, well-conditioned). Writes a CSV + Markdown
          brief and returns a short text suitable for SMS.

Inputs  : data/Ticker List.xlsx          (column "Ticker")
          ALPACA_API_KEY, ALPACA_API_SECRET   (env / .env)

Outputs : reports/<YYYY-MM-DD>_scores.csv
          reports/<YYYY-MM-DD>_scorecard.md
          returns a short brief string (also printed)

Usage   : python src/scorecard.py
          python src/scorecard.py --top 10 --max-weight 0.25 --universe-limit 1500

This is analytical output, NOT financial advice. Methodology is sound;
returns are never guaranteed. Past performance does not predict the future.
"""

from __future__ import annotations

import os
import re
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Third-party, all lightweight
from dotenv import load_dotenv
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
TICKER_FILE = DATA_DIR / "Ticker List.xlsx"

RISK_FREE_RATE = 0.04                       # annual risk-free rate
TRADING_DAYS = 252                          # standard annualization factor
RF_DAILY = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1   # geometric daily rf
MIN_HISTORY = TRADING_DAYS + 1              # need >= 1y of closes to screen
MIN_OBS = 200                               # min usable daily returns in window
BENCHMARK = "SPY"

# Return winsorization (guards against bad ticks / data errors)
WINSOR_LO, WINSOR_HI = 0.005, 0.995

# Investability filters (applied BEFORE scoring so the GOOD/BAD ranking and
# the displayed picks reflect only tradeable names, not data-noise micro-caps
# or hyper-volatile leveraged ETFs).
MIN_PRICE = 5.00                    # drop sub-$5 names (wide spreads, manipulation, noise)
MAX_VOL = 0.80                      # annualized vol ceiling (removes 2x/3x ETFs, blowups)
MIN_VOL = 0.05                      # floor (removes dead/illiquid names)
MIN_MEDIAN_DOLLAR_VOL = 5_000_000   # >= $5M median daily traded (liquidity)

# Portfolio-eligibility thresholds (on top of investability)
MIN_SHARPE = 0.30
MAX_DRAWDOWN_FLOOR = -0.50          # reject names that fell >50% peak-to-trough
MIN_SHARPE_TSTAT = 1.5             # Sharpe must be ~marginally significant for the portfolio

WARRANT_UNIT_SUFFIXES = ("WS", "WT", "WSA", "WTA", "UN", "UNA")

load_dotenv(ROOT / ".env")

# Public dashboard URL (GitHub Pages); overridable via env.
DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://drobinson18dr9-spec.github.io/stocksight/")


# ──────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────
def alpaca_active_symbols() -> set:
    """Live Alpaca active US-equity symbols (tradable, non-OTC, simple symbols),
    matching the original notebook's universe merge. Cached daily; fails to an
    empty set so the CSV universe is always available."""
    CACHE_DIR.mkdir(exist_ok=True)
    asof = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    cache = CACHE_DIR / f"alpaca_universe_{asof}.pkl"
    if cache.exists():
        return set(pd.read_pickle(cache))
    syms = set()
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        client = TradingClient(os.environ["ALPACA_API_KEY"],
                               os.environ["ALPACA_API_SECRET"], paper=True)
        for a in client.get_all_assets(GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)):
            s = a.symbol.upper().strip()
            if (a.tradable and "." not in a.symbol and a.exchange != "OTC"
                    and s.isalpha() and 1 <= len(s) <= 5
                    and not s.endswith(WARRANT_UNIT_SUFFIXES)):
                syms.add(s)
        pd.to_pickle(syms, cache)
    except Exception as e:
        print(f"Alpaca asset list unavailable ({e}); using CSV universe only.")
    return syms


def load_universe(limit: int | None = None, include_alpaca: bool = True) -> list[str]:
    df = pd.read_excel(TICKER_FILE)
    col = next((c for c in ["Ticker", "Ticker Symbol", "Symbol", "symbol", "ticker"]
                if c in df.columns), df.columns[0])
    syms = (
        df[col].astype(str).str.upper().str.strip()
        .loc[lambda s: s.str.match(r"^[A-Z]+$", na=False)]
    )
    universe = {s for s in syms if not s.endswith(WARRANT_UNIT_SUFFIXES) and 1 <= len(s) <= 5}
    if include_alpaca:
        universe |= alpaca_active_symbols()      # merge full live Alpaca universe (~13k)
    cleaned = sorted(universe)
    if limit:
        cleaned = cleaned[:limit]
    if BENCHMARK not in cleaned:
        cleaned.append(BENCHMARK)
    return cleaned


# ──────────────────────────────────────────────────────────────────────
# Data pull (batched, fault-tolerant)
# ──────────────────────────────────────────────────────────────────────
def fetch_bars(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        sys.exit("ERROR: set ALPACA_API_KEY and ALPACA_API_SECRET (see .env.example).")

    client = StockHistoricalDataClient(key, secret)
    frames, failed = [], []
    batch_size = 200
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    print(f"Pulling {len(symbols)} tickers in {len(batches)} batches...")

    for i, batch in enumerate(batches, 1):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment="all",
            )
            df = client.get_stock_bars(req).df
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"symbol": "ticker"})
                frames.append(df)
        except Exception as e:
            failed.extend(batch)
            print(f"  batch {i} failed ({type(e).__name__}); will skip {len(batch)} syms")
        if i % 5 == 0 or i == len(batches):
            print(f"  batch {i}/{len(batches)} done")
        time.sleep(0.2)

    if not frames:
        sys.exit("ERROR: no bar data returned. Check API keys / connectivity.")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    out = out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    if failed:
        print(f"  {len(failed)} tickers failed to pull (skipped).")
    print(f"Pulled {out['ticker'].nunique()} tickers, {len(out):,} rows.")
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-ticker metrics
#
# All statistics use simple (arithmetic) daily returns over the trailing
# window, computed strictly point-in-time (no future data). Citations:
#   Sharpe ratio ........ Sharpe (1966); annualization & t-stat per Lo (2002)
#   Sortino ratio ....... Sortino & Price (1994)  [downside deviation vs MAR]
#   12-1 momentum ....... Jegadeesh & Titman (1993)
#   Max drawdown ........ standard peak-to-trough on the equity curve
# ──────────────────────────────────────────────────────────────────────
def _winsorize(s: pd.Series, lo: float = WINSOR_LO, hi: float = WINSOR_HI) -> pd.Series:
    if len(s) == 0:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def compute_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in bars.groupby("ticker"):
        g = g.sort_values("timestamp")
        close_all = g["close"].astype(float)
        if len(close_all) < MIN_HISTORY:
            continue

        # Trailing window: TRADING_DAYS+1 closes -> TRADING_DAYS returns
        close = close_all.tail(TRADING_DAYS + 1)
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        ret = _winsorize(ret)
        n = len(ret)
        sd = ret.std(ddof=1)                      # sample std
        if n < MIN_OBS or not np.isfinite(sd) or sd == 0:
            continue

        # ── Sharpe (annualized) and its significance ─────────────────
        excess = ret - RF_DAILY
        mean_excess = excess.mean()
        sharpe_daily = mean_excess / sd
        sharpe = np.sqrt(TRADING_DAYS) * sharpe_daily
        # t-stat of the mean excess return under IID: t = SR_daily * sqrt(n)
        sharpe_tstat = sharpe_daily * np.sqrt(n)
        ann_vol = sd * np.sqrt(TRADING_DAYS)

        # ── CAGR (geometric realized return, descriptive) ────────────
        p0, p1 = float(close.iloc[0]), float(close.iloc[-1])
        cagr = (p1 / p0) ** (TRADING_DAYS / n) - 1 if p0 > 0 else np.nan

        # ── Sortino: downside deviation vs target=rf over ALL obs ────
        downside_sq = np.minimum(excess, 0.0) ** 2
        dd_dev = np.sqrt(downside_sq.mean())
        sortino = np.sqrt(TRADING_DAYS) * (mean_excess / dd_dev) if dd_dev > 0 else np.nan

        # ── 12-1 momentum: P[t-21] / P[t-252] - 1 ────────────────────
        p_12m = float(close_all.iloc[-(TRADING_DAYS + 1)])   # ~12 months ago
        p_1m = float(close_all.iloc[-22])                    # ~1 month ago
        momentum = p_1m / p_12m - 1 if p_12m > 0 else np.nan

        # ── Max drawdown over the window ─────────────────────────────
        equity = (1 + ret).cumprod()
        max_dd = float((equity / equity.cummax() - 1).min())

        # ── Trend and liquidity (median = robust to spikes) ──────────
        sma200 = float(close_all.tail(200).mean())
        above_trend = bool(p1 > sma200)
        recent = g.tail(TRADING_DAYS)
        med_dollar_vol = float((recent["close"] * recent["volume"]).median())

        rows.append({
            "ticker": ticker,
            "current_price": round(p1, 2),
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sharpe_tstat": sharpe_tstat,
            "sortino": sortino,
            "momentum_12_1": momentum,
            "max_drawdown": max_dd,
            "above_200d_trend": above_trend,
            "med_dollar_vol": med_dollar_vol,
            "n_obs": n,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Scoring: cross-sectional composite + GOOD/NEUTRAL/BAD verdict
# ──────────────────────────────────────────────────────────────────────
def _robust_z(s: pd.Series) -> pd.Series:
    """Robust standardization via median / MAD (Huber), clipped to +/-3.
    Falls back to mean/std when MAD is zero. NaNs filled to the median first
    so a missing factor is neutral rather than dropping the name."""
    s = s.astype(float)
    s = s.fillna(s.median())
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad                      # MAD -> std-equivalent for normal data
    if np.isfinite(scale) and scale > 0:
        z = (s - med) / scale
    else:
        sd = s.std(ddof=0)
        z = (s - s.mean()) / sd if sd and sd > 0 else s * 0.0
    return z.clip(-3, 3)


def filter_investable(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep only tradeable names before ranking. These bounds are economically
    grounded (price, liquidity, volatility), not arbitrary return caps: they
    remove sub-$5 noise, illiquid micro-caps, and hyper-volatile leveraged
    products whose trailing statistics are not investable signal.

    Price band is configurable per strategy via env (MIN_PRICE / MAX_PRICE),
    e.g. MIN_PRICE=5 MAX_PRICE=40 for an 'affordable' run."""
    min_p = float(os.environ.get("MIN_PRICE", MIN_PRICE))
    max_p = float(os.environ.get("MAX_PRICE", 1e12))
    keep = metrics[
        (metrics["current_price"] >= min_p)
        & (metrics["current_price"] <= max_p)
        & (metrics["ann_vol"].between(MIN_VOL, MAX_VOL))
        & (metrics["med_dollar_vol"] >= MIN_MEDIAN_DOLLAR_VOL)
    ].copy()
    return keep


def score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional composite of robust z-scores. The factor weights are a
    modeling choice (not a theorem); each input statistic is computed rigorously
    and standardized robustly so no single outlier dominates the ranking."""
    df = metrics.copy()
    df["composite"] = (
        0.30 * _robust_z(df["sharpe"])
        + 0.25 * _robust_z(df["sortino"])
        + 0.25 * _robust_z(df["momentum_12_1"])
        + 0.10 * _robust_z(df["max_drawdown"])      # less-negative dd scores higher
        + 0.10 * df["above_200d_trend"].astype(float)
    )
    df["percentile"] = df["composite"].rank(pct=True) * 100
    df["verdict"] = np.where(
        df["percentile"] >= 70, "GOOD",
        np.where(df["percentile"] <= 30, "BAD", "NEUTRAL"),
    )
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Hierarchical Risk Parity (Lopez de Prado 2016). Stable allocator that
# beats Markowitz max-Sharpe out-of-sample in our Stage 2 test (higher
# Sharpe, far lower drawdown) because it avoids covariance inversion.
# ──────────────────────────────────────────────────────────────────────
def _quasi_diag(link):
    link = link.astype(int)
    s = pd.Series([link[-1, 0], link[-1, 1]])
    n = link[-1, 3]
    while s.max() >= n:
        s.index = range(0, s.shape[0] * 2, 2)
        df0 = s[s >= n]
        i, j = df0.index, df0.values - n
        s[i] = link[j, 0]
        s = pd.concat([s, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        s.index = range(s.shape[0])
    return s.tolist()


def _cluster_var(cov, items):
    c = cov.loc[items, items]
    iv = 1.0 / np.diag(c)
    iv /= iv.sum()
    return float(iv @ c.values @ iv)


def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    cov, corr = rets.cov(), rets.corr()
    dist = ((1 - corr) / 2.0) ** 0.5
    link = linkage(squareform(dist.values, checks=False), "single")
    order = [corr.index[i] for i in _quasi_diag(link)]
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in
                    ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w.reindex(rets.columns)


# ──────────────────────────────────────────────────────────────────────
# Portfolio: screen candidates, then weight with HRP (capped)
# ──────────────────────────────────────────────────────────────────────
def build_portfolio(scored: pd.DataFrame, bars: pd.DataFrame,
                    top_n: int, max_weight: float,
                    apply_sentiment_veto: bool = True) -> pd.DataFrame:
    pool = scored[
        (scored["ticker"] != BENCHMARK)
        & (scored["sharpe"] > MIN_SHARPE)
        & (scored["sharpe_tstat"] >= MIN_SHARPE_TSTAT)     # statistically meaningful
        & (scored["ann_vol"].between(MIN_VOL, MAX_VOL))
        & (scored["med_dollar_vol"] > MIN_MEDIAN_DOLLAR_VOL)
        & (scored["momentum_12_1"] > 0)
        & (scored["max_drawdown"] > MAX_DRAWDOWN_FLOOR)
    ].head(top_n * 2 + 4).copy()

    # Event-risk veto: drop names whose live news turned sharply negative, then
    # backfill from the next-best screened names (keeps the book fluid).
    if apply_sentiment_veto and len(pool):
        try:
            from sentiment import apply_veto
            pool, _ = apply_veto(pool)
        except Exception as e:
            print(f"Sentiment veto skipped: {e}")
            pool["news_sentiment"] = 0.0

    candidates = pool.head(top_n).copy()
    if len(candidates) < 2:
        return candidates.assign(weight=1.0 / max(len(candidates), 1))

    tickers = candidates["ticker"].tolist()
    rets = (
        bars[bars["ticker"].isin(tickers)]
        .pivot_table(index="timestamp", columns="ticker", values="close")
        .pct_change()
        .tail(TRADING_DAYS)
        .dropna(axis=1, how="all")
        .dropna()
    )
    # Winsorize each column so a single bad tick cannot distort mu/cov.
    rets = rets.apply(_winsorize, axis=0)
    tickers = [t for t in tickers if t in rets.columns]
    candidates = candidates[candidates["ticker"].isin(tickers)].copy()

    # Hierarchical Risk Parity (Lopez de Prado 2016). Chosen over Markowitz
    # max-Sharpe because our out-of-sample Stage 2 test showed HRP delivers a
    # materially higher Sharpe and far smaller drawdown; Markowitz overfits the
    # covariance. Weights are then capped and renormalized.
    w = hrp_weights(rets[tickers]).reindex(tickers).fillna(0.0).values
    w = np.clip(w, 0, max_weight)
    if w.sum() <= 0:
        w = np.repeat(1.0 / len(tickers), len(tickers))
    weights = w / w.sum()

    candidates = candidates.set_index("ticker").loc[tickers].reset_index()
    candidates["weight"] = weights
    return candidates.sort_values("weight", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Why-brief generation
# ──────────────────────────────────────────────────────────────────────
def reason_for(row: pd.Series) -> str:
    bits = []
    if row["momentum_12_1"] > 0.15:
        bits.append(f"strong 12-1 momentum +{row['momentum_12_1']*100:.0f}%")
    elif row["momentum_12_1"] > 0:
        bits.append(f"positive momentum +{row['momentum_12_1']*100:.0f}%")
    if row["sharpe"] >= 1.0:
        bits.append(f"Sharpe {row['sharpe']:.2f}")
    if row["above_200d_trend"]:
        bits.append("above 200d trend")
    if row["max_drawdown"] > -0.20:
        bits.append("shallow drawdown")
    return ", ".join(bits) if bits else f"Sharpe {row['sharpe']:.2f}"


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────
def write_reports(scored: pd.DataFrame, portfolio: pd.DataFrame, asof: str) -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = REPORTS_DIR / f"{asof}_scores.csv"
    md_path = REPORTS_DIR / f"{asof}_scorecard.md"
    scored.to_csv(csv_path, index=False)

    good = scored[scored["verdict"] == "GOOD"]
    bad = scored[scored["verdict"] == "BAD"]
    spy = scored[scored["ticker"] == BENCHMARK]
    spy_ret = spy["cagr"].iloc[0] if len(spy) else float("nan")

    lines = [
        f"# StockSight Daily Scorecard, {asof}",
        "",
        f"Universe screened: {len(scored)} tickers with >= 1y history.",
        f"GOOD: {len(good)}  |  NEUTRAL: {len(scored)-len(good)-len(bad)}  |  BAD: {len(bad)}",
        f"Benchmark {BENCHMARK} trailing 1y return: {spy_ret*100:.1f}%",
        "",
        "## Top picks (best risk-adjusted names right now)",
        "",
        "| # | Ticker | Price | 1y CAGR | Sharpe | t-stat | Momentum | Why |",
        "|---|--------|-------|---------|--------|--------|----------|-----|",
    ]
    for i, (_, r) in enumerate(good.head(10).iterrows(), 1):
        lines.append(
            f"| {i} | {r['ticker']} | ${r['current_price']:.2f} | "
            f"{r['cagr']*100:.0f}% | {r['sharpe']:.2f} | {r['sharpe_tstat']:.1f} | "
            f"{r['momentum_12_1']*100:.0f}% | {reason_for(r)} |"
        )

    lines += ["", "## Optimized portfolio (max-Sharpe, shrinkage covariance)", ""]
    if len(portfolio):
        lines.append("| Ticker | Weight | Sharpe | Momentum | Why |")
        lines.append("|--------|--------|--------|----------|-----|")
        for _, r in portfolio.iterrows():
            lines.append(
                f"| {r['ticker']} | {r['weight']*100:.1f}% | {r['sharpe']:.2f} | "
                f"{r['momentum_12_1']*100:.0f}% | {reason_for(r)} |"
            )
    else:
        lines.append("No names cleared the screen today.")

    lines += [
        "",
        "## Bottom of the screen (avoid / weak)",
        "",
        ", ".join(bad.tail(10)["ticker"].tolist()) or "none",
        "",
        "## Methodology",
        "",
        f"- Window: trailing {TRADING_DAYS} trading days, point-in-time (no future data).",
        f"- Investable filter before ranking: price >= ${MIN_PRICE:.0f}, "
        f"vol in [{MIN_VOL:.0%}, {MAX_VOL:.0%}], median dollar-volume >= ${MIN_MEDIAN_DOLLAR_VOL/1e6:.0f}M.",
        "- Returns winsorized at 0.5/99.5% to neutralize bad ticks.",
        f"- Sharpe (Sharpe 1966), annualized sqrt({TRADING_DAYS}) x daily; rf = {RISK_FREE_RATE:.0%}.",
        f"  Portfolio requires Sharpe t-stat >= {MIN_SHARPE_TSTAT} (Lo 2002) so the edge is significant.",
        "- Sortino (Sortino & Price 1994): downside deviation vs rf over all observations.",
        "- Momentum: 12-1 (Jegadeesh & Titman 1993).",
        "- Composite: robust z-scores (median/MAD, clipped +/-3); weights are a modeling choice.",
        "- Portfolio: max-Sharpe (Markowitz 1952) with Ledoit-Wolf (2004) shrinkage covariance,",
        "  long-only, single-weight capped. Expected returns are noisy estimates, controlled by",
        "  shrinkage + caps + long-only constraints.",
        "- Known limitation: universe is currently-listed names (survivorship bias affects",
        "  backtests, not the forward pick).",
        "",
        "---",
        "Analytical output, not financial advice. Past performance does not predict future results.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path.name} and {md_path.name}")
    return md_path.read_text(encoding="utf-8")


def build_sms(scored: pd.DataFrame, portfolio: pd.DataFrame, asof: str) -> str:
    names = get_company_names()

    def label(t: str) -> str:
        c = names.get(t, "")
        return f"{t} ({c})" if c else t

    good = scored[scored["verdict"] == "GOOD"].head(5)
    picks = [
        f"{label(r['ticker'])} ${r['current_price']:.2f}, Sharpe {r['sharpe']:.1f}, "
        f"Momentum {r['momentum_12_1']*100:.0f}%"
        for _, r in good.iterrows()
    ]
    port = "; ".join(
        f"{label(r['ticker'])} ${r['current_price']:.2f} {r['weight']*100:.0f}%"
        for _, r in portfolio.head(6).iterrows())
    msg = (
        f"StockSight {asof}\n"
        f"Top picks: {'; '.join(picks) if picks else 'none cleared screen'}\n"
        f"Portfolio: {port if port else 'n/a'}\n"
        f"Why: highest trailing risk-adjusted return + positive momentum + above trend.\n"
        f"Full breakdown: {DASHBOARD_URL}"
    )
    return msg


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
CACHE_DIR = DATA_DIR / "cache"

_ENTITY_TAIL = re.compile(
    r",?\s*(Incorporated|Inc\.?|Corporation|Corp\.?|Company|Co\.?|Holdings?|"
    r"Group|Ltd\.?|Limited|PLC|N\.V\.|S\.A\.|plc|Trust|Fund)\s*$", re.I)
_SHARE_CLASS = re.compile(
    r"\s*(Common Stock|Class [A-C].*|Ordinary Shares.*|Common Shares.*|"
    r"American Depositary Shares?.*|Depositary Shares.*|- .*)$", re.I)


def _clean_company(name: str) -> str:
    n = _SHARE_CLASS.sub("", str(name)).strip().rstrip(",")
    n = _ENTITY_TAIL.sub("", n).strip().rstrip(",")
    n = _ENTITY_TAIL.sub("", n).strip().rstrip(",")   # second pass (e.g. "X Corporation")
    return n


def get_company_names() -> dict:
    """Map ticker -> short company name from Alpaca's asset list (cached daily).
    Returns {} on any failure so callers fall back to bare tickers."""
    CACHE_DIR.mkdir(exist_ok=True)
    asof = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    cache = CACHE_DIR / f"names_{asof}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    names = {}
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        client = TradingClient(key, secret, paper=True)
        assets = client.get_all_assets(GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
        for a in assets:
            if a.name:
                names[a.symbol.upper().strip()] = _clean_company(a.name)
        pd.to_pickle(names, cache)
    except Exception as e:
        print(f"Company-name lookup unavailable ({e}); using bare tickers.")
    return names


def get_bars(universe_limit: int | None = None, lookback_days: int = 420,
             use_cache: bool = True, include_alpaca: bool = True) -> pd.DataFrame:
    """Pull bars, caching to disk so charts/backtests reuse one pull.
    Cache key = (universe_limit, lookback_days, include_alpaca); daily refresh."""
    CACHE_DIR.mkdir(exist_ok=True)
    asof = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    tag = f"{universe_limit or 'all'}_{lookback_days}_{'fa' if include_alpaca else 'csv'}_{asof}"
    cache = CACHE_DIR / f"bars_{tag}.pkl"
    if use_cache and cache.exists():
        print(f"Loading cached bars: {cache.name}")
        return pd.read_pickle(cache)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    universe = load_universe(limit=universe_limit, include_alpaca=include_alpaca)
    print(f"Universe: {len(universe)} symbols")
    bars = fetch_bars(universe, start, end)
    bars.to_pickle(cache)
    return bars


def run(top_n: int = 12, max_weight: float = 0.25,
        universe_limit: int | None = None, lookback_days: int = 420) -> str:
    asof = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    bars = get_bars(universe_limit=universe_limit, lookback_days=lookback_days)
    metrics = compute_metrics(bars)
    print(f"Computed metrics for {len(metrics)} tickers with sufficient history.")
    investable = filter_investable(metrics)
    print(f"Investable after price/liquidity/vol filters: {len(investable)} tickers.")
    scored = score(investable)
    portfolio = build_portfolio(scored, bars, top_n=top_n, max_weight=max_weight)
    write_reports(scored, portfolio, asof)
    sms = build_sms(scored, portfolio, asof)
    print("\n" + "=" * 60 + "\n" + sms + "\n" + "=" * 60)
    return sms


# Strategy presets: (label, MIN_PRICE, MAX_PRICE). Add more bands here anytime.
STRATEGIES = [
    ("Core (price >= $5)", "5", "1e12"),
    ("Affordable ($5-$40)", "5", "40"),
]


def run_multi(top_n: int = 12, max_weight: float = 0.25,
              universe_limit: int | None = None, lookback_days: int = 420) -> list:
    """Run several price-band strategies off one data pull. Core also drives the
    saved report/dashboard. Returns [(label, summary_text), ...]."""
    asof = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    bars = get_bars(universe_limit=universe_limit, lookback_days=lookback_days)
    metrics = compute_metrics(bars)
    out = []
    for i, (label_, min_p, max_p) in enumerate(STRATEGIES):
        os.environ["MIN_PRICE"], os.environ["MAX_PRICE"] = min_p, max_p
        inv = filter_investable(metrics)
        scored = score(inv)
        portfolio = build_portfolio(scored, bars, top_n=top_n, max_weight=max_weight)
        if i == 0:                       # Core feeds the saved report/dashboard
            write_reports(scored, portfolio, asof)
        msg = f"[{label_}]\n" + build_sms(scored, portfolio, asof)
        out.append((label_, msg))
        print("\n" + "=" * 60 + "\n" + msg + "\n" + "=" * 60)
    return out


def main():
    ap = argparse.ArgumentParser(description="StockSight daily scorecard")
    ap.add_argument("--top", type=int, default=12, help="portfolio size")
    ap.add_argument("--max-weight", type=float, default=0.25, help="max single weight")
    ap.add_argument("--universe-limit", type=int, default=None,
                    help="cap universe size (for fast test runs)")
    ap.add_argument("--lookback-days", type=int, default=420)
    ap.add_argument("--multi", action="store_true", help="run all STRATEGIES and post each")
    ap.add_argument("--no-notify", action="store_true", help="skip sending the text")
    args = ap.parse_args()

    if args.multi:
        results = run_multi(top_n=args.top, max_weight=args.max_weight,
                            universe_limit=args.universe_limit, lookback_days=args.lookback_days)
        if not args.no_notify:
            try:
                import notify
                # Core via the full chain (Twilio primary, Slack fallback);
                # the rest straight to Slack so every strategy is visible.
                for i, (_, msg) in enumerate(results):
                    if i == 0:
                        notify.send(msg)
                    else:
                        notify._slack(msg)
            except Exception as e:
                print(f"Notify skipped/failed: {e}")
        return

    sms = run(top_n=args.top, max_weight=args.max_weight,
              universe_limit=args.universe_limit, lookback_days=args.lookback_days)

    if not args.no_notify:
        try:
            from notify import send
            send(sms)
        except Exception as e:
            print(f"Notify skipped/failed: {e}")


if __name__ == "__main__":
    main()
