"""
StockSight fundamentals (free, no API key) via yfinance / Yahoo.

Pulls valuation, profitability, growth, leverage, and Wall-Street analyst
targets for a list of tickers, and derives:
  - analyst_upside  = mean target / current price - 1
  - quality_score   = cross-sectional blend of margins, ROE, growth, low
                      leverage, and analyst upside (robust z-scores)

This is the qualitative->quantitative fundamental layer that complements the
price-based screen. Fails open (NaN) per field so a missing value never breaks
the pipeline.

Usage: python src/fundamentals.py --tickers WDC MU INDV
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd

FIELDS = ["marketCap", "trailingPE", "forwardPE", "profitMargins",
          "returnOnEquity", "revenueGrowth", "debtToEquity",
          "targetMeanPrice", "currentPrice", "recommendationKey",
          "numberOfAnalystOpinions"]


def fetch(tickers) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for t in tickers:
        d = {"ticker": t}
        try:
            info = yf.Ticker(t).info or {}
            for f in FIELDS:
                d[f] = info.get(f)
        except Exception:
            for f in FIELDS:
                d[f] = None
        rows.append(d)
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    cp = pd.to_numeric(df.get("currentPrice"), errors="coerce")
    tp = pd.to_numeric(df.get("targetMeanPrice"), errors="coerce")
    df["analyst_upside"] = (tp / cp - 1).round(3)
    return df


def _rz(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 0:
        z = (s - med) / scale
    else:
        sd = s.std(ddof=0)
        z = (s - s.mean()) / sd if sd and sd > 0 else s * 0
    return z.clip(-3, 3).fillna(0)


def quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quality_score"] = (
        0.25 * _rz(df["profitMargins"])
        + 0.25 * _rz(df["returnOnEquity"])
        + 0.20 * _rz(df["revenueGrowth"])
        - 0.15 * _rz(df["debtToEquity"])        # less leverage scores higher
        + 0.15 * _rz(df["analyst_upside"])
    ).round(3)
    return df


def get(tickers) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        return quality_score(fetch(tickers))
    except Exception as e:
        print(f"Fundamentals unavailable ({e}).")
        return pd.DataFrame({"ticker": list(tickers)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    args = ap.parse_args()
    d = get(args.tickers)
    cols = ["ticker", "trailingPE", "profitMargins", "returnOnEquity",
            "revenueGrowth", "debtToEquity", "analyst_upside",
            "recommendationKey", "quality_score"]
    print(d[[c for c in cols if c in d.columns]].to_string(index=False))
