"""
Precompute the per-ticker prediction data that powers the dashboard's
Forecasts explorer. For each name it runs the full model panel (predict.py):
walk-forward actual-vs-predicted, error metrics, and a forward forecast.
Writes assets/predict/<TICKER>.json and assets/predict/index.json.

A static site cannot run Python on demand, so we precompute for the portfolio
names (and optionally top GOOD names). Arbitrary tickers still work via the
local CLI: python src/predict.py --ticker XYZ

Usage: python src/build_explorer.py [--tickers AAPL MSFT ...] [--test-days 20]
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import scorecard as sc
import predict as pr

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "predict"
MIN_HISTORY_FORECAST = pr.MIN_HISTORY_FORECAST    # recent-IPO floor (~6 months)


def _yf_bars(tickers, start, end):
    """Daily bars from Yahoo for tickers Alpaca lacks (OTC/ADRs). Returns a
    long DataFrame matching the Alpaca schema (ticker, timestamp, close, volume)."""
    import yfinance as yf
    frames = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start.strftime("%Y-%m-%d"),
                                     end=end.strftime("%Y-%m-%d"), auto_adjust=True)
            if len(h) >= MIN_HISTORY_FORECAST:
                df = h.reset_index()[["Date", "Close", "Volume"]]
                df.columns = ["timestamp", "close", "volume"]
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df["ticker"] = t
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["timestamp", "close", "volume", "ticker"])


def portfolio_tickers(top_n=12, extra_good=28) -> list[str]:
    """Portfolio names plus the top GOOD-rated names, so the explorer covers a
    meaningful default set (any other ticker is available on demand)."""
    bars = sc.get_bars()
    m = sc.compute_metrics(bars)
    inv = sc.filter_investable(m)
    s = sc.score(inv)
    p = sc.build_portfolio(s, bars, top_n=top_n, max_weight=0.25,
                           apply_sentiment_veto=False)
    names = list(p["ticker"])
    good = s[(s["verdict"] == "GOOD") & (s["ticker"] != sc.BENCHMARK)]["ticker"].tolist()
    for g in good:
        if g not in names:
            names.append(g)
        if len(names) >= top_n + extra_good:
            break
    return names


def main(tickers=None, test_days=20, horizon=15,
         all_universe=False, shard=0, num_shards=1):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if not tickers:
        if all_universe:
            tickers = sorted(set(sc.load_universe()))     # every modelable name
        else:
            tickers = portfolio_tickers()
    if num_shards > 1:
        tickers = tickers[shard::num_shards]              # this runner's slice
    # Skip Windows-reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9):
    # their <ticker>.json files are invalid paths on Windows checkouts.
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    tickers = [t for t in tickers if t.upper() not in reserved]
    print(f"Precomputing forecasts: {len(tickers)} names "
          f"(shard {shard}/{num_shards}, all_universe={all_universe})")

    # One batched pull for all names. Pull as much history as Alpaca IEX allows
    # (it caps ~2020-07, ~6y); more history trains the walk-forward models better.
    # Asking for 12y is harmless: the API returns whatever it has.
    start = datetime.now(timezone.utc) - timedelta(days=int(12 * 365))
    end = datetime.now(timezone.utc) - timedelta(days=1)
    try:
        bars = sc.fetch_bars(tickers, start, end)
    except SystemExit:                       # Alpaca returned nothing (e.g. all OTC)
        bars = pd.DataFrame(columns=["ticker", "timestamp", "close", "volume"])

    # STALENESS GATE: Alpaca's feed can lag (it was capping all tickers ~2 weeks
    # behind). If its latest bar is more than 4 days old, drop it entirely so every
    # ticker falls through to yfinance, which carries current data.
    if len(bars):
        try:
            mx = pd.to_datetime(bars["timestamp"]).max()
            age = (pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
                   - pd.Timestamp(mx).tz_localize(None)).days
            if age > 4:
                print(f"  Alpaca data stale (latest {pd.Timestamp(mx).date()}, "
                      f"{age}d old); routing ALL tickers to yfinance for current data")
                bars = bars.iloc[0:0]
        except Exception as e:
            print(f"  staleness check skipped: {e}")

    # Fallback: any ticker Alpaca lacks (or all of them, if stale) -> Yahoo.
    have = {t for t, g in bars.groupby("ticker") if len(g) >= MIN_HISTORY_FORECAST} if len(bars) else set()
    missing = [t for t in tickers if t not in have]
    if missing:
        yb = _yf_bars(missing, start, end)
        if len(yb):
            bars = pd.concat([bars, yb], ignore_index=True)
            print(f"  yfinance fallback supplied {yb['ticker'].nunique()} names")

    done = []
    for t in tickers:
        s = bars[bars["ticker"] == t].sort_values("timestamp")
        if len(s) < MIN_HISTORY_FORECAST:
            print(f"  {t}: insufficient history, skipped")
            continue
        prices = pd.Series(s["close"].values,
                           index=pd.to_datetime(s["timestamp"].values)).astype(float)
        try:
            wf = pr.walk_forward(prices, test_days)
            for name in pr.MODELS:
                wf[f"var_{name}"] = (wf["actual"] - wf[name]).round(2)
            metrics = pr.error_metrics(wf)
            fwd = pr.forward_forecast(prices, horizon)
            payload = {
                "ticker": t,
                "last_close": round(float(prices.iloc[-1]), 2),
                "last_date": prices.index[-1].strftime("%Y-%m-%d"),
                "models": list(pr.MODELS),
                "walk_forward": json.loads(wf.assign(date=wf["date"].astype(str)).to_json(orient="records")),
                "error_metrics": json.loads(metrics.to_json(orient="records")),
                "forward": json.loads(fwd.assign(date=fwd["date"].astype(str)).to_json(orient="records")),
            }
            (ASSET_DIR / f"{t}.json").write_text(json.dumps(payload), encoding="utf-8")
            done.append(t)
            print(f"  {t}: done")
        except Exception as e:
            print(f"  {t}: failed ({e})")

    # Merge into the existing index so on-demand-requested tickers persist
    existing = []
    idx = ASSET_DIR / "index.json"
    if idx.exists():
        try:
            existing = json.loads(idx.read_text()).get("tickers", [])
        except Exception:
            existing = []
    all_tickers = sorted(set(existing) | set(done))
    idx.write_text(json.dumps({"tickers": all_tickers}), encoding="utf-8")
    print(f"Wrote {len(done)} forecast files; index now {len(all_tickers)} tickers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--test-days", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--all-universe", action="store_true",
                    help="compute every modelable name in the universe")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    main(tickers=args.tickers, test_days=args.test_days, horizon=args.horizon,
         all_universe=args.all_universe, shard=args.shard, num_shards=args.num_shards)
