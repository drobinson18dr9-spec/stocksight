"""
Multi-source data integrity + fundamentals layer.

Cross-checks each ticker's latest price across independent free sources
(Alpaca, FMP, Finnhub, Tiingo) so a single bad feed can't introduce a blip,
and pulls fundamentals/analyst signal (FMP ratios, Finnhub recommendation,
yfinance) for a fuller, less-guessy picture.

All calls are best-effort: a missing key or failed source is skipped, never
fatal. Keys read from env: FMP_API_KEY, FINNHUB_API_KEY, TIINGO_API_KEY.

Usage: python src/sources.py --tickers AAPL NVDA WDC
"""

from __future__ import annotations
import os
import argparse
import statistics

import requests

TIMEOUT = 15


def _get(url, **kw):
    try:
        r = requests.get(url, timeout=TIMEOUT, **kw)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def price_finnhub(sym):
    k = os.environ.get("FINNHUB_API_KEY")
    if not k:
        return None
    d = _get(f"https://finnhub.io/api/v1/quote?symbol={sym}&token={k}")
    return float(d["c"]) if d and d.get("c") else None


def price_fmp(sym):
    k = os.environ.get("FMP_API_KEY")
    if not k:
        return None
    d = _get(f"https://financialmodelingprep.com/api/v3/quote-short/{sym}?apikey={k}")
    return float(d[0]["price"]) if isinstance(d, list) and d and d[0].get("price") else None


def price_tiingo(sym):
    k = os.environ.get("TIINGO_API_KEY")
    if not k:
        return None
    d = _get(f"https://api.tiingo.com/iex/{sym}?token={k}")
    if isinstance(d, list) and d:
        return float(d[0].get("tngoLast") or d[0].get("last") or 0) or None
    return None


def price_alpaca(sym):
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not (key and sec):
        return None
    d = _get(f"https://data.alpaca.markets/v2/stocks/{sym}/trades/latest",
             headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    return float(d["trade"]["p"]) if d and d.get("trade") else None


def price_crosscheck(sym: str, tol: float = 0.02) -> dict:
    """Latest price from each source + agreement flag (within `tol`)."""
    srcs = {"alpaca": price_alpaca(sym), "fmp": price_fmp(sym),
            "finnhub": price_finnhub(sym), "tiingo": price_tiingo(sym)}
    vals = [v for v in srcs.values() if v]
    med = statistics.median(vals) if vals else None
    max_dev = max(abs(v / med - 1) for v in vals) if (med and len(vals) > 1) else 0.0
    return {"ticker": sym, "sources": srcs, "median": round(med, 2) if med else None,
            "n_sources": len(vals), "max_deviation": round(max_dev, 4),
            "agree": bool(med and max_dev <= tol)}


def finnhub_recommendation(sym):
    k = os.environ.get("FINNHUB_API_KEY")
    if not k:
        return None
    d = _get(f"https://finnhub.io/api/v1/stock/recommendation?symbol={sym}&token={k}")
    if isinstance(d, list) and d:
        r = d[0]
        return {"strongBuy": r.get("strongBuy"), "buy": r.get("buy"),
                "hold": r.get("hold"), "sell": r.get("sell"),
                "strongSell": r.get("strongSell"), "period": r.get("period")}
    return None


def fmp_ratios(sym):
    k = os.environ.get("FMP_API_KEY")
    if not k:
        return None
    d = _get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}?apikey={k}")
    if isinstance(d, list) and d:
        r = d[0]
        return {k2: r.get(k2) for k2 in
                ("peRatioTTM", "returnOnEquityTTM", "debtEquityRatioTTM",
                 "netProfitMarginTTM", "currentRatioTTM")}
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    args = ap.parse_args()
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    for t in args.tickers:
        cc = price_crosscheck(t)
        print(f"\n{t}  median ${cc['median']}  sources={cc['n_sources']}  "
              f"max_dev={cc['max_deviation']*100:.2f}%  agree={cc['agree']}")
        print("   prices:", {k: v for k, v in cc["sources"].items() if v})
        rec = finnhub_recommendation(t)
        if rec:
            print("   analyst:", rec)
        rat = fmp_ratios(t)
        if rat:
            print("   ratios:", {k: (round(v, 2) if isinstance(v, (int, float)) else v)
                                  for k, v in rat.items() if v is not None})
