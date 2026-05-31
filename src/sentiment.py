"""
StockSight sentiment + event-risk engine (free sources, no API keys).

Turns qualitative news into a quantitative per-ticker signal and an event veto:
  - Pulls each ticker's live news headlines from Yahoo Finance per-ticker RSS.
  - Scores every headline with VADER (Hutto & Gilbert 2014), a lexicon tuned
    for short financial/social text. (FinBERT is a heavier upgrade; VADER is
    light, dependency-free beyond nltk, and reliable in the cloud.)
  - Aggregates a mean sentiment, a worst-headline score, and a recent count.
  - VETO: if a name has enough recent coverage AND it is sharply negative,
    the name is flagged so the portfolio drops it. This is the mechanism that
    ejects a stock when bad news hits (e.g. a bankruptcy/fraud headline),
    so the buy-list stays fluid and reacts to real-world events.

Returns neutral (0, no veto) for names with no/low coverage so the screen is
never silently starved.

Usage: python src/sentiment.py --tickers TSLA WDC MU
"""

from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

# Veto thresholds (tunable). A real event (bankruptcy/fraud) floods the feed
# with negative coverage, so we trigger on a negative REGIME (mean tone or a
# majority of strongly-negative headlines), never on one noisy headline.
VETO_MEAN = -0.25            # mean VADER compound below this => negative regime
VETO_MIN_HEADLINES = 3       # require enough coverage to trust the signal
NEG_HEADLINE = -0.4          # a headline this negative counts as "negative"
VETO_SHARE_NEG = 0.5         # >= half of recent headlines negative => veto

YAHOO_RSS = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
             "?s={sym}&region=US&lang=en-US")


def _vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _yahoo_headlines(sym, cutoff):
    """Per-ticker headlines from Yahoo RSS -> [(text, source), ...]."""
    import feedparser
    out = []
    try:
        for e in feedparser.parse(YAHOO_RSS.format(sym=sym)).entries:
            pub = e.get("published_parsed") or e.get("updated_parsed")
            if pub and datetime(*pub[:6], tzinfo=timezone.utc) < cutoff:
                continue
            text = f"{e.get('title','')}. {e.get('summary','')[:200]}".strip()
            if text and text != ".":
                out.append((text, "yahoo"))
    except Exception:
        pass
    return out


def _finnhub_headlines(sym, lookback_days):
    """Per-ticker headlines from Finnhub company-news (aggregates Benzinga,
    Reuters, MarketWatch, etc.) -> [(text, source), ...]. Free tier."""
    import os
    import requests
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    frm = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = []
    try:
        r = requests.get("https://finnhub.io/api/v1/company-news",
                         params={"symbol": sym, "from": frm, "to": to, "token": key},
                         timeout=20)
        if r.status_code == 200:
            for a in r.json()[:60]:
                text = f"{a.get('headline','')}. {a.get('summary','')[:200]}".strip()
                if text and text != ".":
                    out.append((text, a.get("source", "finnhub")))
    except Exception:
        pass
    return out


def ticker_sentiment(symbols, lookback_days: int = 21) -> pd.DataFrame:
    """Multi-source per-ticker headline sentiment (Yahoo RSS + Finnhub company
    news, which itself aggregates Benzinga/Reuters/MarketWatch/etc.), VADER-scored."""
    sia = _vader()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []
    for sym in symbols:
        items = _yahoo_headlines(sym, cutoff) + _finnhub_headlines(sym, lookback_days)
        seen, scores, srcs, worst, n_neg = set(), [], set(), 0.0, 0
        for text, src in items:
            key = text[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            c = sia.polarity_scores(text)["compound"]
            scores.append(c)
            srcs.add(src)
            worst = min(worst, c)
            if c < NEG_HEADLINE:
                n_neg += 1
        time.sleep(0.05)
        n = len(scores)
        mean_s = float(pd.Series(scores).mean()) if scores else 0.0
        share_neg = n_neg / n if n else 0.0
        veto = bool(n >= VETO_MIN_HEADLINES
                    and (mean_s < VETO_MEAN or share_neg >= VETO_SHARE_NEG))
        rows.append({
            "ticker": sym,
            "news_sentiment": round(mean_s, 3),
            "worst_headline": round(worst, 3),
            "share_negative": round(share_neg, 2),
            "headline_count": n,
            "n_sources": len(srcs),
            "event_veto": veto,
        })
    return pd.DataFrame(rows)


def apply_veto(candidates: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    """Attach sentiment columns to candidate names and drop vetoed ones.
    Returns (kept_df, full_sentiment_df). Fails open: if news fetch errors,
    nothing is vetoed."""
    syms = candidates["ticker"].tolist()
    if not syms:
        return candidates.assign(news_sentiment=0.0, event_veto=False), pd.DataFrame()
    try:
        sent = ticker_sentiment(syms, lookback_days)
    except Exception as e:
        print(f"Sentiment unavailable ({e}); no veto applied.")
        return candidates.assign(news_sentiment=0.0, event_veto=False), pd.DataFrame()
    merged = candidates.merge(sent, on="ticker", how="left")
    merged["event_veto"] = merged["event_veto"].fillna(False)
    merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)
    kept = merged[~merged["event_veto"]].copy()
    dropped = merged[merged["event_veto"]]["ticker"].tolist()
    if dropped:
        print(f"Event-risk veto removed {len(dropped)}: {dropped}")
    return kept, sent


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--lookback-days", type=int, default=21)
    args = ap.parse_args()
    df = ticker_sentiment(args.tickers, args.lookback_days)
    print(df.to_string(index=False))
