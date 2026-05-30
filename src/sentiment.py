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


def ticker_sentiment(symbols, lookback_days: int = 21) -> pd.DataFrame:
    import feedparser
    sia = _vader()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []
    for sym in symbols:
        scores, worst, n_recent, n_neg = [], 0.0, 0, 0
        try:
            feed = feedparser.parse(YAHOO_RSS.format(sym=sym))
            for e in feed.entries:
                pub = e.get("published_parsed") or e.get("updated_parsed")
                if pub:
                    dt = datetime(*pub[:6], tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                title = e.get("title", "")
                summary = e.get("summary", "")[:200]
                text = f"{title}. {summary}".strip()
                if not text:
                    continue
                c = sia.polarity_scores(text)["compound"]
                scores.append(c)
                worst = min(worst, c)
                n_recent += 1
                if c < NEG_HEADLINE:
                    n_neg += 1
        except Exception:
            pass
        time.sleep(0.05)
        mean_s = float(pd.Series(scores).mean()) if scores else 0.0
        share_neg = n_neg / n_recent if n_recent else 0.0
        veto = bool(n_recent >= VETO_MIN_HEADLINES
                    and (mean_s < VETO_MEAN or share_neg >= VETO_SHARE_NEG))
        rows.append({
            "ticker": sym,
            "news_sentiment": round(mean_s, 3),
            "worst_headline": round(worst, 3),
            "share_negative": round(share_neg, 2),
            "headline_count": n_recent,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--lookback-days", type=int, default=21)
    args = ap.parse_args()
    df = ticker_sentiment(args.tickers, args.lookback_days)
    print(df.to_string(index=False))
