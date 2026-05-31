"""
StockSight per-ticker sentiment + event-risk veto, multi-source and decision-grade.

Sources: Yahoo Finance RSS (no key) + Finnhub company-news (aggregates Benzinga,
Reuters, MarketWatch, etc.) + Seeking Alpha per-ticker RSS (no key). Scored
with VADER (Hutto & Gilbert 2014).

What's returned per ticker (the score columns that actually drive decisions):
  sent_overall   : mean VADER compound across deduped headlines        [-1, +1]
  sent_yahoo     : per-source mean (NaN if no headlines from that source)
  sent_finnhub   : per-source mean
  sent_seeking   : per-source mean
  sources_agree  : True iff every source with >=2 headlines agrees in sign
  worst_headline : the single most-negative score (event-shock detector)
  share_negative : fraction of headlines below NEG_HEADLINE threshold
  sent_momentum  : sentiment in the last 3 days minus the prior window
                   (>0 means news is getting BETTER, <0 means worse)
  event_veto     : drop this name from the portfolio (real event-risk signal)

VETO rule (calibrated for genuine bad-news floods, not single scary headlines):
  >= VETO_MIN_HEADLINES total AND
  (mean tone < VETO_MEAN  OR  >=50% headlines strongly negative  OR
   sent_momentum < -0.30 AND share_negative > 0.4)
"""

from __future__ import annotations
import argparse
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# Tunables
VETO_MEAN = -0.25
VETO_MIN_HEADLINES = 3
NEG_HEADLINE = -0.4
VETO_SHARE_NEG = 0.5
MOMENTUM_VETO = -0.30

YAHOO_RSS = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
             "?s={sym}&region=US&lang=en-US")
SA_RSS = "https://seekingalpha.com/api/sa/combined/{sym}.xml"


def _vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _rss_items(url, source, cutoff):
    """Return [(dt, text, source), ...] from any RSS-ish feed."""
    import feedparser
    out = []
    try:
        for e in feedparser.parse(url).entries[:60]:
            pub = e.get("published_parsed") or e.get("updated_parsed")
            if not pub:
                continue
            dt = datetime(*pub[:6], tzinfo=timezone.utc)
            if dt < cutoff:
                continue
            text = f"{e.get('title','')}. {e.get('summary','')[:200]}".strip()
            if text and text != ".":
                out.append((dt, text, source))
    except Exception:
        pass
    return out


def _finnhub_items(sym, lookback_days, cutoff):
    """[(dt, text, source), ...] from Finnhub company-news REST."""
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
            for a in r.json()[:80]:
                ts = a.get("datetime")
                if not ts:
                    continue
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                if dt < cutoff:
                    continue
                text = f"{a.get('headline','')}. {a.get('summary','')[:200]}".strip()
                if text and text != ".":
                    out.append((dt, text, "finnhub"))
    except Exception:
        pass
    return out


def _mean(xs):
    return float(np.mean(xs)) if xs else np.nan


def ticker_sentiment(symbols, lookback_days: int = 21) -> pd.DataFrame:
    sia = _vader()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    recent_cut = datetime.now(timezone.utc) - timedelta(days=3)
    rows = []
    for sym in symbols:
        items = (_rss_items(YAHOO_RSS.format(sym=sym), "yahoo", cutoff)
                 + _finnhub_items(sym, lookback_days, cutoff)
                 + _rss_items(SA_RSS.format(sym=sym), "seekingalpha", cutoff))

        # Dedup by first 80 chars (same story syndicates across sources).
        seen = set()
        scored = []                                # list of (dt, score, source)
        for dt, text, src in items:
            k = text[:80].lower()
            if k in seen:
                continue
            seen.add(k)
            c = sia.polarity_scores(text)["compound"]
            scored.append((dt, c, src))

        by_src = {"yahoo": [], "finnhub": [], "seekingalpha": []}
        recent_scores, older_scores, all_scores = [], [], []
        worst, n_neg = 0.0, 0
        for dt, c, src in scored:
            by_src.setdefault(src, []).append(c)
            all_scores.append(c)
            (recent_scores if dt >= recent_cut else older_scores).append(c)
            worst = min(worst, c)
            if c < NEG_HEADLINE:
                n_neg += 1

        n = len(all_scores)
        mean_overall = _mean(all_scores) if n else 0.0
        share_neg = n_neg / n if n else 0.0
        per_src = {s: _mean(v) for s, v in by_src.items()}
        # Agreement: sources with >= 2 headlines must all share sign
        signing_srcs = [v for v in per_src.values() if not np.isnan(v) and abs(v) > 0.05]
        agree = bool(signing_srcs) and all(np.sign(v) == np.sign(signing_srcs[0])
                                           for v in signing_srcs)
        # Momentum: recent (last 3d) minus older (prior window)
        sent_mom = (_mean(recent_scores) - _mean(older_scores)
                    if recent_scores and older_scores else np.nan)
        time.sleep(0.05)
        veto = bool(
            n >= VETO_MIN_HEADLINES
            and (mean_overall < VETO_MEAN
                 or share_neg >= VETO_SHARE_NEG
                 or (np.isfinite(sent_mom) and sent_mom < MOMENTUM_VETO
                     and share_neg > 0.4))
        )
        rows.append({
            "ticker": sym,
            "sent_overall": round(float(mean_overall), 3),
            "sent_yahoo": round(per_src.get("yahoo"), 3) if not np.isnan(per_src.get("yahoo", np.nan)) else None,
            "sent_finnhub": round(per_src.get("finnhub"), 3) if not np.isnan(per_src.get("finnhub", np.nan)) else None,
            "sent_seeking": round(per_src.get("seekingalpha"), 3) if not np.isnan(per_src.get("seekingalpha", np.nan)) else None,
            "sources_agree": agree,
            "worst_headline": round(worst, 3),
            "share_negative": round(share_neg, 2),
            "sent_momentum": round(float(sent_mom), 3) if np.isfinite(sent_mom) else None,
            "n_headlines": n,
            "event_veto": veto,
        })
    return pd.DataFrame(rows)


def apply_veto(candidates: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    syms = candidates["ticker"].tolist()
    if not syms:
        return (candidates.assign(sent_overall=0.0, event_veto=False),
                pd.DataFrame())
    try:
        sent = ticker_sentiment(syms, lookback_days)
    except Exception as e:
        print(f"Sentiment unavailable ({e}); no veto applied.")
        return (candidates.assign(sent_overall=0.0, event_veto=False),
                pd.DataFrame())
    merged = candidates.merge(sent, on="ticker", how="left")
    merged["event_veto"] = merged["event_veto"].fillna(False)
    merged["sent_overall"] = merged["sent_overall"].fillna(0.0)
    # Back-compat: callers reading "news_sentiment"
    merged["news_sentiment"] = merged["sent_overall"]
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
    cols = ["ticker", "sent_overall", "sent_yahoo", "sent_finnhub", "sent_seeking",
            "sources_agree", "sent_momentum", "worst_headline", "share_negative",
            "n_headlines", "event_veto"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
