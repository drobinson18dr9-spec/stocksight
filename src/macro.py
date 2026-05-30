"""
StockSight macro context (free, no API key) via Yahoo market indices (yfinance):
  ^TNX 10y Treasury yield, ^IRX 13-week T-bill, ^VIX volatility index.
Derives the 10y-3m yield-curve spread (a standard recession indicator,
Estrella & Mishkin) and a simple risk regime flag (risk-off when the curve
is inverted or VIX is elevated). Yahoo is reliable where FRED's CSV is slow.

Usage: python src/macro.py
"""

from __future__ import annotations


def _last(sym: str):
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="5d")
        return round(float(h["Close"].iloc[-1]), 2) if len(h) else None
    except Exception:
        return None


def get() -> dict:
    t10 = _last("^TNX")     # 10y yield (percent)
    t3m = _last("^IRX")     # 13-week T-bill (percent)
    vix = _last("^VIX")
    spread = round(t10 - t3m, 2) if (t10 is not None and t3m is not None) else None
    inverted = spread is not None and spread < 0
    high_vix = vix is not None and vix > 25
    return {
        "treasury_10y": t10,
        "tbill_3m": t3m,
        "vix": vix,
        "yield_curve_spread": spread,
        "regime": "risk-off" if (inverted or high_vix) else "risk-on",
        "regime_reason": ("inverted yield curve" if inverted
                          else "elevated VIX" if high_vix else "curve normal, VIX calm"),
    }


if __name__ == "__main__":
    m = get()
    for k, v in m.items():
        print(f"  {k}: {v}")
