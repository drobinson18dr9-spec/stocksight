"""
Macro regime + fixed-income signals (research-backed).

  - Term spread (10y - 3m) -> NY Fed / Estrella-Mishkin 12-month recession prob:
        P(recession) = Phi(-0.5333 - 0.6330 * spread)
    Use probabilistically (it false-flagged Oct-2022..Dec-2024), not as a switch.
  - High-yield credit spread (FRED BAMLH0A0HYM2) and MOVE index (bond vol) as
    additional risk-off signals.
  - Bond ETFs (TLT/IEF/AGG/SHY/LQD/HYG) are the investable fixed-income sleeve /
    risk-off allocation.

Free sources: FRED API (DGS10, DGS3MO, BAMLH0A0HYM2) with FRED_API_KEY, plus
Yahoo (^MOVE) fallback. Fails open to None.
"""

from __future__ import annotations
import os
from scipy.stats import norm
import requests

BOND_ETFS = ["SHY", "IEF", "TLT", "AGG", "LQD", "HYG"]   # short->long, agg, IG, HY
FRED = "https://api.stlouisfed.org/fred/series/observations"


def _fred_latest(series_id: str):
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(FRED, params={"series_id": series_id, "api_key": key,
                                       "file_type": "json", "sort_order": "desc",
                                       "limit": 1}, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        v = obs[0]["value"] if obs else "."
        return float(v) if v not in (".", "", None) else None
    except Exception:
        return None


def _yf_last(symbol: str):
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="5d")
        return float(h["Close"].iloc[-1]) if len(h) else None
    except Exception:
        return None


def recession_prob(spread: float) -> float:
    """Estrella-Mishkin 12-month-ahead recession probability from the term spread."""
    return float(norm.cdf(-0.5333 - 0.6330 * spread))


def get() -> dict:
    t10 = _fred_latest("DGS10") or _yf_last("^TNX")
    t3m = _fred_latest("DGS3MO") or (_yf_last("^IRX"))
    hy = _fred_latest("BAMLH0A0HYM2")            # HY option-adjusted spread (%)
    move = _yf_last("^MOVE")                      # bond-market implied vol
    spread = round(t10 - t3m, 2) if (t10 is not None and t3m is not None) else None
    prec = round(recession_prob(spread), 3) if spread is not None else None

    inverted = spread is not None and spread < 0
    wide_credit = hy is not None and hy > 5.0     # HY OAS > 5% = stress
    high_move = move is not None and move > 120    # elevated bond vol
    risk_off = bool(inverted or wide_credit or high_move)
    reasons = []
    if inverted: reasons.append("inverted curve")
    if wide_credit: reasons.append("wide HY credit spread")
    if high_move: reasons.append("elevated MOVE")
    return {
        "treasury_10y": t10, "tbill_3m": t3m, "term_spread": spread,
        "recession_prob_12m": prec, "hy_credit_spread": hy, "move_index": move,
        "regime": "risk-off" if risk_off else "risk-on",
        "regime_reason": ", ".join(reasons) if reasons else "curve normal, credit calm, MOVE calm",
        "bond_etfs": BOND_ETFS,
    }


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    print(json.dumps(get(), indent=2))
