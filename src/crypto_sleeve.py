"""
Crypto in the math (research-backed, Johansson & Boyd 2024).

Treats crypto as a real asset sleeve, not a price card:
  - Daily history via yfinance (BTC-USD, ETH-USD, ...), annualized on 365 (24/7).
  - Same risk-adjusted screen as equities but 365-day, plus Student-t VaR
    because crypto is leptokurtic (kurtosis 6-16; Gaussian VaR understates risk).
  - HRP weights WITHIN the crypto sleeve.
  - combine(): 90/10 traditional/crypto split with a HARD 10% combined crypto
    cap, the cap (not a diversification assumption) is what delivers the benefit;
    crypto-equity correlation is NOT reliably low (that claim was refuted).

All point-in-time; no lookahead. Live spot can come from coinbase.py/robinhood.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import scorecard as sc
import metrics as mx

CRYPTO_PPY = 365
DEFAULT_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
                    "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD", "MATIC-USD"]
CRYPTO_CAP = 0.10          # hard combined crypto weight cap (Johansson & Boyd)


def fetch_crypto(symbols=DEFAULT_UNIVERSE, lookback_days=420) -> pd.DataFrame:
    import yfinance as yf
    from datetime import datetime, timedelta, timezone
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    frames = []
    for s in symbols:
        try:
            h = yf.Ticker(s).history(start=start, auto_adjust=True)
            if len(h) >= 200:
                df = h.reset_index()[["Date", "Close", "Volume"]]
                df.columns = ["timestamp", "close", "volume"]
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df["ticker"] = s
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def crypto_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    """Risk-adjusted metrics per coin on a 365-day year, Student-t VaR."""
    rows = []
    rf_d = (1 + sc.RISK_FREE_RATE) ** (1 / CRYPTO_PPY) - 1
    for t, g in bars.groupby("ticker"):
        g = g.sort_values("timestamp")
        close = g["close"].astype(float)
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        ret = sc._winsorize(ret)
        if len(ret) < 120 or ret.std(ddof=1) == 0:
            continue
        excess = ret - rf_d
        sharpe = np.sqrt(CRYPTO_PPY) * excess.mean() / ret.std(ddof=1)
        p_now, p_0 = float(close.iloc[-1]), float(close.iloc[0])
        p_1m = float(close.iloc[-31]) if len(close) > 31 else p_0
        mom = p_1m / p_0 - 1 if p_0 > 0 else np.nan      # ~12-1 on crypto calendar
        rows.append({
            "ticker": t, "current_price": round(p_now, 2),
            "sharpe": round(float(sharpe), 3),
            "ann_vol": round(mx.ann_vol(ret, CRYPTO_PPY), 3),
            "momentum": round(float(mom), 3) if np.isfinite(mom) else np.nan,
            "cvar_95": round(mx.cvar(ret, 0.05), 4),
            "var_95_studentT": round(mx.student_t_var(ret, 0.05), 4),
            "kurtosis": round(float(ret.kurtosis()) + 3, 2),   # Pearson (Gaussian=3)
        })
    return pd.DataFrame(rows)


def crypto_hrp(bars: pd.DataFrame, names: list[str]) -> pd.Series:
    rets = (bars[bars["ticker"].isin(names)]
            .pivot_table(index="timestamp", columns="ticker", values="close")
            .pct_change().tail(CRYPTO_PPY).dropna(axis=1, how="all").dropna())
    rets = rets.apply(sc._winsorize, axis=0)
    names = [n for n in names if n in rets.columns]
    if len(names) < 2:
        return pd.Series({n: 1.0 / max(len(names), 1) for n in names})
    return sc.hrp_weights(rets[names])


def combine(equity_weights: pd.Series, crypto_weights: pd.Series,
            crypto_cap: float = CRYPTO_CAP) -> pd.DataFrame:
    """90/10 split with a hard combined-crypto cap. Equity scaled to (1-cap),
    crypto scaled to cap. The cap is the risk control, not a low-correlation bet."""
    eq = equity_weights / equity_weights.sum() * (1 - crypto_cap)
    cr = crypto_weights / crypto_weights.sum() * crypto_cap if len(crypto_weights) else pd.Series(dtype=float)
    out = pd.concat([
        pd.DataFrame({"asset": eq.index, "weight": eq.values, "sleeve": "equity"}),
        pd.DataFrame({"asset": cr.index, "weight": cr.values, "sleeve": "crypto"}),
    ], ignore_index=True)
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


def build(top_n: int = 5):
    sc.set_live_risk_free()
    bars = fetch_crypto()
    if bars.empty:
        print("No crypto data."); return pd.DataFrame(), pd.Series(dtype=float)
    m = crypto_metrics(bars)
    m = m[(m["sharpe"] > 0) & (m["momentum"] > 0)].sort_values("sharpe", ascending=False)
    picks = m.head(top_n)["ticker"].tolist()
    w = crypto_hrp(bars, picks)
    return m, w


if __name__ == "__main__":
    m, w = build()
    print("Crypto screen (365-day, Student-t VaR):")
    print(m.to_string(index=False))
    print("\nCrypto sleeve HRP weights:")
    print((w * 100).round(1).to_string())
