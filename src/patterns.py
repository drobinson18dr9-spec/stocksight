"""
Technical chart patterns (point-in-time, no lookahead).

  - Golden cross / death cross: 50-day SMA crossing the 200-day SMA. Precise,
    well-defined, widely studied as a trend-following signal.
  - Cup-and-handle: a rounded base + small handle + breakout. NOTE: this pattern
    has no universal mathematical definition, so this is a transparent HEURISTIC
    (rim symmetry, cup depth band, shallow handle, breakout proximity), not an
    exact detector. Use as a soft flag, not gospel.
  - MA stack / trend state for context.

Every signal uses only data up to the evaluation bar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma_cross(close: pd.Series, fast: int = 50, slow: int = 200) -> dict:
    """Golden/death cross state from 50/200 SMA. Returns current relationship,
    the most recent cross type, and trading days since it occurred."""
    c = pd.Series(close).astype(float).dropna()
    if len(c) < slow + 2:
        return {"signal": "n/a", "above": None, "days_since_cross": None}
    sf = c.rolling(fast).mean()
    ss = c.rolling(slow).mean()
    state = np.sign((sf - ss).dropna())            # +1 fast above slow, -1 below
    above = bool(state.iloc[-1] > 0)
    changes = state[state.diff().fillna(0) != 0]   # bars where the cross flipped
    if len(changes) == 0:
        return {"signal": "none", "above": above, "days_since_cross": None}
    last_idx = changes.index[-1]
    days_since = int(len(state) - state.index.get_loc(last_idx) - 1)
    signal = "golden_cross" if changes.iloc[-1] > 0 else "death_cross"
    return {"signal": signal, "above": above, "days_since_cross": days_since,
            "fast": fast, "slow": slow}


def cup_and_handle(close: pd.Series, window: int = 130) -> dict:
    """Heuristic cup-and-handle over the trailing `window` bars (~6 months).
    Returns detected flag + the geometric checks, so it is auditable, not magic."""
    c = pd.Series(close).astype(float).dropna()
    if len(c) < window:
        return {"detected": False, "reason": "insufficient history"}
    w = c.tail(window).reset_index(drop=True)
    n = len(w)
    third = n // 3
    left_rim = w[:third].max()
    cup_bottom = w[third:2 * third].min()
    right_region = w[2 * third:]
    right_rim = right_region.max()
    rim = (left_rim + right_rim) / 2
    if rim <= 0:
        return {"detected": False, "reason": "bad data"}

    cup_depth = (rim - cup_bottom) / rim                     # how deep the cup is
    rim_symmetry = abs(left_rim - right_rim) / rim           # rims should be similar
    # handle = pullback after the right rim is reached
    rr_pos = int(right_region.idxmax())
    handle = right_region.loc[rr_pos:]
    handle_depth = (right_rim - handle.min()) / right_rim if len(handle) > 1 else 0.0
    breakout = w.iloc[-1] >= right_rim * 0.97               # near/through the rim

    detected = bool(
        0.10 <= cup_depth <= 0.50 and          # cup neither too shallow nor a crash
        rim_symmetry <= 0.07 and               # rims roughly level
        0 < handle_depth <= cup_depth / 2 and  # handle shallower than the cup
        breakout
    )
    return {
        "detected": detected,
        "cup_depth": round(float(cup_depth), 3),
        "rim_symmetry": round(float(rim_symmetry), 3),
        "handle_depth": round(float(handle_depth), 3),
        "near_breakout": bool(breakout),
    }


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               atr_len: int = 10, mult: float = 3.0) -> dict:
    """Supertrend (ATR-based) discrete trend state. Deterministic from OHLC.
    Returns trend ('up'/'down'), the line value, and distance of close to it."""
    h, l, c = (pd.Series(high).astype(float), pd.Series(low).astype(float),
               pd.Series(close).astype(float))
    if len(c) < atr_len + 2:
        return {"trend": "n/a"}
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()
    hl2 = (h + l) / 2
    ub = (hl2 + mult * atr).values
    lb = (hl2 - mult * atr).values
    cv = c.values
    n = len(cv)
    valid = ~np.isnan(ub)
    if valid.sum() < 2:
        return {"trend": "n/a"}
    first = int(np.argmax(valid))               # first bar with a valid ATR band
    fub = np.full(n, np.nan); flb = np.full(n, np.nan); trend = np.ones(n)
    fub[first], flb[first] = ub[first], lb[first]
    for i in range(first + 1, n):
        # Carry the band forward; reset cleanly if the prior band was NaN.
        fub[i] = ub[i] if (np.isnan(fub[i-1]) or ub[i] < fub[i-1] or cv[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (np.isnan(flb[i-1]) or lb[i] > flb[i-1] or cv[i-1] < flb[i-1]) else flb[i-1]
        if trend[i-1] > 0:                       # was uptrend (line = lower band)
            trend[i] = -1 if cv[i] < flb[i] else 1
        else:                                    # was downtrend (line = upper band)
            trend[i] = 1 if cv[i] > fub[i] else -1
    line = float(flb[-1] if trend[-1] > 0 else fub[-1])
    return {"trend": "up" if trend[-1] > 0 else "down",
            "supertrend": round(line, 2),
            "dist": round(float(cv[-1] / line - 1), 4) if line else None}


def pct_to_52w_high(close: pd.Series, window: int = 252) -> float:
    """George-Hwang nearness to the 52-week high: close / max(close, 252d)."""
    c = pd.Series(close).astype(float).dropna()
    hi = c.tail(window).max()
    return float(c.iloc[-1] / hi) if hi > 0 else float("nan")


def signals(close: pd.Series, high: pd.Series | None = None,
            low: pd.Series | None = None) -> dict:
    """Compact technical-signal bundle for one name."""
    x = sma_cross(close)
    cup = cup_and_handle(close)
    out = {
        "ma_signal": x["signal"],
        "ma_above_200": x.get("above"),
        "days_since_cross": x.get("days_since_cross"),
        "cup_and_handle": cup["detected"],
        "pct_to_52w_high": round(pct_to_52w_high(close), 3),
    }
    if high is not None and low is not None:
        out["supertrend"] = supertrend(high, low, close)["trend"]
    return out


if __name__ == "__main__":
    # golden cross: long downtrend then sustained uptrend
    down = np.linspace(100, 60, 220)
    up = np.linspace(60, 130, 120)
    s = pd.Series(np.concatenate([down, up]))
    print("uptrend-after-downtrend ->", sma_cross(s))
    print("cup test ->", cup_and_handle(pd.Series(
        np.concatenate([np.linspace(100, 100, 10), np.linspace(100, 75, 40),
                        np.linspace(75, 100, 40), np.linspace(100, 95, 20),
                        np.linspace(95, 101, 20)]))))
