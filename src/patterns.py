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


def signals(close: pd.Series) -> dict:
    """Compact technical-signal bundle for one name."""
    x = sma_cross(close)
    cup = cup_and_handle(close)
    return {
        "ma_signal": x["signal"],
        "ma_above_200": x.get("above"),
        "days_since_cross": x.get("days_since_cross"),
        "cup_and_handle": cup["detected"],
    }


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
