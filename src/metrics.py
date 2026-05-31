"""
StockSight complete risk/performance metrics, research-backed (2026).

Every function is a textbook/peer-reviewed definition, with `ppy` (periods per
year) so the SAME code is correct for equities (252) and crypto (365, 24/7).

References:
  CVaR/ES coherent & subadditive ... Artzner et al. (1999); Rockafellar-Uryasev (2000)
  Calmar ........................... Young (1991)
  Omega ............................ Keating & Shadwick (2002)
  Ulcer index ...................... Martin & McCann (1989)
  Jensen's alpha / beta ............ Jensen (1968), CAPM
  Information ratio ................ Grinold & Kahn
  Student-t VaR for fat tails ...... crypto kurtosis 6-16 (Empirical Economics 2023)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

EQUITY_PPY = 252
CRYPTO_PPY = 365      # crypto trades 24/7/365


def _eq(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def cagr(returns: pd.Series, ppy: int = EQUITY_PPY) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan
    ending = _eq(r).iloc[-1]
    if ending <= 0:                       # a <=-100% path: total loss (audit fix)
        return -1.0
    return float(ending ** (ppy / len(r)) - 1)


def ann_vol(returns: pd.Series, ppy: int = EQUITY_PPY) -> float:
    return float(pd.Series(returns).dropna().std(ddof=1) * np.sqrt(ppy))


def max_drawdown(returns: pd.Series) -> float:
    eq = _eq(pd.Series(returns).dropna())
    return float((eq / eq.cummax() - 1).min())


def calmar(returns: pd.Series, ppy: int = EQUITY_PPY) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, ppy) / abs(mdd)) if mdd < 0 else np.nan


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss number at confidence (1-alpha)."""
    r = pd.Series(returns).dropna()
    return float(-np.quantile(r, alpha)) if len(r) else np.nan


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR / Expected Shortfall: mean loss in the worst-alpha tail.
    Coherent & subadditive (Rockafellar-Uryasev). Positive = loss."""
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return np.nan
    # Average exactly the worst-alpha observations (sorted), so ties at the
    # quantile don't pull in extra mass and bias ES toward zero (audit fix).
    k = max(1, int(np.ceil(alpha * len(r))))
    worst = np.sort(r.values)[:k]
    return float(-worst.mean())


def student_t_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Parametric VaR under a fitted Student-t (fat tails). For crypto sleeves
    where Gaussian VaR understates risk (kurtosis 6-16)."""
    r = pd.Series(returns).dropna()
    if len(r) < 30:
        return np.nan
    df, loc, scale = stats.t.fit(r)
    return float(-stats.t.ppf(alpha, df, loc=loc, scale=scale))


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    r = pd.Series(returns).dropna()
    return float(np.sqrt((np.minimum(r - mar, 0.0) ** 2).mean()))


def omega(returns: pd.Series, mar: float = 0.0) -> float:
    """Omega ratio: prob-weighted gains over losses about a threshold (MAR)."""
    r = pd.Series(returns).dropna() - mar
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else np.inf


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer index: RMS of percentage drawdowns (depth AND duration of pain)."""
    eq = _eq(pd.Series(returns).dropna())
    dd = (eq / eq.cummax() - 1) * 100
    return float(np.sqrt((dd ** 2).mean()))


def beta(returns: pd.Series, market: pd.Series) -> float:
    a, b = pd.Series(returns).align(pd.Series(market), join="inner")
    a, b = a.dropna(), b.dropna()
    a, b = a.align(b, join="inner")
    if len(a) < 5 or b.var() == 0:
        return np.nan
    return float(np.cov(a, b)[0, 1] / np.var(b, ddof=1))


def jensens_alpha(returns: pd.Series, market: pd.Series,
                  rf_annual: float = 0.04, ppy: int = EQUITY_PPY) -> float:
    """Annualized CAPM alpha: actual excess return minus beta * market excess."""
    a, b = pd.Series(returns).align(pd.Series(market), join="inner")
    a, b = a.dropna(), b.dropna()
    a, b = a.align(b, join="inner")
    if len(a) < 5:
        return np.nan
    rf_d = (1 + rf_annual) ** (1 / ppy) - 1
    bt = beta(a, b)
    alpha_d = (a.mean() - rf_d) - bt * (b.mean() - rf_d)
    return float(alpha_d * ppy)


def information_ratio(returns: pd.Series, benchmark: pd.Series,
                      ppy: int = EQUITY_PPY) -> float:
    """Active return per unit tracking error vs a benchmark."""
    a, b = pd.Series(returns).align(pd.Series(benchmark), join="inner")
    a, b = a.dropna(), b.dropna()
    a, b = a.align(b, join="inner")
    active = a - b
    te = active.std(ddof=1)
    return float((active.mean() * ppy) / (te * np.sqrt(ppy))) if te > 0 else np.nan


def full_report(returns: pd.Series, market: pd.Series | None = None,
                rf_annual: float = 0.04, ppy: int = EQUITY_PPY) -> dict:
    """One call -> the complete metric set for a return stream."""
    r = pd.Series(returns).dropna()
    rf_d = (1 + rf_annual) ** (1 / ppy) - 1
    vol = ann_vol(r, ppy)
    sharpe = ((r.mean() - rf_d) / r.std(ddof=1) * np.sqrt(ppy)) if r.std(ddof=1) > 0 else np.nan
    dd = downside_deviation(r, rf_d)
    sortino = ((r.mean() - rf_d) / dd * np.sqrt(ppy)) if dd > 0 else np.nan
    out = {
        "CAGR": cagr(r, ppy), "ann_vol": vol, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar(r, ppy), "Omega": omega(r), "Ulcer": ulcer_index(r),
        "max_drawdown": max_drawdown(r),
        "VaR_95": var_historical(r, 0.05), "CVaR_95": cvar(r, 0.05),
        "VaR_95_studentT": student_t_var(r, 0.05),
    }
    if market is not None:
        out["beta"] = beta(r, market)
        out["jensens_alpha"] = jensens_alpha(r, market, rf_annual, ppy)
        out["information_ratio"] = information_ratio(r, market, ppy)
    return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in out.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.012, 600))
    m = pd.Series(rng.normal(0.0004, 0.010, 600))
    import json
    print(json.dumps(full_report(r, m), indent=2))
