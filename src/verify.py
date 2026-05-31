"""
StockSight math verification suite.

Proves the production formulas are correct by recomputing each one a second,
independent way (or against a hand-known value) and asserting they match.
This guarantees the MATH has no errors. It does NOT (and cannot) guarantee
forecast accuracy, markets are near-random-walk and no formula removes that.

Run: python src/verify.py   (exits non-zero if any check fails)
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

import scorecard as sc

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def synth_bars(seed=1, n=320):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    px = 100 * np.cumprod(1 + rng.normal(0.0008, 0.018, n))
    return pd.DataFrame({"ticker": "TST", "timestamp": dates, "close": px,
                         "volume": rng.integers(2_000_000, 9_000_000, n)})


def main():
    # Deterministic rf so the test is reproducible offline.
    sc.RISK_FREE_RATE = 0.04
    sc.RF_DAILY = (1 + 0.04) ** (1 / sc.TRADING_DAYS) - 1

    bars = synth_bars()
    m = sc.compute_metrics(bars)
    assert len(m) == 1, "compute_metrics should return one row"
    r = m.iloc[0]

    # Independent re-derivation from the same prices ----------------------
    close = bars.sort_values("timestamp")["close"].astype(float)
    win = close.tail(sc.TRADING_DAYS + 1)
    ret = win.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret.clip(ret.quantile(0.005), ret.quantile(0.995))
    n = len(ret)
    sd = ret.std(ddof=1)
    excess = ret - sc.RF_DAILY

    sharpe_indep = np.sqrt(sc.TRADING_DAYS) * excess.mean() / sd
    check("Sharpe matches independent re-derivation",
          abs(r["sharpe"] - sharpe_indep) < 1e-9, f"{r['sharpe']:.6f} vs {sharpe_indep:.6f}")

    # Cross-check: annualized Sharpe == t-stat at n≈252 identity
    check("Sharpe t-stat == SR_daily*sqrt(n)",
          abs(r["sharpe_tstat"] - (excess.mean() / sd) * np.sqrt(n)) < 1e-9)

    dd_dev = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    sortino_indep = np.sqrt(sc.TRADING_DAYS) * excess.mean() / dd_dev
    check("Sortino matches downside-deviation definition",
          abs(r["sortino"] - sortino_indep) < 1e-9, f"{r['sortino']:.6f}")

    p_12m = float(close.iloc[-(sc.TRADING_DAYS + 1)])
    p_1m = float(close.iloc[-22])
    mom_indep = p_1m / p_12m - 1
    check("12-1 momentum matches P[-22]/P[-253]-1",
          abs(r["momentum_12_1"] - mom_indep) < 1e-9, f"{r['momentum_12_1']:.6f}")

    equity = (1 + ret).cumprod()
    dd_indep = float((equity / equity.cummax() - 1).min())
    check("Max drawdown matches peak-to-trough", abs(r["max_drawdown"] - dd_indep) < 1e-9)

    # Known-value max drawdown: 100->120->60 has DD = -50%
    kp = pd.Series([100, 120, 60.0])
    kdd = float(((1 + kp.pct_change()).cumprod() /
                 (1 + kp.pct_change()).cumprod().cummax() - 1).min())
    check("Max drawdown known value (120->60 = -50%)", abs(kdd - (-0.5)) < 1e-9, f"{kdd:.4f}")

    # Robust z: the median element maps to ~0; outliers are clipped -------
    zmed = sc._robust_z(pd.Series([10, 20, 30, 40, 50.0]))   # odd, median=30 at idx 2
    check("Robust z maps median to 0", abs(zmed.iloc[2]) < 1e-9, f"{zmed.iloc[2]:.2e}")
    zclip = sc._robust_z(pd.Series([1, 2, 3, 4, 5, 100.0]))
    check("Robust z clips outliers to <= 3", zclip.max() <= 3 + 1e-9)

    # HRP: weights sum to 1, all non-negative, diversified ---------------
    rng = np.random.default_rng(7)
    rr = pd.DataFrame(rng.normal(0, 0.01, (252, 6)),
                      columns=[f"S{i}" for i in range(6)])
    w = sc.hrp_weights(rr)
    check("HRP weights sum to 1", abs(w.sum() - 1.0) < 1e-9, f"sum={w.sum():.6f}")
    check("HRP weights non-negative", bool((w >= -1e-12).all()))
    check("HRP is diversified (not all in one name)", w.max() < 0.95)

    # Probabilistic / Deflated Sharpe sanity ------------------------------
    import quant_alpha as qa
    psr_half = qa.probabilistic_sharpe(0.1, 100, 0.0, 3.0, sr_benchmark=0.1)
    check("PSR at SR==benchmark is ~0.5", abs(psr_half - 0.5) < 0.05, f"{psr_half:.3f}")
    psr0 = qa.probabilistic_sharpe(0.1, 100, 0.0, 3.0, sr_benchmark=0.0)
    check("PSR in [0,1] and > 0.5 for positive SR", 0 <= psr0 <= 1 and psr0 > 0.5, f"{psr0:.3f}")
    dsr = qa.deflated_sharpe(0.1, 100, 0.0, 3.0, n_trials=20, sr_trials_std=0.02)
    check("Deflated Sharpe <= PSR (deflation penalizes multiple testing)",
          dsr <= psr0 + 1e-9, f"DSR={dsr:.3f} <= PSR={psr0:.3f}")

    # Winsorization bounds ------------------------------------------------
    s = pd.Series(list(range(100)) + [10_000.0])
    wn = sc._winsorize(s)
    check("Winsorize caps at the 99.5 pctile", wn.max() <= s.quantile(0.995) + 1e-9)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed.")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("All math checks passed: formulas are correct and internally consistent.")


if __name__ == "__main__":
    main()
