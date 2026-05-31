"""
StockSight Stage 2: risk-based portfolio construction.

Implements and compares, out-of-sample, the standard allocators a quant desk
uses to turn a set of selected names into a "best stocks to buy" weighting:

  - Equal weight ........... naive baseline
  - Inverse volatility ..... risk-weighted baseline
  - Max-Sharpe (Markowitz) . current StockSight method, Ledoit-Wolf cov
  - HRP .................... Hierarchical Risk Parity (Lopez de Prado 2016):
                             correlation-distance clustering + recursive
                             bisection + inverse-variance. No matrix inversion,
                             so it is stable where Markowitz is fragile.

Each month we pick a fixed candidate set (top names by momentum x low-vol from
point-in-time data), allocate with each method on trailing 252d returns, hold
one month, and chain the returns. Output: CAGR / vol / Sharpe / max-DD per
method vs SPY, so the BEST RISK-ADJUSTED construction is visible, not assumed.

Usage: python src/portfolio_construct.py --years 5 --n 30
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

import scorecard as sc

REPORTS = Path(__file__).resolve().parents[1] / "reports"
CHART_DIR = REPORTS / "charts"
HOLD = 21
WIN = 252


# ── Allocators ─────────────────────────────────────────────────────────
def w_equal(rets):
    n = rets.shape[1]
    return pd.Series(np.repeat(1 / n, n), index=rets.columns)


def w_inverse_vol(rets):
    iv = 1.0 / rets.std().replace(0, np.nan)      # guard zero-variance (audit fix)
    iv = iv.fillna(0.0)
    s = iv.sum()
    return iv / s if s > 0 else pd.Series(1.0 / rets.shape[1], index=rets.columns)


def w_max_sharpe(rets):
    mu = rets.mean().values * WIN
    cov = LedoitWolf().fit(rets.values).covariance_ * WIN
    n = len(mu)

    def neg_sharpe(w):
        v = np.sqrt(w @ cov @ w)
        return -(w @ mu - sc.RISK_FREE_RATE) / v if v > 0 else 0.0
    res = minimize(neg_sharpe, np.repeat(1/n, n), method="SLSQP",
                   bounds=[(0, 0.25)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x if res.success else np.repeat(1/n, n), 0, None)
    return pd.Series(w / w.sum(), index=rets.columns)


def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov, items):
    c = cov.loc[items, items]
    iv = 1.0 / np.diag(c)
    iv /= iv.sum()
    return float(iv @ c.values @ iv)


def w_hrp(rets):
    cov, corr = rets.cov(), rets.corr()
    dist = ((1 - corr) / 2.0) ** 0.5
    link = linkage(squareform(dist.values, checks=False), "single")
    sort_ix = [corr.index[i] for i in _quasi_diag(link)]
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in
                    ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w.reindex(rets.columns)


def _ivp(cov_df):
    iv = 1.0 / np.diag(cov_df.values)
    iv /= iv.sum()
    return pd.Series(iv, index=cov_df.index)


def w_nco(rets):
    """Nested Clustered Optimization (Lopez de Prado 2019): inverse-variance
    weights WITHIN correlation clusters, inverse-variance ACROSS clusters, then
    the dot product. Contains instability inside clusters where Markowitz blows up."""
    from scipy.cluster.hierarchy import fcluster
    cov, corr = rets.cov(), rets.corr()
    n = len(corr)
    if n < 3:
        return _ivp(cov)
    dist = ((1 - corr) / 2.0) ** 0.5
    link = linkage(squareform(dist.values, checks=False), "ward")
    k = max(2, int(np.sqrt(n)))
    labels = fcluster(link, k, criterion="maxclust")
    clusters = {}
    for asset, lab in zip(corr.index, labels):
        clusters.setdefault(lab, []).append(asset)
    inner = pd.Series(0.0, index=corr.index)
    cl_rets = pd.DataFrame(index=rets.index)
    for lab, members in clusters.items():
        wi = _ivp(cov.loc[members, members])
        inner[members] = wi.values
        cl_rets[lab] = (rets[members] * wi).sum(axis=1)
    outer = _ivp(cl_rets.cov())
    final = inner.copy()
    for lab, members in clusters.items():
        final[members] *= outer[lab]
    return (final / final.sum()).reindex(rets.columns)


def vol_target_scalar(returns, target_vol=0.10, ppy=252, cap=2.0):
    """Leverage/dilution scalar to hit a target annual vol (Man Group 2018:
    raises Sharpe on equity/credit, cuts tails everywhere). Capped, no shorting."""
    rv = pd.Series(returns).dropna().std(ddof=1) * np.sqrt(ppy)
    return float(min(target_vol / rv, cap)) if rv > 0 else 1.0


ALLOCATORS = {"EqualWeight": w_equal, "InverseVol": w_inverse_vol,
              "MaxSharpe": w_max_sharpe, "HRP": w_hrp, "NCO": w_nco}


def stats(r, ppy):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (ppy / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(ppy)
    sharpe = (r.mean() * ppy - sc.RISK_FREE_RATE) / vol if vol > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sharpe, maxdd


def run(years=5, n=30):
    bars = sc.get_bars(lookback_days=2225)
    close = bars.pivot_table(index="timestamp", columns="ticker", values="close").sort_index()
    vol = bars.pivot_table(index="timestamp", columns="ticker", values="volume").sort_index()
    rets = close.pct_change()
    # point-in-time selection score: momentum x low-vol, liquid, price>=5
    mom = close.shift(21) / close.shift(252) - 1
    lv = -rets.rolling(60).std()
    dvol = (close * vol).rolling(60).median()
    elig = (close >= sc.MIN_PRICE) & (dvol >= sc.MIN_MEDIAN_DOLLAR_VOL)

    dates = close.index
    start = max(WIN + 1, len(dates) - int(years * 252))
    rebal = list(range(start, len(dates) - HOLD, HOLD))
    ppy = 252 / HOLD

    series = {k: [] for k in ALLOCATORS}
    series["SPY"] = []
    idx = []
    for i in rebal:
        # Select on data through i-1, enter at close[i] (no same-day lookahead — audit fix)
        sc_score = (mom.iloc[i - 1].rank() + lv.iloc[i - 1].rank()).where(elig.iloc[i - 1])
        picks = sc_score.dropna().nlargest(n).index.tolist()
        train = rets[picks].iloc[i - WIN:i].dropna(axis=1, how="any")
        picks = [p for p in picks if p in train.columns]
        if len(picks) < 5:
            continue
        train = train[picks]
        fwd = (close[picks].iloc[i + HOLD] / close[picks].iloc[i] - 1)
        valid = fwd.notna()                       # drop names missing a fwd return
        fwd = fwd[valid]
        picks_v = list(fwd.index)
        if len(picks_v) < 5:
            continue
        for name, fn in ALLOCATORS.items():
            try:
                w = fn(train).reindex(picks_v).fillna(0)
                if w.sum() <= 0:
                    series[name].append(np.nan); continue
                w = w / w.sum()
                series[name].append(float((w.values * fwd.values).sum()))
            except Exception:
                series[name].append(np.nan)
        if sc.BENCHMARK in close.columns:
            series["SPY"].append(float(close[sc.BENCHMARK].iloc[i + HOLD] /
                                       close[sc.BENCHMARK].iloc[i] - 1))
        else:
            series["SPY"].append(np.nan)
        idx.append(dates[i])

    out = {}
    rows = []
    for name in list(ALLOCATORS) + ["SPY"]:
        s = pd.Series(series[name], index=idx)
        cagr, v, sh, dd = stats(s, ppy)
        out[name] = {"CAGR_%": round(cagr*100, 2), "Vol_%": round(v*100, 2),
                     "Sharpe": round(sh, 3), "MaxDD_%": round(dd*100, 2)}
        rows.append((name, out[name]))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "construction_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== Stage 2: portfolio construction comparison ({len(idx)} months, top {n} names) ===")
    print(f"  {'Method':<12} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    for name, st in rows:
        print(f"  {name:<12} {st['CAGR_%']:>7}% {st['Vol_%']:>7}% {st['Sharpe']:>8} {st['MaxDD_%']:>7}%")
    best = max(ALLOCATORS, key=lambda k: out[k]["Sharpe"])
    print(f"\n  Best risk-adjusted construction (Sharpe): {best}")
    _plot(series, idx)
    return out


def _plot(series, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"EqualWeight": "#999", "InverseVol": "#8856a7",
              "MaxSharpe": "#2c7fb8", "HRP": "#1a9850", "SPY": "black"}
    for name in series:
        s = pd.Series(series[name], index=idx).dropna()
        eq = (1 + s).cumprod() * 100
        ls = "--" if name == "SPY" else "-"
        ax.plot(eq.index, eq, label=name, color=colors.get(name), lw=2, ls=ls)
    ax.set_yscale("log")
    ax.set_title("Stage 2: portfolio construction methods vs SPY (out-of-sample)")
    ax.set_ylabel("Growth of $100 (log)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(CHART_DIR / "construction_compare.png", dpi=110); plt.close(fig)
    print("Saved chart: construction_compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    run(years=args.years, n=args.n)
