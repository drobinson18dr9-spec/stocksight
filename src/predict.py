"""
StockSight multi-model price prediction engine.

For any ticker it runs a panel of genuine, citable forecasting models and
produces:
  1. A walk-forward ACTUAL vs PREDICTED table over a historical test window
     (each model is trained only on data prior to each date, no lookahead),
     with honest variance = actual - predicted and standard error metrics.
  2. A forward forecast for future dates (predictions only, no actuals exist).

Models (all real, with references):
  - Random Walk + drift .... Hyndman & Athanasopoulos, FPP3 (naive benchmark)
  - ARIMA .................. Box & Jenkins (1970), via statsmodels
  - ETS (Holt-Winters) ..... Holt (1957), Winters (1960), damped trend
  - Theta .................. Assimakopoulos & Nikolopoulos (2000)
  - Ridge (lagged) ......... Hoerl & Kennard (1970) on lagged log-returns
  - AR-GARCH ............... Bollerslev (1986) mean+variance, via arch

Error metrics: MAE, RMSE, MAPE, directional accuracy (all standard).
Variance column = actual - predicted (signed forecast error).

Usage:
  python src/predict.py --ticker AAPL --test-days 40 --horizon 21
Outputs: reports/predict/<TICKER>.json  + reports/charts/predict_<TICKER>.png
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import scorecard as sc

PRED_DIR = Path(__file__).resolve().parents[1] / "reports" / "predict"
CHART_DIR = Path(__file__).resolve().parents[1] / "reports" / "charts"
MIN_HISTORY_FORECAST = 120     # allow recent IPOs (~6 months) to be forecast


# ──────────────────────────────────────────────────────────────────────
# Individual models. Each takes a price Series (train) and horizon h,
# returns an array of h predicted PRICES. Returns NaNs on failure.
# ──────────────────────────────────────────────────────────────────────
def m_random_walk(train: pd.Series, h: int) -> np.ndarray:
    lr = np.log(train).diff().dropna()
    mu = lr.mean()
    last = float(train.iloc[-1])
    return last * np.exp(np.cumsum(np.repeat(mu, h)))


def m_arima(train: pd.Series, h: int, order=(1, 1, 1)) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    log_p = np.log(train.values)
    fit = ARIMA(log_p, order=order).fit()
    fc = fit.forecast(h)
    return np.exp(np.asarray(fc))


def m_ets(train: pd.Series, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(train.values, trend="add", damped_trend=True).fit()
    return np.asarray(fit.forecast(h))


def m_theta(train: pd.Series, h: int) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    fit = ThetaModel(train.values, period=1, deseasonalize=False).fit()
    return np.asarray(fit.forecast(h))


def m_ridge(train: pd.Series, h: int, lags=(1, 2, 3, 5, 10)) -> np.ndarray:
    from sklearn.linear_model import Ridge
    r = np.log(train).diff().dropna()
    X, y = [], []
    maxlag = max(lags)
    for i in range(maxlag, len(r)):
        X.append([r.iloc[i - l] for l in lags])
        y.append(r.iloc[i])
    if len(X) < 30:
        return np.full(h, np.nan)
    model = Ridge(alpha=1.0).fit(np.array(X), np.array(y))
    hist = list(r.values)
    preds, price = [], float(train.iloc[-1])
    for _ in range(h):
        feat = [hist[-l] for l in lags]
        nxt = float(model.predict([feat])[0])
        hist.append(nxt)
        price *= np.exp(nxt)
        preds.append(price)
    return np.array(preds)


def m_argarch(train: pd.Series, h: int) -> np.ndarray:
    from arch import arch_model
    r = np.log(train).diff().dropna() * 100.0
    fit = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    fc = fit.forecast(horizon=h, reindex=False)
    mean_r = np.asarray(fc.mean.values[-1]) / 100.0
    price, preds = float(train.iloc[-1]), []
    for rr in mean_r:
        price *= np.exp(rr)
        preds.append(price)
    return np.array(preds)


MODELS = {
    "RandomWalk": m_random_walk,
    "ARIMA": m_arima,
    "ETS": m_ets,
    "Theta": m_theta,
    "Ridge": m_ridge,
    "AR-GARCH": m_argarch,
}


def _safe(fn, train, h):
    try:
        out = np.asarray(fn(train, h), dtype=float)
        if out.shape[0] != h or not np.all(np.isfinite(out)):
            return np.full(h, np.nan)
        return out
    except Exception:
        return np.full(h, np.nan)


# ──────────────────────────────────────────────────────────────────────
# Walk-forward actual-vs-predicted (1-step-ahead, expanding window)
# ──────────────────────────────────────────────────────────────────────
def walk_forward(prices: pd.Series, test_days: int) -> pd.DataFrame:
    n = len(prices)
    # Warmup = up to 252 bars, but shrink for short-history (recent-IPO) names so
    # there's still a test window; keep >= 60 bars of training (audit/IPO fix).
    start = min(max(60, n - test_days), n - 5)
    rows = []
    for origin in range(start, n):
        train = prices.iloc[:origin]
        actual = float(prices.iloc[origin])
        row = {"date": prices.index[origin], "actual": round(actual, 2)}
        for name, fn in MODELS.items():
            pred = _safe(fn, train, 1)[0]
            row[name] = round(float(pred), 2) if np.isfinite(pred) else None
        rows.append(row)
    return pd.DataFrame(rows)


def error_metrics(wf: pd.DataFrame) -> pd.DataFrame:
    out = []
    actual = wf["actual"].values
    prev = wf["actual"].shift(1).values
    for name in MODELS:
        pred = wf[name].values
        mask = ~pd.isna(pred) & ~pd.isna(actual)
        a, p = actual[mask], pred[mask].astype(float)
        if len(a) < 5:
            continue
        err = a - p
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        mape = np.mean(np.abs(err / a)) * 100
        # directional accuracy: did the model get up/down vs prior actual right?
        pv = prev[mask]
        dmask = ~pd.isna(pv)
        dir_acc = (np.sign(p[dmask] - pv[dmask]) == np.sign(a[dmask] - pv[dmask])).mean() * 100 \
            if dmask.sum() else np.nan
        out.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                    "MAPE_%": round(mape, 2), "DirAcc_%": round(dir_acc, 1),
                    "n": int(len(a))})
    return pd.DataFrame(out).sort_values("RMSE").reset_index(drop=True)


def forward_forecast(prices: pd.Series, h: int) -> pd.DataFrame:
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=h)
    out = {"date": future_dates}
    for name, fn in MODELS.items():
        # ARIMA: small AIC grid for the real forward call (quality over speed)
        if name == "ARIMA":
            out[name] = np.round(_best_arima(prices, h), 2)
        else:
            out[name] = np.round(_safe(fn, prices, h), 2)
    df = pd.DataFrame(out)
    df["ensemble_mean"] = df[list(MODELS)].mean(axis=1, skipna=True).round(2)
    return df


def _best_arima(prices: pd.Series, h: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    log_p = np.log(prices.values)
    best, best_aic = None, np.inf
    for p in (0, 1, 2):
        for d in (1,):
            for q in (0, 1, 2):
                try:
                    fit = ARIMA(log_p, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best, best_aic = fit, fit.aic
                except Exception:
                    continue
    if best is None:
        return np.full(h, np.nan)
    return np.exp(np.asarray(best.forecast(h)))


def run(ticker: str, test_days: int = 40, horizon: int = 21) -> dict:
    ticker = ticker.upper().strip()
    from datetime import datetime, timedelta, timezone
    start = datetime.now(timezone.utc) - timedelta(days=int(3 * 365) + 60)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    bars = sc.fetch_bars([ticker], start, end)   # targeted single-ticker pull
    s = bars[bars["ticker"] == ticker].sort_values("timestamp")
    if len(s) < MIN_HISTORY_FORECAST:
        raise SystemExit(f"{ticker}: not enough history ({len(s)} bars).")
    prices = pd.Series(s["close"].values, index=pd.to_datetime(s["timestamp"].values)).astype(float)

    wf = walk_forward(prices, test_days)
    # signed variance (actual - predicted) per model, for the table
    for name in MODELS:
        wf[f"var_{name}"] = (wf["actual"] - wf[name]).round(2)
    metrics = error_metrics(wf)
    fwd = forward_forecast(prices, horizon)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "last_close": round(float(prices.iloc[-1]), 2),
        "last_date": prices.index[-1].strftime("%Y-%m-%d"),
        "walk_forward": json.loads(wf.assign(date=wf["date"].astype(str)).to_json(orient="records")),
        "error_metrics": json.loads(metrics.to_json(orient="records")),
        "forward": json.loads(fwd.assign(date=fwd["date"].astype(str)).to_json(orient="records")),
        "models": list(MODELS),
    }
    (PRED_DIR / f"{ticker}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== {ticker} model accuracy (walk-forward, {len(wf)} days, 1-step) ===")
    print(metrics.to_string(index=False))
    print(f"\n=== Forward {horizon}-day forecast (no actuals exist yet) ===")
    print(fwd.head(10).to_string(index=False))
    _plot(ticker, prices, wf, fwd)
    return payload


def _plot(ticker, prices, wf, fwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    hist = prices.tail(120)
    ax.plot(hist.index, hist.values, color="black", lw=2, label="Actual")
    for name in MODELS:
        ax.plot(wf["date"], wf[name], lw=1, alpha=0.8, label=f"{name} (1-step)")
    for name in MODELS:
        ax.plot(fwd["date"], fwd[name], lw=1, ls="--", alpha=0.7)
    ax.plot(fwd["date"], fwd["ensemble_mean"], color="red", lw=2.5, ls="--", label="Ensemble (forward)")
    ax.axvline(prices.index[-1], color="gray", ls=":", alpha=0.7)
    ax.set_title(f"{ticker}: actual vs model predictions (left of line = backtest, right = forecast)")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"predict_{ticker}.png", dpi=110)
    plt.close(fig)
    print(f"Saved chart: predict_{ticker}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--test-days", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=21)
    args = ap.parse_args()
    run(args.ticker, test_days=args.test_days, horizon=args.horizon)
