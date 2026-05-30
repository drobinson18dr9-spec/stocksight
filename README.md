# StockSight

Two tracks:

- **`src/scorecard.py`** — the daily engine. Screens the ticker universe on
  trailing risk-adjusted performance, labels each name GOOD / NEUTRAL / BAD,
  and builds a max-Sharpe portfolio from the top names using a Ledoit-Wolf
  shrinkage covariance. Writes a CSV + Markdown brief and texts you a summary.
  Light dependencies, runs in minutes, safe to schedule.
- **`research/stocksight_full.py`** — the full 12-model research notebook
  (ARIMA, GARCH, LSTM, Prophet, ensemble, Monte Carlo, etc.). Heavy; run by
  hand when you want the deep bake-off. Not meant for the cron.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in ALPACA_API_KEY / ALPACA_API_SECRET
python src/scorecard.py --universe-limit 300   # quick test
python src/scorecard.py                         # full universe
```

Outputs land in `reports/`. The screen verdict and portfolio also print to the
console and go out as a text if a delivery method is configured (see
`.env.example` and `src/notify.py`).

## Schedule in the cloud (fires whether your computer is on or off)

1. Push this repo to GitHub.
2. Repo Settings -> Secrets and variables -> Actions -> add `ALPACA_API_KEY`,
   `ALPACA_API_SECRET`, and the secrets for your chosen delivery method.
3. The workflow in `.github/workflows/daily-scorecard.yml` runs Mon-Fri at
   13:00 UTC. Trigger it manually from the Actions tab to test.

## Methodology

Per ticker over the trailing 252 trading days: annualized return, annualized
volatility, Sharpe (rf = 4%), Sortino, 12-1 momentum, max drawdown, 200-day
trend, dollar-volume liquidity. Composite is a cross-sectional z-score blend
(Sharpe 30%, Sortino 25%, momentum 25%, drawdown 10%, trend 10%); top/bottom
30% become GOOD/BAD. Portfolio = max-Sharpe optimization on the screened names
with a shrinkage covariance for stability, long-only, capped single weight.

Analytical output, not financial advice. Past performance does not predict
future results.
