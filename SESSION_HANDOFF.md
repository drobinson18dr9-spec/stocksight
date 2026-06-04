# StockSight, Session Handoff

Read this first in any new session. It is the complete state of the StockSight
project so nothing from the build sessions is lost. (No em dashes anywhere, per
operator rule.)

## What StockSight is

A quantitative multi-asset stock SELECTION + alerting system. It screens a large
US equity universe (plus crypto and bond sleeves), ranks names on risk-adjusted
and technical signals, builds a portfolio, runs multi-model price forecasts,
scores news sentiment, and posts summaries to Slack/SMS on a schedule. Runs
itself on GitHub Actions (cloud) whether the user's computer is on or off.

Honest bottom line (do not oversell): the math is verified correct (39 self-tests
gate every run), but out-of-sample backtests show it does NOT reliably beat SPY.
It is a disciplined, risk-managed, transparent selector, not a guaranteed
money-maker. Say this plainly; the user values honesty over hype.

## Where everything lives

- Local project root: `C:\Users\drobi\stocksight`
- GitHub repo (private): https://github.com/drobinson18dr9-spec/stocksight  (default branch `master`)
- Live dashboard (GitHub Pages): https://drobinson18dr9-spec.github.io/stocksight/
- Forecasts explorer: https://drobinson18dr9-spec.github.io/stocksight/forecasts.html
- Policy pages (for A2P): /privacy.html, /terms.html, /optin.html
- Python env: Anaconda `py313` (`C:\Users\drobi\anaconda3\envs\py313\python.exe`)
- GitHub CLI: `C:\Program Files\GitHub CLI\gh.exe`

## Source modules (in `src/`)

- `scorecard.py` — core: universe load (CSV xlsx + live Alpaca merge, ~13.5k),
  daily-bar pull + cache, `compute_metrics` (Sharpe/Sortino/12-1 momentum/52w-high
  /drawdown/fast 1mo,3mo/recent-IPO branch), robust z-score `score`, investability
  filter, `build_portfolio` (HRP allocator + news veto), `build_sms`, `run_multi`
  (Core + Affordable strategies, `--cadence` label), live risk-free rate from T-bill.
- `metrics.py` — CVaR/ES, Calmar, Omega, Ulcer, Jensen alpha, beta, info ratio,
  Student-t VaR; crypto-aware (365 vs 252 annualization).
- `patterns.py` — golden/death cross, cup-and-handle (low-trust heuristic),
  Supertrend, 52-week-high helper.
- `crypto_sleeve.py` — crypto in the math: 365-day metrics, Student-t VaR, HRP,
  90/10 split + 10% combined-crypto cap (Johansson & Boyd).
- `regime.py` — term-spread recession prob (Estrella-Mishkin), HY credit spread,
  MOVE, bond ETFs, risk-on/off.
- `portfolio_construct.py` — EqualWeight/InverseVol/MaxSharpe/HRP/NCO comparison +
  volatility-target scalar.
- `validation.py` — PBO via CSCV, Hansen SPA test.
- `quant_alpha.py` — XGBoost cross-sectional ranker, embargoed walk-forward, PSR/DSR.
- `backtest.py` — top vs bottom decile vs SPY, out-of-sample.
- `predict.py` — per-ticker 6-model forecasts (RW/ARIMA/ETS/Theta/Ridge/AR-GARCH),
  walk-forward actual-vs-predicted-vs-variance + forward forecast.
- `build_explorer.py` — precompute per-ticker forecast JSON (shardable for the
  full-universe parallel run); yfinance fallback for OTC/ADRs and recent IPOs.
- `dashboard.py` — builds index.html + forecasts.html (interactive Plotly, client-side
  loads per-ticker JSON; mobile-responsive; Build-portfolio tab).
- `sentiment.py` — multi-source per-ticker news sentiment (Yahoo RSS + Finnhub +
  Seeking Alpha), per-source scores, source-agreement, 3-day sentiment momentum,
  event-risk veto.
- `sources.py` — multi-source price cross-check (Alpaca/FMP/Finnhub/Tiingo) + Finnhub analyst.
- `coinbase.py` — read-only crypto prices (JWT ES256). `robinhood.py` — read-only crypto (Ed25519).
- `notify.py` — Twilio (primary, polls delivery; DISABLE_TWILIO env skips it) ->
  email-to-SMS -> Telegram -> Slack fallback chain.
- `policy_pages.py` — privacy/terms/optin HTML for A2P.
- `verify.py` — 39 math cross-checks; gates every cloud run.
- `macro.py` — live macro (yields, VIX, regime) and `risk_free_rate()` (13wk T-bill).
- `research/stocksight_full.py` — the original Colab notebook (on-demand, not wired).

## Workflows (`.github/workflows/`)

- `daily-scorecard.yml` — three cron cadences: Daily (Mon-Fri 15:00 UTC), Weekly
  (Mon 15:30), Monthly (1st 16:00). Maps `github.event.schedule` to a `--cadence`
  label; runs verify.py gate, then `scorecard.py --multi`, posts to Slack. Twilio
  skipped via `DISABLE_TWILIO=1` until A2P clears. Also builds explorer + dashboard
  and deploys Pages.
- `deploy.yml` — rebuild dashboard + deploy Pages, NO notifications (use for UI deploys).
- `forecast-ticker.yml` — on-demand single-ticker forecast (workflow_dispatch input).
- `forecast-all.yml` — sharded parallel precompute of the whole modelable universe.

## Data sources + secrets

Secrets live in `.env` (local, gitignored) AND as GitHub Actions repo secrets.
NEVER commit secret values. Keys configured: ALPACA_API_KEY/SECRET, FINNHUB_API_KEY,
TIINGO_API_KEY, ALPHAVANTAGE_API_KEY, FRED_API_KEY, FMP_API_KEY (free tier blocks
quote/DCF/analyst), TWILIO_SID/TOKEN/FROM, SMS_TO, SLACK_WEBHOOK_URL,
ROBINHOOD_API_KEY/PRIVATE_KEY/PUBLIC_KEY, COINBASE_API_KEY_NAME/PRIVATE_KEY.
Free, no-key: Yahoo (yfinance/RSS), SEC EDGAR, GDELT, FRED CSV, Seeking Alpha RSS.

## Alerting status

- Slack is the WORKING channel (free, push to phone). Channel #stocksight.
- Twilio SMS is blocked by A2P 10DLC. History: rejected for CTA, then for P2P.
  Root cause: describing it as "personal/sole recipient" reads as person-to-person.
  Fix in progress: reframe as an A2P notification APP with opt-in subscribers
  (privacy/terms/optin pages updated; campaign description must drop personal
  language; use case "Low Volume Mixed"; declare stock + task/webhook message types).
  If it rejects again, stop and rely on Slack/Telegram.

## What was researched + audited (multi-agent)

- 2x deep-research workflows (cited): crypto/bond/metrics/construction/sources;
  technical signals (added 52-week-high George-Hwang, Supertrend; rejected
  head-and-shoulders/candlesticks as failing OOS) + free news sources.
- Audit workflow: found + fixed 24 real bugs (drawdown on winsorized returns,
  CAGR period count, CVaR ties, regime unit mismatch, lookahead in construction
  backtest, DSR dispersion, etc.).
- Expert panel: 9 fixes (weekly->daily cadence with fast signals, recent-IPO
  inclusion, mobile/desktop CSS, $1000 realism).

## Known limitations (be honest about these)

- Survivorship bias: universe is currently-listed names; inflates backtest and the
  GOOD-minus-BAD spread. A point-in-time universe with delisted returns is needed
  to remove it; not done.
- Does not beat SPY out-of-sample (Deflated Sharpe < 0.95). Slight positive but
  noise-dominated at small size; $1000 over a month is not a reliable "hit."
- FMP free tier blocks the valuable endpoints.
- Recent IPOs (<120 bars) have limited/none forecast history.
- A2P SMS not yet approved (P2P classification problem).

## How to run

- Local: `cp .env.example .env` (fill keys); `pip install -r requirements.txt`;
  `python src/verify.py` (math gate); `python src/scorecard.py --multi --no-notify`
  (fast test: add `--universe-limit 300`).
- Deploy UI only (no texts): trigger `deploy.yml`.
- Any ticker forecast locally: `python src/predict.py --ticker XYZ`.
- Push uses gh with a Personal Access Token (user supplies; revoke after).

## Operating rules with this user

- No em dashes or en dashes anywhere (chat, code, pages). Hard rule.
- Be brutally honest about performance; never claim it beats the market.
- Do not burn Twilio: keep DISABLE_TWILIO=1 until A2P approves; deploy via deploy.yml.
- The user opts into heavy multi-agent work ("use agents", "ultracode"); use Workflow for audits/research.

## Open / next

- Resubmit A2P with the mixed-use, non-P2P framing (one more try, then stop).
- Optional: split cadence (daily alerts + weekly rebalance) if daily feels samey.
- Optional: wire crypto prices + analyst consensus into the dashboard/Slack.
- Optional: point-in-time universe to remove survivorship bias (hard, needs paid data).
