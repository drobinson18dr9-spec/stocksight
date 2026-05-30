"""
Builds the StockSight web dashboard: a single self-contained interactive HTML
page (Plotly charts you can hover, zoom, and toggle). Published to GitHub Pages
by the daily workflow; the alert text links to it.

Tabs: Picks table | Risk vs Return | Picks vs SPY | Portfolio | Score spread |
Backtest (if reports/backtest_equity.csv exists).

Usage: python src/dashboard.py [--universe-limit N]
Output: reports/site/index.html
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import scorecard as sc
import fundamentals as fund
import macro as macro_mod

SITE = Path(__file__).resolve().parents[1] / "reports" / "site"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
COLORS = {"GOOD": "#1a9850", "NEUTRAL": "#9e9e9e", "BAD": "#d73027"}


def _div(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                       config={"responsive": True, "displaylogo": False})


def build(universe_limit=None):
    SITE.mkdir(parents=True, exist_ok=True)
    bars = sc.get_bars(universe_limit=universe_limit)
    metrics = sc.compute_metrics(bars)
    investable = sc.filter_investable(metrics)
    scored = sc.score(investable)
    portfolio = sc.build_portfolio(scored, bars, top_n=12, max_weight=0.25)
    asof = bars["timestamp"].max().strftime("%Y-%m-%d")
    funds = fund.get(portfolio["ticker"].tolist())
    mac = macro_mod.get()

    good = scored[scored["verdict"] == "GOOD"]
    bad = scored[scored["verdict"] == "BAD"]
    spy = scored[scored["ticker"] == sc.BENCHMARK]
    spy_ret = spy["cagr"].iloc[0] * 100 if len(spy) else float("nan")

    # ── Chart 1: risk vs return ─────────────────────────────────────────
    f1 = go.Figure()
    for v in ["NEUTRAL", "BAD", "GOOD"]:
        d = scored[scored["verdict"] == v]
        f1.add_scatter(
            x=d["ann_vol"] * 100, y=d["cagr"] * 100, mode="markers", name=v,
            marker=dict(size=6, color=COLORS[v], opacity=0.55),
            text=d["ticker"], customdata=np.stack([d["sharpe"], d["momentum_12_1"]*100], axis=-1),
            hovertemplate="<b>%{text}</b><br>vol %{x:.0f}%<br>return %{y:.0f}%<br>"
                          "Sharpe %{customdata[0]:.2f}<br>mom %{customdata[1]:.0f}%<extra></extra>")
    if len(spy):
        f1.add_scatter(x=spy["ann_vol"]*100, y=spy["cagr"]*100, mode="markers", name="SPY",
                       marker=dict(size=18, color="black", symbol="star"), text=["SPY"],
                       hovertemplate="<b>SPY</b><br>vol %{x:.0f}%<br>return %{y:.0f}%<extra></extra>")
    f1.update_layout(title="Risk vs return (trailing 1y), hover any dot",
                     xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
                     yaxis_range=[-100, min(float(scored["cagr"].quantile(0.99)*100), 400)],
                     template="plotly_white", height=560)

    # ── Chart 2: picks vs SPY ───────────────────────────────────────────
    f2 = go.Figure()
    picks = portfolio.head(6)["ticker"].tolist() or scored.head(6)["ticker"].tolist()
    for t in picks + [sc.BENCHMARK]:
        d = bars[bars["ticker"] == t].sort_values("timestamp").tail(252)
        if len(d) < 50:
            continue
        norm = d["close"] / d["close"].iloc[0] * 100
        f2.add_scatter(x=d["timestamp"], y=norm, mode="lines", name=t,
                       line=dict(width=3 if t == sc.BENCHMARK else 1.6,
                                 dash="dash" if t == sc.BENCHMARK else "solid"))
    f2.update_layout(title="Top picks vs SPY, growth of 100 (click legend to toggle)",
                     yaxis_title="Growth of 100", template="plotly_white", height=560)

    # ── Chart 3: portfolio weights ──────────────────────────────────────
    p = portfolio[portfolio["weight"] > 0.005].sort_values("weight")
    f3 = go.Figure(go.Bar(x=p["weight"]*100, y=p["ticker"], orientation="h",
                          marker_color="#2c7fb8",
                          text=[f"{w*100:.1f}%" for w in p["weight"]], textposition="outside"))
    f3.update_layout(title="Optimized max-Sharpe portfolio weights",
                     xaxis_title="Weight (%)", template="plotly_white", height=560)

    # ── Chart 4: score distribution ─────────────────────────────────────
    f4 = go.Figure()
    for v in ["BAD", "NEUTRAL", "GOOD"]:
        f4.add_histogram(x=scored.loc[scored["verdict"] == v, "composite"], name=v,
                         marker_color=COLORS[v], opacity=0.75, nbinsx=60)
    f4.update_layout(title="Composite score spread across investable names",
                     barmode="overlay", xaxis_title="Composite score",
                     yaxis_title="Number of stocks", template="plotly_white", height=560)

    # ── Chart 5: backtest (optional) ────────────────────────────────────
    bt_div = ""
    bt_csv = REPORTS / "backtest_equity.csv"
    if not bt_csv.exists():
        bt_csv = Path(__file__).resolve().parents[1] / "assets" / "backtest_equity.csv"
    if bt_csv.exists():
        eq = pd.read_csv(bt_csv, index_col=0, parse_dates=True)
        f5 = go.Figure()
        f5.add_scatter(x=eq.index, y=eq["GOOD"], name="GOOD top decile", line=dict(color=COLORS["GOOD"], width=2.5))
        f5.add_scatter(x=eq.index, y=eq["SPY"], name="SPY", line=dict(color="black", width=2.5, dash="dash"))
        f5.add_scatter(x=eq.index, y=eq["BAD"], name="BAD bottom decile", line=dict(color=COLORS["BAD"], width=2.5))
        f5.update_layout(title="Out-of-sample backtest: $100 grown, top vs bottom decile vs SPY",
                         yaxis_type="log", yaxis_title="Growth of $100 (log)",
                         template="plotly_white", height=560)
        bt_div = f"<div id='backtest' class='chart'>{_div(f5)}</div>"

    # ── Picks table ─────────────────────────────────────────────────────
    rows = ""
    for i, (_, r) in enumerate(portfolio.iterrows(), 1):
        ns = r.get("news_sentiment", 0.0)
        ns = 0.0 if pd.isna(ns) else ns
        tone = "#1a9850" if ns > 0.05 else ("#d73027" if ns < -0.05 else "#666")
        rows += (f"<tr><td>{i}</td><td><b>{r['ticker']}</b></td><td>{r['weight']*100:.1f}%</td>"
                 f"<td>${r['current_price']:.2f}</td><td>{r['sharpe']:.2f}</td>"
                 f"<td>{r['momentum_12_1']*100:.0f}%</td>"
                 f"<td style='color:{tone}'>{ns:+.2f}</td></tr>")
    table = f"""<table>
      <tr><th>#</th><th>Ticker</th><th>Weight</th><th>Price</th><th>Sharpe</th><th>Momentum (1yr)</th><th>News tone</th></tr>
      {rows}</table>"""

    # Fundamentals table (yfinance: valuation, quality, analyst targets)
    fmap = funds.set_index("ticker") if len(funds) else None

    def _fv(t, col, pct=False):
        if fmap is None or t not in fmap.index:
            return "-"
        v = fmap.loc[t].get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if pct:
            return f"{v*100:.1f}%"
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

    frows = ""
    for _, r in portfolio.iterrows():
        t = r["ticker"]
        rating = fmap.loc[t].get("recommendationKey") if (fmap is not None and t in fmap.index) else None
        frows += (f"<tr><td><b>{t}</b></td><td>{_fv(t,'trailingPE')}</td>"
                  f"<td>{_fv(t,'profitMargins',pct=True)}</td><td>{_fv(t,'returnOnEquity',pct=True)}</td>"
                  f"<td>{_fv(t,'revenueGrowth',pct=True)}</td><td>{_fv(t,'analyst_upside',pct=True)}</td>"
                  f"<td>{rating or '-'}</td><td>{_fv(t,'quality_score')}</td></tr>")
    fund_table = (f"<h3 style='margin:6px 10px'>Fundamentals & analyst view (free, yfinance)</h3>"
                  f"<table><tr><th>Ticker</th><th>P/E</th><th>Margin</th><th>ROE</th>"
                  f"<th>Rev growth</th><th>Analyst upside</th><th>Rating</th><th>Quality</th></tr>{frows}</table>")

    # Macro banner
    def _mv(k, suffix=""):
        v = mac.get(k)
        return f"{v}{suffix}" if v is not None else "n/a"
    macro_banner = (f"Macro: 10y {_mv('treasury_10y','%')} | 3m {_mv('tbill_3m','%')} | "
                    f"10y-3m curve {_mv('yield_curve_spread')} | VIX {_mv('vix')} | "
                    f"regime <b>{mac.get('regime','?')}</b> ({mac.get('regime_reason','')})")

    # ── Assemble page ───────────────────────────────────────────────────
    tabs = [("picks", "Portfolio picks"), ("fundamentals", "Fundamentals"),
            ("risk", "Risk vs Return"), ("vsspy", "Picks vs SPY"),
            ("weights", "Weights"), ("spread", "Score spread")]
    if bt_div:
        tabs.append(("backtest", "Backtest"))
    tab_btns = "".join(f"<button onclick=\"show('{tid}')\">{label}</button>" for tid, label in tabs)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight, {asof}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f7f8fa;color:#1a1a2e}}
 header{{background:#0f1b3d;color:#fff;padding:18px 22px}}
 header h1{{margin:0;font-size:22px}} header p{{margin:6px 0 0;opacity:.85;font-size:14px}}
 .summary{{display:flex;gap:14px;flex-wrap:wrap;padding:16px 22px}}
 .card{{background:#fff;border-radius:10px;padding:12px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 .card .n{{font-size:24px;font-weight:700}} .good{{color:#1a9850}} .bad{{color:#d73027}}
 .tabs{{padding:0 22px;display:flex;gap:8px;flex-wrap:wrap}}
 .tabs button{{border:0;background:#e6e9f0;padding:9px 14px;border-radius:8px;cursor:pointer;font-size:14px}}
 .tabs button:hover{{background:#d3d8e6}}
 .chart{{display:none;margin:14px 22px;background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 #picks{{display:block}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #eee}} th{{background:#f0f2f7}}
 footer{{padding:18px 22px;color:#777;font-size:12px}}
</style></head><body>
<header>
  <h1>StockSight Daily Scorecard</h1>
  <p>{asof} · {len(scored)} investable names screened · SPY trailing 1y {spy_ret:.0f}%</p>
  <p style="font-size:13px">{macro_banner}</p>
</header>
<div class="summary">
  <div class="card"><div class="n good">{len(good)}</div>GOOD</div>
  <div class="card"><div class="n">{len(scored)-len(good)-len(bad)}</div>NEUTRAL</div>
  <div class="card"><div class="n bad">{len(bad)}</div>BAD</div>
  <div class="card"><div class="n">{len(portfolio)}</div>in portfolio</div>
</div>
<div class="tabs">{tab_btns}<a href="forecasts.html" style="text-decoration:none"><button style="background:#0f1b3d;color:#fff">Model Forecasts &rarr;</button></a></div>
<div id="picks" class="chart"><h3 style="margin:6px 10px">Optimized portfolio</h3>{table}</div>
<div id="fundamentals" class="chart">{fund_table}</div>
<div id="risk" class="chart">{_div(f1, first=True)}</div>
<div id="vsspy" class="chart">{_div(f2)}</div>
<div id="weights" class="chart">{_div(f3)}</div>
<div id="spread" class="chart">{_div(f4)}</div>
{bt_div}
<footer>Built by StockSight. Methodology: trailing 252d Sharpe (Lo 2002 t-stat gate),
Sortino, 12-1 momentum, robust z-scoring, max-Sharpe with Ledoit-Wolf shrinkage.
For research and education. Do your own diligence.</footer>
<script>
 function show(id){{
   document.querySelectorAll('.chart').forEach(e=>e.style.display='none');
   var el=document.getElementById(id); el.style.display='block';
   window.dispatchEvent(new Event('resize'));
 }}
 show('picks');
</script>
</body></html>"""

    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote dashboard: {out}")
    build_forecasts()


def build_forecasts():
    """Generate forecasts.html from precomputed assets/predict/*.json:
    a ticker picker, per-ticker actual-vs-model chart, error metrics,
    forward forecast, and the actual/predicted/variance table."""
    pred_dir = Path(__file__).resolve().parents[1] / "assets" / "predict"
    import json
    if not pred_dir.exists():
        return
    # Serve every precomputed ticker file present (robust to parallel merges).
    tickers = sorted(f.stem for f in pred_dir.glob("*.json") if f.stem != "index")
    if not tickers:
        return
    SITE.mkdir(parents=True, exist_ok=True)

    import shutil
    # Serve each ticker's compact data file from our own site; the chart is
    # drawn client-side on selection (so the page stays small and can offer
    # the whole universe without pre-rendering thousands of charts).
    site_pred = SITE / "predict"
    site_pred.mkdir(parents=True, exist_ok=True)
    served = []
    for t in tickers:
        f = pred_dir / f"{t}.json"
        if f.exists():
            shutil.copy(f, site_pred / f"{t}.json")
            served.append(t)
    (site_pred / "index.json").write_text(json.dumps({"tickers": served}), encoding="utf-8")

    all_syms = sorted(set(sc.load_universe()))
    datalist = "".join(f"<option value='{t}'></option>" for t in all_syms)
    pre_js = "[" + ",".join(f'"{t}"' for t in served) + "]"
    first = served[0] if served else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockSight Forecasts</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f7f8fa;color:#1a1a2e}}
 header{{background:#0f1b3d;color:#fff;padding:16px 22px}} header a{{color:#9ec5ff}}
 .bar{{padding:14px 22px}} input{{font-size:16px;padding:8px 12px;border-radius:8px;border:1px solid #ccc;width:260px}}
 .card{{margin:0 22px 22px;background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 16px}}
 th,td{{padding:6px 9px;text-align:left;border-bottom:1px solid #eee}} th{{background:#f0f2f7}}
 h4{{margin:14px 6px 4px}} #msg{{margin:0 22px;color:#a33}}
</style></head><body>
<header><h1>StockSight Forecasts explorer</h1>
<p><a href="index.html">&larr; back to scorecard</a> &middot; type any of {len(all_syms)} tickers. Dotted lines and the forward table are future predictions (no actuals yet).</p></header>
<div class="bar"><label>Ticker: </label>
  <input id="picker" list="alltickers" placeholder="type a symbol, e.g. AAPL" autocomplete="off" value="{first}">
  <datalist id="alltickers">{datalist}</datalist>
  <span style="margin-left:10px;color:#666">{len(served)} loaded; more added each run</span>
</div>
<div id="msg"></div>
<div class="card" id="chartcard"><div id="chart" style="height:520px"></div></div>
<div class="card" id="tables"></div>
<script>
 var PRE = {pre_js};
 function fmt(v){{return (v==null||isNaN(v))?'-':Number(v).toFixed(2);}}
 function render(d){{
   document.getElementById('msg').textContent='';
   var M=d.models, wf=d.walk_forward, fwd=d.forward, traces=[];
   traces.push({{x:wf.map(r=>r.date),y:wf.map(r=>r.actual),name:'ACTUAL',mode:'lines',line:{{color:'black',width:3}}}});
   M.forEach(m=>{{
     traces.push({{x:wf.map(r=>r.date),y:wf.map(r=>r[m]),name:m,mode:'lines',line:{{width:1.3}}}});
     traces.push({{x:fwd.map(r=>r.date),y:fwd.map(r=>r[m]),name:m+' fwd',mode:'lines',line:{{width:1,dash:'dot'}},showlegend:false}});
   }});
   traces.push({{x:fwd.map(r=>r.date),y:fwd.map(r=>r.ensemble_mean),name:'Ensemble (forecast)',mode:'lines',line:{{color:'red',width:2.5,dash:'dash'}}}});
   Plotly.newPlot('chart',traces,{{title:d.ticker+': actual vs model predictions (dotted = future forecast)',template:'plotly_white',yaxis:{{title:'Price ($)'}},margin:{{t:40}}}},{{responsive:true,displaylogo:false}});
   var em='<h4>Model accuracy (walk-forward backtest)</h4><table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Dir. acc</th></tr>';
   d.error_metrics.forEach(r=>{{em+='<tr><td>'+r.model+'</td><td>'+r.MAE+'</td><td>'+r.RMSE+'</td><td>'+r['MAPE_%']+'%</td><td>'+r['DirAcc_%']+'%</td></tr>';}});
   em+='</table>';
   var fcols=M.concat(['ensemble_mean']);
   var ft='<h4>Forward forecast (future dates, no actuals exist)</h4><table><tr><th>Date</th>'+fcols.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
   fwd.forEach(r=>{{ft+='<tr><td>'+r.date+'</td>'+fcols.map(c=>'<td>'+fmt(r[c])+'</td>').join('')+'</tr>';}});
   ft+='</table>';
   var rec=wf.slice(-8), vh=M.map(m=>'<th>'+m+'</th><th>&Delta;'+m+'</th>').join('');
   var vt='<h4>Actual vs predicted vs variance (&Delta; = actual - predicted)</h4><div style="overflow-x:auto"><table><tr><th>Date</th><th>Actual</th>'+vh+'</tr>';
   rec.forEach(r=>{{vt+='<tr><td>'+r.date+'</td><td><b>'+fmt(r.actual)+'</b></td>'+M.map(m=>{{var v=r['var_'+m];return '<td>'+fmt(r[m])+'</td><td style="color:#888">'+(v==null?'-':(v>=0?'+':'')+Number(v).toFixed(2))+'</td>';}}).join('')+'</tr>';}});
   vt+='</table></div>';
   document.getElementById('tables').innerHTML=em+ft+vt;
 }}
 function show(t){{
   t=(t||'').toUpperCase().trim();
   if(PRE.indexOf(t)>=0){{
     fetch('predict/'+t+'.json').then(r=>r.json()).then(render)
       .catch(e=>{{document.getElementById('msg').textContent='Could not load '+t;}});
   }} else {{
     Plotly.purge('chart'); document.getElementById('tables').innerHTML='';
     document.getElementById('msg').textContent = t ? ('No forecast loaded for '+t+' yet. The set expands automatically each run.') : '';
   }}
 }}
 document.getElementById('picker').addEventListener('change',function(){{show(this.value);}});
 show('{first}');
</script></body></html>"""
    (SITE / "forecasts.html").write_text(page, encoding="utf-8")
    print(f"Wrote forecasts explorer (client-side): {len(served)} tickers served")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-limit", type=int, default=None)
    args = ap.parse_args()
    build(universe_limit=args.universe_limit)
