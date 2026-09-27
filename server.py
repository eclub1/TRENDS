"""
ETH Trend AI — FastAPI web server.
Serves a live dashboard at / and a JSON API at /api/analysis
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.fetcher import fetch_ohlc, fetch_market_data
from src.indicators import add_all_indicators
from src.trend_detector import analyze

app = FastAPI(title="ETH Trend AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def run_analysis(days: int = 90) -> dict:
    ohlc_df = fetch_ohlc(days=days)
    market = fetch_market_data()
    df_ind = add_all_indicators(ohlc_df)
    result = analyze(df_ind)

    return {
        "market": market,
        "trend": result.trend,
        "reversal_risk": result.reversal_risk,
        "confidence": round(result.confidence, 1),
        "score": round(result.total_score, 2),
        "recommendation": result.recommendation,
        "signals": [
            {
                "name": s.name,
                "value": round(s.value, 2),
                "score": round(s.score, 2),
                "interpretation": s.interpretation,
            }
            for s in result.signals
        ],
        "reversal_warnings": result.reversal_reasons,
    }


@app.get("/api/analysis")
async def api_analysis(days: int = 90):
    try:
        data = run_analysis(days)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse(content=HTML_TEMPLATE)


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ETH Trend AI</title>
  <style>
    :root {
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #7d8590;
      --green: #3fb950;
      --red: #f85149;
      --yellow: #d29922;
      --blue: #58a6ff;
      --purple: #bc8cff;
      --cyan: #39d353;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
      padding: 24px 16px;
    }
    .container { max-width: 960px; margin: 0 auto; }

    /* Header */
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .logo { font-size: 1.5rem; font-weight: 700; color: var(--blue); }
    .logo span { color: var(--muted); font-weight: 400; font-size: 0.9rem; margin-left: 8px; }
    .refresh-info { font-size: 0.8rem; color: var(--muted); }
    #last-updated { color: var(--text); }

    /* Price card */
    .price-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }
    .price-main { display: flex; align-items: baseline; gap: 12px; }
    .price-value { font-size: 2.2rem; font-weight: 700; color: var(--text); }
    .badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .badge.up { background: rgba(63,185,80,0.15); color: var(--green); }
    .badge.down { background: rgba(248,81,73,0.15); color: var(--red); }
    .badge.flat { background: rgba(210,153,34,0.15); color: var(--yellow); }
    .price-stats { display: flex; gap: 24px; flex-wrap: wrap; }
    .stat { display: flex; flex-direction: column; gap: 2px; }
    .stat-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-value { font-size: 0.95rem; font-weight: 600; }

    /* Trend summary row */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    @media (max-width: 600px) { .summary-grid { grid-template-columns: 1fr; } }
    .summary-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      text-align: center;
    }
    .summary-card .label {
      font-size: 0.72rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .summary-card .value { font-size: 1.4rem; font-weight: 700; }
    .trend-STRONG_UPTREND   { color: #3fb950; }
    .trend-UPTREND          { color: #56d364; }
    .trend-NEUTRAL          { color: #d29922; }
    .trend-DOWNTREND        { color: #f08080; }
    .trend-STRONG_DOWNTREND { color: #f85149; }
    .risk-LOW      { color: var(--green); }
    .risk-MEDIUM   { color: var(--yellow); }
    .risk-HIGH     { color: #f08080; }
    .risk-CRITICAL { color: var(--red); }

    /* Score bar */
    .score-bar-wrap {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 20px;
    }
    .score-bar-label {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .score-track {
      background: #21262d;
      border-radius: 8px;
      height: 10px;
      position: relative;
      overflow: hidden;
    }
    .score-fill {
      height: 100%;
      border-radius: 8px;
      transition: width 0.6s ease;
    }
    .score-val {
      margin-top: 6px;
      font-size: 0.85rem;
      font-weight: 600;
    }

    /* Signals table */
    .section-title {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 10px;
    }
    .signals-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 20px;
    }
    table { width: 100%; border-collapse: collapse; }
    thead th {
      padding: 10px 16px;
      text-align: left;
      font-size: 0.72rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
      background: #0d1117;
    }
    tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
    tbody tr:last-child { border-bottom: none; }
    tbody tr:hover { background: rgba(255,255,255,0.03); }
    tbody td { padding: 12px 16px; font-size: 0.875rem; vertical-align: middle; }
    .sig-name { font-weight: 600; }
    .sig-score-pos { color: var(--green); font-weight: 700; }
    .sig-score-neg { color: var(--red); font-weight: 700; }
    .sig-score-neu { color: var(--yellow); font-weight: 700; }
    .sig-interp { color: var(--muted); font-size: 0.82rem; }

    /* Warnings */
    .warnings-card {
      background: rgba(248,81,73,0.07);
      border: 1px solid rgba(248,81,73,0.3);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 20px;
    }
    .warning-item {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 0.875rem;
      color: #f08080;
      margin-bottom: 6px;
    }
    .warning-item:last-child { margin-bottom: 0; }
    .no-warnings { color: var(--green); font-size: 0.875rem; }

    /* Recommendation */
    .rec-card {
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 20px;
      border-left: 4px solid var(--blue);
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 4px solid var(--blue);
    }
    .rec-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .rec-text { font-size: 1rem; font-weight: 500; line-height: 1.5; }

    /* Footer */
    .footer { text-align: center; color: var(--muted); font-size: 0.78rem; margin-top: 8px; }

    /* Loading */
    .loading {
      display: flex; align-items: center; justify-content: center;
      height: 60vh; flex-direction: column; gap: 16px;
      color: var(--muted);
    }
    .spinner {
      width: 36px; height: 36px;
      border: 3px solid var(--border);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hidden { display: none; }

    /* Pulse dot */
    .live-dot {
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      margin-right: 6px;
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">
        ⚡ ETH Trend AI
        <span>Real-time Ethereum trend &amp; reversal detector</span>
      </div>
      <div class="refresh-info">
        <span class="live-dot"></span>
        Auto-refresh every 5 min &nbsp;·&nbsp; Last: <span id="last-updated">—</span>
      </div>
    </div>

    <div id="loading" class="loading">
      <div class="spinner"></div>
      <div>Fetching ETH market data…</div>
    </div>

    <div id="content" class="hidden">
      <!-- Price card -->
      <div class="price-card">
        <div class="price-main">
          <div class="price-value" id="price">—</div>
          <div id="badge-24h" class="badge flat">—</div>
          <div id="badge-7d" class="badge flat">7d —</div>
        </div>
        <div class="price-stats">
          <div class="stat">
            <div class="stat-label">24h High</div>
            <div class="stat-value" id="high24h">—</div>
          </div>
          <div class="stat">
            <div class="stat-label">24h Low</div>
            <div class="stat-value" id="low24h">—</div>
          </div>
          <div class="stat">
            <div class="stat-label">Volume 24h</div>
            <div class="stat-value" id="vol24h">—</div>
          </div>
          <div class="stat">
            <div class="stat-label">Market Cap</div>
            <div class="stat-value" id="mcap">—</div>
          </div>
        </div>
      </div>

      <!-- Trend summary -->
      <div class="summary-grid">
        <div class="summary-card">
          <div class="label">Trend</div>
          <div class="value" id="trend-val">—</div>
        </div>
        <div class="summary-card">
          <div class="label">Reversal Risk</div>
          <div class="value" id="risk-val">—</div>
        </div>
        <div class="summary-card">
          <div class="label">Confidence</div>
          <div class="value" id="conf-val" style="color:var(--blue)">—</div>
        </div>
      </div>

      <!-- Score bar -->
      <div class="score-bar-wrap">
        <div class="score-bar-label">Aggregate Signal Score (−10 bearish → +10 bullish)</div>
        <div class="score-track">
          <div class="score-fill" id="score-fill" style="width:50%;background:var(--blue);"></div>
        </div>
        <div class="score-val" id="score-val">—</div>
      </div>

      <!-- Signals -->
      <div class="section-title">Signal Breakdown</div>
      <div class="signals-card">
        <table>
          <thead>
            <tr>
              <th>Indicator</th>
              <th>Value</th>
              <th>Score</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody id="signals-body"></tbody>
        </table>
      </div>

      <!-- Warnings -->
      <div class="section-title">Reversal Warnings</div>
      <div class="warnings-card" id="warnings-card">
        <div id="warnings-body"></div>
      </div>

      <!-- Recommendation -->
      <div class="rec-card">
        <div class="rec-label">🤖 AI Recommendation</div>
        <div class="rec-text" id="rec-text">—</div>
      </div>

      <div class="footer">
        ⚠ For informational purposes only. Not financial advice. Always do your own research.
      </div>
    </div>
  </div>

  <script>
    const fmt = (n, dec=2) => n?.toLocaleString('en-US', {minimumFractionDigits:dec, maximumFractionDigits:dec});
    const fmtB = n => n >= 1e9 ? '$' + (n/1e9).toFixed(2) + 'B' : n >= 1e6 ? '$' + (n/1e6).toFixed(0) + 'M' : '$' + n;

    const TREND_EMOJI = {
      STRONG_UPTREND: '🚀', UPTREND: '📈', NEUTRAL: '➡️',
      DOWNTREND: '📉', STRONG_DOWNTREND: '🔻'
    };

    async function load() {
      try {
        const res = await fetch('/api/analysis');
        const d = await res.json();
        if (d.error) throw new Error(d.error);

        const m = d.market;

        // Price
        document.getElementById('price').textContent = '$' + fmt(m.price);

        const b24 = document.getElementById('badge-24h');
        b24.textContent = (m.change_24h >= 0 ? '+' : '') + fmt(m.change_24h) + '%';
        b24.className = 'badge ' + (m.change_24h > 0 ? 'up' : m.change_24h < 0 ? 'down' : 'flat');

        const b7 = document.getElementById('badge-7d');
        b7.textContent = '7d ' + (m.change_7d >= 0 ? '+' : '') + fmt(m.change_7d) + '%';
        b7.className = 'badge ' + (m.change_7d > 0 ? 'up' : m.change_7d < 0 ? 'down' : 'flat');

        document.getElementById('high24h').textContent = '$' + fmt(m.high_24h, 0);
        document.getElementById('low24h').textContent  = '$' + fmt(m.low_24h, 0);
        document.getElementById('vol24h').textContent  = fmtB(m.volume_24h);
        document.getElementById('mcap').textContent    = fmtB(m.market_cap);

        // Trend summary
        const trendEl = document.getElementById('trend-val');
        trendEl.textContent = (TREND_EMOJI[d.trend] || '') + ' ' + d.trend.replace(/_/g,' ');
        trendEl.className = 'value trend-' + d.trend;

        const riskEl = document.getElementById('risk-val');
        riskEl.textContent = d.reversal_risk;
        riskEl.className = 'value risk-' + d.reversal_risk;

        document.getElementById('conf-val').textContent = d.confidence + '%';

        // Score bar
        const pct = Math.min(100, Math.max(0, (d.score + 10) / 20 * 100));
        const fill = document.getElementById('score-fill');
        fill.style.width = pct + '%';
        fill.style.background = d.score > 0 ? 'var(--green)' : d.score < 0 ? 'var(--red)' : 'var(--yellow)';
        const scoreEl = document.getElementById('score-val');
        scoreEl.textContent = (d.score > 0 ? '+' : '') + d.score;
        scoreEl.style.color = d.score > 0 ? 'var(--green)' : d.score < 0 ? 'var(--red)' : 'var(--yellow)';

        // Signals
        const tbody = document.getElementById('signals-body');
        tbody.innerHTML = d.signals.map(s => {
          const sc = s.score > 0 ? `<span class="sig-score-pos">+${s.score}</span>`
                   : s.score < 0 ? `<span class="sig-score-neg">${s.score}</span>`
                   : `<span class="sig-score-neu">0.0</span>`;
          return `<tr>
            <td class="sig-name">${s.name}</td>
            <td>${s.value}</td>
            <td>${sc}</td>
            <td class="sig-interp">${s.interpretation}</td>
          </tr>`;
        }).join('');

        // Warnings
        const wb = document.getElementById('warnings-body');
        const wc = document.getElementById('warnings-card');
        if (d.reversal_warnings.length === 0) {
          wb.innerHTML = '<div class="no-warnings">✅ No reversal warnings detected.</div>';
          wc.style.background = 'rgba(63,185,80,0.07)';
          wc.style.borderColor = 'rgba(63,185,80,0.3)';
        } else {
          wb.innerHTML = d.reversal_warnings.map(w =>
            `<div class="warning-item"><span>⚠</span><span>${w}</span></div>`
          ).join('');
          wc.style.background = 'rgba(248,81,73,0.07)';
          wc.style.borderColor = 'rgba(248,81,73,0.3)';
        }

        // Recommendation
        document.getElementById('rec-text').textContent = d.recommendation;

        // Timestamp
        document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();

        // Show content
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('content').classList.remove('hidden');

      } catch (err) {
        document.getElementById('loading').innerHTML =
          `<div style="color:var(--red);text-align:center;">
            ⚠ Failed to load data<br><small>${err.message}</small><br><br>
            <button onclick="load()" style="padding:8px 20px;background:var(--blue);color:#fff;border:none;border-radius:6px;cursor:pointer;">Retry</button>
          </div>`;
      }
    }

    load();
    setInterval(load, 5 * 60 * 1000); // refresh every 5 minutes
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
