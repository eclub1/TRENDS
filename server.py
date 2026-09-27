"""
ETH Trend AI — FastAPI web server.
Serves a TradingView-style dashboard at / and JSON API at /api/analysis and /api/candles
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.fetcher import fetch_ohlc, fetch_market_data
from src.indicators import add_all_indicators
from src.trend_detector import analyze

app = FastAPI(title="ETH Trend AI", version="2.0.0")

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

    # Build candle series for the chart
    candles = []
    for _, row in df_ind.iterrows():
        candles.append({
            "time": int(row["timestamp"].timestamp()),
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
            # indicator overlays
            "ema9":   round(float(row["ema_9"]),   2),
            "ema21":  round(float(row["ema_21"]),  2),
            "ema50":  round(float(row["ema_50"]),  2),
            "ema200": round(float(row["ema_200"]), 2),
            "bb_upper": round(float(row["bb_upper"]), 2),
            "bb_mid":   round(float(row["bb_mid"]),   2),
            "bb_lower": round(float(row["bb_lower"]), 2),
            # sub-panel indicators
            "rsi":       round(float(row["rsi"]),       2),
            "macd":      round(float(row["macd"]),      2),
            "macd_sig":  round(float(row["macd_signal"]),2),
            "macd_hist": round(float(row["macd_hist"]), 2),
            "stoch_k":   round(float(row["stoch_k"]),   2),
            "stoch_d":   round(float(row["stoch_d"]),   2),
        })

    return {
        "market": market,
        "candles": candles,
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/analysis")
async def api_analysis(days: int = 90):
    try:
        data = run_analysis(days)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # Use path relative to CWD (works in Docker where CWD = /app)
    base = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
