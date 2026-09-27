"""
ETH Trend AI — FastAPI web server.
Serves a TradingView-style dashboard at / and JSON API at /api/analysis
"""

import os
import traceback
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


import math
import numpy as np
from fastapi.responses import Response
import json


class SafeJSONResponse(Response):
    """JSONResponse that handles numpy int64/float64 and NaN/Inf values."""
    media_type = "application/json"

    def render(self, content) -> bytes:
        return json.dumps(content, cls=_SafeEncoder).encode("utf-8")


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        # Also sanitize plain Python floats
        if isinstance(obj, float):
            return "null" if (math.isnan(obj) or math.isinf(obj)) else super().encode(obj)
        return super().encode(obj)


def _safe(v):
    """Convert NaN/Inf floats and numpy scalars to safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def run_analysis(days: int = 90) -> dict:
    logger.info(f"Fetching OHLC data for {days} days...")
    ohlc_df = fetch_ohlc(days=days)
    logger.info(f"Got {len(ohlc_df)} candles. Fetching market data...")
    market = fetch_market_data()
    logger.info("Computing indicators...")
    df_ind = add_all_indicators(ohlc_df)
    logger.info(f"Running trend analysis on {len(df_ind)} rows...")
    result = analyze(df_ind)
    logger.info(f"Analysis complete: {result.trend}")

    candles = []
    for _, row in df_ind.iterrows():
        candles.append({
            "time":     int(row["timestamp"].timestamp()),
            "open":     _safe(round(float(row["open"]),  2)),
            "high":     _safe(round(float(row["high"]),  2)),
            "low":      _safe(round(float(row["low"]),   2)),
            "close":    _safe(round(float(row["close"]), 2)),
            "ema9":     _safe(round(float(row["ema_9"]),    2)),
            "ema21":    _safe(round(float(row["ema_21"]),   2)),
            "ema50":    _safe(round(float(row["ema_50"]),   2)),
            "ema200":   _safe(round(float(row["ema_200"]),  2)),
            "bb_upper": _safe(round(float(row["bb_upper"]), 2)),
            "bb_mid":   _safe(round(float(row["bb_mid"]),   2)),
            "bb_lower": _safe(round(float(row["bb_lower"]), 2)),
            "rsi":      _safe(round(float(row["rsi"]),       2)),
            "macd":     _safe(round(float(row["macd"]),      2)),
            "macd_sig": _safe(round(float(row["macd_signal"]), 2)),
            "macd_hist":_safe(round(float(row["macd_hist"]), 2)),
            "stoch_k":  _safe(round(float(row["stoch_k"]),   2)),
            "stoch_d":  _safe(round(float(row["stoch_d"]),   2)),
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
        return SafeJSONResponse(content=data)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Analysis failed:\n{tb}")
        return SafeJSONResponse(status_code=500, content={"error": str(e), "detail": tb})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    base = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
