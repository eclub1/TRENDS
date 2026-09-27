"""
ETH Trend AI — FastAPI web server.
Endpoints:
  GET  /              → TradingView-style dashboard
  GET  /health        → health check
  GET  /api/analysis  → JSON market analysis
  POST /api/chat      → AI chat (Groq LLaMA)
"""

import os
import math
import json
import traceback
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests as req_lib
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.fetcher import fetch_ohlc, fetch_market_data
from src.indicators import add_all_indicators
from src.trend_detector import analyze

app = FastAPI(title="ETH Trend AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"


# ── Safe JSON response ────────────────────────────────────────────────────────

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        # Patch plain Python floats inline
        for chunk in super().iterencode(obj, _one_shot):
            yield chunk


class SafeJSONResponse(Response):
    media_type = "application/json"
    def render(self, content) -> bytes:
        return json.dumps(content, cls=_SafeEncoder, allow_nan=False,
                          default=lambda o: None).encode("utf-8")


def _safe(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ── Analysis ──────────────────────────────────────────────────────────────────

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


def _build_system_prompt(analysis: dict) -> str:
    m = analysis["market"]
    sigs = "\n".join(
        f"  - {s['name']}: score {s['score']:+.1f} — {s['interpretation']}"
        for s in analysis["signals"]
    )
    warnings = "\n".join(f"  - {w}" for w in analysis["reversal_warnings"]) or "  None"

    return f"""You are an expert crypto market analyst AI embedded in the ETH Trend AI platform.
You have access to real-time Ethereum market data and technical analysis. Use it to answer user questions clearly and helpfully.

CURRENT ETH MARKET DATA:
- Price: ${m['price']:,.2f}
- 24h Change: {m['change_24h']:+.2f}%
- 7d Change: {m['change_7d']:+.2f}%
- 24h High: ${m['high_24h']:,.2f}
- 24h Low: ${m['low_24h']:,.2f}
- Volume 24h: ${m['volume_24h']:,.0f}

TREND ANALYSIS:
- Trend: {analysis['trend'].replace('_', ' ')}
- Reversal Risk: {analysis['reversal_risk']}
- Confidence: {analysis['confidence']}%
- Aggregate Score: {analysis['score']:+.2f} (scale: -10 bearish to +10 bullish)
- AI Recommendation: {analysis['recommendation']}

SIGNAL BREAKDOWN:
{sigs}

REVERSAL WARNINGS:
{warnings}

INSTRUCTIONS:
- Always ground your answers in the data above.
- Be direct and specific — reference actual indicator values.
- If asked about buying/selling, give a nuanced technical view but always remind the user this is not financial advice.
- Keep responses concise (3-5 sentences max unless a detailed explanation is asked for).
- Never make up data not present above.
- Today's date: 2026-09-27.
"""


# ── Routes ────────────────────────────────────────────────────────────────────

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
        return SafeJSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat")
async def api_chat(request: Request):
    if not GROQ_API_KEY:
        return SafeJSONResponse(status_code=503, content={"error": "GROQ_API_KEY not configured."})

    body = await request.json()
    user_message: str = body.get("message", "").strip()
    history: list   = body.get("history", [])   # [{role, content}, ...]

    if not user_message:
        return SafeJSONResponse(status_code=400, content={"error": "Empty message."})

    # Get current analysis as context (uses cache — no extra API calls)
    try:
        analysis = run_analysis(days=90)
        system_prompt = _build_system_prompt(analysis)
    except Exception:
        system_prompt = "You are an expert crypto analyst. Answer questions about Ethereum markets."

    messages = [{"role": "system", "content": system_prompt}]
    # Include recent history (last 10 turns to stay within context)
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.4,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        return SafeJSONResponse(content={"reply": reply})
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Groq API error:\n{tb}")
        return SafeJSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    base = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
