"""
ETH Trend AI — FastAPI web server v4.0
Endpoints:
  GET  /              → dashboard
  GET  /health        → health check
  GET  /api/analysis  → full analysis JSON (trend + ML + MTF + patterns + prediction)
  POST /api/chat      → Groq AI chat
  POST /api/calculate → position size calculator
"""

import os, math, json, traceback, logging
import numpy as np
import requests as req_lib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.fetcher      import fetch_ohlc, fetch_market_data
from src.indicators   import add_all_indicators
from src.trend_detector import analyze
from src.predictor    import predict
from src.ml_engine    import train_and_predict
from src.multi_timeframe import analyze_mtf
from src.patterns     import detect_patterns
from src.signal_engine import generate_signal, generate_chart_signals

app = FastAPI(title="ETH Trend AI", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"


# ── Safe JSON ─────────────────────────────────────────────────────────────────
class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class SafeJSONResponse(Response):
    media_type = "application/json"
    def render(self, content) -> bytes:
        return json.dumps(content, cls=_SafeEncoder, allow_nan=False,
                          default=lambda o: None).encode("utf-8")

def _safe(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v


# ── Analysis ──────────────────────────────────────────────────────────────────
def run_analysis(days: int = 90) -> dict:
    logger.info("Fetching OHLC + market data...")
    ohlc_df = fetch_ohlc(days=days)
    market  = fetch_market_data()

    logger.info("Computing indicators...")
    df_ind = add_all_indicators(ohlc_df)

    logger.info("Running trend analysis...")
    result = analyze(df_ind)

    logger.info("Running ML engine...")
    try:
        ml = train_and_predict(df_ind)
        ml_data = {
            "direction":          ml.direction,
            "probability":        ml.probability,
            "model_accuracy":     ml.model_accuracy,
            "signal_strength":    ml.signal_strength,
            "feature_importances": ml.feature_importances,
        }
    except Exception as e:
        logger.warning(f"ML failed: {e}")
        ml_data = {"direction":"NEUTRAL","probability":50,"model_accuracy":0,"signal_strength":"WEAK","feature_importances":[]}

    logger.info("Running multi-timeframe analysis...")
    try:
        mtf = analyze_mtf()
        mtf_data = {
            "confluence":       mtf.confluence,
            "confluence_score": mtf.confluence_score,
            "trade_quality":    mtf.trade_quality,
            "summary":          mtf.summary,
            "signals": [
                {"label": s.label, "trend": s.trend, "score": round(s.score,2),
                 "rsi": s.rsi, "ema_align": s.ema_align, "key_point": s.key_point}
                for s in mtf.signals
            ],
        }
    except Exception as e:
        logger.warning(f"MTF failed: {e}")
        mtf_data = {"confluence":"NEUTRAL","confluence_score":0,"trade_quality":"D","summary":"MTF data unavailable","signals":[]}

    logger.info("Detecting patterns...")
    try:
        pats = detect_patterns(df_ind)
        patterns_data = [
            {"name": p.name, "type": p.type, "description": p.description, "strength": p.strength}
            for p in pats
        ]
    except Exception as e:
        logger.warning(f"Patterns failed: {e}")
        patterns_data = []

    logger.info("Running prediction engine...")
    try:
        pred = predict(df_ind)
        pred_data = {
            "forecast": pred.forecast, "forecast_horizon": pred.forecast_horizon,
            "forecast_confidence": pred.forecast_confidence, "bias": pred.bias,
            "current_price": pred.current_price,
            "entry_low": pred.entry_low, "entry_high": pred.entry_high, "entry_note": pred.entry_note,
            "tp1": pred.tp1, "tp2": pred.tp2, "tp3": pred.tp3,
            "tp1_pct": pred.tp1_pct, "tp2_pct": pred.tp2_pct, "tp3_pct": pred.tp3_pct,
            "stop_loss": pred.stop_loss, "stop_pct": pred.stop_pct,
            "rr1": pred.rr1, "rr2": pred.rr2, "rr3": pred.rr3,
            "summary": pred.summary,
            "supports":    [{"price":s.price,"label":s.label,"strength":s.strength} for s in pred.supports],
            "resistances": [{"price":r.price,"label":r.label,"strength":r.strength} for r in pred.resistances],
        }
    except Exception as e:
        logger.warning(f"Predictor failed: {e}")
        pred_data = {}

    logger.info("Generating real-time signals...")
    try:
        ml_dir  = ml_data.get("direction", "NEUTRAL")
        ml_prob = ml_data.get("probability", 50.0)
        sig = generate_signal(df_ind, ml_dir, ml_prob)
        chart_sigs = generate_chart_signals(df_ind)
        signal_data = {
            "action":        sig.action,
            "strength":      sig.strength,
            "price":         sig.price,
            "reason":        sig.reason,
            "confirmations": sig.confirmations,
            "total_checks":  sig.total_checks,
            "entry":         sig.entry,
            "stop_loss":     sig.stop_loss,
            "take_profit":   sig.take_profit,
            "urgency":       sig.urgency,
        }
        chart_signal_data = [
            {"time": cs.time, "price": cs.price, "action": cs.action, "strength": cs.strength}
            for cs in chart_sigs
        ]
    except Exception as e:
        logger.warning(f"Signal engine failed: {e}")
        signal_data = {"action":"HOLD","strength":"WEAK","price":0,"reason":"Signal engine error","confirmations":0,"total_checks":8,"urgency":"WAIT"}
        chart_signal_data = []

    # Build candle series
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

    logger.info(f"Done: {result.trend} | ML:{ml_data['direction']} {ml_data['probability']}% | MTF:{mtf_data['confluence']} | Patterns:{len(patterns_data)}")

    return {
        "market": market, "candles": candles,
        "trend": result.trend, "reversal_risk": result.reversal_risk,
        "confidence": round(result.confidence, 1), "score": round(result.total_score, 2),
        "recommendation": result.recommendation,
        "signals": [{"name":s.name,"value":round(s.value,2),"score":round(s.score,2),"interpretation":s.interpretation} for s in result.signals],
        "reversal_warnings": result.reversal_reasons,
        "ml": ml_data,
        "mtf": mtf_data,
        "patterns": patterns_data,
        "prediction": pred_data,
        "signal": signal_data,
        "chart_signals": chart_signal_data,
    }


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt(d: dict) -> str:
    m   = d["market"]
    p   = d.get("prediction", {})
    ml  = d.get("ml", {})
    mtf = d.get("mtf", {})
    pats = d.get("patterns", [])

    sigs = "\n".join(f"  - {s['name']}: {s['score']:+.1f} — {s['interpretation']}" for s in d["signals"])
    warns = "\n".join(f"  - {w}" for w in d["reversal_warnings"]) or "  None"
    pat_str = "\n".join(f"  - {p2['name']} ({p2['type']}, {p2['strength']}): {p2['description']}" for p2 in pats) or "  None detected"
    mtf_str = "\n".join(f"  - {s['label']}: {s['trend']} (RSI {s['rsi']}) — {s['key_point']}" for s in mtf.get("signals", []))
    sup_str = "\n".join(f"  - ${s['price']:,.2f} ({s['label']})" for s in p.get("supports", []))
    res_str = "\n".join(f"  - ${r['price']:,.2f} ({r['label']})" for r in p.get("resistances", []))

    return f"""You are an advanced crypto market analyst AI embedded in the ETH Trend AI platform.
You have access to real-time Ethereum market data, technical indicators, machine learning predictions, multi-timeframe analysis, and pattern recognition. Use all of it to give precise, actionable answers.

═══ LIVE ETH DATA ═══
Price: ${m['price']:,.2f} | 24h: {m['change_24h']:+.2f}% | 7d: {m['change_7d']:+.2f}%
High/Low 24h: ${m['high_24h']:,.2f} / ${m['low_24h']:,.2f} | Volume: ${m['volume_24h']:,.0f}

═══ TREND ANALYSIS ═══
Trend: {d['trend'].replace('_',' ')} | Risk: {d['reversal_risk']} | Confidence: {d['confidence']}% | Score: {d['score']:+.2f}/10
Recommendation: {d['recommendation']}

Signals:
{sigs}

Reversal Warnings:
{warns}

═══ ML ENGINE (Random Forest + Gradient Boosting) ═══
Prediction: {ml.get('direction','N/A')} | Confidence: {ml.get('probability',0):.1f}% | Signal Strength: {ml.get('signal_strength','N/A')}
Model Accuracy (cross-val): {ml.get('model_accuracy',0):.1f}%
Top features driving prediction: {', '.join(f["feature"] for f in ml.get('feature_importances',[])[:3])}

═══ MULTI-TIMEFRAME CONFLUENCE ═══
Overall: {mtf.get('confluence','N/A')} | Score: {mtf.get('confluence_score',0):+d}/6 | Trade Quality: {mtf.get('trade_quality','N/A')}
{mtf_str}
Summary: {mtf.get('summary','')}

═══ DETECTED PATTERNS ═══
{pat_str}

═══ PREDICTION ENGINE ═══
Forecast: {p.get('forecast','N/A')} ({p.get('forecast_horizon','')}) — {p.get('forecast_confidence',0):.0f}% confidence
Bias: {p.get('bias','N/A')}
Entry Zone: ${p.get('entry_low',0):,.2f} – ${p.get('entry_high',0):,.2f}
TP1: ${p.get('tp1',0):,.2f} ({p.get('tp1_pct',0):+.1f}%) R:R 1:{p.get('rr1',0):.1f}
TP2: ${p.get('tp2',0):,.2f} ({p.get('tp2_pct',0):+.1f}%) R:R 1:{p.get('rr2',0):.1f}
TP3: ${p.get('tp3',0):,.2f} ({p.get('tp3_pct',0):+.1f}%) R:R 1:{p.get('rr3',0):.1f}
Stop Loss: ${p.get('stop_loss',0):,.2f} ({p.get('stop_pct',0):+.1f}%)

Support Levels:
{sup_str}
Resistance Levels:
{res_str}

Summary: {p.get('summary','')}

═══ INSTRUCTIONS ═══
- Always reference actual values from the data above when answering.
- When the ML, MTF, and technical signals all agree = high conviction answer.
- When they disagree = explain the conflict honestly and what to watch for.
- For entry/exit questions: give the exact levels from the Prediction Engine.
- Always end answers about trades with: "This is not financial advice — always manage your risk."
- Keep answers focused: 3-6 sentences unless a detailed explanation is asked.
- Today: 2026-09-27.
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/analysis")
async def api_analysis(days: int = 90):
    try:
        return SafeJSONResponse(content=run_analysis(days))
    except Exception as e:
        logger.error(traceback.format_exc())
        return SafeJSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat")
async def api_chat(request: Request):
    if not GROQ_API_KEY:
        return SafeJSONResponse(status_code=503, content={"error": "GROQ_API_KEY not configured."})
    body = await request.json()
    user_message = body.get("message", "").strip()
    history      = body.get("history", [])
    if not user_message:
        return SafeJSONResponse(status_code=400, content={"error": "Empty message."})
    try:
        analysis = run_analysis(days=90)
        system_prompt = _build_system_prompt(analysis)
    except Exception:
        system_prompt = "You are an expert crypto analyst for Ethereum. Answer clearly and precisely."

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 600, "temperature": 0.35},
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        return SafeJSONResponse(content={"reply": reply})
    except Exception as e:
        logger.error(traceback.format_exc())
        return SafeJSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/calculate")
async def api_calculate(request: Request):
    """
    Position size calculator.
    Input JSON:
      { "balance": 1000, "risk_pct": 2 }   (risk_pct = % of balance to risk per trade)
    Uses current prediction (entry, SL, TP1/2/3) from cached analysis.
    Returns full trade plan with lot size, dollar risk, profit targets, recommendations.
    """
    body = await request.json()
    balance  = float(body.get("balance", 0))
    risk_pct = float(body.get("risk_pct", 2))   # default 2% risk per trade

    if balance <= 0:
        return SafeJSONResponse(status_code=400, content={"error": "Balance must be greater than 0."})
    if risk_pct <= 0 or risk_pct > 100:
        return SafeJSONResponse(status_code=400, content={"error": "Risk % must be between 0.1 and 100."})

    try:
        analysis = run_analysis(days=90)
        pred = analysis.get("prediction", {})
        market = analysis.get("market", {})

        entry     = float(pred.get("entry_high", market.get("price", 0)))
        stop_loss = float(pred.get("stop_loss", 0))
        tp1       = float(pred.get("tp1", 0))
        tp2       = float(pred.get("tp2", 0))
        tp3       = float(pred.get("tp3", 0))
        bias      = pred.get("bias", "WAIT")
        forecast  = pred.get("forecast", "NEUTRAL")

        if entry <= 0 or stop_loss <= 0:
            return SafeJSONResponse(status_code=500, content={"error": "Prediction data not available yet. Try refreshing."})

        # ── Core calculations ─────────────────────────────────────────────────
        risk_dollar   = round(balance * risk_pct / 100, 2)       # $ you're willing to lose
        risk_per_eth  = abs(entry - stop_loss)                    # $ risk per 1 ETH
        lot_size      = round(risk_dollar / risk_per_eth, 6) if risk_per_eth > 0 else 0  # ETH to buy
        position_value = round(lot_size * entry, 2)               # total $ in trade
        position_pct   = round(position_value / balance * 100, 1) # % of account used

        # Profit at each TP
        def profit(tp):
            return round(lot_size * abs(tp - entry), 2)
        def roi(tp):
            return round(profit(tp) / balance * 100, 2)

        tp1_profit = profit(tp1); tp1_roi = roi(tp1)
        tp2_profit = profit(tp2); tp2_roi = roi(tp2)
        tp3_profit = profit(tp3); tp3_roi = roi(tp3)

        # New balance at each TP
        bal_tp1 = round(balance + tp1_profit, 2)
        bal_tp2 = round(balance + tp2_profit, 2)
        bal_tp3 = round(balance + tp3_profit, 2)
        bal_sl  = round(balance - risk_dollar, 2)

        # ── Recommendation ────────────────────────────────────────────────────
        if bias == "WAIT" or forecast == "NEUTRAL":
            action = "⏳ WAIT — No clear trade setup right now. Hold your capital."
        elif bias == "LONG":
            action = f"🟢 BUY {lot_size} ETH at ~${entry:,.2f}"
        else:
            action = f"🔴 SELL/SHORT {lot_size} ETH at ~${entry:,.2f}"

        # Risk warning
        if risk_pct > 5:
            risk_warning = f"⚠ You are risking {risk_pct}% per trade. Professional traders risk 1-2% max. Consider reducing risk."
        elif risk_pct > 3:
            risk_warning = f"⚡ {risk_pct}% risk is above conservative levels (1-2%). Acceptable for experienced traders."
        else:
            risk_warning = f"✅ {risk_pct}% risk is within safe limits (1-2% recommended)."

        # Max trades before account blown (at this risk %)
        max_losses = math.floor(math.log(0.5) / math.log(1 - risk_pct / 100)) if risk_pct < 100 else 1

        return SafeJSONResponse(content={
            # Inputs
            "balance":        balance,
            "risk_pct":       risk_pct,
            "risk_dollar":    risk_dollar,
            # Trade setup
            "entry":          entry,
            "stop_loss":      stop_loss,
            "bias":           bias,
            "forecast":       forecast,
            "action":         action,
            # Position
            "lot_size":       lot_size,
            "position_value": position_value,
            "position_pct":   position_pct,
            "risk_per_eth":   round(risk_per_eth, 2),
            # Targets
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "tp1_profit": tp1_profit, "tp2_profit": tp2_profit, "tp3_profit": tp3_profit,
            "tp1_roi": tp1_roi, "tp2_roi": tp2_roi, "tp3_roi": tp3_roi,
            "bal_tp1": bal_tp1, "bal_tp2": bal_tp2, "bal_tp3": bal_tp3,
            "bal_sl": bal_sl,
            # Risk info
            "risk_warning": risk_warning,
            "max_losses_before_50pct_drawdown": max_losses,
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return SafeJSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
