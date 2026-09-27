"""
Prediction engine: forecast trend direction, entry zones, exit targets, and stop loss.

Approach:
- Trend forecast: EMA slope momentum + MACD trajectory + RSI direction
- Support levels: recent swing lows, EMA confluence, lower Bollinger Band
- Resistance levels: recent swing highs, upper Bollinger Band, ATR projection
- Entry zone: best buy zone (support confluence + momentum confirmation)
- Take-profit targets: TP1 (conservative), TP2 (moderate), TP3 (aggressive)
- Stop loss: ATR-based below last swing low
- Risk/Reward: auto-calculated per TP level
- Confidence: weighted signal agreement
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class PriceLevel:
    price: float
    label: str
    strength: str  # WEAK / MODERATE / STRONG


@dataclass
class Prediction:
    # Trend forecast
    forecast: str           # BULLISH / BEARISH / NEUTRAL
    forecast_horizon: str   # "Next 1-3 candles" etc.
    forecast_confidence: float  # 0-100

    # Current price context
    current_price: float

    # Entry
    entry_low:  float
    entry_high: float
    entry_note: str

    # Targets
    tp1: float
    tp2: float
    tp3: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float

    # Stop loss
    stop_loss: float
    stop_pct: float

    # Risk/reward (using entry midpoint)
    rr1: float
    rr2: float
    rr3: float

    # Support / resistance
    supports:    List[PriceLevel] = field(default_factory=list)
    resistances: List[PriceLevel] = field(default_factory=list)

    # Summary
    summary: str = ""
    bias: str = ""   # LONG / SHORT / WAIT


def _swing_lows(closes: pd.Series, window: int = 3) -> List[float]:
    """Find local minima in price series."""
    lows = []
    arr = closes.values
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i - window: i + window + 1]):
            lows.append(float(arr[i]))
    return sorted(set(round(v, 2) for v in lows))


def _swing_highs(closes: pd.Series, window: int = 3) -> List[float]:
    """Find local maxima in price series."""
    highs = []
    arr = closes.values
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i - window: i + window + 1]):
            highs.append(float(arr[i]))
    return sorted(set(round(v, 2) for v in highs), reverse=True)


def _ema_slope(series: pd.Series, lookback: int = 3) -> float:
    """Percentage slope of last N values."""
    if len(series) < lookback + 1:
        return 0.0
    start = float(series.iloc[-(lookback + 1)])
    end   = float(series.iloc[-1])
    if start == 0:
        return 0.0
    return (end - start) / start * 100


def predict(df: pd.DataFrame) -> Prediction:
    """
    Run full prediction on the indicators DataFrame.
    df must come from indicators.add_all_indicators().
    """
    if len(df) < 5:
        raise ValueError("Not enough data for prediction.")

    row   = df.iloc[-1]
    close = df["close"]
    price = float(row["close"])

    # ── Trend forecast ────────────────────────────────────────────────────────
    scores = []

    # EMA 9 slope
    slope9 = _ema_slope(df["ema_9"], 3)
    scores.append(1 if slope9 > 0.2 else (-1 if slope9 < -0.2 else 0))

    # EMA 21 slope
    slope21 = _ema_slope(df["ema_21"], 3)
    scores.append(1 if slope21 > 0.1 else (-1 if slope21 < -0.1 else 0))

    # MACD histogram direction (is it growing or shrinking?)
    if len(df) >= 3:
        hist_now  = float(df["macd_hist"].iloc[-1])
        hist_prev = float(df["macd_hist"].iloc[-2])
        scores.append(1 if hist_now > hist_prev else (-1 if hist_now < hist_prev else 0))
    
    # MACD above/below zero
    scores.append(1 if float(row["macd"]) > 0 else -1)

    # RSI direction
    if len(df) >= 3:
        rsi_now  = float(df["rsi"].iloc[-1])
        rsi_prev = float(df["rsi"].iloc[-2])
        if rsi_now > rsi_prev and rsi_now < 70:
            scores.append(1)
        elif rsi_now < rsi_prev and rsi_now > 30:
            scores.append(-1)
        else:
            scores.append(0)

    # Price vs EMA 50
    scores.append(1 if price > float(row["ema_50"]) else -1)

    # EMA alignment
    scores.append(int(np.clip(float(row["ema_align"]) / 3, -1, 1)))

    total = sum(scores)
    max_score = len(scores)
    forecast_confidence = round(abs(total) / max_score * 100, 1)

    if total >= 3:
        forecast = "BULLISH"
    elif total <= -3:
        forecast = "BEARISH"
    else:
        forecast = "NEUTRAL"

    # Horizon based on data granularity (90d data = daily candles)
    forecast_horizon = "Next 1–3 days"

    # ── ATR for dynamic levels ────────────────────────────────────────────────
    atr = float(row["atr"]) if not np.isnan(float(row["atr"])) else price * 0.02

    # ── Support levels ────────────────────────────────────────────────────────
    bb_lower  = float(row["bb_lower"])
    ema_200   = float(row["ema_200"])
    ema_50    = float(row["ema_50"])
    ema_21    = float(row["ema_21"])
    swing_lows_list = _swing_lows(close, window=2)

    support_candidates = []

    # EMA supports (only below current price)
    for ema_val, label in [(ema_21, "EMA 21"), (ema_50, "EMA 50"), (ema_200, "EMA 200")]:
        if ema_val < price:
            support_candidates.append(PriceLevel(round(ema_val, 2), label, "STRONG"))

    # Bollinger lower band
    if bb_lower < price:
        support_candidates.append(PriceLevel(round(bb_lower, 2), "BB Lower", "MODERATE"))

    # Recent swing lows below price
    for sl in swing_lows_list[-4:]:
        if sl < price * 0.99:
            support_candidates.append(PriceLevel(sl, "Swing Low", "MODERATE"))

    # Sort by price descending (nearest support first)
    supports = sorted(support_candidates, key=lambda x: x.price, reverse=True)[:4]

    # ── Resistance levels ─────────────────────────────────────────────────────
    bb_upper = float(row["bb_upper"])
    swing_highs_list = _swing_highs(close, window=2)

    resistance_candidates = []

    # BB upper
    if bb_upper > price:
        resistance_candidates.append(PriceLevel(round(bb_upper, 2), "BB Upper", "MODERATE"))

    # Recent swing highs above price
    for sh in swing_highs_list[:4]:
        if sh > price * 1.01:
            resistance_candidates.append(PriceLevel(sh, "Swing High", "MODERATE"))

    # ATR-based projection
    atr_target = round(price + 2 * atr, 2)
    resistance_candidates.append(PriceLevel(atr_target, "2× ATR Target", "WEAK"))

    resistances = sorted(resistance_candidates, key=lambda x: x.price)[:4]

    # ── Entry zone ────────────────────────────────────────────────────────────
    if forecast == "BULLISH":
        # Best entry: near nearest support or current price if already at support
        nearest_sup = supports[0].price if supports else price - atr
        entry_low  = round(max(nearest_sup, price - atr * 1.2), 2)
        entry_high = round(price + atr * 0.3, 2)  # slight premium ok
        entry_note = f"Buy dip toward ${entry_low:,.2f}–${entry_high:,.2f} (near {supports[0].label if supports else 'support'})"
        bias = "LONG"
    elif forecast == "BEARISH":
        # Short entry: near nearest resistance or current price
        nearest_res = resistances[0].price if resistances else price + atr
        entry_low  = round(price - atr * 0.3, 2)
        entry_high = round(min(nearest_res, price + atr * 1.2), 2)
        entry_note = f"Short near ${entry_high:,.2f} (resistance at {resistances[0].label if resistances else 'level'})"
        bias = "SHORT"
    else:
        entry_low  = round(price - atr, 2)
        entry_high = round(price + atr, 2)
        entry_note = "Wait for clearer direction before entering"
        bias = "WAIT"

    entry_mid = (entry_low + entry_high) / 2

    # ── Take-profit targets ───────────────────────────────────────────────────
    if forecast == "BULLISH":
        # TP1: nearest resistance or 1× ATR
        tp1 = round(resistances[0].price if resistances else price + atr, 2)
        tp2 = round(resistances[1].price if len(resistances) > 1 else price + 2 * atr, 2)
        tp3 = round(resistances[2].price if len(resistances) > 2 else price + 3 * atr, 2)
    elif forecast == "BEARISH":
        tp1 = round(supports[0].price if supports else price - atr, 2)
        tp2 = round(supports[1].price if len(supports) > 1 else price - 2 * atr, 2)
        tp3 = round(supports[2].price if len(supports) > 2 else price - 3 * atr, 2)
    else:
        tp1 = round(price + atr, 2)
        tp2 = round(price + 2 * atr, 2)
        tp3 = round(price + 3 * atr, 2)

    def pct_from_entry(target):
        if entry_mid == 0:
            return 0.0
        return round((target - entry_mid) / entry_mid * 100, 2)

    tp1_pct = pct_from_entry(tp1)
    tp2_pct = pct_from_entry(tp2)
    tp3_pct = pct_from_entry(tp3)

    # ── Stop loss ─────────────────────────────────────────────────────────────
    if forecast in ("BULLISH", "NEUTRAL"):
        # Stop below nearest support, at least 1× ATR below entry
        last_swing_low = supports[-1].price if supports else entry_mid - atr * 1.5
        stop_loss = round(min(last_swing_low - atr * 0.3, entry_mid - atr * 1.5), 2)
    else:
        last_swing_high = resistances[-1].price if resistances else entry_mid + atr * 1.5
        stop_loss = round(max(last_swing_high + atr * 0.3, entry_mid + atr * 1.5), 2)

    stop_pct = round((stop_loss - entry_mid) / entry_mid * 100, 2)

    # ── Risk/Reward ───────────────────────────────────────────────────────────
    risk = abs(entry_mid - stop_loss)

    def rr(tp):
        reward = abs(tp - entry_mid)
        return round(reward / risk, 2) if risk > 0 else 0.0

    rr1 = rr(tp1)
    rr2 = rr(tp2)
    rr3 = rr(tp3)

    # ── Summary ───────────────────────────────────────────────────────────────
    direction_word = {"BULLISH": "upward", "BEARISH": "downward", "NEUTRAL": "sideways"}[forecast]
    summary = (
        f"ETH is forecasted to move {direction_word} with {forecast_confidence:.0f}% confidence. "
        f"{'Enter long' if bias == 'LONG' else 'Enter short' if bias == 'SHORT' else 'Wait'} "
        f"between ${entry_low:,.2f}–${entry_high:,.2f}. "
        f"TP1 ${tp1:,.2f} ({tp1_pct:+.1f}%), TP2 ${tp2:,.2f} ({tp2_pct:+.1f}%), TP3 ${tp3:,.2f} ({tp3_pct:+.1f}%). "
        f"Stop loss at ${stop_loss:,.2f} ({stop_pct:+.1f}%). "
        f"Best R:R = 1:{rr2:.1f} at TP2."
    )

    return Prediction(
        forecast=forecast,
        forecast_horizon=forecast_horizon,
        forecast_confidence=forecast_confidence,
        current_price=round(price, 2),
        entry_low=entry_low,
        entry_high=entry_high,
        entry_note=entry_note,
        tp1=tp1, tp2=tp2, tp3=tp3,
        tp1_pct=tp1_pct, tp2_pct=tp2_pct, tp3_pct=tp3_pct,
        stop_loss=stop_loss,
        stop_pct=stop_pct,
        rr1=rr1, rr2=rr2, rr3=rr3,
        supports=supports,
        resistances=resistances,
        summary=summary,
        bias=bias,
    )
