"""
Prediction engine: forecast trend direction, entry zones, exit targets, and stop loss.

Logic:
- Stop loss: 1.5× ATR below entry (tight, practical)
- TP1: 1× ATR above entry  (conservative, ~1:0.7 R:R minimum)
- TP2: 2× ATR above entry  (moderate)
- TP3: nearest resistance or 3× ATR (aggressive)
- R:R expressed as reward:risk (e.g. 1:2 means risk $1 to make $2)
- Entry zone: current price ± 0.5× ATR (realistic immediate entry)
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class PriceLevel:
    price: float
    label: str
    strength: str  # WEAK / MODERATE / STRONG


@dataclass
class Prediction:
    forecast: str            # BULLISH / BEARISH / NEUTRAL
    forecast_horizon: str
    forecast_confidence: float

    current_price: float
    entry_low:  float
    entry_high: float
    entry_note: str
    bias: str                # LONG / SHORT / WAIT

    tp1: float; tp2: float; tp3: float
    tp1_pct: float; tp2_pct: float; tp3_pct: float

    stop_loss: float
    stop_pct: float

    # R:R = reward / risk (reward-to-risk ratio, e.g. 2.1 means 1:2.1)
    rr1: float; rr2: float; rr3: float

    supports:    List[PriceLevel] = field(default_factory=list)
    resistances: List[PriceLevel] = field(default_factory=list)
    summary: str = ""


def _swing_lows(series: pd.Series, window: int = 3) -> List[float]:
    arr = series.values
    lows = []
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i - window: i + window + 1]):
            lows.append(float(arr[i]))
    return sorted(set(round(v, 2) for v in lows))


def _swing_highs(series: pd.Series, window: int = 3) -> List[float]:
    arr = series.values
    highs = []
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i - window: i + window + 1]):
            highs.append(float(arr[i]))
    return sorted(set(round(v, 2) for v in highs), reverse=True)


def _ema_slope(series: pd.Series, lookback: int = 3) -> float:
    if len(series) < lookback + 1:
        return 0.0
    start = float(series.iloc[-(lookback + 1)])
    end   = float(series.iloc[-1])
    return (end - start) / start * 100 if start != 0 else 0.0


def predict(df: pd.DataFrame) -> Prediction:
    if len(df) < 5:
        raise ValueError("Not enough data for prediction.")

    row   = df.iloc[-1]
    price = float(row["close"])

    # ── ATR (use 2% of price as fallback) ────────────────────────────────────
    atr_raw = float(row["atr"])
    atr = atr_raw if (not np.isnan(atr_raw) and atr_raw > 0) else price * 0.02

    # ── Trend forecast score ─────────────────────────────────────────────────
    scores = []
    scores.append(1 if _ema_slope(df["ema_9"],  3) > 0.2  else (-1 if _ema_slope(df["ema_9"],  3) < -0.2  else 0))
    scores.append(1 if _ema_slope(df["ema_21"], 3) > 0.1  else (-1 if _ema_slope(df["ema_21"], 3) < -0.1  else 0))

    if len(df) >= 3:
        h_now  = float(df["macd_hist"].iloc[-1])
        h_prev = float(df["macd_hist"].iloc[-2])
        scores.append(1 if h_now > h_prev else (-1 if h_now < h_prev else 0))

    scores.append(1 if float(row["macd"]) > 0 else -1)

    if len(df) >= 3:
        r_now  = float(df["rsi"].iloc[-1])
        r_prev = float(df["rsi"].iloc[-2])
        if r_now > r_prev and r_now < 70:   scores.append(1)
        elif r_now < r_prev and r_now > 30: scores.append(-1)
        else:                               scores.append(0)

    scores.append(1 if price > float(row["ema_50"]) else -1)
    scores.append(int(np.clip(float(row["ema_align"]) / 3, -1, 1)))

    total = sum(scores)
    forecast_confidence = round(abs(total) / len(scores) * 100, 1)
    forecast = "BULLISH" if total >= 3 else "BEARISH" if total <= -3 else "NEUTRAL"
    forecast_horizon = "Next 1–3 days"

    # ── Entry zone: tight band around current price ───────────────────────────
    # For a LONG: entry is current price down to 0.5× ATR below (buy the dip)
    # For a SHORT: entry is current price up to 0.5× ATR above (sell the rip)
    if forecast == "BULLISH":
        entry_low  = round(price - atr * 0.5, 2)
        entry_high = round(price, 2)
        bias = "LONG"
        entry_note = f"Enter long near current price. Ideal dip entry: ${entry_low:,.2f}"
    elif forecast == "BEARISH":
        entry_low  = round(price, 2)
        entry_high = round(price + atr * 0.5, 2)
        bias = "SHORT"
        entry_note = f"Enter short near current price. Ideal rip entry: ${entry_high:,.2f}"
    else:
        entry_low  = round(price - atr * 0.3, 2)
        entry_high = round(price + atr * 0.3, 2)
        bias = "WAIT"
        entry_note = "No clear directional edge. Wait for confirmation."

    entry_mid = (entry_low + entry_high) / 2

    # ── Stop loss: 1.5× ATR from entry mid (tight and practical) ─────────────
    if bias == "LONG":
        stop_loss = round(entry_mid - atr * 1.5, 2)
    elif bias == "SHORT":
        stop_loss = round(entry_mid + atr * 1.5, 2)
    else:
        stop_loss = round(entry_mid - atr * 1.5, 2)

    risk      = abs(entry_mid - stop_loss)          # dollar risk per unit
    stop_pct  = round((stop_loss - entry_mid) / entry_mid * 100, 2)

    # ── Take-profit targets: ATR multiples from entry ─────────────────────────
    # Also look at nearest resistance/support for realistic targets
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])
    s_highs  = _swing_highs(df["close"], window=2)
    s_lows   = _swing_lows(df["close"],  window=2)

    if bias == "LONG":
        # TP1: 1× ATR up, TP2: 2× ATR up, TP3: nearest resistance or 3× ATR
        tp1_base = round(entry_mid + atr * 1.0, 2)
        tp2_base = round(entry_mid + atr * 2.0, 2)
        tp3_base = round(entry_mid + atr * 3.0, 2)

        # Nudge TP3 toward nearest real resistance above entry
        res_above = [h for h in s_highs if h > entry_mid + atr]
        if res_above:
            tp3_base = round(min(res_above[0], tp3_base), 2)
        if bb_upper > tp2_base:
            tp3_base = round(max(tp3_base, bb_upper), 2)

        tp1, tp2, tp3 = tp1_base, tp2_base, tp3_base

    elif bias == "SHORT":
        tp1 = round(entry_mid - atr * 1.0, 2)
        tp2 = round(entry_mid - atr * 2.0, 2)
        tp3 = round(entry_mid - atr * 3.0, 2)
        sup_below = [l for l in s_lows if l < entry_mid - atr]
        if sup_below:
            tp3 = round(max(sup_below[-1], tp3), 2)

    else:
        tp1 = round(entry_mid + atr * 1.0, 2)
        tp2 = round(entry_mid + atr * 2.0, 2)
        tp3 = round(entry_mid + atr * 3.0, 2)

    # ── % from entry mid ──────────────────────────────────────────────────────
    def pct(target):
        return round((target - entry_mid) / entry_mid * 100, 2)

    tp1_pct = pct(tp1)
    tp2_pct = pct(tp2)
    tp3_pct = pct(tp3)

    # ── R:R = reward / risk (higher = better) ────────────────────────────────
    def rr(tp):
        reward = abs(tp - entry_mid)
        return round(reward / risk, 2) if risk > 0 else 0.0

    rr1, rr2, rr3 = rr(tp1), rr(tp2), rr(tp3)

    # ── Support & resistance levels ───────────────────────────────────────────
    ema_21  = float(row["ema_21"])
    ema_50  = float(row["ema_50"])
    ema_200 = float(row["ema_200"])

    support_candidates = []
    for val, label in [(ema_21, "EMA 21"), (ema_50, "EMA 50"), (ema_200, "EMA 200")]:
        if val < price:
            support_candidates.append(PriceLevel(round(val, 2), label, "STRONG"))
    if bb_lower < price:
        support_candidates.append(PriceLevel(round(bb_lower, 2), "BB Lower", "MODERATE"))
    for sl in s_lows[-3:]:
        if sl < price * 0.99:
            support_candidates.append(PriceLevel(sl, "Swing Low", "MODERATE"))
    supports = sorted(support_candidates, key=lambda x: x.price, reverse=True)[:4]

    resistance_candidates = []
    if bb_upper > price:
        resistance_candidates.append(PriceLevel(round(bb_upper, 2), "BB Upper", "MODERATE"))
    for sh in s_highs[:3]:
        if sh > price * 1.005:
            resistance_candidates.append(PriceLevel(sh, "Swing High", "MODERATE"))
    resistance_candidates.append(PriceLevel(round(entry_mid + 3 * atr, 2), "3× ATR Target", "WEAK"))
    resistances = sorted(resistance_candidates, key=lambda x: x.price)[:4]

    # ── Summary ───────────────────────────────────────────────────────────────
    direction = {"BULLISH": "upward", "BEARISH": "downward", "NEUTRAL": "sideways"}[forecast]
    summary = (
        f"ETH is forecasted to move {direction} with {forecast_confidence:.0f}% confidence. "
        f"{'Enter long' if bias == 'LONG' else 'Enter short' if bias == 'SHORT' else 'Wait'} "
        f"between ${entry_low:,.2f}–${entry_high:,.2f}. "
        f"TP1 ${tp1:,.2f} ({tp1_pct:+.1f}%, R:R 1:{rr1}), "
        f"TP2 ${tp2:,.2f} ({tp2_pct:+.1f}%, R:R 1:{rr2}), "
        f"TP3 ${tp3:,.2f} ({tp3_pct:+.1f}%, R:R 1:{rr3}). "
        f"Stop loss ${stop_loss:,.2f} ({stop_pct:+.1f}%)."
    )

    return Prediction(
        forecast=forecast,
        forecast_horizon=forecast_horizon,
        forecast_confidence=forecast_confidence,
        current_price=round(price, 2),
        entry_low=entry_low,
        entry_high=entry_high,
        entry_note=entry_note,
        bias=bias,
        tp1=tp1, tp2=tp2, tp3=tp3,
        tp1_pct=tp1_pct, tp2_pct=tp2_pct, tp3_pct=tp3_pct,
        stop_loss=stop_loss,
        stop_pct=stop_pct,
        rr1=rr1, rr2=rr2, rr3=rr3,
        supports=supports,
        resistances=resistances,
        summary=summary,
    )
