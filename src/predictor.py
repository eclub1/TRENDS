"""
Prediction engine with mode-aware levels.

Swing mode (daily candles):
  - Horizon: Next 1-3 days
  - SL: 0.5× ATR, TP: 1.5×/2.5×/4× ATR

Day trade mode (15min candles):
  - Horizon: Next 1-4 hours
  - SL: 0.8× ATR (slightly wider for noise), TP: 1.5×/2.5×/4× ATR
  - ATR on 15min is ~$15-40, so targets are realistic intraday levels
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class PriceLevel:
    price: float
    label: str
    strength: str


@dataclass
class Prediction:
    forecast: str
    forecast_horizon: str
    forecast_confidence: float
    current_price: float
    entry_low: float
    entry_high: float
    entry_note: str
    bias: str
    tp1: float; tp2: float; tp3: float
    tp1_pct: float; tp2_pct: float; tp3_pct: float
    stop_loss: float
    stop_pct: float
    rr1: float; rr2: float; rr3: float
    supports: List[PriceLevel] = field(default_factory=list)
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


def predict(df: pd.DataFrame, mode: str = "swing") -> Prediction:
    if len(df) < 5:
        raise ValueError("Not enough data for prediction.")

    row   = df.iloc[-1]
    price = float(row["close"])

    # ATR fallback: 0.5% of price on 15min, 2% on daily
    fallback_pct = 0.005 if mode == "daytrade" else 0.02
    atr_raw = float(row["atr"])
    atr = atr_raw if (not np.isnan(atr_raw) and atr_raw > 0) else price * fallback_pct

    # Horizon label
    horizon = "Next 1–4 hours" if mode == "daytrade" else "Next 1–3 days"

    # ── Trend forecast ────────────────────────────────────────────────────────
    scores = []
    scores.append(1 if _ema_slope(df["ema_9"],  3) > 0.1 else (-1 if _ema_slope(df["ema_9"],  3) < -0.1 else 0))
    scores.append(1 if _ema_slope(df["ema_21"], 3) > 0.05 else (-1 if _ema_slope(df["ema_21"], 3) < -0.05 else 0))
    if len(df) >= 3:
        h_now, h_prev = float(df["macd_hist"].iloc[-1]), float(df["macd_hist"].iloc[-2])
        scores.append(1 if h_now > h_prev else (-1 if h_now < h_prev else 0))
    scores.append(1 if float(row["macd"]) > 0 else -1)
    if len(df) >= 3:
        r_now, r_prev = float(df["rsi"].iloc[-1]), float(df["rsi"].iloc[-2])
        if r_now > r_prev and r_now < 70:   scores.append(1)
        elif r_now < r_prev and r_now > 30: scores.append(-1)
        else:                               scores.append(0)
    scores.append(1 if price > float(row["ema_50"]) else -1)
    scores.append(int(np.clip(float(row["ema_align"]) / 3, -1, 1)))

    total = sum(scores)
    forecast_confidence = round(abs(total) / len(scores) * 100, 1)
    forecast = "BULLISH" if total >= 3 else "BEARISH" if total <= -3 else "NEUTRAL"

    # ── Support / resistance ──────────────────────────────────────────────────
    s_highs  = _swing_highs(df["close"], window=2)
    s_lows   = _swing_lows(df["close"],  window=2)
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])

    # ── Entry zone ────────────────────────────────────────────────────────────
    dip = atr * 0.15 if mode == "daytrade" else atr * 0.2

    if forecast == "BULLISH":
        entry_low  = round(price - dip, 2)
        entry_high = round(price, 2)
        bias = "LONG"
        entry_note = f"Enter long at market ~${price:,.2f}. Acceptable dip to ${entry_low:,.2f}"
    elif forecast == "BEARISH":
        entry_low  = round(price, 2)
        entry_high = round(price + dip, 2)
        bias = "SHORT"
        entry_note = f"Enter short at market ~${price:,.2f}. Acceptable rip to ${entry_high:,.2f}"
    else:
        entry_low  = round(price - dip * 0.8, 2)
        entry_high = round(price + dip * 0.8, 2)
        bias = "WAIT"
        entry_note = "No clear edge. Wait for confirmation before entering."

    entry = entry_high if bias == "LONG" else entry_low if bias == "SHORT" else price

    # ── Stop loss ─────────────────────────────────────────────────────────────
    # Day trade: slightly wider stop (0.8×) to avoid being stopped out by noise
    # Swing:     tight stop (0.5×)
    sl_mult = 0.8 if mode == "daytrade" else 0.5

    if bias == "LONG":
        stop_loss = round(entry - atr * sl_mult, 2)
    elif bias == "SHORT":
        stop_loss = round(entry + atr * sl_mult, 2)
    else:
        stop_loss = round(entry - atr * sl_mult, 2)

    risk     = abs(entry - stop_loss)
    stop_pct = round((stop_loss - entry) / entry * 100, 2)

    # ── Take-profit targets ───────────────────────────────────────────────────
    if bias == "LONG":
        tp1 = round(entry + atr * 1.5, 2)
        tp2 = round(entry + atr * 2.5, 2)
        tp3 = round(entry + atr * 4.0, 2)
        res_above = [h for h in s_highs if h > entry + atr]
        if res_above:
            tp3 = round(max(tp3, min(res_above[0], entry + atr * 6)), 2)
    elif bias == "SHORT":
        tp1 = round(entry - atr * 1.5, 2)
        tp2 = round(entry - atr * 2.5, 2)
        tp3 = round(entry - atr * 4.0, 2)
        sup_below = [l for l in s_lows if l < entry - atr]
        if sup_below:
            tp3 = round(min(tp3, max(sup_below[-1], entry - atr * 6)), 2)
    else:
        tp1 = round(entry + atr * 1.5, 2)
        tp2 = round(entry + atr * 2.5, 2)
        tp3 = round(entry + atr * 4.0, 2)

    # ── % and R:R ─────────────────────────────────────────────────────────────
    def pct(target):
        return round((target - entry) / entry * 100, 2)

    tp1_pct = pct(tp1); tp2_pct = pct(tp2); tp3_pct = pct(tp3)

    def rr(tp):
        reward = abs(tp - entry)
        return round(reward / risk, 1) if risk > 0 else 0.0

    rr1, rr2, rr3 = rr(tp1), rr(tp2), rr(tp3)

    # ── Support & resistance levels ───────────────────────────────────────────
    ema_21 = float(row["ema_21"]); ema_50 = float(row["ema_50"]); ema_200 = float(row["ema_200"])
    support_candidates = []
    for val, label in [(ema_21,"EMA 21"),(ema_50,"EMA 50"),(ema_200,"EMA 200")]:
        if val < price:
            support_candidates.append(PriceLevel(round(val,2), label, "STRONG"))
    if bb_lower < price:
        support_candidates.append(PriceLevel(round(bb_lower,2), "BB Lower", "MODERATE"))
    for sl in s_lows[-3:]:
        if sl < price * 0.999:
            support_candidates.append(PriceLevel(sl, "Swing Low", "MODERATE"))
    supports = sorted(support_candidates, key=lambda x: x.price, reverse=True)[:4]

    resistance_candidates = []
    if bb_upper > price:
        resistance_candidates.append(PriceLevel(round(bb_upper,2), "BB Upper", "MODERATE"))
    for sh in s_highs[:3]:
        if sh > price * 1.001:
            resistance_candidates.append(PriceLevel(sh, "Swing High", "MODERATE"))
    resistance_candidates.append(PriceLevel(round(entry + 4*atr,2), f"4× ATR", "WEAK"))
    resistances = sorted(resistance_candidates, key=lambda x: x.price)[:4]

    # ── Summary ───────────────────────────────────────────────────────────────
    direction = {"BULLISH":"upward","BEARISH":"downward","NEUTRAL":"sideways"}[forecast]
    mode_label = "Day trade" if mode == "daytrade" else "Swing"
    summary = (
        f"{mode_label}: ETH forecast {direction} ({forecast_confidence:.0f}% confidence). "
        f"Entry ~${entry:,.2f}. SL ${stop_loss:,.2f} ({stop_pct:+.1f}%). "
        f"TP1 ${tp1:,.2f} ({tp1_pct:+.1f}%, R:R 1:{rr1}) | "
        f"TP2 ${tp2:,.2f} ({tp2_pct:+.1f}%, R:R 1:{rr2}) | "
        f"TP3 ${tp3:,.2f} ({tp3_pct:+.1f}%, R:R 1:{rr3})."
    )

    return Prediction(
        forecast=forecast, forecast_horizon=horizon,
        forecast_confidence=forecast_confidence,
        current_price=round(price,2),
        entry_low=entry_low, entry_high=entry_high, entry_note=entry_note,
        bias=bias,
        tp1=tp1, tp2=tp2, tp3=tp3,
        tp1_pct=tp1_pct, tp2_pct=tp2_pct, tp3_pct=tp3_pct,
        stop_loss=stop_loss, stop_pct=stop_pct,
        rr1=rr1, rr2=rr2, rr3=rr3,
        supports=supports, resistances=resistances,
        summary=summary,
    )
