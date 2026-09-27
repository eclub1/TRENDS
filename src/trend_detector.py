"""
Trend detection and reversal prediction engine.

Uses a multi-signal scoring system combining:
- EMA alignment (structural trend)
- MACD crossover (momentum shift)
- RSI (overbought/oversold)
- Bollinger Band squeeze/expansion (volatility breakout)
- Stochastic oscillator (short-term momentum)
- Rate of change (price momentum)

Each signal contributes a score. The aggregate score determines:
  STRONG_UPTREND, UPTREND, NEUTRAL, DOWNTREND, STRONG_DOWNTREND

Reversal detection watches for divergence between price and momentum,
MACD zero-line crosses, and RSI extreme exits.
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class TrendSignal:
    name: str
    value: float
    score: float          # positive = bullish, negative = bearish
    interpretation: str


@dataclass
class TrendResult:
    trend: str            # STRONG_UPTREND / UPTREND / NEUTRAL / DOWNTREND / STRONG_DOWNTREND
    reversal_risk: str    # LOW / MEDIUM / HIGH / CRITICAL
    confidence: float     # 0-100
    total_score: float    # raw aggregate score
    signals: List[TrendSignal] = field(default_factory=list)
    reversal_reasons: List[str] = field(default_factory=list)
    recommendation: str = ""


def _classify_trend(score: float) -> str:
    if score >= 4:
        return "STRONG_UPTREND"
    elif score >= 1.5:
        return "UPTREND"
    elif score <= -4:
        return "STRONG_DOWNTREND"
    elif score <= -1.5:
        return "DOWNTREND"
    else:
        return "NEUTRAL"


def _score_ema_alignment(row: pd.Series) -> TrendSignal:
    align = row["ema_align"]  # -3 to +3
    score = align * 0.8  # weight: ±2.4 max
    if align == 3:
        interp = "All EMAs bullishly stacked (9>21>50>200)"
    elif align == 2:
        interp = "Mostly bullish EMA structure"
    elif align == 1:
        interp = "Slight bullish EMA lean"
    elif align == 0:
        interp = "Mixed EMA signals — no clear trend"
    elif align == -1:
        interp = "Slight bearish EMA lean"
    elif align == -2:
        interp = "Mostly bearish EMA structure"
    else:
        interp = "All EMAs bearishly stacked (9<21<50<200)"
    return TrendSignal("EMA Alignment", align, score, interp)


def _score_macd(row: pd.Series) -> TrendSignal:
    macd = row["macd"]
    hist = row["macd_hist"]
    # MACD above zero = bullish, histogram positive and growing = momentum
    if macd > 0 and hist > 0:
        score = 1.5
        interp = f"MACD bullish: line above zero, histogram positive ({hist:.2f})"
    elif macd > 0 and hist < 0:
        score = 0.3
        interp = f"MACD above zero but histogram weakening — momentum fading"
    elif macd < 0 and hist < 0:
        score = -1.5
        interp = f"MACD bearish: line below zero, histogram negative ({hist:.2f})"
    elif macd < 0 and hist > 0:
        score = -0.3
        interp = f"MACD below zero but histogram recovering — watch for crossover"
    else:
        score = 0
        interp = "MACD near zero — transition zone"
    return TrendSignal("MACD", macd, score, interp)


def _score_rsi(row: pd.Series) -> TrendSignal:
    rsi = row["rsi"]
    if rsi >= 70:
        score = -1.0
        interp = f"RSI overbought ({rsi:.1f}) — reversal risk elevated"
    elif rsi >= 60:
        score = 0.8
        interp = f"RSI strong bullish zone ({rsi:.1f})"
    elif rsi >= 50:
        score = 0.3
        interp = f"RSI above midline ({rsi:.1f}) — mild bullish bias"
    elif rsi >= 40:
        score = -0.3
        interp = f"RSI below midline ({rsi:.1f}) — mild bearish bias"
    elif rsi >= 30:
        score = -0.8
        interp = f"RSI weak bearish zone ({rsi:.1f})"
    else:
        score = 1.0  # oversold = potential bounce
        interp = f"RSI oversold ({rsi:.1f}) — bounce watch, reversal possible"
    return TrendSignal("RSI", rsi, score, interp)


def _score_bollinger(row: pd.Series) -> TrendSignal:
    bb_pct = row["bb_pct"]
    bb_width = row["bb_width"]
    squeeze = bb_width < 0.05  # tight bands = coiling for breakout

    if squeeze:
        score = 0
        interp = f"BB squeeze detected (width={bb_width:.3f}) — breakout imminent, direction unclear"
    elif bb_pct > 0.9:
        score = -0.8
        interp = f"Price near upper BB ({bb_pct:.0%}) — overextended, pullback risk"
    elif bb_pct > 0.6:
        score = 0.8
        interp = f"Price in upper BB zone ({bb_pct:.0%}) — bullish momentum"
    elif bb_pct > 0.4:
        score = 0
        interp = f"Price at BB midpoint ({bb_pct:.0%}) — neutral"
    elif bb_pct > 0.1:
        score = -0.8
        interp = f"Price in lower BB zone ({bb_pct:.0%}) — bearish momentum"
    else:
        score = 0.5
        interp = f"Price near lower BB ({bb_pct:.0%}) — oversold, possible bounce"
    return TrendSignal("Bollinger Bands", bb_pct, score, interp)


def _score_stochastic(row: pd.Series) -> TrendSignal:
    k = row["stoch_k"]
    d = row["stoch_d"]
    if k > 80:
        score = -0.5
        interp = f"Stochastic overbought (K={k:.1f}) — momentum peak"
    elif k < 20:
        score = 0.5
        interp = f"Stochastic oversold (K={k:.1f}) — momentum trough"
    elif k > d and k > 50:
        score = 0.7
        interp = f"Stochastic bullish crossover in positive zone (K={k:.1f})"
    elif k < d and k < 50:
        score = -0.7
        interp = f"Stochastic bearish crossover in negative zone (K={k:.1f})"
    else:
        score = 0
        interp = f"Stochastic neutral (K={k:.1f}, D={d:.1f})"
    return TrendSignal("Stochastic", k, score, interp)


def _score_roc(row: pd.Series) -> TrendSignal:
    roc = row["roc_10"]
    if roc > 10:
        score = 1.0
        interp = f"Strong upward momentum +{roc:.1f}% over 10 periods"
    elif roc > 3:
        score = 0.5
        interp = f"Moderate positive momentum +{roc:.1f}%"
    elif roc > -3:
        score = 0
        interp = f"Flat momentum ({roc:.1f}%)"
    elif roc > -10:
        score = -0.5
        interp = f"Moderate negative momentum {roc:.1f}%"
    else:
        score = -1.0
        interp = f"Strong downward momentum {roc:.1f}% over 10 periods"
    return TrendSignal("Rate of Change", roc, score, interp)


def _detect_reversals(df: pd.DataFrame, row: pd.Series, signals: List[TrendSignal]) -> tuple:
    """
    Detect potential reversal conditions. Returns (risk_level, reasons).
    """
    reasons = []
    risk_points = 0

    # --- Bearish divergence: price making higher high but RSI making lower high ---
    if len(df) >= 10:
        recent = df.tail(10)
        price_trend = recent["close"].iloc[-1] > recent["close"].iloc[0]
        rsi_trend = recent["rsi"].iloc[-1] > recent["rsi"].iloc[0]
        if price_trend and not rsi_trend and row["rsi"] > 55:
            reasons.append("Bearish RSI divergence: price rising but RSI falling")
            risk_points += 2

    # --- Bullish divergence: price making lower low but RSI recovering ---
        if not price_trend and rsi_trend and row["rsi"] < 45:
            reasons.append("Bullish RSI divergence: price falling but RSI recovering")
            risk_points += 2

    # --- MACD zero-line cross imminent ---
    macd = row["macd"]
    hist = row["macd_hist"]
    if abs(macd) < 5 and abs(hist) > abs(macd) * 0.5:
        reasons.append(f"MACD near zero-line crossing (macd={macd:.2f})")
        risk_points += 1

    # --- EMA crossover imminent ---
    ema9_ema21_gap = abs(row["ema_9"] - row["ema_21"]) / row["close"] * 100
    if ema9_ema21_gap < 0.3:
        reasons.append(f"EMA 9/21 crossover imminent (gap={ema9_ema21_gap:.2f}%)")
        risk_points += 1

    # --- RSI extreme exit (leaving overbought/oversold) ---
    if len(df) >= 3:
        prev_rsi = df["rsi"].iloc[-2]
        curr_rsi = row["rsi"]
        if prev_rsi >= 70 and curr_rsi < 70:
            reasons.append("RSI exiting overbought zone — bearish signal")
            risk_points += 2
        elif prev_rsi <= 30 and curr_rsi > 30:
            reasons.append("RSI exiting oversold zone — bullish signal")
            risk_points += 2

    # --- Bollinger squeeze breakout ---
    if row["bb_width"] < 0.04:
        reasons.append("Bollinger Band squeeze: volatility coiling, expect breakout soon")
        risk_points += 1

    # --- Stochastic overbought/oversold extremes ---
    if row["stoch_k"] > 85:
        reasons.append(f"Stochastic extreme overbought ({row['stoch_k']:.1f})")
        risk_points += 1
    elif row["stoch_k"] < 15:
        reasons.append(f"Stochastic extreme oversold ({row['stoch_k']:.1f})")
        risk_points += 1

    if risk_points == 0:
        risk_level = "LOW"
    elif risk_points <= 2:
        risk_level = "MEDIUM"
    elif risk_points <= 4:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return risk_level, reasons


def _make_recommendation(trend: str, reversal_risk: str, row: pd.Series) -> str:
    rsi = row["rsi"]
    if trend == "STRONG_UPTREND" and reversal_risk in ("LOW", "MEDIUM"):
        return "Trend is strong and intact. Momentum favors longs. Watch for reversal signals before exiting."
    elif trend == "UPTREND" and reversal_risk == "HIGH":
        return "Uptrend showing fatigue. Consider tightening stops or partial profit-taking."
    elif trend in ("UPTREND", "STRONG_UPTREND") and reversal_risk == "CRITICAL":
        return "Strong reversal signals in an uptrend. High probability of pullback — caution advised."
    elif trend == "STRONG_DOWNTREND" and reversal_risk in ("LOW", "MEDIUM"):
        return "Downtrend dominant. Avoid longs. Bearish bias should persist."
    elif trend == "DOWNTREND" and reversal_risk == "HIGH":
        return "Downtrend showing signs of exhaustion. Watch for bullish reversal confirmation."
    elif trend in ("DOWNTREND", "STRONG_DOWNTREND") and reversal_risk == "CRITICAL":
        return "Extreme oversold conditions in downtrend. Potential reversal bounce — watch closely."
    elif trend == "NEUTRAL":
        if rsi > 55:
            return "Neutral trend with mild bullish tilt. Wait for clearer directional signal."
        elif rsi < 45:
            return "Neutral trend with mild bearish tilt. No high-conviction entry."
        else:
            return "Market consolidating. No strong directional bias. Stay patient."
    return "Monitor closely. Mixed signals present."


def analyze(df: pd.DataFrame) -> TrendResult:
    """
    Run full trend analysis on the indicators DataFrame.

    Args:
        df: DataFrame output from indicators.add_all_indicators()

    Returns:
        TrendResult with trend classification, reversal risk, signals, and recommendation.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — cannot analyze.")

    row = df.iloc[-1]

    # Score each signal
    signals = [
        _score_ema_alignment(row),
        _score_macd(row),
        _score_rsi(row),
        _score_bollinger(row),
        _score_stochastic(row),
        _score_roc(row),
    ]

    total_score = sum(s.score for s in signals)
    trend = _classify_trend(total_score)

    # Reversal detection
    reversal_risk, reversal_reasons = _detect_reversals(df, row, signals)

    # Confidence: how many signals agree with the trend direction
    expected_sign = 1 if "UPTREND" in trend else (-1 if "DOWNTREND" in trend else 0)
    if expected_sign != 0:
        agreeing = sum(1 for s in signals if np.sign(s.score) == expected_sign)
        confidence = (agreeing / len(signals)) * 100
    else:
        neutral = sum(1 for s in signals if abs(s.score) < 0.4)
        confidence = (neutral / len(signals)) * 100

    recommendation = _make_recommendation(trend, reversal_risk, row)

    return TrendResult(
        trend=trend,
        reversal_risk=reversal_risk,
        confidence=confidence,
        total_score=total_score,
        signals=signals,
        reversal_reasons=reversal_reasons,
        recommendation=recommendation,
    )
