"""
Pattern recognition engine.

Detects:
CANDLESTICK PATTERNS:
  - Doji (indecision)
  - Hammer / Hanging Man (reversal)
  - Bullish/Bearish Engulfing
  - Morning Star / Evening Star
  - Shooting Star
  - Three White Soldiers / Three Black Crows

CHART PATTERNS:
  - Golden Cross (EMA 50 crosses above EMA 200)
  - Death Cross (EMA 50 crosses below EMA 200)
  - Higher Highs + Higher Lows (uptrend structure)
  - Lower Highs + Lower Lows (downtrend structure)
  - Bollinger Band Squeeze (volatility coiling)
  - RSI Bullish/Bearish Divergence
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class Pattern:
    name: str
    type: str        # BULLISH / BEARISH / NEUTRAL
    description: str
    strength: str    # STRONG / MODERATE / WEAK


def _doji(o, h, l, c, threshold=0.05):
    body  = abs(c - o)
    range_ = h - l
    return range_ > 0 and body / range_ < threshold


def _hammer(o, h, l, c):
    body   = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick > body * 2 and upper_wick < body * 0.5 and body / range_ > 0.1)


def _shooting_star(o, h, l, c):
    body   = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick > body * 2 and lower_wick < body * 0.5 and body / range_ > 0.1)


def detect_patterns(df: pd.DataFrame) -> List[Pattern]:
    patterns = []
    if len(df) < 4:
        return patterns

    # Use last 4 candles
    c = df.iloc[-1]
    p1 = df.iloc[-2]
    p2 = df.iloc[-3]
    p3 = df.iloc[-4]

    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    o1, h1, l1, c1 = float(p1["open"]), float(p1["high"]), float(p1["low"]), float(p1["close"])
    o2, h2, l2, c2 = float(p2["open"]), float(p2["high"]), float(p2["low"]), float(p2["close"])

    # ── Candlestick patterns ─────────────────────────────────────────────────

    # Doji
    if _doji(o, h, l, cl):
        patterns.append(Pattern("Doji", "NEUTRAL",
            "Open and close nearly equal — market indecision. Watch for breakout direction.",
            "MODERATE"))

    # Hammer (bullish reversal after downtrend)
    if _hammer(o, h, l, cl) and cl > o:
        patterns.append(Pattern("Hammer", "BULLISH",
            "Long lower wick shows buyers rejected lower prices — potential bullish reversal.",
            "STRONG"))

    # Hanging Man (bearish after uptrend)
    if _hammer(o, h, l, cl) and cl < o:
        patterns.append(Pattern("Hanging Man", "BEARISH",
            "Hammer shape in uptrend signals sellers are emerging — watch for reversal.",
            "MODERATE"))

    # Shooting Star (bearish)
    if _shooting_star(o, h, l, cl):
        patterns.append(Pattern("Shooting Star", "BEARISH",
            "Long upper wick shows buyers were rejected at highs — bearish reversal signal.",
            "STRONG"))

    # Bullish Engulfing
    if cl > o and c1 < o1 and o <= c1 and cl >= o1:
        patterns.append(Pattern("Bullish Engulfing", "BULLISH",
            "Green candle fully engulfs previous red candle — strong bullish reversal.",
            "STRONG"))

    # Bearish Engulfing
    if cl < o and c1 > o1 and o >= c1 and cl <= o1:
        patterns.append(Pattern("Bearish Engulfing", "BEARISH",
            "Red candle fully engulfs previous green candle — strong bearish reversal.",
            "STRONG"))

    # Morning Star (3-candle bullish reversal)
    if (c2 < o2 and                             # candle -3: bearish
        abs(c1 - o1) < abs(c2 - o2) * 0.5 and  # candle -2: small body (star)
        cl > o and cl > (o2 + c2) / 2):         # candle -1: bullish, closes above midpoint
        patterns.append(Pattern("Morning Star", "BULLISH",
            "Three-candle reversal: bearish → indecision → bullish. Strong bottom signal.",
            "STRONG"))

    # Evening Star (3-candle bearish reversal)
    if (c2 > o2 and                             # candle -3: bullish
        abs(c1 - o1) < abs(c2 - o2) * 0.5 and  # candle -2: small body (star)
        cl < o and cl < (o2 + c2) / 2):         # candle -1: bearish, closes below midpoint
        patterns.append(Pattern("Evening Star", "BEARISH",
            "Three-candle reversal: bullish → indecision → bearish. Strong top signal.",
            "STRONG"))

    # Three White Soldiers (strong uptrend continuation)
    if (cl > o and c1 > o1 and c2 > o2 and
        cl > c1 > c2 and o > o1 > o2):
        patterns.append(Pattern("Three White Soldiers", "BULLISH",
            "Three consecutive rising green candles — strong bullish momentum.",
            "STRONG"))

    # Three Black Crows (strong downtrend continuation)
    if (cl < o and c1 < o1 and c2 < o2 and
        cl < c1 < c2 and o < o1 < o2):
        patterns.append(Pattern("Three Black Crows", "BEARISH",
            "Three consecutive falling red candles — strong bearish momentum.",
            "STRONG"))

    # ── Chart patterns ───────────────────────────────────────────────────────

    if len(df) >= 10:
        closes   = df["close"]
        ema_50   = df["ema_50"]
        ema_200  = df["ema_200"]
        highs    = df["high"]
        lows     = df["low"]
        rsi      = df["rsi"]
        bb_width = df["bb_width"]

        # Golden Cross
        if (float(ema_50.iloc[-1]) > float(ema_200.iloc[-1]) and
            float(ema_50.iloc[-3]) <= float(ema_200.iloc[-3])):
            patterns.append(Pattern("Golden Cross", "BULLISH",
                "EMA 50 crossed above EMA 200 — major long-term bullish signal.",
                "STRONG"))

        # Death Cross
        if (float(ema_50.iloc[-1]) < float(ema_200.iloc[-1]) and
            float(ema_50.iloc[-3]) >= float(ema_200.iloc[-3])):
            patterns.append(Pattern("Death Cross", "BEARISH",
                "EMA 50 crossed below EMA 200 — major long-term bearish signal.",
                "STRONG"))

        # Higher Highs + Higher Lows (last 5 candles)
        recent_highs = highs.iloc[-5:].values
        recent_lows  = lows.iloc[-5:].values
        if all(recent_highs[i] > recent_highs[i-1] for i in range(1, 5)) and \
           all(recent_lows[i]  > recent_lows[i-1]  for i in range(1, 5)):
            patterns.append(Pattern("Higher Highs & Higher Lows", "BULLISH",
                "Consistent HH/HL structure — classic uptrend confirmation.",
                "STRONG"))

        # Lower Highs + Lower Lows
        if all(recent_highs[i] < recent_highs[i-1] for i in range(1, 5)) and \
           all(recent_lows[i]  < recent_lows[i-1]  for i in range(1, 5)):
            patterns.append(Pattern("Lower Highs & Lower Lows", "BEARISH",
                "Consistent LH/LL structure — classic downtrend confirmation.",
                "STRONG"))

        # Bollinger Band Squeeze
        avg_width = float(bb_width.iloc[-20:].mean())
        curr_width = float(bb_width.iloc[-1])
        if curr_width < avg_width * 0.6:
            patterns.append(Pattern("Bollinger Band Squeeze", "NEUTRAL",
                f"Bands are {((1 - curr_width/avg_width)*100):.0f}% narrower than average — "
                "volatility is coiling. Expect a sharp breakout soon.",
                "STRONG"))

        # Bearish RSI divergence (price new high, RSI lower high)
        price_higher = float(closes.iloc[-1]) > float(closes.iloc[-5])
        rsi_lower    = float(rsi.iloc[-1])    < float(rsi.iloc[-5])
        if price_higher and rsi_lower and float(rsi.iloc[-1]) > 55:
            patterns.append(Pattern("Bearish RSI Divergence", "BEARISH",
                "Price making new highs but RSI declining — momentum fading, reversal risk.",
                "STRONG"))

        # Bullish RSI divergence
        price_lower = float(closes.iloc[-1]) < float(closes.iloc[-5])
        rsi_higher  = float(rsi.iloc[-1])    > float(rsi.iloc[-5])
        if price_lower and rsi_higher and float(rsi.iloc[-1]) < 45:
            patterns.append(Pattern("Bullish RSI Divergence", "BULLISH",
                "Price making new lows but RSI rising — selling momentum weakening.",
                "STRONG"))

    return patterns
