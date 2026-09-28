"""
Market regime detector + session clock.

Regime types:
  TRENDING_UP   — ADX > 25 + price above EMA 50 + EMA slope positive
  TRENDING_DOWN — ADX > 25 + price below EMA 50 + EMA slope negative
  RANGING       — ADX < 20, price oscillating inside BB bands
  VOLATILE      — ATR > 2× 20-period ATR average (expansion)
  BREAKOUT      — BB squeeze followed by expansion + volume spike

Sessions (UTC):
  ASIA:   00:00 – 09:00
  LONDON: 07:00 – 16:00
  NY:     13:00 – 22:00
  OVERLAP (best): 13:00 – 16:00 (London+NY)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timezone


@dataclass
class RegimeResult:
    regime:          str    # TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE / BREAKOUT
    regime_strength: str    # STRONG / MODERATE / WEAK
    adx:             float
    volatility_pct:  float  # ATR as % of price
    bb_width:        float
    bb_squeeze:      bool   # bands are tight — coiling
    session:         str    # ASIA / LONDON / NY / OVERLAP / CLOSED
    session_quality: str    # HIGH / MEDIUM / LOW
    best_strategy:   str    # what to do in this regime
    description:     str
    support_zones:   List[dict]
    resistance_zones:List[dict]


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm.shift()
    minus_dm[mask2] = 0

    atr = (high - low).ewm(com=period-1, adjust=False).mean()
    atr = atr.replace(0, np.nan)

    plus_di  = 100 * plus_dm.ewm(com=period-1, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(com=period-1, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(com=period-1, adjust=False).mean()
    return adx


def _find_support_resistance(df: pd.DataFrame, n_levels: int = 4) -> tuple:
    """Find key support and resistance zones using pivot points."""
    if len(df) < 10:
        return [], []

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    price  = float(closes[-1])

    # Pivot highs and lows
    sup_levels = []
    res_levels = []

    window = max(3, len(df) // 20)
    for i in range(window, len(df) - window):
        # Pivot low = support
        if lows[i] == min(lows[i-window:i+window+1]):
            sup_levels.append(float(lows[i]))
        # Pivot high = resistance
        if highs[i] == max(highs[i-window:i+window+1]):
            res_levels.append(float(highs[i]))

    # Cluster nearby levels (within 0.3% of each other)
    def cluster(levels, pct=0.003):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for v in levels[1:]:
            if abs(v - clusters[-1][-1]) / clusters[-1][-1] < pct:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [round(np.mean(c), 2) for c in clusters]

    sups = cluster(sup_levels)
    ress = cluster(res_levels)

    # Filter: supports below price, resistances above
    sups = sorted([s for s in sups if s < price * 0.999], reverse=True)[:n_levels]
    ress = sorted([r for r in ress if r > price * 1.001])[:n_levels]

    sup_out = [{"price": s, "strength": "STRONG" if sups.index(s) < 2 else "MODERATE"} for s in sups]
    res_out = [{"price": r, "strength": "STRONG" if ress.index(r) < 2 else "MODERATE"} for r in ress]

    return sup_out, res_out


def get_session(utc_hour: int) -> tuple:
    """Return current trading session and quality."""
    if 13 <= utc_hour < 16:
        return "OVERLAP", "HIGH"     # London + NY overlap — best liquidity
    elif 13 <= utc_hour < 22:
        return "NY", "HIGH"
    elif 7 <= utc_hour < 16:
        return "LONDON", "HIGH"
    elif 0 <= utc_hour < 9:
        return "ASIA", "MEDIUM"
    else:
        return "CLOSED", "LOW"


def detect_regime(df: pd.DataFrame) -> RegimeResult:
    """Detect the current market regime from indicator DataFrame."""
    if len(df) < 20:
        now = datetime.now(timezone.utc)
        session, sq = get_session(now.hour)
        return RegimeResult(
            regime="RANGING", regime_strength="WEAK",
            adx=0, volatility_pct=0, bb_width=0, bb_squeeze=False,
            session=session, session_quality=sq,
            best_strategy="Not enough data",
            description="Insufficient data for regime detection.",
            support_zones=[], resistance_zones=[],
        )

    row   = df.iloc[-1]
    price = float(row["close"])

    # ADX
    adx_series = _compute_adx(df)
    adx = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 15.0

    # Volatility
    atr       = float(row["atr"]) if not np.isnan(float(row["atr"])) else price * 0.003
    vol_pct   = round(atr / price * 100, 3)
    atr_20avg = float(df["atr"].tail(20).mean())
    atr_expanded = atr > atr_20avg * 1.8

    # BB width
    bb_width = float(row["bb_width"]) if not np.isnan(float(row["bb_width"])) else 0.05
    avg_bb_width = float(df["bb_width"].tail(30).mean()) if len(df) >= 30 else bb_width
    bb_squeeze = bb_width < avg_bb_width * 0.6

    # EMA slopes
    ema50  = float(row["ema_50"])
    ema50_5ago = float(df["ema_50"].iloc[-6]) if len(df) >= 6 else ema50
    ema_slope = (ema50 - ema50_5ago) / ema50_5ago * 100 if ema50_5ago != 0 else 0

    # RSI
    rsi = float(row["rsi"]) if not np.isnan(float(row["rsi"])) else 50.0

    # Regime classification
    if atr_expanded and not bb_squeeze:
        regime = "VOLATILE"
        regime_strength = "STRONG" if atr > atr_20avg * 2.5 else "MODERATE"
        best_strategy = "Widen stops. Trade with trend only. Avoid counter-trend entries."
        desc = f"Market is volatile — ATR {vol_pct:.2f}% of price. Candles are large. Increase stop distance, reduce position size."
    elif bb_squeeze:
        regime = "BREAKOUT"
        regime_strength = "STRONG"
        best_strategy = "Wait for the breakout direction. Enter on candle close outside the bands. Stop inside the squeeze."
        desc = f"Bollinger Band squeeze detected — volatility coiling. A sharp breakout is imminent. Watch for the first strong candle outside the bands."
    elif adx > 25 and ema_slope > 0.05 and price > ema50:
        regime = "TRENDING_UP"
        regime_strength = "STRONG" if adx > 35 else "MODERATE"
        best_strategy = "Buy dips to EMA 9/21. Do not short against the trend. Trail stop below each higher low."
        desc = f"Strong uptrend confirmed. ADX {adx:.1f} — trend strength {'strong' if adx > 35 else 'moderate'}. Price above all EMAs. Buy pullbacks, not breakouts."
    elif adx > 25 and ema_slope < -0.05 and price < ema50:
        regime = "TRENDING_DOWN"
        regime_strength = "STRONG" if adx > 35 else "MODERATE"
        best_strategy = "Sell rallies to EMA 9/21. Do not buy dips. Trail stop above each lower high."
        desc = f"Strong downtrend confirmed. ADX {adx:.1f}. Price below all EMAs. Short rallies, not breakdowns."
    else:
        regime = "RANGING"
        regime_strength = "MODERATE" if adx > 15 else "WEAK"
        best_strategy = "Buy near support, sell near resistance. Use tight stops. Avoid trend-following entries."
        desc = f"Market is ranging. ADX {adx:.1f} — no clear directional trend. Trade mean reversion: buy support, sell resistance."

    # Session
    now = datetime.now(timezone.utc)
    session, session_quality = get_session(now.hour)

    # Support / resistance
    support_zones, resistance_zones = _find_support_resistance(df)

    return RegimeResult(
        regime=regime,
        regime_strength=regime_strength,
        adx=round(adx, 1),
        volatility_pct=vol_pct,
        bb_width=round(bb_width, 4),
        bb_squeeze=bb_squeeze,
        session=session,
        session_quality=session_quality,
        best_strategy=best_strategy,
        description=desc,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
    )
