"""
Multi-timeframe (MTF) confluence analyzer.

Fetches ETH data at three timeframes using CoinGecko:
  - Short:  7 days  → 30-min candles  (intraday momentum)
  - Medium: 30 days → 4-hour candles  (swing trend)
  - Long:   90 days → daily candles   (macro trend)

For each timeframe, computes a simplified trend bias score.
Confluence = how many timeframes agree on direction.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.fetcher import fetch_ohlc
from src.indicators import add_all_indicators
from src.trend_detector import analyze


@dataclass
class TimeframeSignal:
    label: str        # "Daily", "4H", "30min"
    days: int
    trend: str        # BULLISH / BEARISH / NEUTRAL
    score: float
    rsi: float
    ema_align: int
    key_point: str    # one-line summary


@dataclass
class MTFResult:
    signals: List[TimeframeSignal]
    confluence: str       # STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR
    confluence_score: int # -3 to +3
    summary: str
    trade_quality: str    # A / B / C / D — overall trade setup quality


def _bias_from_df(df: pd.DataFrame, label: str, days: int) -> TimeframeSignal:
    """Compute bias for a single timeframe."""
    try:
        df_ind = add_all_indicators(df)
        if df_ind.empty:
            raise ValueError("Empty after indicators")
        result = analyze(df_ind)
        row = df_ind.iloc[-1]
        rsi = round(float(row["rsi"]), 1)
        ema_align = int(row["ema_align"])

        # Map trend to BULLISH/BEARISH/NEUTRAL
        trend_map = {
            "STRONG_UPTREND": "BULLISH", "UPTREND": "BULLISH",
            "NEUTRAL": "NEUTRAL",
            "DOWNTREND": "BEARISH", "STRONG_DOWNTREND": "BEARISH",
        }
        bias = trend_map.get(result.trend, "NEUTRAL")

        # One-line key point
        if bias == "BULLISH":
            key = f"EMA stacked bullish, RSI {rsi}"
        elif bias == "BEARISH":
            key = f"EMA bearish structure, RSI {rsi}"
        else:
            key = f"Mixed signals, RSI {rsi}"

        return TimeframeSignal(
            label=label,
            days=days,
            trend=bias,
            score=result.total_score,
            rsi=rsi,
            ema_align=ema_align,
            key_point=key,
        )
    except Exception as e:
        return TimeframeSignal(
            label=label, days=days,
            trend="NEUTRAL", score=0, rsi=50, ema_align=0,
            key_point=f"Data unavailable: {str(e)[:40]}",
        )


def analyze_mtf() -> MTFResult:
    """Fetch and analyze all three timeframes."""
    timeframes = [
        (90,  "Daily"),
        (30,  "4H"),
        (7,   "30min"),
    ]

    signals = []
    for days, label in timeframes:
        try:
            df = fetch_ohlc(days=days)
            sig = _bias_from_df(df, label, days)
        except Exception as e:
            sig = TimeframeSignal(
                label=label, days=days,
                trend="NEUTRAL", score=0, rsi=50, ema_align=0,
                key_point=f"Fetch failed: {str(e)[:40]}",
            )
        signals.append(sig)

    # Confluence scoring
    scores = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
    # Weight: daily=3, 4H=2, 30min=1
    weights = [3, 2, 1]
    total = sum(scores[s.trend] * w for s, w in zip(signals, weights))

    if total >= 5:
        confluence = "STRONG_BULL"
    elif total >= 2:
        confluence = "BULL"
    elif total <= -5:
        confluence = "STRONG_BEAR"
    elif total <= -2:
        confluence = "BEAR"
    else:
        confluence = "NEUTRAL"

    # Trade quality
    bull_count = sum(1 for s in signals if s.trend == "BULLISH")
    bear_count = sum(1 for s in signals if s.trend == "BEARISH")

    if bull_count == 3 or bear_count == 3:
        trade_quality = "A"   # All 3 agree — best setup
    elif bull_count == 2 or bear_count == 2:
        trade_quality = "B"   # 2/3 agree — decent setup
    elif bull_count == 1 or bear_count == 1:
        trade_quality = "C"   # Only 1 agrees — weak setup
    else:
        trade_quality = "D"   # All neutral — no trade

    # Summary
    direction = "bullish" if total > 0 else "bearish" if total < 0 else "neutral"
    summary = (
        f"Multi-timeframe analysis shows {direction} confluence (score: {total:+d}/6). "
        f"Daily: {signals[0].trend}, 4H: {signals[1].trend}, 30min: {signals[2].trend}. "
        f"Trade setup quality: {trade_quality}."
    )

    return MTFResult(
        signals=signals,
        confluence=confluence,
        confluence_score=total,
        summary=summary,
        trade_quality=trade_quality,
    )
