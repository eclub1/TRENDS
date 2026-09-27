"""
Multi-timeframe confluence analyzer.

Swing mode  (default): Daily / 4H / 30min via CoinGecko
Day trade mode:         4H / 1H / 15min via Binance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.fetcher import fetch_ohlc, fetch_binance_ohlc
from src.indicators import add_all_indicators
from src.trend_detector import analyze


@dataclass
class TimeframeSignal:
    label: str
    trend: str        # BULLISH / BEARISH / NEUTRAL
    score: float
    rsi: float
    ema_align: int
    key_point: str


@dataclass
class MTFResult:
    signals: List[TimeframeSignal]
    confluence: str
    confluence_score: int
    summary: str
    trade_quality: str


def _bias(df: pd.DataFrame, label: str) -> TimeframeSignal:
    try:
        df_ind = add_all_indicators(df)
        if df_ind.empty:
            raise ValueError("Empty after indicators")
        result = analyze(df_ind)
        row = df_ind.iloc[-1]
        rsi = round(float(row["rsi"]), 1)
        ema_align = int(row["ema_align"])
        trend_map = {
            "STRONG_UPTREND":"BULLISH","UPTREND":"BULLISH",
            "NEUTRAL":"NEUTRAL",
            "DOWNTREND":"BEARISH","STRONG_DOWNTREND":"BEARISH",
        }
        bias = trend_map.get(result.trend, "NEUTRAL")
        key = (
            f"EMA bullish, RSI {rsi}" if bias == "BULLISH" else
            f"EMA bearish, RSI {rsi}" if bias == "BEARISH" else
            f"Mixed signals, RSI {rsi}"
        )
        return TimeframeSignal(label=label, trend=bias, score=result.total_score,
                               rsi=rsi, ema_align=ema_align, key_point=key)
    except Exception as e:
        return TimeframeSignal(label=label, trend="NEUTRAL", score=0, rsi=50,
                               ema_align=0, key_point=f"Unavailable: {str(e)[:40]}")


def analyze_mtf(mode: str = "swing") -> MTFResult:
    """
    Fetch and analyze three timeframes based on mode.

    mode='swing'    → Daily (90d CoinGecko), 4H (30d CoinGecko), 30min (7d CoinGecko)
    mode='daytrade' → 4H (Binance), 1H (Binance), 15min (Binance)
    """
    signals = []

    if mode == "daytrade":
        configs = [
            ("4H",    lambda: fetch_binance_ohlc("4h",  200)),
            ("1H",    lambda: fetch_binance_ohlc("1h",  200)),
            ("15min", lambda: fetch_binance_ohlc("15m", 200)),
        ]
        weights = [3, 2, 1]
    else:
        configs = [
            ("Daily", lambda: fetch_ohlc(days=90)),
            ("4H",    lambda: fetch_ohlc(days=30)),
            ("30min", lambda: fetch_ohlc(days=7)),
        ]
        weights = [3, 2, 1]

    for label, fetch_fn in configs:
        try:
            df = fetch_fn()
            sig = _bias(df, label)
        except Exception as e:
            sig = TimeframeSignal(label=label, trend="NEUTRAL", score=0, rsi=50,
                                  ema_align=0, key_point=f"Fetch failed: {str(e)[:40]}")
        signals.append(sig)

    score_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
    total = sum(score_map[s.trend] * w for s, w in zip(signals, weights))

    if total >= 5:   confluence = "STRONG_BULL"
    elif total >= 2: confluence = "BULL"
    elif total <= -5:confluence = "STRONG_BEAR"
    elif total <= -2:confluence = "BEAR"
    else:            confluence = "NEUTRAL"

    bull = sum(1 for s in signals if s.trend == "BULLISH")
    bear = sum(1 for s in signals if s.trend == "BEARISH")
    trade_quality = "A" if (bull==3 or bear==3) else "B" if (bull==2 or bear==2) else "C" if (bull==1 or bear==1) else "D"

    direction = "bullish" if total > 0 else "bearish" if total < 0 else "neutral"
    tf_names  = " / ".join(s.label for s in signals)
    summary = (
        f"{mode.upper()} mode — {tf_names} confluence: {direction} (score {total:+d}/6). "
        f"Setup quality: {trade_quality}."
    )

    return MTFResult(signals=signals, confluence=confluence,
                     confluence_score=total, summary=summary, trade_quality=trade_quality)
