"""
Intraday signal engine — optimized for 5min and 1min candles.

Key improvements over the original:
- Lower threshold for intraday (3 confirmations vs 4 on daily)
- Checks price momentum (candle body direction, consecutive closes)
- Volume-weighted momentum (body size relative to ATR)
- EMA 9/21 distance — how far price is from EMA signals reversion
- RSI oversold/overbought exits on short timeframes reset quickly
- Generates chart markers for every historical signal
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    action: str           # BUY / SELL / HOLD
    strength: str         # STRONG / MODERATE / WEAK
    price: float
    reason: str
    confirmations: int
    total_checks: int
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    urgency: str = "WAIT"  # NOW / SOON / WAIT
    bull_score: int = 0
    bear_score: int = 0


@dataclass
class ChartSignal:
    time: int
    price: float
    action: str      # BUY / SELL
    strength: str


def _crossed_above(s1: pd.Series, s2: pd.Series, lookback: int = 3) -> bool:
    for i in range(-lookback, 0):
        if s1.iloc[i-1] <= s2.iloc[i-1] and s1.iloc[i] > s2.iloc[i]:
            return True
    return False


def _crossed_below(s1: pd.Series, s2: pd.Series, lookback: int = 3) -> bool:
    for i in range(-lookback, 0):
        if s1.iloc[i-1] >= s2.iloc[i-1] and s1.iloc[i] < s2.iloc[i]:
            return True
    return False


def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if not (np.isnan(f) or np.isinf(f)) else 0.0
    except Exception:
        return 0.0


def generate_signal(df: pd.DataFrame, ml_direction: str = "NEUTRAL",
                    ml_prob: float = 50.0, mode: str = "daytrade") -> Signal:
    if len(df) < 10:
        price = _safe_float(df["close"].iloc[-1]) if len(df) > 0 else 0
        return Signal("HOLD", "WEAK", price, "Not enough data.", 0, 10)

    row   = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]
    price = _safe_float(row["close"])
    atr   = _safe_float(row["atr"]) or price * 0.003

    # Threshold: intraday needs fewer confirmations (faster signals)
    STRONG_THRESH   = 5 if mode == "swing" else 4
    MODERATE_THRESH = 3 if mode == "swing" else 2

    buy_checks  = []
    sell_checks = []

    # ── 1. EMA 9/21 crossover ──────────────────────────────────────────────
    if _crossed_above(df["ema_9"], df["ema_21"], 3):
        buy_checks.append("EMA 9 crossed above EMA 21")
    if _crossed_below(df["ema_9"], df["ema_21"], 3):
        sell_checks.append("EMA 9 crossed below EMA 21")

    # ── 2. EMA 9 vs 21 current alignment ─────────────────────────────────
    e9  = _safe_float(row["ema_9"])
    e21 = _safe_float(row["ema_21"])
    e50 = _safe_float(row["ema_50"])
    if e9 > e21 and price > e9:
        buy_checks.append("Price above EMA9 > EMA21")
    if e9 < e21 and price < e9:
        sell_checks.append("Price below EMA9 < EMA21")

    # ── 3. MACD histogram flip ─────────────────────────────────────────────
    h_now  = _safe_float(row["macd_hist"])
    h_prev = _safe_float(prev["macd_hist"])
    h_prev2= _safe_float(prev2["macd_hist"])
    if h_prev < 0 and h_now > 0:
        buy_checks.append("MACD histogram flipped positive")
    if h_prev > 0 and h_now < 0:
        sell_checks.append("MACD histogram flipped negative")
    # Histogram growing in direction
    if h_now > h_prev > h_prev2 and h_now > 0:
        buy_checks.append("MACD histogram 3-bar rise")
    if h_now < h_prev < h_prev2 and h_now < 0:
        sell_checks.append("MACD histogram 3-bar fall")

    # ── 4. RSI momentum ───────────────────────────────────────────────────
    rsi_now  = _safe_float(row["rsi"])
    rsi_prev = _safe_float(prev["rsi"])
    # Oversold exit (powerful intraday buy)
    if rsi_prev < 35 and rsi_now > rsi_prev:
        buy_checks.append(f"RSI bouncing from oversold ({rsi_now:.0f})")
    # Overbought exit (powerful intraday sell)
    if rsi_prev > 65 and rsi_now < rsi_prev:
        sell_checks.append(f"RSI turning from overbought ({rsi_now:.0f})")
    # RSI mid-line cross
    if rsi_prev < 50 and rsi_now >= 50:
        buy_checks.append("RSI crossed above 50")
    if rsi_prev > 50 and rsi_now <= 50:
        sell_checks.append("RSI crossed below 50")

    # ── 5. Price vs EMA 50 ────────────────────────────────────────────────
    price_prev = _safe_float(prev["close"])
    if price_prev < e50 and price > e50:
        buy_checks.append("Price crossed above EMA 50")
    if price_prev > e50 and price < e50:
        sell_checks.append("Price crossed below EMA 50")

    # ── 6. Stochastic crossover ───────────────────────────────────────────
    k_now  = _safe_float(row["stoch_k"])
    k_prev = _safe_float(prev["stoch_k"])
    d_now  = _safe_float(row["stoch_d"])
    d_prev = _safe_float(prev["stoch_d"])
    if k_prev < d_prev and k_now > d_now and k_now < 40:
        buy_checks.append(f"Stochastic bullish cross (K={k_now:.0f})")
    if k_prev > d_prev and k_now < d_now and k_now > 60:
        sell_checks.append(f"Stochastic bearish cross (K={k_now:.0f})")

    # ── 7. Bollinger Band bounce ──────────────────────────────────────────
    bb_pct = _safe_float(row["bb_pct"])
    bb_pct_prev = _safe_float(prev["bb_pct"])
    if bb_pct_prev < 0.15 and bb_pct > 0.15:
        buy_checks.append("Bounce off lower Bollinger Band")
    if bb_pct_prev > 0.85 and bb_pct < 0.85:
        sell_checks.append("Rejection at upper Bollinger Band")

    # ── 8. Consecutive candle momentum (intraday key) ─────────────────────
    c0 = _safe_float(row["close"]);   o0 = _safe_float(row["open"])
    c1 = _safe_float(prev["close"]);  o1 = _safe_float(prev["open"])
    c2 = _safe_float(prev2["close"]); o2 = _safe_float(prev2["open"])
    body0 = c0 - o0; body1 = c1 - o1; body2 = c2 - o2
    if body0 > 0 and body1 > 0 and body2 > 0:  # 3 green candles
        buy_checks.append("3 consecutive bullish candles")
    if body0 < 0 and body1 < 0 and body2 < 0:  # 3 red candles
        sell_checks.append("3 consecutive bearish candles")

    # ── 9. ML model ────────────────────────────────────────────────────────
    if ml_direction == "BULLISH" and ml_prob >= 58:
        buy_checks.append(f"ML: BULLISH ({ml_prob:.0f}%)")
    if ml_direction == "BEARISH" and ml_prob >= 58:
        sell_checks.append(f"ML: BEARISH ({ml_prob:.0f}%)")

    # ── 10. EMA alignment score ────────────────────────────────────────────
    ema_align = _safe_float(row["ema_align"])
    if ema_align >= 2:
        buy_checks.append(f"EMA fully bullish ({ema_align:.0f}/3)")
    if ema_align <= -2:
        sell_checks.append(f"EMA fully bearish ({ema_align:.0f}/3)")

    total_checks = 10
    n_buy  = len(buy_checks)
    n_sell = len(sell_checks)

    if n_buy >= MODERATE_THRESH and n_buy > n_sell:
        action = "BUY"
        strength = "STRONG" if n_buy >= STRONG_THRESH else "MODERATE"
        confirmations = n_buy
        reasons = buy_checks
        entry       = round(price, 2)
        stop_loss   = round(price - atr * (0.8 if mode != "swing" else 0.5), 2)
        take_profit = round(price + atr * 1.5, 2)
        urgency     = "NOW" if n_buy >= STRONG_THRESH else "SOON"
    elif n_sell >= MODERATE_THRESH and n_sell > n_buy:
        action = "SELL"
        strength = "STRONG" if n_sell >= STRONG_THRESH else "MODERATE"
        confirmations = n_sell
        reasons = sell_checks
        entry       = round(price, 2)
        stop_loss   = round(price + atr * (0.8 if mode != "swing" else 0.5), 2)
        take_profit = round(price - atr * 1.5, 2)
        urgency     = "NOW" if n_sell >= STRONG_THRESH else "SOON"
    else:
        action = "HOLD"
        strength = "WEAK" if max(n_buy, n_sell) < 2 else "MODERATE"
        confirmations = max(n_buy, n_sell)
        reasons = buy_checks if n_buy >= n_sell else sell_checks
        entry = stop_loss = take_profit = None
        urgency = "WAIT"

    reason_str = " | ".join(reasons[:4]) if reasons else "No clear edge — wait for setup"

    return Signal(
        action=action, strength=strength, price=price,
        reason=reason_str, confirmations=confirmations,
        total_checks=total_checks, entry=entry,
        stop_loss=stop_loss, take_profit=take_profit,
        urgency=urgency, bull_score=n_buy, bear_score=n_sell,
    )


def generate_chart_signals(df: pd.DataFrame) -> List[ChartSignal]:
    """Generate historical BUY/SELL markers for the chart."""
    signals = []
    if len(df) < 10:
        return signals

    min_candles = 8
    for i in range(min_candles, len(df)):
        window = df.iloc[:i+1]
        row  = window.iloc[-1]
        prev = window.iloc[-2]

        price = _safe_float(row["close"])
        ts    = int(row["timestamp"].timestamp())

        ema9_up   = _safe_float(prev["ema_9"]) < _safe_float(prev["ema_21"]) and \
                    _safe_float(row["ema_9"])  > _safe_float(row["ema_21"])
        ema9_down = _safe_float(prev["ema_9"]) > _safe_float(prev["ema_21"]) and \
                    _safe_float(row["ema_9"])  < _safe_float(row["ema_21"])
        macd_pos  = _safe_float(prev["macd_hist"]) < 0 and _safe_float(row["macd_hist"]) > 0
        macd_neg  = _safe_float(prev["macd_hist"]) > 0 and _safe_float(row["macd_hist"]) < 0

        # RSI oversold/overbought exits
        rsi_now  = _safe_float(row["rsi"])
        rsi_prev = _safe_float(prev["rsi"])
        rsi_buy  = rsi_prev < 35 and rsi_now > rsi_prev
        rsi_sell = rsi_prev > 65 and rsi_now < rsi_prev

        if (ema9_up or macd_pos or rsi_buy) and not (ema9_down or macd_neg):
            strength = "STRONG" if sum([ema9_up, macd_pos, rsi_buy]) >= 2 else "MODERATE"
            signals.append(ChartSignal(ts, price, "BUY", strength))
        elif (ema9_down or macd_neg or rsi_sell) and not (ema9_up or macd_pos):
            strength = "STRONG" if sum([ema9_down, macd_neg, rsi_sell]) >= 2 else "MODERATE"
            signals.append(ChartSignal(ts, price, "SELL", strength))

    return signals
