"""
Real-time signal engine.
Produces a current BUY / SELL / HOLD signal with exact entry/exit price,
reason, urgency, and a historical series of past signals for chart markers.

Signal logic (requires multiple confirmations to avoid noise):
BUY  when: ≥4 of these are true:
  - EMA 9 crossed above EMA 21 in last 3 candles
  - MACD histogram turning positive (was negative, now positive)
  - RSI > 50 and rising from below 50
  - Price crossed above EMA 50
  - Stochastic K crossed above D from below 30 (oversold exit)
  - BB %B recovering from below 0.2
  - ML predicts BULLISH with >60% confidence

SELL when: ≥4 of these are true:
  - EMA 9 crossed below EMA 21 in last 3 candles
  - MACD histogram turning negative
  - RSI < 50 and falling from above 50
  - Price crossed below EMA 50
  - Stochastic K crossed below D from above 70 (overbought exit)
  - BB %B dropping from above 0.8
  - ML predicts BEARISH with >60% confidence

HOLD: fewer than 4 confirmations either way.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    action: str          # BUY / SELL / HOLD
    strength: str        # STRONG / MODERATE / WEAK
    price: float
    reason: str          # human-readable explanation
    confirmations: int   # how many sub-signals agreed
    total_checks: int
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    urgency: str = "NORMAL"   # NOW / SOON / WAIT


@dataclass
class ChartSignal:
    """A past signal point for drawing on the chart."""
    time: int        # unix timestamp
    price: float
    action: str      # BUY / SELL
    strength: str


def _crossed_above(s1: pd.Series, s2: pd.Series, lookback: int = 3) -> bool:
    """True if s1 crossed above s2 within the last `lookback` candles."""
    for i in range(-lookback, 0):
        prev = s1.iloc[i-1] <= s2.iloc[i-1]
        curr = s1.iloc[i]   >  s2.iloc[i]
        if prev and curr:
            return True
    return False


def _crossed_below(s1: pd.Series, s2: pd.Series, lookback: int = 3) -> bool:
    for i in range(-lookback, 0):
        prev = s1.iloc[i-1] >= s2.iloc[i-1]
        curr = s1.iloc[i]   <  s2.iloc[i]
        if prev and curr:
            return True
    return False


def generate_signal(df: pd.DataFrame, ml_direction: str = "NEUTRAL",
                    ml_prob: float = 50.0) -> Signal:
    """Generate the current real-time signal."""
    if len(df) < 5:
        return Signal("HOLD", "WEAK", float(df["close"].iloc[-1]),
                      "Not enough data.", 0, 7)

    row  = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(row["close"])
    atr   = float(row["atr"]) if not np.isnan(float(row["atr"])) else price * 0.02

    buy_checks  = []
    sell_checks = []

    # 1. EMA 9/21 cross
    if _crossed_above(df["ema_9"], df["ema_21"], 3):
        buy_checks.append("EMA 9 crossed above EMA 21 ✓")
    if _crossed_below(df["ema_9"], df["ema_21"], 3):
        sell_checks.append("EMA 9 crossed below EMA 21 ✓")

    # 2. MACD histogram flip
    hist_now  = float(row["macd_hist"])
    hist_prev = float(prev["macd_hist"])
    if hist_prev < 0 and hist_now > 0:
        buy_checks.append("MACD histogram flipped positive ✓")
    if hist_prev > 0 and hist_now < 0:
        sell_checks.append("MACD histogram flipped negative ✓")
    # Also: histogram growing = momentum
    if hist_now > hist_prev and hist_now > 0:
        buy_checks.append("MACD momentum accelerating ✓")
    if hist_now < hist_prev and hist_now < 0:
        sell_checks.append("MACD momentum decelerating ✓")

    # 3. RSI
    rsi_now  = float(row["rsi"])
    rsi_prev = float(prev["rsi"])
    if rsi_now > 50 and rsi_prev <= 50:
        buy_checks.append("RSI crossed above 50 ✓")
    if rsi_now < 50 and rsi_prev >= 50:
        sell_checks.append("RSI crossed below 50 ✓")
    if rsi_now > 55 and rsi_now < 70 and rsi_now > rsi_prev:
        buy_checks.append(f"RSI bullish zone ({rsi_now:.0f}) and rising ✓")
    if rsi_now < 45 and rsi_now > 30 and rsi_now < rsi_prev:
        sell_checks.append(f"RSI bearish zone ({rsi_now:.0f}) and falling ✓")

    # 4. Price vs EMA 50 cross
    ema50_now  = float(row["ema_50"])
    ema50_prev = float(prev["ema_50"])
    price_prev = float(prev["close"])
    if price_prev < ema50_prev and price > ema50_now:
        buy_checks.append("Price crossed above EMA 50 ✓")
    if price_prev > ema50_prev and price < ema50_now:
        sell_checks.append("Price crossed below EMA 50 ✓")

    # 5. Stochastic
    k_now  = float(row["stoch_k"])
    k_prev = float(prev["stoch_k"])
    d_now  = float(row["stoch_d"])
    d_prev = float(prev["stoch_d"])
    if k_prev < d_prev and k_now > d_now and k_prev < 30:
        buy_checks.append(f"Stochastic bullish crossover from oversold ({k_now:.0f}) ✓")
    if k_prev > d_prev and k_now < d_now and k_prev > 70:
        sell_checks.append(f"Stochastic bearish crossover from overbought ({k_now:.0f}) ✓")

    # 6. Bollinger Band %B
    bb_pct_now  = float(row["bb_pct"])  if not np.isnan(float(row["bb_pct"]))  else 0.5
    bb_pct_prev = float(prev["bb_pct"]) if not np.isnan(float(prev["bb_pct"])) else 0.5
    if bb_pct_prev < 0.2 and bb_pct_now > 0.2:
        buy_checks.append("Price recovered from lower Bollinger Band ✓")
    if bb_pct_prev > 0.8 and bb_pct_now < 0.8:
        sell_checks.append("Price rejected at upper Bollinger Band ✓")

    # 7. ML model
    if ml_direction == "BULLISH" and ml_prob >= 60:
        buy_checks.append(f"ML model predicts BULLISH ({ml_prob:.0f}% confidence) ✓")
    if ml_direction == "BEARISH" and ml_prob >= 60:
        sell_checks.append(f"ML model predicts BEARISH ({ml_prob:.0f}% confidence) ✓")

    # 8. EMA alignment
    ema_align = float(row["ema_align"])
    if ema_align >= 2:
        buy_checks.append(f"EMA structure bullish (alignment {ema_align:.0f}/3) ✓")
    if ema_align <= -2:
        sell_checks.append(f"EMA structure bearish (alignment {ema_align:.0f}/3) ✓")

    total_checks = 8
    n_buy  = len(buy_checks)
    n_sell = len(sell_checks)

    # Decision threshold
    STRONG_THRESH   = 5
    MODERATE_THRESH = 3

    if n_buy >= MODERATE_THRESH and n_buy > n_sell:
        action = "BUY"
        strength = "STRONG" if n_buy >= STRONG_THRESH else "MODERATE"
        confirmations = n_buy
        reasons = buy_checks
        entry      = round(price, 2)
        stop_loss  = round(price - atr * 0.5, 2)
        take_profit= round(price + atr * 1.5, 2)
        urgency    = "NOW" if n_buy >= STRONG_THRESH else "SOON"
    elif n_sell >= MODERATE_THRESH and n_sell > n_buy:
        action = "SELL"
        strength = "STRONG" if n_sell >= STRONG_THRESH else "MODERATE"
        confirmations = n_sell
        reasons = sell_checks
        entry      = round(price, 2)
        stop_loss  = round(price + atr * 0.5, 2)
        take_profit= round(price - atr * 1.5, 2)
        urgency    = "NOW" if n_sell >= STRONG_THRESH else "SOON"
    else:
        action = "HOLD"
        strength = "WEAK"
        confirmations = max(n_buy, n_sell)
        reasons = buy_checks if n_buy > n_sell else sell_checks
        entry = stop_loss = take_profit = None
        urgency = "WAIT"

    reason_str = " | ".join(reasons[:4]) if reasons else "Mixed signals — no clear edge"

    return Signal(
        action=action, strength=strength, price=price,
        reason=reason_str, confirmations=confirmations,
        total_checks=total_checks, entry=entry,
        stop_loss=stop_loss, take_profit=take_profit,
        urgency=urgency,
    )


def generate_chart_signals(df: pd.DataFrame) -> List[ChartSignal]:
    """
    Walk through historical candles and mark where BUY/SELL signals occurred.
    Uses a simplified version (EMA cross + MACD flip) for chart history.
    """
    signals = []
    if len(df) < 10:
        return signals

    for i in range(5, len(df)):
        window = df.iloc[:i+1]
        row   = window.iloc[-1]
        prev  = window.iloc[-2]

        price = float(row["close"])
        ts    = int(row["timestamp"].timestamp())

        ema9_cross_up   = float(prev["ema_9"]) < float(prev["ema_21"]) and float(row["ema_9"]) > float(row["ema_21"])
        ema9_cross_down = float(prev["ema_9"]) > float(prev["ema_21"]) and float(row["ema_9"]) < float(row["ema_21"])
        macd_flip_pos   = float(prev["macd_hist"]) < 0 and float(row["macd_hist"]) > 0
        macd_flip_neg   = float(prev["macd_hist"]) > 0 and float(row["macd_hist"]) < 0

        if ema9_cross_up or macd_flip_pos:
            strength = "STRONG" if (ema9_cross_up and macd_flip_pos) else "MODERATE"
            signals.append(ChartSignal(ts, price, "BUY", strength))
        elif ema9_cross_down or macd_flip_neg:
            strength = "STRONG" if (ema9_cross_down and macd_flip_neg) else "MODERATE"
            signals.append(ChartSignal(ts, price, "SELL", strength))

    return signals
