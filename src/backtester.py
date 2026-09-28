"""
Backtesting engine.

Simulates the signal engine on historical candles and computes:
- Win rate, profit factor, Sharpe ratio, max drawdown
- Per-trade log with entry/exit/P&L
- Best and worst performing hours of day
- Equity curve for charting

Strategy tested:
  BUY when signal engine fires BUY (≥2 confirmations on intraday, ≥3 on swing)
  SELL when signal engine fires SELL
  Exit: hit TP1 (1.5× ATR) or SL (0.8× ATR) — whichever comes first
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from src.indicators import add_all_indicators
from src.signal_engine import generate_signal, _safe_float


@dataclass
class Trade:
    entry_time:  str
    exit_time:   str
    direction:   str   # LONG / SHORT
    entry_price: float
    exit_price:  float
    stop_loss:   float
    take_profit: float
    pnl_pct:     float  # % gain/loss
    pnl_r:       float  # in R (risk units)
    result:      str    # WIN / LOSS / BREAKEVEN
    reason:      str    # signal reason


@dataclass
class BacktestResult:
    total_trades:    int
    win_rate:        float   # 0-100
    profit_factor:   float   # gross profit / gross loss
    sharpe_ratio:    float
    max_drawdown:    float   # % from peak
    avg_win_r:       float   # average win in R
    avg_loss_r:      float   # average loss in R
    expectancy:      float   # expected R per trade
    total_return:    float   # % total return on $1000
    best_hour:       int     # UTC hour with best win rate
    worst_hour:      int
    win_hours:       List[dict]  # {hour, win_rate, trades}
    trades:          List[dict]  # serialized trade list
    equity_curve:    List[dict]  # {time, equity}
    summary:         str


def _simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                    atr: float, sl_mult: float = 0.8, tp_mult: float = 1.5) -> Optional[Trade]:
    """Simulate a single trade: forward-walk to find TP or SL hit."""
    if entry_idx >= len(df) - 1:
        return None

    entry_row  = df.iloc[entry_idx]
    entry_price = _safe_float(entry_row["close"])
    entry_time  = str(entry_row["timestamp"])

    if atr <= 0 or entry_price <= 0:
        return None

    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    if direction == "LONG":
        stop_loss   = entry_price - sl_dist
        take_profit = entry_price + tp_dist
    else:
        stop_loss   = entry_price + sl_dist
        take_profit = entry_price - tp_dist

    # Walk forward candles looking for SL or TP hit
    for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
        row = df.iloc[j]
        high  = _safe_float(row["high"])
        low   = _safe_float(row["low"])
        ts    = str(row["timestamp"])

        if direction == "LONG":
            if low <= stop_loss:
                exit_price = stop_loss
                result = "LOSS"
                exit_time = ts
                break
            if high >= take_profit:
                exit_price = take_profit
                result = "WIN"
                exit_time = ts
                break
        else:
            if high >= stop_loss:
                exit_price = stop_loss
                result = "LOSS"
                exit_time = ts
                break
            if low <= take_profit:
                exit_price = take_profit
                result = "WIN"
                exit_time = ts
                break
    else:
        # Timed out — exit at last close
        last = df.iloc[min(entry_idx + 99, len(df) - 1)]
        exit_price = _safe_float(last["close"])
        exit_time  = str(last["timestamp"])
        result = "WIN" if (
            (direction == "LONG"  and exit_price > entry_price) or
            (direction == "SHORT" and exit_price < entry_price)
        ) else "LOSS"

    if direction == "LONG":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        pnl_r   = (exit_price - entry_price) / sl_dist
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100
        pnl_r   = (entry_price - exit_price) / sl_dist

    return Trade(
        entry_time=entry_time, exit_time=exit_time,
        direction=direction,
        entry_price=round(entry_price, 2), exit_price=round(exit_price, 2),
        stop_loss=round(stop_loss, 2), take_profit=round(take_profit, 2),
        pnl_pct=round(pnl_pct, 3), pnl_r=round(pnl_r, 3),
        result=result, reason="",
    )


def run_backtest(df_raw: pd.DataFrame, mode: str = "daytrade",
                 sl_mult: float = 0.8, tp_mult: float = 1.5) -> BacktestResult:
    """
    Run full backtest on a DataFrame of OHLC candles.

    Args:
        df_raw:   Raw OHLC DataFrame (no indicators yet)
        mode:     'swing' | 'daytrade' | 'scalp'
        sl_mult:  Stop loss multiplier of ATR
        tp_mult:  Take profit multiplier of ATR
    """
    # Add indicators
    df = add_all_indicators(df_raw)
    if len(df) < 20:
        return BacktestResult(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [],
            "Not enough data for backtest."
        )

    trades: List[Trade] = []
    in_trade = False
    last_signal_bar = -5  # prevent re-entry on same bar

    # Minimum confirmations per mode
    min_conf = 2 if mode in ("daytrade", "scalp") else 3

    for i in range(10, len(df) - 1):
        if in_trade:
            continue

        # Only generate signal if not too soon after last
        if i - last_signal_bar < 3:
            continue

        window = df.iloc[:i+1]
        sig = generate_signal(window, mode=mode)

        if sig.action in ("BUY", "SELL") and sig.confirmations >= min_conf:
            direction = "LONG" if sig.action == "BUY" else "SHORT"
            atr = _safe_float(df.iloc[i]["atr"]) or _safe_float(df.iloc[i]["close"]) * 0.003

            trade = _simulate_trade(df, i, direction, atr, sl_mult, tp_mult)
            if trade:
                trade.reason = sig.reason
                trades.append(trade)
                last_signal_bar = i

    if not trades:
        return BacktestResult(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [],
            "No signals generated during backtest period."
        )

    # ── Compute statistics ────────────────────────────────────────────────────
    wins   = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]

    win_rate      = round(len(wins) / len(trades) * 100, 1)
    gross_profit  = sum(t.pnl_r for t in wins)
    gross_loss    = abs(sum(t.pnl_r for t in losses)) or 0.001
    profit_factor = round(gross_profit / gross_loss, 2)

    avg_win_r  = round(np.mean([t.pnl_r for t in wins]),  2) if wins   else 0
    avg_loss_r = round(np.mean([t.pnl_r for t in losses]),2) if losses else 0
    expectancy = round(win_rate/100 * avg_win_r + (1 - win_rate/100) * avg_loss_r, 3)

    # Equity curve (starting at $1000, risking 1% per trade = 1R = $10)
    equity = 1000.0
    equity_curve = [{"time": trades[0].entry_time, "equity": equity}]
    for t in trades:
        equity += t.pnl_r * 10  # 1R = $10 on $1000 account at 1% risk
        equity_curve.append({"time": t.exit_time, "equity": round(equity, 2)})

    total_return = round((equity - 1000) / 1000 * 100, 2)

    # Max drawdown
    peak = 1000.0
    max_dd = 0.0
    eq = 1000.0
    for t in trades:
        eq += t.pnl_r * 10
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    max_drawdown = round(max_dd, 2)

    # Sharpe ratio (simplified: mean R / std R * sqrt(252))
    r_series = [t.pnl_r for t in trades]
    if len(r_series) > 1 and np.std(r_series) > 0:
        sharpe = round(np.mean(r_series) / np.std(r_series) * (252 ** 0.5), 2)
    else:
        sharpe = 0.0

    # Best/worst hour analysis
    hour_stats: dict = {}
    for t in trades:
        try:
            ts = pd.Timestamp(t.entry_time)
            h = ts.hour
        except Exception:
            h = 0
        if h not in hour_stats:
            hour_stats[h] = {"wins": 0, "total": 0}
        hour_stats[h]["total"] += 1
        if t.result == "WIN":
            hour_stats[h]["wins"] += 1

    win_hours = []
    for h, s in sorted(hour_stats.items()):
        if s["total"] >= 2:
            win_hours.append({
                "hour": h,
                "win_rate": round(s["wins"] / s["total"] * 100, 0),
                "trades": s["total"],
            })

    best_hour  = max(win_hours, key=lambda x: x["win_rate"])["hour"] if win_hours else 0
    worst_hour = min(win_hours, key=lambda x: x["win_rate"])["hour"] if win_hours else 0

    summary = (
        f"Backtest: {len(trades)} trades | Win rate {win_rate}% | "
        f"Profit factor {profit_factor} | Expectancy {expectancy:+.2f}R | "
        f"Max drawdown {max_drawdown}% | Total return {total_return:+.1f}% on $1000 at 1% risk."
    )

    return BacktestResult(
        total_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        expectancy=expectancy,
        total_return=total_return,
        best_hour=best_hour,
        worst_hour=worst_hour,
        win_hours=win_hours,
        trades=[{
            "entry_time":  t.entry_time,
            "exit_time":   t.exit_time,
            "direction":   t.direction,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "stop_loss":   t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_pct":     t.pnl_pct,
            "pnl_r":       t.pnl_r,
            "result":      t.result,
        } for t in trades[-50:]],  # last 50 trades
        equity_curve=equity_curve,
        summary=summary,
    )
