"""
Technical indicators engine for ETH trend analysis.
Computes EMA, RSI, MACD, Bollinger Bands, ATR, OBV, and Stochastic oscillator.
"""

import pandas as pd
import numpy as np


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price approximation (uses OHLC midpoint as typical price)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    # OHLC data from CoinGecko doesn't include volume; use cumulative avg instead
    return typical.rolling(window=20).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the OHLC DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close]

    Returns:
        DataFrame enriched with all indicator columns.
    """
    df = df.copy()
    close = df["close"]

    # Trend EMAs
    df["ema_9"] = compute_ema(close, 9)
    df["ema_21"] = compute_ema(close, 21)
    df["ema_50"] = compute_ema(close, 50)
    df["ema_200"] = compute_ema(close, 200)

    # Momentum
    df["rsi"] = compute_rsi(close, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    # Volatility
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(close)
    df["atr"] = compute_atr(df)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Oscillator
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df)

    # Price position relative to bands (0 = lower band, 1 = upper band)
    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (close - df["bb_lower"]) / band_range

    # Momentum score: rate of change
    df["roc_10"] = close.pct_change(10) * 100  # 10-period rate of change

    # EMA alignment score (-3 to +3): bullish when short EMAs > long EMAs
    df["ema_align"] = (
        (df["ema_9"] > df["ema_21"]).astype(int)
        + (df["ema_21"] > df["ema_50"]).astype(int)
        + (df["ema_50"] > df["ema_200"]).astype(int)
        - (df["ema_9"] < df["ema_21"]).astype(int)
        - (df["ema_21"] < df["ema_50"]).astype(int)
        - (df["ema_50"] < df["ema_200"]).astype(int)
    )

    # Drop rows where indicators haven't warmed up yet
    df = df.dropna(subset=["ema_200", "macd", "rsi", "stoch_k"]).reset_index(drop=True)

    return df
