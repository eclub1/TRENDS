"""
ETH price data fetcher using CoinGecko public API (no API key required).
"""

import requests
import pandas as pd
from datetime import datetime


COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_ohlc(days: int = 90) -> pd.DataFrame:
    """
    Fetch ETH OHLC candlestick data from CoinGecko.
    Granularity: 1-2 days → 30min candles, 3-30 days → 4h candles, >30 days → daily candles.

    Args:
        days: Number of days of historical data to fetch (max 365 for free tier).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close
    """
    url = f"{COINGECKO_BASE}/coins/ethereum/ohlc"
    params = {"vs_currency": "usd", "days": days}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch OHLC data: {e}")

    raw = resp.json()
    if not raw:
        raise ValueError("Empty OHLC response from CoinGecko.")

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_market_data() -> dict:
    """
    Fetch current ETH market snapshot: price, volume, market cap, 24h change.

    Returns:
        dict with current market stats.
    """
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": "ethereum",
        "order": "market_cap_desc",
        "per_page": 1,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch market data: {e}")

    data = resp.json()
    if not data:
        raise ValueError("Empty market data response.")

    coin = data[0]
    return {
        "price": coin.get("current_price", 0),
        "market_cap": coin.get("market_cap", 0),
        "volume_24h": coin.get("total_volume", 0),
        "change_1h": coin.get("price_change_percentage_1h_in_currency", 0),
        "change_24h": coin.get("price_change_percentage_24h_in_currency", 0),
        "change_7d": coin.get("price_change_percentage_7d_in_currency", 0),
        "high_24h": coin.get("high_24h", 0),
        "low_24h": coin.get("low_24h", 0),
        "last_updated": coin.get("last_updated", ""),
    }
